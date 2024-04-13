"""
model, tokenizer:
    'deberta-v3-small': PretrainedModel('deberta-v3-small', 'spm.model', 'spm'),
    'deberta-v3-base': PretrainedModel('deberta-v3-base', 'spm.model', 'spm'),
    'deberta-v3-large': PretrainedModel('deberta-v3-large', 'spm.model', 'spm'),
    'mdeberta-v3-base': PretrainedModel('mdeberta-v3-base', 'spm.model', 'spm'),
    'deberta-v3-xsmall': PretrainedModel('deberta-v3-xsmall', 'spm.model', 'spm'),
"""

from bisect import bisect
import math
import numpy as np
import random
import torch

from torch.utils.data import Dataset, DataLoader
import lightning as L
from datasets import concatenate_datasets

from ..config.config import Config
from .third.utils import NGramMaskGenerator
from .third.fetch_dataset import fetch_dataset
from .third.prep_dataset import split_sentences, tokenize

from ..config.config import Config
from .third.utils import NGramMaskGenerator


class Masker:
    def __init__(self, config:Config, tokenizer, **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        self.mask_generator = NGramMaskGenerator(tokenizer, config.mask_lm_prob, config.max_seq_len, config.max_preds_per_seq, **kwargs)

    def from_text_to_inputs(self, text, rng=random, return_origin=False, **kwargs):
        tokens = self.tokenizer.tokenize(text)
        tokens = self._truncate_tokens(tokens, self.config.max_seq_len-2)
        tokens = self._add_eos_bos(tokens)
        output = self.from_tokens_to_inputs(tokens, rng, **kwargs)
        if return_origin:
            original_inputs_ids = self.tokenizer(text, add_special_tokens=True)['input_ids']
            output['original_input_ids'] = original_inputs_ids
        return output
    
    def from_tokens_to_inputs(self, tokens, rng=random, **kwargs):
        masked_tokens, target_labels = self._generate_masked_tokens(tokens, rng, **kwargs)
        masked_input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        output = {
            'input_ids': masked_input_ids,
            'attention_mask': [1]*len(masked_input_ids),
            'labels': target_labels,
        }
        return output
    
    def _add_eos_bos(self, tokens):
        tokens.insert(0, self.tokenizer.bos_token)
        tokens.append(self.tokenizer.eos_token)
        return tokens

    def _add_ids_at_first_and_last(self, ids:list, add_eos_bos=False, fill=0):
        if add_eos_bos:
            ids.insert(0, self.tokenizer.bos_token_id)
            ids.append(self.tokenizer.eos_token_id)
        else:
            ids.insert(0, fill)
            ids.append(fill)

    def _truncate_tokens(self, tokens, max_length):
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        return tokens
    
    def _generate_masked_tokens(self, tokens, rng=random, **kwargs):
        return self.mask_generator.mask_tokens(tokens, rng, **kwargs)


class ReplaceTaskPrepare:
    def __init__(self, config, tokenizer, **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        self.masker = Masker(config, tokenizer, **kwargs)

    @staticmethod
    def collate_fn(batch):
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }

    def get_generator_inputs(self, text=None, tokens=None, rng=random, **kwargs):
        if text is not None:
            masked_data = self.masker.from_text_to_inputs(text, rng, **kwargs)
        elif tokens is not None:
            masked_data = self.masker.from_tokens_to_inputs(tokens, rng, **kwargs)
        else:
            ValueError("Please provide either text or tokens")
        return masked_data
    
    def get_discriminator_inputs(self, masked_data, logits, is_stochastic=False, rng=random, **kwargs):
        masked_input_ids = self._replace_masked_tokens(masked_data, logits, is_stochastic, **kwargs)
        masked_data['input_ids'] = masked_input_ids
        new_labels = self._check_labels(masked_data)
        masked_data['labels'] = new_labels
        return masked_data
    
    def _replace_masked_tokens(self, masked_data, logits, is_stochastic=False, rng=random, **kwargs):
        masked_input_ids = masked_data['input_ids']
        if is_stochastic:
            probs = torch.softmax(logits, dim=-1)
            sampled_ids = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            sampled_ids = torch.argmax(logits, dim=-1)
        masked_input_ids = torch.where(masked_data['labels'] > 0, sampled_ids, masked_input_ids)
        return masked_input_ids

    def _check_labels(self, masked_data):
        masked_input_ids = masked_data['input_ids']
        labels = masked_data['labels']
        new_labels = torch.zeros_like(labels)
        new_labels = torch.where(labels > 0, labels != masked_input_ids, new_labels)
        return new_labels



class LMDataModule(L.LightningDataModule):
    def __init__(self, config:Config, tokenizer, **kwargs):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.mask_generator = NGramMaskGenerator(tokenizer, config.mask_lm_prob, config.max_seq_len, config.max_preds_per_seq, **kwargs)
        self.train_dataset = None
        self.dataset_names = ['wikipedia', 'bookcorpus']

    def prepare_data(self) -> None:
        for dataset_name in self.dataset_names:
            fetch_dataset(dataset_name)

    def setup(self, stage='train'):
        datasets = [fetch_dataset(dataset_name)['train'] for dataset_name in self.dataset_names]
        for idx, dataset in enumerate(datasets):
            dataset = dataset.map(
                split_sentences,
                batched=True,
                num_proc=12,
                remove_columns=dataset.column_names)
            datasets[idx] = dataset
        self.train_dataset = concatenate_datasets(datasets)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            collate_fn=self.get_generator_input_collate_fn(),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,)
    
    def get_generator_input_collate_fn(self, rng=random, **kwargs):
        def preprocess_per_sample(sample):
            sample = self.tokenizer.convert_ids_to_tokens(sample)
            masked_sample, target_labels = self.mask_generator.mask_tokens(sample, rng, **kwargs)
            masked_sample = self.tokenizer.convert_tokens_to_ids(masked_sample)
            return {
                'input_ids': masked_sample,
                'labels': target_labels,
                }
        def collate_fn(batch):
            batch = [sample['text'] for sample in batch]
            batch = tokenize(batch, self.config.max_seq_len)['input_ids']
            batch = list(map(preprocess_per_sample, batch))
            batch_input_ids = {'input_ids': [x['input_ids'] for x in batch]}
            batch_labels = {'input_ids': [x['labels'] for x in batch]}
            batch_input_ids = self.tokenizer.pad(
                batch_input_ids,
                padding='longest',
                return_tensors='pt',
                return_attention_mask=True)
            batch_labels = self.tokenizer.pad(
                batch_labels,
                padding='longest',
                return_tensors='pt',
                return_attention_mask=False)
            return {
                'input_ids': batch_input_ids['input_ids'],
                'attention_mask': batch_input_ids['attention_mask'],
                'labels': batch_labels['input_ids'],
            }
        return collate_fn
    
    
    
