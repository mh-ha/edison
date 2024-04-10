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

from .config import Config
from .utils import NGramMaskGenerator


class Masker:
    def __init__(self, config:Config, tokenizer, **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        self.mask_generator = NGramMaskGenerator(tokenizer, config.mask_lm_prob, config.max_seq_len, config.max_preds_per_seq, **kwargs)

    def from_text_to_inputs(self, text, rng=random, **kwargs):
        tokens = self.tokenizer.tokenize(text)
        tokens = self._truncate_tokens(tokens, self.config.max_seq_len-2)
        tokens = self._add_eos_bos(tokens)
        masked_tokens, target_labels = self._generate_masked_tokens(tokens, rng, **kwargs)
        original_inputs_ids = self.tokenizer(text, add_special_tokens=True, return_tensors='pt')['input_ids']
        masked_input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        masked_input_ids = torch.tensor(masked_input_ids)
        output = {
            'input_ids': masked_input_ids.unsqueeze(0),
            'attention_mask': torch.ones_like(masked_input_ids).unsqueeze(0),
            'labels': torch.tensor(target_labels).unsqueeze(0),
            'original_input_ids': original_inputs_ids,
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


class ReplaceTaskData:
    def __init__(self, config, tokenizer, **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        self.masker = Masker(config, tokenizer, **kwargs)

    def get_generator_inputs(self, text, rng=random, **kwargs):
        masked_data = self.masker.from_text_to_inputs(text, rng, **kwargs)
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








# class MaskTaskDataset(Dataset):

# class ReplaceTaskDataset(Dataset):