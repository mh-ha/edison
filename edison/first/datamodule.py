import random

import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from datasets import concatenate_datasets

from ..config.config import Config
from .second.third.utils import NGramMaskGenerator
from .second.third.fetch_dataset import fetch_dataset
from .second.third.prep_dataset import split_sentences, tokenize

from ..config.config import Config
from .second.third.utils import NGramMaskGenerator


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
            num_workers=self.config.num_trainloader_workers,
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
                'input_ids': batch_input_ids['input_ids'].to(torch.int32),
                'attention_mask': batch_input_ids['attention_mask'].to(torch.int32),
                'labels': batch_labels['input_ids'].to(torch.int32),
            }
        return collate_fn