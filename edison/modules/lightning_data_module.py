import random
import os
import json
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
import lightning as L
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, default_data_collator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase
from transformers.models.bart.modeling_bart import BartForConditionalGeneration, shift_tokens_right

from ..config.config import Config
from .tokens import TokenConverter

def get_dataset(dataset_name, data_path=None):
    if dataset_name == 'roc':
        if data_path is None:
            data_path = 'edison/datasets/ROCstory'
        dataset = load_dataset("text", data_files={f'{split}': os.path.join(data_path, f'roc_{split}.json') for split in ['train', 'valid']})
        dataset = process_roc_dataset(dataset)
    elif dataset_name == 'ag_news':
        dataset = load_dataset('pietrolesci/ag_news', 'original')
        train_ds = dataset['train']
        train_val_ds = train_ds.train_test_split(test_size=1000, seed=42)
        train_val_ds['valid'] = train_val_ds['test']
        train_val_ds['test'] = dataset['test']
        dataset = process_ag_news_dataset(train_val_ds)
    elif dataset_name == 'xsum':
        dataset = load_dataset('xsum')
        dataset['valid'] = dataset['validation']
        del(dataset['validation'])
        dataset = process_xsum_dataset(dataset)
    elif dataset_name == 'qqp':
        if data_path is None:
            data_path = 'edison/datasets/qqp'
        dataset = load_dataset("text", data_files={f'{split}': os.path.join(data_path, f'{split}.jsonl') for split in ['train', 'valid', 'test']})
        dataset = process_qqp_dataset(dataset)
    else:
        raise NotImplementedError
    return dataset


def get_dataloader(config:Config, dataset, decoder_start_token_id, tokenizer, max_seq_len, mode='diffusion', shuffle=True, context_tokenizer=None):
    def tokenization(example):
        if mode == 'diffusion' and config.dataset_name in {'xsum', 'qqp'}:
            assert context_tokenizer is not None
            source = example['context']
            target = example['text']

            if config.dataset_name in {'qqp',}:
                cond_inputs = context_tokenizer(source, padding="max_length", truncation=True, max_length=max_seq_len)
            elif config.dataset_name in {'xsum',}:
                cond_inputs = context_tokenizer(source, padding="max_length", truncation=True, max_length=max_seq_len*4)
            else:
                raise NotImplementedError

            model_inputs = tokenizer(text_target=target, padding="max_length", truncation=True, max_length=max_seq_len)
            
            # Add model target to model inputs
            for k in cond_inputs.keys():
                model_inputs[f'cond_{k}'] = cond_inputs[k]

            return model_inputs
        else:
            text = example["text"]
        return tokenizer(text, padding="max_length", truncation=True, max_length=max_seq_len)

    collate_fn=DataCollatorForBartDenoisingLM(tokenizer, decoder_start_token_id)
    
    if config.dataset_name in {'xsum', 'qqp'}:
        dataset = dataset.map(tokenization, remove_columns=['text', 'context'], batched=True, num_proc=None)
    else:
        dataset = dataset.map(tokenization, remove_columns='text')
            
    dl = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=config.train_batch_size,
            shuffle=shuffle,
            pin_memory = True,
            num_workers = 4
        )
    return dl

def get_xtdataloader(config:Config, dataset, decoder_start_token_id, tokenizer, max_seq_len, min_buffer_size, mode='diffusion', shuffle=True, context_tokenizer=None):
    def tokenization(example):
        if mode == 'diffusion' and config.dataset_name in {'xsum', 'qqp'}:
            assert context_tokenizer is not None
            source = example['context']
            target = example['text']

            if config.dataset_name in {'qqp',}:
                cond_inputs = context_tokenizer(source, padding="max_length", truncation=True, max_length=max_seq_len)
            elif config.dataset_name in {'xsum',}:
                cond_inputs = context_tokenizer(source, padding="max_length", truncation=True, max_length=max_seq_len*4)
            else:
                raise NotImplementedError

            model_inputs = tokenizer(text_target=target, padding="max_length", truncation=True, max_length=max_seq_len)
            
            # Add model target to model inputs
            for k in cond_inputs.keys():
                model_inputs[f'cond_{k}'] = cond_inputs[k]

            return model_inputs
        else:
            text = example["text"]
        
        # add extra padding tokens for buffer
        text = text[:max_seq_len]
        return tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_seq_len+min_buffer_size)

    collate_fn=DataCollatorForBartDenoisingLM(tokenizer, decoder_start_token_id, config)
    
    if config.dataset_name in {'xsum', 'qqp'}:
        dataset = dataset.map(tokenization, remove_columns=['text', 'context'], batched=True, num_proc=None)
    else:
        dataset = dataset.map(tokenization, remove_columns='text')
            
    dl = DataLoader(
            dataset,
            collate_fn=collate_fn,
            batch_size=config.train_batch_size,
            shuffle=shuffle,
            pin_memory = True,
            num_workers = 4
        )
    return dl

def process_roc_dataset(dataset):
    def extract_roc_text(example):
        text = example['text']
        assert text[:2] == '["'
        assert text[-2:] == '"]'
        sentences = text[2:-2]
        return {'text': sentences}
    dataset = dataset.map(extract_roc_text, )
    dataset = dataset.shuffle(seed=42)
    # Hold out some validation samples for testing
    val_test_ds = dataset['valid'].train_test_split(train_size=1000, shuffle=False)
    dataset['valid'] = val_test_ds['train']
    dataset['test'] = val_test_ds['test']
    return dataset

def process_ag_news_dataset(dataset):
    def process_ag_news_text(example):
        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example["description"].strip()), 'label':example['label']-1}
    dataset = dataset.map(process_ag_news_text, remove_columns=['title', 'description', 'class'])
    return dataset

def process_xsum_dataset(dataset):
    def process_xsum_text(example):
        return {'text': PreTrainedTokenizerBase.clean_up_tokenization(example["summary"].strip()), 'context':PreTrainedTokenizerBase.clean_up_tokenization(example["document"].strip())}
    dataset = dataset.map(process_xsum_text, remove_columns=['summary', 'document', 'id'])
    dataset = dataset.shuffle(seed=42)
    return dataset

def process_qqp_dataset(dataset):
    def process_qqp_text(example):
        dict_example = json.loads(example['text'])
        dict_example['text'] = dict_example['trg']
        dict_example['context'] = dict_example['src']
        del dict_example['trg']
        del dict_example['src']
        return dict_example
    dataset = dataset.map(process_qqp_text, )
    dataset = dataset.shuffle(seed=42)
    return dataset

def parse_metadata(metadata):
    if type(metadata) == list:
        return ' | '.join(metadata)
    elif type(metadata) == float:
        return 'Positive' if metadata > 0.5 else 'Negative'


@dataclass
class DataCollatorForBartDenoisingLM:
    """
    Data collator used for BART denoising language modeling.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data
    """

    tokenizer: PreTrainedTokenizerBase
    decoder_start_token_id: int
    config: Config = None
    vocab = tokenizer.vocab()

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:
        batch = BatchEncoding(
            {k: torch.LongTensor([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )
        """
        2. padding -> buffer 변환
            (max_seq_len, max_all_len 따로 -> 최소한의 buffer 크기 확보)
        4. 여기서는 buffer word를 추가하는 것만, p와 c는 diffusion 입력 직전에 계산
            (using attention_mask(c), consistent position(p))
        """

        batch["labels"] = batch["input_ids"].clone()
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )

        batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100

        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.tokenizer.pad_token_id).long()
        
        # add buffer words
        if self.config is not None:
            sampled_words = self.sample_words_from_vocab_and_batch(batch)
            batch = self.replace_pad_to_sampled_word(batch, sampled_words)
        return batch
    
    def sample_words_from_vocab_and_batch(self, batch):
        num_sampling = torch.sum((batch["attention_mask"] == 0).long())
        batch_words = set(batch["input_ids"][batch["attention_mask"]]) - {self.tokenizer.pad_token_id}
        vocab_words = self.vocab
        ratio = self.config.buffer_sampling_ratio
        #TODO: sample from batch_words, vocab_words using ratio
        sampled_words =
        return sampled_words
    
    def replace_pad_to_sampled_word(self, batch, sampled_words):
        #TODO: replace pad_token_ids to sampled words
        
        return batch