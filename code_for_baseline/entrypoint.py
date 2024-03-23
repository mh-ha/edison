from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn, optim, utils
from torch.nn import functional as F
import lightning as L

from transformers.models.bart.modeling_bart import BartForConditionalGeneration, shift_tokens_right
from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerBase

from code_for_baseline.layers import PerceiverAutoEncoder

class LatentGenerator(L.LightningModule):
    def __init__(
            self,
            pretrained_model: BartForConditionalGeneration,
            autoencoder: PerceiverAutoEncoder,
            ):
        super().__init__()
        self.lm = pretrained_model
        self.lm_encoder = pretrained_model.get_encoder()
        self.lm_decoder = pretrained_model.get_decoder()
        self.autoencoder = autoencoder

    # required
    def training_step(self, batch, batch_idx):
        # print(f'1: {x}')
        x = self.lm_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        x = self.autoencoder.encode(x['last_hidden_state'], attention_mask=batch['attention_mask'])
        x = self.autoencoder.decode(x)
        loss = self.lm(labels=batch['labels'], encoder_outputs=x).loss
        self.log('train_loss', loss)
        return loss
    
    # required
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
    # optional
    def validation_step(self, batch, batch_idx):
        # print(f'1: {x}')
        x = self.lm_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        x = self.autoencoder.encode(x['last_hidden_state'], attention_mask=batch['attention_mask'])
        x = self.autoencoder.decode(x)
        loss = self.lm(labels=batch['labels'], encoder_outputs=x).loss
        self.log('train_loss', loss)
        return loss

class Diffusion(L.LightningModule):
    def __init__(
            self,
            diffusion_model,
            ):
        super().__init__()
        self.diffusion_model = diffusion_model

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

    def __call__(self, examples: List[Dict[str, List[int]]]) -> BatchEncoding:
        batch = BatchEncoding(
            {k: torch.LongTensor([examples[i][k] for i in range(len(examples))]) for k, v in examples[0].items()}
        )
        batch["labels"] = batch["input_ids"].clone()
        batch["decoder_input_ids"] = shift_tokens_right(
            batch["labels"], self.tokenizer.pad_token_id, self.decoder_start_token_id
        )
        batch['labels'][batch['labels'] == self.tokenizer.pad_token_id] = -100
        batch["attention_mask"] = (batch["input_ids"] != self.tokenizer.pad_token_id).long()
        batch["decoder_attention_mask"] = (batch["decoder_input_ids"] != self.tokenizer.pad_token_id).long()

        return batch


enc_dec_model = 'facebook/bart-base'
max_seq_len = 128
dataset_name = 'roc'
train_batch_size = 32

pretrained_model = BartForConditionalGeneration.from_pretrained(enc_dec_model)
autoencoder = PerceiverAutoEncoder(
    dim_lm=pretrained_model.config.d_model,
    dim_ae=64,
    depth=3,
    num_encoder_latents=32,
    num_decoder_latents=32,
    max_seq_len=max_seq_len,
    transformer_decoder=True,
    l2_normalize_latents=False,
)
latent_generator = LatentGenerator(pretrained_model, autoencoder)
tokenizer = AutoTokenizer.from_pretrained(enc_dec_model)
model_config = pretrained_model.config

from datasets import load_dataset
import os
roc_data_path = 'code_for_baseline/datasets/ROCstory'
dataset = load_dataset("text", data_files={f'{split}': os.path.join(roc_data_path, f'roc_{split}.json') for split in ['train', 'valid']})
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
prep_dataset = process_roc_dataset(dataset)
map_prep_dataset = prep_dataset.map(lambda x: tokenizer(x['text'], padding="max_length", truncation=True, max_length=max_seq_len), batched=True, remove_columns=['text'], num_proc=4)
# map_prep_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

from torch.utils.data import DataLoader
# collate_fn = lambda x: tokenizer.pad(x, return_tensors='pt')
collate_fn = DataCollatorForBartDenoisingLM(tokenizer, model_config.decoder_start_token_id)
train_dataloader = DataLoader(map_prep_dataset['train'], batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_dataloader = DataLoader(map_prep_dataset['valid'], batch_size=32, shuffle=False, collate_fn=collate_fn)
test_dataloader = DataLoader(map_prep_dataset['test'], batch_size=32, shuffle=False, collate_fn=collate_fn)
# train_dataloader = DataLoader(prep_dataset['train'], batch_size=32, shuffle=True, collate_fn=collate_fn)
# valid_dataloader = DataLoader(prep_dataset['valid'], batch_size=32, shuffle=False, collate_fn=collate_fn)
# test_dataloader = DataLoader(prep_dataset['test'], batch_size=32, shuffle=False, collate_fn=collate_fn)


trainer = L.Trainer(
    accelerator='gpu',
    # strategy='ddp',
    max_epochs=10,
    # precision=16,
)
trainer.fit(latent_generator, train_dataloader, valid_dataloader)