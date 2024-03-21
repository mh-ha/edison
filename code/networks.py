import torch
from torch import nn, optim, utils
from torch.nn import functional as F
import lightning as L

from transformers import AutoTokenizer
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from layers import PerceiverAutoEncoder
from data import get_dataset

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
        x = self.lm_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        x = self.autoencoder.encode(x, attention_mask=batch['attention_mask'])
        x = self.autoencoder.decode(x)
        loss = self.lm(labels=batch['labels'], encoder_outputs=x).loss
        self.log('train_loss', loss)
        return loss
    
    # required
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
    # optional
    def validation_step(self, batch, batch_idx):
        x = self.lm_encoder(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        x = self.autoencoder.encode(x, attention_mask=batch['attention_mask'])
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




enc_dec_model = 'facebook/bart-base'
max_seq_len = 64
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
train_dataset, valid_dataset = get_dataset(
    tokenizer,
    max_seq_len,
    dataset_name,
    enc_dec_model,
    train_batch_size,
    model_config,
    eval=False,
)

trainer = L.Trainer(
    max_epochs=10,
    gpus=1,
    precision=16,
    progress_bar_refresh_rate=1,
)
trainer.fit(latent_generator, train_dataset, valid_dataset)
