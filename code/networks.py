import torch
from torch import nn, optim, utils
from torch.nn import functional as F
import lightning as L

from transformers import AutoTokenizer
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

from layers import PerceiverResampler

class LatentGenerator(L.LightningModule):
    def __init__(
            self,
            pretrained_model,
            autoencoder,
            ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.autoencoder = autoencoder

    # required
    def training_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        y_hat = self.decoder(z)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    # required
    def loss(self, y_hat, y):
        return F.mse_loss(y_hat, y)
    
    # required
    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)
    
    # optional
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        y_hat = self.decoder(z)
        loss = self.loss(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
pretrained_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
autoencoder = PerceiverResampler(
    dim=1024, dim_latent=256, depth=6, dim_head=64,
    num_latents=8, max_seq_len=64, ff_mult=4, l2_normalize_latents=True
    )
latent_generator = LatentGenerator(pretrained_model, autoencoder)
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

class Diffusion(L.LightningModule):
    def __init__(
            self,
            diffusion_model,
            ):
        super().__init__()
        self.diffusion_model = diffusion_model
