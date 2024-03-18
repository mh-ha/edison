import torch
from torch import nn, optim, utils
from torch.nn import functional as F
import lightning as L

from transformers import AutoTokenizer
from transformers.models.bart.modeling_bart import BartForConditionalGeneration

class LD4LG(L.LightningModule):
    def __init__(
            self,
            pretrained_model,
            encoder,
            decoder,
            diffusion_model,
            ):
        super().__init__()
        self.pretrained_model = pretrained_model
        self.encoder = encoder
        self.decoder = decoder
        self.diffusion_model = diffusion_model

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
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-base')

class Diffusion(L.LightningModule):
    def __init__(
            self,
            diffusion_model,
            ):
        super().__init__()
        self.diffusion_model = diffusion_model

"""
latent AE
Diffusion
각각 구축해서 각자 훈련
-> 기존 코드에 의존하지 말고 일단 내가 원하는 대로 구현부터
"""