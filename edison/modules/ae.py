import torch
from torch import nn
import lightning as L

from ..config.config import Config
from ..layers.optimizer import AdamW

class Autoencoder(L.LightningModule):
    def __init__(self, config:Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.compress_model = None
        self.reconstruct_model = None
        self.loss = None
        
    def forward_compressor(self, inputs):
        return self.compress_model(inputs)
    
    def forward_reconstructor(self, inputs):
        return self.reconstruct_model(inputs)
    
    def forward(self, inputs):
        comp_output = self.forward_compressor(inputs)
        recon_output = self.forward_reconstructor(comp_output)
        return recon_output

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.loss(output, target)
        return loss

    def configure_optimizers(self):
        return {
            'optimizer': AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay),
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.StepLR(self.optimizers(), step_size=1, gamma=0.1),
                'interval': 'epoch',
                'frequency': 1,}
            }