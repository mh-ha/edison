

import lightning as L
import torch

from ..config.config import Config


class LD4LGAE(L.LightningModule):
    def __init__(
        self,
        config:Config,
        lm,
        ae,
        ):
        super().__init__()
        self.save_hyperparameters("config")
        self.lm = lm
        self.ae = ae
        #TODO: loss
        self.loss = None
        
    #TODO
    def forward(self, inputs):
        return self.model(inputs)

    #TODO
    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.forward(inputs)
        loss = self.loss(output, target)
        return loss

    #TODO
    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)