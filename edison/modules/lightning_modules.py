

import lightning as L
import torch

from ..config.config import Config


class LD4LGAE(L.LightningModule):
    def __init__(
        self,
        config:Config,
        lm:torch.nn.Module,
        ae:torch.nn.Module,
        ):
        super().__init__()
        self.save_hyperparameters("config")
        self.config = config
        self.lm = lm
        # Freeze LM
        for param in lm.parameters():
            param.requires_grad = False
        self.ae = ae
        
    def forward(self, batch):
        """
        Only Encode forward
        """
        inputs = batch['input_ids']
        attention_masks = batch['attention_mask']
        with torch.no_grad():
            # LM encoder outputs
            encoder_outputs = self.lm.get_encoder()(
                input_ids = inputs,
                attention_mask = attention_masks)
        encoder_outputs = self.ae.encode(
            encoder_outputs,
            attention_mask=attention_masks)
        return encoder_outputs

    def training_step(self, batch, batch_idx):
        inputs = batch['input_ids']
        attention_masks = batch['attention_mask']
        targets = batch['labels']
        # print(f"start: {inputs.shape} {attention_masks.shape} {targets.shape}")
        # LM encoder outputs
        encoder_outputs = self.lm.get_encoder()(
            input_ids = inputs,
            attention_mask = attention_masks)
        # print(f"LM encoder outputs: {encoder_outputs['last_hidden_state'].shape}")
        # AE encoder, decoder outputs
        encoder_outputs = self.ae(
            encoder_outputs['last_hidden_state'],
            attention_mask=attention_masks)
        # print(f"AE outputs - {encoder_outputs.shape}")
        # LM decoder outputs (loss)
        outputs = self.lm(
            labels=targets,
            encoder_outputs=encoder_outputs)
        loss = outputs.loss
        # print(f"decoder logits outputs: {outputs.logits.shape}")
        # print(f"loss: {loss}")
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.ae.parameters(), lr=self.config.learning_rate)