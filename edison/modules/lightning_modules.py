"""

    LM Encoder      (batch, seq_len_lm, d_model)
    -> AE Encoder   (batch, seq_len_ae=32, d_ae=64)
    -> Diffusion    (batch, seq_len_ae, d_ae) (내부적으로 dimention for diffusion 있음)
    -> AE Decoder   (batch, seq_len_ae, d_ae)
    -> LM Decoder   (batch, seq_len, d_model)
    
edison
    - embed -> LM : c=0 무시
    - -> AE -> latent
        1. only sentence
        2. sentence with buffer together
        3. sentence and buffer seperately
    - -> Diffusion
        context diffusion       (batch, seq_len_ae, d_ae) -> (batch, seq_len_ae, d_diff) -> (batch, seq_len_ae, d_ae)
        embedding diffusion     (batch, seq_len_lm, d_embed(=d_model?)) -> (batch, seq_len_lm, d_diff) -> (batch, seq_len_lm, d_embed(=d_model?))

TODO
    1. data to xt_data logic
        result: input_ids that replace pad_ids to sampled buffer words
    2. build LM, AE -> LM은 그대로 써도 됨, AE는 조정 필요(sentence 부분, buffer 부분 나눠서 계산하는 로직 등 옵션 3개)
        result: AE with 3 options
    3. build Diffusions -> LD4LG와 비슷한 것, embedding을 위한 것 각각
        result: context diffusion, embedding diffusion
    4. processing Diffusions together -> 여러 개의 옵션 중 하나 선택 가능하도록
        result: diffusion training logic
"""


import lightning as L
import torch

from ..config.config import Config
from .diffusion import GaussianDiffusion


class LD4LGAE(L.LightningModule):
    def __init__(
        self,
        config:Config,
        lm:torch.nn.Module,
        ae:torch.nn.Module,
        ):
        super().__init__()
        self.save_hyperparameters('config')
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
        encoder_outputs = self.lm.get_encoder()(
            input_ids = inputs,
            attention_mask = attention_masks)
        encoder_outputs = self.ae.encode(
            encoder_outputs['last_hidden_state'],
            attention_mask=attention_masks)
        return encoder_outputs
    
    def encode(self, input_ids, attention_masks):
        encoder_outputs = self.lm.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_masks)
        encoder_outputs = self.ae.encode(
            encoder_outputs['last_hidden_state'],
            attention_mask=attention_masks)
        return encoder_outputs

    def decode(self, encoder_outputs):
        decoder_outputs = self.ae.decode(encoder_outputs['last_hidden_state'])
        outputs = self.lm(encoder_outputs=decoder_outputs)
        return outputs['logits']

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


class LD4LGDiffusion(L.LightningModule):
    def __init__(
        self,
        config:Config,
        autoencoder:LD4LGAE,
        ):
        super().__init__()
        self.save_hyperparameters('config')
        self.config = config
        self.autoencoder = autoencoder
        self.autoencoder.freeze()
        self.diffusion_model = GaussianDiffusion(config=config)
        
    def forward(self, encoder_outputs, class_id=None):
        mask = torch.ones(
            encoder_outputs.shape[0],
            self.config.num_encoder_latents,
            dtype=torch.bool,
            device=encoder_outputs.device,)
        return self.diffusion_model(
            txt_latent=encoder_outputs,
            mask=mask,
            class_id=class_id
            )
        
    def training_step(self, batch, batch_idx):
        inputs = batch['input_ids']
        attention_masks = batch['attention_mask']
        class_id = batch['label'] if 'label' in batch else None
        encoder_outputs = self.autoencoder.encode(inputs, attention_masks)
        loss = self.forward(encoder_outputs, class_id)
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.diffusion_model.parameters(), lr=self.config.learning_rate)


#TODO
class EdisonAE(L.LightningModule):
    def __init__(
        self,
        config:Config,
        lm:torch.nn.Module,
        ae:torch.nn.Module,
        ):
        self.save_hyperparameters('config')
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
        encoder_outputs = self.lm.get_encoder()(
            input_ids = inputs,
            attention_mask = attention_masks)
        encoder_outputs = self.ae.encode(
            encoder_outputs['last_hidden_state'],
            attention_mask=attention_masks)
        return encoder_outputs
    
    def encode(self, input_ids, attention_masks):
        encoder_outputs = self.lm.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_masks)
        encoder_outputs = self.ae.encode(
            encoder_outputs['last_hidden_state'],
            attention_mask=attention_masks)
        return encoder_outputs

    def decode(self, encoder_outputs):
        decoder_outputs = self.ae.decode(encoder_outputs['last_hidden_state'])
        outputs = self.lm(encoder_outputs=decoder_outputs)
        return outputs['logits']

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

#TODO
class EdisonDiffusion(L.LightningModule):
    def __init__(
        self,
        config:Config,
        autoencoder:LD4LGAE,
        ):
        super().__init__()
        self.save_hyperparameters('config')
        self.config = config
        self.autoencoder = autoencoder
        self.autoencoder.freeze()
        self.context_diffusion_model = GaussianDiffusion(config=config)
        self.embedding_diffusion_model = GaussianDiffusion(config=config)
        
    def forward(self, encoder_outputs, class_id=None):
        mask = torch.ones(
            encoder_outputs.shape[0],
            self.config.num_encoder_latents,
            dtype=torch.bool,
            device=encoder_outputs.device,)
        return self.diffusion_model(
            txt_latent=encoder_outputs,
            mask=mask,
            class_id=class_id
            )
        
    def training_step(self, batch, batch_idx):
        inputs = batch['input_ids']
        attention_masks = batch['attention_mask']
        class_id = batch['label'] if 'label' in batch else None
        encoder_outputs = self.autoencoder.encode(inputs, attention_masks)
        loss = self.forward(encoder_outputs, class_id)
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.diffusion_model.parameters(), lr=self.config.learning_rate)
