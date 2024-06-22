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
    1. data to xt_data logic    ##TODO 구현 완료, 테스트 필요
        result: input_ids that replace pad_ids to sampled buffer words
    2. build LM, AE -> LM은 그대로 써도 됨, AE는 조정 필요(sentence 부분, buffer 부분 나눠서 계산하는 로직 등 옵션 3개) ##TODO 구현 완료, 테스트 필요
        result: AE with 3 options
    3. build Diffusions -> LD4LG와 비슷한 것, embedding을 위한 것 각각  ##TODO 구현 완료, 테스트 필요
        result: context diffusion, embedding diffusion
    4. processing Diffusions together -> 여러 개의 옵션 중 하나 선택 가능하도록 ##TODO 일부(same) 구현 완료, 테스트 필요
        result: diffusion training logic
            same: context, embedding 동시에 처리 (이전 context, embedding latent 사용, seq2seq_cond로 사용 -> cross attention)
            context_first: context 먼저 처리, embedding은 context latent 사용
            alternately: context, embedding 번갈아가면서 처리
            
    5. p, c 처리 로직
        1) XT-attention (like disentangle transformer) 구현     ##TODO 구현 완료, 테스트 필요
        2) PE: RPE, CPE 구현    
        https://www.notion.so/Edison-c16538b822a14728bb8dddba142a83de?pvs=4#e389c1f15c6e4212b4884f92b93b1463
        result: word, p, c, processing logic
"""


import lightning as L
import torch
from torch import nn
from einops import repeat

from ..config.config import Config
from .diffusion import GaussianDiffusion
from .edison_diffusion import EdisonGaussianDiffusion
from .positional_embedding import SinusoidalPosEmb, ConsciousnessEmbedding


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
            input_ids=inputs,
            attention_mask=attention_masks)
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
        print(f"start: {inputs.shape} {attention_masks.shape} {targets.shape}")
        # LM encoder outputs
        encoder_outputs = self.lm.get_encoder()(
            input_ids=inputs,
            attention_mask=attention_masks)
        print(f"LM encoder outputs: {encoder_outputs['last_hidden_state'].shape}")
        # AE encoder, decoder outputs
        encoder_outputs = self.ae(
            encoder_outputs['last_hidden_state'],
            attention_mask=attention_masks)
        print(f"AE outputs - {encoder_outputs.shape}")
        # LM decoder outputs (loss)
        outputs = self.lm(
            labels=targets,
            encoder_outputs=encoder_outputs)
        loss = outputs.loss
        print(f"decoder logits outputs: {outputs.logits.shape}")
        # print(f"loss: {loss}")
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.ae.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=50000)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }


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
        optimizer = torch.optim.AdamW(self.diffusion_model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=50000)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }


class EdisonAE(L.LightningModule):
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
        self.lm_input_embeddings = lm.get_input_embeddings()
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
    
    def encode(self, input_ids, attention_masks, return_embeddings=False):
        encoder_outputs = self.lm.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_masks)
        encoder_outputs = self.ae.encode(
            encoder_outputs['last_hidden_state'],
            attention_mask=attention_masks)
        if return_embeddings:
            embeddings = self.lm_input_embeddings(input_ids)
            return encoder_outputs, embeddings
        return encoder_outputs

    def decode(self, encoder_outputs):
        ae_decoder_outputs = self.ae.decode(encoder_outputs)
        outputs_c1 = self.lm(encoder_outputs=ae_decoder_outputs['latents_c1'])
        outputs_c0 = self.lm(encoder_outputs=ae_decoder_outputs['latents_c0'])
        return {'logits_c1':outputs_c1['logits'], 'logits_c0':outputs_c0['logits']}

    def training_step(self, batch, batch_idx):
        inputs = batch['input_ids']
        attention_masks = batch['attention_mask']
        targets = batch['labels']
        targets_c0 = batch['labels_c0']
        # LM encoder outputs
        encoder_outputs = self.lm.get_encoder()(
            input_ids = inputs,
            attention_mask = attention_masks)
        # AE encoder, decoder outputs
        ae_encoder_outputs = self.ae.encode(
            encoder_outputs['last_hidden_state'],
            attention_mask=attention_masks)
        ae_decoder_outputs = self.ae.decode(ae_encoder_outputs)
        # LM decoder outputs (loss)
        outputs_c1 = self.lm(labels=targets, encoder_outputs=ae_decoder_outputs['latents_c1'])
        outputs_c0 = self.lm(labels=targets_c0, encoder_outputs=ae_decoder_outputs['latents_c0'])
        loss_c1 = outputs_c1.loss
        loss_c0 = outputs_c0.loss
        loss = loss_c1 + loss_c0
        # print(f"loss: {loss}")
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.ae.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=50000)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }


class EdisonDiffusion(L.LightningModule):
    def __init__(
        self,
        config:Config,
        autoencoder:EdisonAE,
        ):
        super().__init__()
        self.save_hyperparameters('config')
        self.config = config
        self.autoencoder = autoencoder
        self.autoencoder.freeze()
        
        self.diffusion_model = EdisonGaussianDiffusion(config=config)
    
    def forward(self, embedding_latents, context_latents, attention_mask, class_id=None):
        # TODO: implement latents_c0 process
        if self.config.use_latents_c0:
            NotImplementedError('latents_c0 not implemented')
        context_latents = context_latents['latents_c1']
        
        embedding_latents_mask = attention_mask
        context_latents_mask = torch.ones(context_latents.shape[:2]).to(context_latents.device)
        # print(embedding_latents.shape, context_latents.shape, embedding_latents_mask.shape, context_latents_mask.shape)
        loss = self.diffusion_model(
            embedding_latents=embedding_latents,
            context_latents=context_latents,
            embedding_latents_mask=embedding_latents_mask,
            class_id=class_id,
            context_latents_mask=context_latents_mask,
        )
        return loss
        
    def training_step(self, batch, batch_idx):
        # print(batch)
        inputs = batch['input_ids']
        attention_mask = batch['attention_mask']
        class_id = batch['label'] if 'label' in batch else None
        context_latents, embedding_latents = self.autoencoder.encode(inputs, attention_mask, return_embeddings=True)
        loss = self(embedding_latents, context_latents, attention_mask, class_id)
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.diffusion_model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=50000)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }
