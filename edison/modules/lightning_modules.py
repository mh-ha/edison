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

from ..config.config import Config
from .diffusion import GaussianDiffusion
from .edison_diffusion import EdisonGaussianDiffusion


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
    
    def encode(self, input_ids, attention_masks, return_embeddings=False):
        encoder_outputs = self.lm.get_encoder()(
            input_ids=input_ids,
            attention_mask=attention_masks)
        encoder_outputs = self.ae.encode(
            encoder_outputs['last_hidden_state'],
            attention_mask=attention_masks)
        if return_embeddings:
            embeddings = self.lm.embeddings.word_embeddings(input_ids)
            return encoder_outputs, embeddings
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


class EdisonDiffusion(L.LightningModule):
    def __init__(
        self,
        config:Config,
        autoencoder:EdisonAE,
        ):
        super().__init__()
        self.save_hyperparameters('config')
        self.config = config
        self.diffusion_mode = config.diffusion_mode
        self.autoencoder = autoencoder
        self.autoencoder.freeze()
        self.context_diffusion_model = EdisonGaussianDiffusion(config=config, diffusion_type='context')
        self.embedding_diffusion_model = EdisonGaussianDiffusion(config=config, diffusion_type='embedding')
        
    def get_position(self, attention_masks):
        """
        Get position of each word in sentence
        buffer words have max value position (i.e. 1.0)
        """
        positions = torch.cumsum(attention_masks, dim=1)
        return positions / positions.max()
    
    def get_consciousness(self, attention_masks):
        """
        Get consciousness of each word in sentence
        buffer words have 0.0 value
        """
        return attention_masks.clone().float()
    
    def forward(self, embedding_latents, context_latents, attention_masks, class_id=None):
        positions = self.get_position(attention_masks)
        consciousness = self.get_consciousness(attention_masks)
        
        attention_mask = torch.ones(
            context_latents.shape[0],
            self.config.num_encoder_latents,
            dtype=torch.bool,
            device=context_latents.device,)
        
        #TODO: p, c 처리 방식 구현 필요
        if self.diffusion_mode == 'same':
            context_latents = self.context_diffusion_model(
                txt_latent=context_latents,
                mask=attention_mask,
                class_id=class_id,
                seq2seq_cond=embedding_latents,
                seq2seq_cond_mask=attention_mask,
                )
            embedding_latents = self.embedding_diffusion_model(
                txt_latent=embedding_latents,
                mask=attention_mask,
                class_id=class_id,
                seq2seq_cond=context_latents,
                seq2seq_cond_mask=attention_mask,
                )
            return context_latents, embedding_latents
        
        #TODO(P1): context와 embedding을 처리하는 로직 필요 - 'context_first', 'alternately'
        elif self.diffusion_mode == 'context_first':
            NotImplementedError("context_first not implemented")
        elif self.diffusion_mode == 'alternately':
            NotImplementedError("alternately not implemented")
            
        else:
            raise ValueError(f"diffusion_mode: {self.diffusion_mode} not supported")
        
    def training_step(self, batch, batch_idx):
        inputs = batch['input_ids']
        attention_masks = batch['attention_mask']
        class_id = batch['label'] if 'label' in batch else None
        context_latents, embedding_latents = self.autoencoder.encode(inputs, attention_masks, return_embeddings=True)
        loss = self.forward(embedding_latents, context_latents, attention_masks, class_id)
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW([
            {'params': self.context_diffusion_model.parameters()},
            {'params': self.embedding_diffusion_model.parameters()},
            ], lr=self.config.learning_rate)
