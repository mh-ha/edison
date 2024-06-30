import lightning as L
import torch

from ..config.config import Config
from .diffusion import GaussianDiffusion
from .edison_diffusion import EdisonGaussianDiffusion


class LD4LGAE(L.LightningModule):
    def __init__(
        self,
        config: Config,
        lm: torch.nn.Module,
        ae: torch.nn.Module,
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
        # print(f"start: {inputs.shape} {attention_masks.shape} {targets.shape}")
        # LM encoder outputs
        encoder_outputs = self.lm.get_encoder()(
            input_ids=inputs,
            attention_mask=attention_masks)
        # print(f"LM encoder outputs: {encoder_outputs['last_hidden_state'].shape}")
        # AE encoder, decoder outputs
        encoder_outputs['last_hidden_state'] = self.ae(
            encoder_outputs['last_hidden_state'],
            attention_mask=attention_masks)
        # print(f"AE outputs - {encoder_outputs['last_hidden_state'].shape}")
        # LM decoder outputs (loss)
        outputs = self.lm(
            labels=targets,
            encoder_outputs=encoder_outputs,
            # output_hidden_states=True,  # Debugging
        )
        loss = outputs.loss
        # print(f"decoder outputs: {outputs.decoder_hidden_states[-1].shape}")
        # print(f"decoder logits outputs: {outputs.logits.shape}")
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
        config: Config,
        autoencoder: LD4LGAE,
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
        loss = self(encoder_outputs, class_id)
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
        config: Config,
        lm: torch.nn.Module,
        ae: torch.nn.Module,
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
            input_ids=inputs,
            attention_mask=attention_masks)
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
        return {'logits_c1': outputs_c1['logits'], 'logits_c0': outputs_c0['logits']}

    def training_step(self, batch, batch_idx):
        inputs = batch['input_ids']
        attention_masks = batch['attention_mask']
        targets = batch['labels']
        targets_c0 = batch['labels_c0']
        # LM encoder outputs
        encoder_outputs = self.lm.get_encoder()(
            input_ids=inputs,
            attention_mask=attention_masks)
        # AE encoder, decoder outputs
        ae_decoder_outputs = self.ae(
            encoder_outputs['last_hidden_state'],
            attention_mask=attention_masks
        )
        # LM decoder outputs (loss)
        encoder_outputs['last_hidden_state'] = ae_decoder_outputs['latents_c1']
        outputs_c1 = self.lm(labels=targets, encoder_outputs=encoder_outputs)
        encoder_outputs['last_hidden_state'] = ae_decoder_outputs['latents_c0']
        outputs_c0 = self.lm(labels=targets_c0, encoder_outputs=encoder_outputs)
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
        config: Config,
        autoencoder: EdisonAE,
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
        # print(f"inputs: {inputs.shape} attention_mask: {attention_mask.shape} class_id: {class_id}")
        context_latents, embedding_latents = self.autoencoder.encode(inputs, attention_mask, return_embeddings=True)
        # print(f"context_latents: {context_latents} \nembedding_latents: {embedding_latents}")
        # print(f"context_latents_c1: {context_latents['latents_c1'].shape} embedding_latents_c1: {embedding_latents.shape}")
        # print(f"context_latents_c0: {context_latents['latents_c0'].shape}")
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

    def generate(self, embedding_latents, context_latents, attention_mask, class_id=None):
        return self.diffusion_model.generate(
            embedding_latents=embedding_latents,
            context_latents=context_latents,
            attention_mask=attention_mask,
            class_id=class_id,
        )
