from typing import Optional

import lightning as L
import torch
from einops import einsum
from transformers.modeling_outputs import BaseModelOutput
from transformers import AutoTokenizer
from tqdm import tqdm

from legacy.etc.config import LD4LGConfig, EdisonConfig
from edison.layers.lm import get_BART
# from edison.layers.edison_autoencoder import EdisonPerceiverAutoEncoder
from edison.layers.ld4lg_diffusion import GaussianDiffusion
# from edison.layers.edison_diffusion import EdisonGaussianDiffusion
from edison.layers.autoencoder import AutoEncoder
from edison.layers.diffusion import DiffusionLayer


class LD4LGAE(L.LightningModule):
    def __init__(
        self,
        config: LD4LGConfig,
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

    def get_decoder_input(self, latents):
        return self.ae.decode(latents)

    def training_step(self, batch, batch_idx):
        inputs = batch['input_ids']
        attention_masks = batch['attention_mask']
        targets = batch['labels']
        # LM encoder outputs
        encoder_outputs = self.lm.get_encoder()(
            input_ids=inputs,
            attention_mask=attention_masks)
        # AE encoder, decoder outputs
        encoder_outputs['last_hidden_state'] = self.ae(
            encoder_outputs['last_hidden_state'],
            attention_mask=attention_masks)
        # LM decoder outputs (loss)
        outputs = self.lm(
            labels=targets,
            encoder_outputs=encoder_outputs,
            # output_hidden_states=True,  # Debugging
        )
        loss = outputs.loss
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
        config: LD4LGConfig,
        autoencoder: LD4LGAE,
        tokenizer: AutoTokenizer,
    ):
        super().__init__()
        self.save_hyperparameters('config')
        self.config = config
        self.autoencoder = autoencoder
        self.autoencoder.freeze()
        self.tokenizer = tokenizer
        # self.diffusion_model = GaussianDiffusion(config=config, device=self.device)
        self.diffusion_model = DiffusionLayer(config=config)

    def forward(self, encoder_outputs, class_id=None):
        mask = torch.ones(
            encoder_outputs.shape[0],
            self.config.num_encoder_latents,
            dtype=torch.bool,
            device=encoder_outputs.device,)
        return self.diffusion_model.training_step(
            latent=encoder_outputs,
            context=None,
            attention_mask=mask,
        )
        # return self.diffusion_model(
        #     txt_latent=encoder_outputs,
        #     mask=mask,
        #     class_id=class_id
        #     )

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
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=250000)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    @torch.no_grad()
    def generate(self, num_samples, seq_len, class_id=None, seed=42, batch_size=8):
        torch.manual_seed(seed)
        generated_texts = []
        for i in tqdm(range(0, num_samples, batch_size)):
            latents, mask = self.diffusion_model.sample(batch_size, [seq_len]*batch_size, class_id=class_id)
            attention_mask = None
            encoder_output = BaseModelOutput(last_hidden_state=self.autoencoder.get_decoder_input(latents.clone()))
            sample_ids = self.autoencoder.lm.generate(encoder_outputs=encoder_output, attention_mask=attention_mask)
            texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
            texts_list = [text.strip() for text in texts_list if len(text.strip()) > 0]
            generated_texts.extend(texts_list)
        return generated_texts


# class EdisonAE(L.LightningModule):
#     def __init__(
#         self,
#         config: Config,
#     ):
#         super().__init__()
#         self.save_hyperparameters('config')
#         self.config = config
#         self.lm, self.tokenizer = get_BART()
#         # Freeze LM
#         for param in self.lm.parameters():
#             param.requires_grad = False
#         self.lm_input_embeddings = self.lm.get_input_embeddings()
#         self.ae = EdisonPerceiverAutoEncoder(
#             dim_lm=self.config.dim_lm,
#             dim_ae=self.config.dim_ae,
#             num_layers=self.config.num_layers,
#             num_encoder_latents=self.config.num_encoder_latents,
#             num_decoder_latents=self.config.num_decoder_latents,
#             transformer_decoder=self.config.transformer_decoder,
#             l2_normalize_latents=self.config.l2_normalize_latents,
#             encoding_mode=self.config.encoding_mode
#         )

#     def forward(self, batch):
#         """
#         Only Encode forward
#         """
#         inputs = batch['input_ids']
#         attention_masks = batch['attention_mask']
#         encoder_outputs = self.lm.get_encoder()(
#             input_ids=inputs,
#             attention_mask=attention_masks)
#         encoder_outputs = self.ae.encode(
#             encoder_outputs['last_hidden_state'],
#             attention_mask=attention_masks)
#         return encoder_outputs

#     def encode(self, input_ids, attention_masks, return_embeddings=False):
#         encoder_outputs = self.lm.get_encoder()(
#             input_ids=input_ids,
#             attention_mask=attention_masks)
#         encoder_outputs = self.ae.encode(
#             encoder_outputs['last_hidden_state'],
#             attention_mask=attention_masks)
#         if return_embeddings:
#             embeddings = self.lm_input_embeddings(input_ids)
#             return encoder_outputs, embeddings
#         return encoder_outputs

#     def decode(self, encoder_outputs):
#         ae_decoder_outputs = self.ae.decode(encoder_outputs)
#         outputs_c1 = self.lm(encoder_outputs=ae_decoder_outputs['latents_c1'])
#         outputs_c0 = self.lm(encoder_outputs=ae_decoder_outputs['latents_c0'])
#         return {'logits_c1': outputs_c1['logits'], 'logits_c0': outputs_c0['logits']}

#     def training_step(self, batch, batch_idx):
#         inputs = batch['input_ids']
#         attention_masks = batch['attention_mask']
#         targets = batch['labels']
#         targets_c0 = batch['labels_c0']
#         # LM encoder outputs
#         encoder_outputs = self.lm.get_encoder()(
#             input_ids=inputs,
#             attention_mask=attention_masks)
#         # AE encoder, decoder outputs
#         ae_decoder_outputs = self.ae(
#             encoder_outputs['last_hidden_state'],
#             attention_mask=attention_masks
#         )
#         # LM decoder outputs (loss)
#         encoder_outputs['last_hidden_state'] = ae_decoder_outputs['latents_c1']
#         outputs_c1 = self.lm(labels=targets, encoder_outputs=encoder_outputs)
#         encoder_outputs['last_hidden_state'] = ae_decoder_outputs['latents_c0']
#         outputs_c0 = self.lm(labels=targets_c0, encoder_outputs=encoder_outputs)
#         loss_c1 = outputs_c1.loss
#         loss_c0 = outputs_c0.loss
#         loss = loss_c1 + loss_c0
#         self.log('loss', loss, on_step=True, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.ae.parameters(), lr=self.config.learning_rate)
#         scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=50000)
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': scheduler,
#         }


# class EdisonDiffusion(L.LightningModule):
#     def __init__(
#         self,
#         config: Config,
#         ae_path: str,
#     ):
#         super().__init__()
#         self.save_hyperparameters('config', 'ae_path')
#         self.config = config
#         self.autoencoder = EdisonAE.load_from_checkpoint(
#                 ae_path,
#                 map_location='cuda' if torch.cuda.is_available() else 'cpu',
#                 strict=False,
#             )
#         self.autoencoder.freeze()
#         self.tokenizer = self.autoencoder.tokenizer
#         self.diffusion_model = EdisonGaussianDiffusion(config=config, device=self.device)

#     def forward(self, embedding_latents, context_latents, attention_mask, class_id=None):
#         if self.config.use_latents_c0:
#             NotImplementedError('latents_c0 not implemented')
#         context_latents = context_latents['latents_c1']

#         embedding_latents_mask = attention_mask
#         context_latents_mask = torch.ones(context_latents.shape[:2]).to(context_latents.device)
#         loss = self.diffusion_model(
#             embedding_latents=embedding_latents,
#             context_latents=context_latents,
#             embedding_latents_mask=embedding_latents_mask,
#             class_id=class_id,
#             context_latents_mask=context_latents_mask,
#         )
#         return loss

#     def training_step(self, batch, batch_idx):
#         inputs = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         class_id = batch['label'] if 'label' in batch else None
#         context_latents, embedding_latents = self.autoencoder.encode(inputs, attention_mask, return_embeddings=True)
#         loss = self(embedding_latents, context_latents, attention_mask, class_id)
#         self.log('loss', loss, on_step=True, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(self.diffusion_model.parameters(), lr=self.config.learning_rate)
#         scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=250000)
#         return {
#             'optimizer': optimizer,
#             'lr_scheduler': scheduler,
#         }

#     @torch.no_grad()
#     def generate(self, num_samples, seq_len, class_id=None, seed=42, batch_size=8):
#         torch.manual_seed(seed)
#         generated_texts = []
#         for i in tqdm(range(0, num_samples, batch_size)):
#             latents, mask = self.diffusion_model.sample(batch_size, [seq_len]*batch_size, class_id=class_id)
#             # decode latents to token_ids
#             sample_ids = einsum(latents, self.autoencoder.lm_input_embeddings.weight, 'b l d, n d -> b l n')
#             sample_ids = sample_ids.argmax(dim=-1)
#             texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
#             texts_list = [text.strip() for text in texts_list if len(text.strip()) > 0]
#             generated_texts.extend(texts_list)
#         return generated_texts


class EdisonAE(L.LightningModule):
    def __init__(
        self,
        config: EdisonConfig,
    ):
        super().__init__()
        self.save_hyperparameters('config')
        self.config = config
        self.lm, self.tokenizer = get_BART()
        # Freeze LM
        for param in self.lm.parameters():
            param.requires_grad = False
        self.lm_input_embeddings = self.lm.get_input_embeddings()
        self.ae = AutoEncoder(
            dim_lm=self.config.dim_lm,
            dim_ae=self.config.dim_ae,
            num_layers=self.config.num_layers,
            num_encoder_latents=self.config.num_encoder_latents,
            num_decoder_latents=self.config.num_decoder_latents,
            transformer_decoder=self.config.transformer_decoder,
            l2_normalize_latents=self.config.l2_normalize_latents,
        )

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
        output = self.lm(encoder_outputs=ae_decoder_outputs)
        return output

    def training_step(self, batch, batch_idx):
        inputs = batch['input_ids']
        attention_masks = batch['attention_mask']
        targets = batch['labels']
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
        encoder_outputs['last_hidden_state'] = ae_decoder_outputs
        output = self.lm(labels=targets, encoder_outputs=encoder_outputs)
        loss = output.loss
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
        config: EdisonConfig,
        ae_path: Optional[str] = None,
        autoencoder: Optional[EdisonAE] = None,
    ):
        super().__init__()
        self.save_hyperparameters('config', 'ae_path')
        self.config = config
        if autoencoder is None:
            if ae_path is None:
                self.autoencoder = EdisonAE(config)
                print("Model initialized.")
            else:
                self.autoencoder = EdisonAE.load_from_checkpoint(ae_path)
                print("Model loaded from checkpoint.")
        else:
            self.autoencoder = autoencoder
        self.autoencoder.freeze()
        self.tokenizer = self.autoencoder.tokenizer
        self.diffusion_model = DiffusionLayer(config=config)

    def forward(self, embedding_latents, context_latents, attention_mask):
        context_latents = context_latents

        loss = self.diffusion_model.training_step(
            latent=embedding_latents,
            context=context_latents,
            attention_mask=attention_mask,
        )
        return loss

    def training_step(self, batch, batch_idx):
        inputs = batch['input_ids']
        attention_mask = batch['attention_mask']
        context_latents, embedding_latents = self.autoencoder.encode(inputs, attention_mask, return_embeddings=True)
        loss = self(embedding_latents, context_latents, attention_mask)
        self.log('loss', loss, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.diffusion_model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0, total_iters=self.config.max_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
        }

    @torch.no_grad()
    def generate(self, num_samples, seq_len, class_id=None, seed=42, batch_size=8):
        torch.manual_seed(seed)
        generated_texts = []
        for i in tqdm(range(0, num_samples, batch_size)):
            latents, mask = self.diffusion_model.sample(batch_size, [seq_len]*batch_size, class_id=class_id)
            # decode latents to token_ids
            sample_ids = einsum(latents, self.autoencoder.lm_input_embeddings.weight, 'b l d, n d -> b l n')
            sample_ids = sample_ids.argmax(dim=-1)
            texts_list = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in sample_ids]
            texts_list = [text.strip() for text in texts_list if len(text.strip()) > 0]
            generated_texts.extend(texts_list)
        return generated_texts
