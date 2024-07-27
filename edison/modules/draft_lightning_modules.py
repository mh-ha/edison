from typing import Optional, Union

import torch
from einops import einsum
from tqdm import tqdm

from edison.configs.base import Config
from edison.layers import get_module as get_layer_module
from edison.layers.lm import get_BART
from edison.layers.base import BaseAutoEncoder, BaseDiffusion, BaseEncoder
from edison.layers.draft_diffusion import Diffusion
from edison.modules import register_module
from edison.modules.base import BaseEdisonAE, BaseEdisonDiffusion   # noqa: F401


@register_module(name='edison_ae')
class EdisonAE(BaseEdisonAE):
    def __init__(
        self,
        config: Config,
    ):
        super().__init__()
        self.save_hyperparameters('config')
        self.config = config

        self.lm, self.tokenizer = get_BART()
        for param in self.lm.parameters():
            param.requires_grad = False
        self.lm_input_embeddings = self.lm.get_input_embeddings()

        self.ae = get_layer_module(module_name="autoencoder")(
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


@register_module(name='edison_diffusion')
class EdisonDiffusion(BaseEdisonDiffusion):
    def __init__(
        self,
        config: Config,
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
        self.diffusion_model = get_layer_module(module_name="diffusion")(config=config)

    def forward(self, embedding_latents, context_latents, attention_mask):
        loss = self.diffusion_model.training_step(
            latent=embedding_latents,
            context=context_latents,
            attention_mask=attention_mask,
        )
        return loss

    def training_step(self, batch, batch_idx):
        inputs = batch['input_ids']
        attention_mask = batch['attention_mask']
        if self.config.model_name == 'LD4LG':
            embedding_latents = self.autoencoder.encode(inputs, attention_mask, return_embeddings=False)
            context_latents = None
        else:
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
