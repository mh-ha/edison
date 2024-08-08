from typing import Optional, List
import math
from random import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm

from edison.configs.base import Config
from edison.layers import register_module
from edison.layers.base import BaseDiffusion
from edison.layers.encoder import Encoder, BaselineEncoder
from edison.layers.positional_embedding import AbsolutePositionalEmbedding
from edison.schemas.model import DiffusionOutput
import edison.utils.utils as utils


@register_module(name="diffusion_layer")
class DiffusionLayer(BaseDiffusion):
    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__()

        # init
        self.config = config
        self.self_condition = config.self_condition
        self.input_dim = config.dim_lm
        self.internal_dim = config.internal_dim
        self.output_dim = config.dim_lm
        self.encoder = Encoder(
            internal_dim=self.internal_dim,
            depth=config.network_depth,
            num_heads=config.num_attn_heads,
            ff_mult=config.ff_mult,
            max_seq_len=config.max_seq_len,
            context_max_seq_len=config.num_encoder_latents,
            num_dense_connections=config.num_dense_connections,
        )

        # layers
        self.pos_emb = None
        self.context_input_proj = self._build_projection(config.dim_ae, self.internal_dim)
        self.time_mlp = self._build_time_mlp(self.internal_dim, self.internal_dim * config.ff_mult)
        self.time_proj = self._build_time_projection(self.internal_dim * config.ff_mult, self.internal_dim)
        if self.self_condition:
            self.input_proj = self._build_projection(self.input_dim * 2, self.internal_dim)
            self.learnable_self_cond = nn.Parameter(torch.randn(1, self.input_dim))
            nn.init.normal_(self.learnable_self_cond, mean=0, std=0.02)
        else:
            self.input_proj = self._build_projection(self.input_dim, self.internal_dim)
            self.learnable_self_cond = None
        self.output_proj = self._build_projection(self.internal_dim, self.output_dim)
        self.norm = nn.LayerNorm(self.internal_dim)

    def forward(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        times: Tensor,
        attention_mask: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
    ) -> DiffusionOutput:
        alpha = utils.time_to_alpha(times)
        encoded = self.encode(
            latent=latent,
            context=context,
            alpha=alpha,
            attention_mask=attention_mask,
            self_cond=self_cond,)
        alpha = utils.time_to_alpha(times, encoded.ndim)
        pred_start = self._predict_start_from_v(latent, alpha, encoded)
        pred_noise = self._predict_noise_from_v(latent, alpha, encoded)
        pred_v = encoded
        return DiffusionOutput(
            pred_start=pred_start,
            pred_noise=pred_noise,
            pred_v=pred_v,
        )

    def encode(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        alpha: Tensor,
        attention_mask: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
    ) -> Tensor:
        # concat self condition with latent
        if self.self_condition:
            if self_cond is None:
                latent_shape = latent.shape
                self_cond = repeat(self.learnable_self_cond, '1 d -> b l d', b=latent_shape[0], l=latent_shape[1])
                latent = torch.cat((latent, self_cond), dim=-1)
            else:
                latent = torch.cat((latent, self_cond), dim=-1)

        # positional embedding (optional)
        if self.pos_emb:
            pos_emb = self.pos_emb(latent)
        else:
            pos_emb = 0.

        # input projection
        latent = self.input_proj(latent)
        if context is not None:
            context = self.context_input_proj(context)

        # add time embedding
        # print(f"[Diffusion.encode] latent: {latent.shape}, alpha: {alpha.shape}")
        time_emb = self.time_mlp(alpha * 1000)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')
        # print(f"[Diffusion.encode] time_emb: {time_emb.shape}")
        latent = latent + self.time_proj(time_emb) + pos_emb

        # encoding
        encoded = self.encoder(
            latent=latent,
            context=context,
            attention_mask=attention_mask,
            time_emb=time_emb,
        )

        # normalization and output projection
        encoded = self.norm(encoded)
        encoded = self.output_proj(encoded)
        return encoded

    @property
    def loss_fn(self):
        loss_types = {'l1': F.l1_loss, 'l2': F.mse_loss, 'smooth_l1': F.smooth_l1_loss}
        if self.config.loss_type in loss_types:
            return loss_types[self.config.loss_type]
        else:
            raise ValueError(f'Invalid loss type: {self.config.loss_type}')

    def training_step(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        # generate times, noise, alpha, target
        times = torch.zeros((latent.shape[0],)).uniform_(0, 1.).to(latent.device)
        noise = torch.randn_like(latent).to(latent.device)
        alpha = utils.time_to_alpha(times, latent.ndim)
        # print(f"[Diffusion.training_step] alpha: {alpha.shape}, noise: {noise.shape}, latent: {latent.shape}, latent.ndim: {latent.ndim}")
        target = alpha.sqrt() * noise - (1 - alpha).sqrt() * latent
        latent = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise
        # TODO: add context process?

        # self-conditioning
        self_cond = None
        if self.self_condition and (random() < self.config.train_self_cond_prob):
            # generate self condition using diffusion model
            with torch.no_grad():
                model_output = self.forward(
                    latent=latent,
                    context=context,
                    times=times,
                    attention_mask=attention_mask,
                )
                self_cond = model_output.pred_start.detach()
                if self.config.l2_normalize_latents:
                    self_cond = F.normalize(self_cond, dim=-1) * math.sqrt(self_cond.shape[-1])

        # predict
        predictions = self.forward(
            latent=latent,
            context=context,
            times=times,
            attention_mask=attention_mask,
            self_cond=self_cond,)

        # calculate loss using pred_v
        pred = predictions.pred_v
        loss = self.loss_fn(pred, target, reduction='mean')
        return loss

    @torch.no_grad()
    def ddpm_sample(
        self,
        batch_size: int,
        lengths: List[int],
    ):
        device = next(self.encoder.parameters()).device
        time_pairs = utils.get_sampling_timesteps(batch_size, self.config.sampling_timesteps, device)

        context_shape = (batch_size, self.config.num_encoder_latents, self.config.dim_ae)
        context_latent = torch.randn(context_shape, device=device)
        # context_mask = [[True] * length + [False] * (self.config.num_encoder_latents - length) for length in lengths]
        # context_mask = torch.tensor(context_mask, dtype=torch.bool, device=device)

        embedding_shape = (batch_size, self.config.max_seq_len, self.config.internal_dim)
        embedding_latent = torch.randn(embedding_shape, device=device)
        embedding_mask = [[True] * length + [False] * (self.config.max_seq_len - length) for length in lengths]
        embedding_mask = torch.tensor(embedding_mask, dtype=torch.bool, device=device)
        self_cond = None

        from tqdm import tqdm
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step', total=self.config.sampling_timesteps):
            embedding_output = self.forward(
                embedding_latent,
                context=context_latent,
                times=time,
                attention_mask=embedding_mask,
                self_cond=self_cond,)
            embedding_x_start = embedding_output.pred_start
            embedding_eps = embedding_output.pred_noise

            alpha = utils.time_to_alpha(time, embedding_latent.ndim)
            alpha_next = utils.time_to_alpha(time_next, embedding_latent.ndim)
            alpha_now = alpha / alpha_next

            if time_next[0] <= 0:
                embedding_latent = embedding_x_start
                continue

            embedding_noise = torch.randn_like(embedding_latent)
            embedding_latent = 1 / alpha_now.sqrt() * (embedding_latent - (1 - alpha_now) / (1 - alpha).sqrt() * embedding_eps)
            embedding_latent = embedding_latent + (torch.sqrt(1 - alpha_now) * embedding_noise)

        return embedding_latent, embedding_mask

    def sample(
        self,
        batch_size: int,
        lengths: List[int],
    ):
        embedding_latent, embedding_mask = self.ddpm_sample(batch_size, lengths)
        return embedding_latent, embedding_mask


@register_module(name="baseline_diffusion_layer")
class BaselineDiffusionLayer(BaseDiffusion):
    def __init__(
        self,
        config: Config,
    ) -> None:
        super().__init__()

        # init
        self.config = config
        self.self_condition = config.self_condition
        self.input_dim = config.dim_ae
        self.internal_dim = config.internal_dim
        self.output_dim = config.dim_ae

        # layers
        self.encoder = BaselineEncoder(
            internal_dim=self.internal_dim,
            depth=config.network_depth,
            num_heads=config.num_attn_heads,
            ff_mult=config.ff_mult,
            max_seq_len=config.num_encoder_latents,
            num_dense_connections=config.num_dense_connections,)
        self.pos_emb = AbsolutePositionalEmbedding(config.internal_dim, config.max_seq_len)
        self.time_mlp = self._build_time_mlp(self.internal_dim, self.internal_dim * config.ff_mult)
        self.time_proj = self._build_time_projection(self.internal_dim * config.ff_mult, self.internal_dim)

        self.input_proj = self._build_projection(self.input_dim * 2, self.internal_dim)
        self.learnable_self_cond = nn.Parameter(torch.randn(1, self.input_dim))
        nn.init.normal_(self.learnable_self_cond, mean=0., std=0.02)
        self.output_proj = self._build_projection(self.internal_dim, self.output_dim)
        self.norm = nn.LayerNorm(self.internal_dim)

    def forward(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        times: Tensor,
        attention_mask: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
    ) -> DiffusionOutput:
        alpha = utils.time_to_alpha(times)
        encoded = self.encode(
            latent=latent,
            context=context,
            alpha=alpha,
            attention_mask=attention_mask,
            self_cond=self_cond,)
        alpha = utils.time_to_alpha(times, encoded.ndim)
        pred_start = self._predict_start_from_v(latent, alpha, encoded)
        pred_noise = self._predict_noise_from_v(latent, alpha, encoded)
        pred_v = encoded
        return DiffusionOutput(
            pred_start=pred_start,
            pred_noise=pred_noise,
            pred_v=pred_v,)

    def encode(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        alpha: Tensor,
        attention_mask: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
    ) -> Tensor:
        # concat self condition with latent
        if self_cond is None:
            latent_shape = latent.shape
            self_cond = repeat(self.learnable_self_cond, '1 d -> b l d', b=latent_shape[0], l=latent_shape[1])
            latent = torch.cat((latent, self_cond), dim=-1)
        else:
            latent = torch.cat((latent, self_cond), dim=-1)

        # positional embedding
        pos_emb = self.pos_emb(latent)

        # input projection
        latent = self.input_proj(latent)

        # add time embedding
        time_emb = self.time_mlp(alpha * 1000)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')
        latent = latent + self.time_proj(time_emb) + pos_emb

        # encoding
        encoded = self.encoder(
            latent=latent,
            context=context,
            attention_mask=attention_mask,
            time_emb=time_emb,)

        # normalization and output projection
        encoded = self.norm(encoded)
        encoded = self.output_proj(encoded)
        return encoded

    @property
    def loss_fn(self):
        loss_types = {'l1': F.l1_loss, 'l2': F.mse_loss, 'smooth_l1': F.smooth_l1_loss}
        if self.config.loss_type in loss_types:
            return loss_types[self.config.loss_type]
        else:
            raise ValueError(f'Invalid loss type: {self.config.loss_type}')

    def training_step(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        # generate times, noise, alpha, target
        times = torch.zeros((latent.shape[0],)).uniform_(0, 1.).to(latent.device)
        noise = torch.randn_like(latent).to(latent.device)
        alpha = utils.time_to_alpha(times, latent.ndim)
        target = alpha.sqrt() * noise - (1 - alpha).sqrt() * latent
        latent = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise
        # TODO: add context process?

        # self-conditioning
        self_cond = None
        if self.self_condition and (random() < self.config.train_self_cond_prob):
            # generate self condition using diffusion model
            with torch.no_grad():
                model_output = self.forward(
                    latent=latent,
                    context=context,
                    times=times,
                    attention_mask=attention_mask,
                )
                self_cond = model_output.pred_start.detach()
                if self.config.l2_normalize_latents:
                    self_cond = F.normalize(self_cond, dim=-1) * math.sqrt(self_cond.shape[-1])

        # predict
        predictions = self.forward(
            latent=latent,
            context=context,
            times=times,
            attention_mask=attention_mask,
            self_cond=self_cond,)

        # calculate loss using pred_v
        pred = predictions.pred_v
        loss = self.loss_fn(pred, target, reduction='mean')
        return loss

    def sample(self, batch_size: int, lengths: List[int]):
        # latents, mask = self.diffusion_model.sample(batch_size, [seq_len]*batch_size)
        latents, latent_mask = self.ddpm_sample(batch_size, lengths)
        return latents, latent_mask

    @torch.no_grad()
    def ddpm_sample(
        self,
        batch_size: int,
        lengths: List[int],
    ):
        device = next(self.encoder.parameters()).device
        time_pairs = utils.get_sampling_timesteps(batch_size, self.config.sampling_timesteps, device)

        latents_shape = (batch_size, self.config.num_encoder_latents, self.config.dim_ae)
        latents = torch.randn(latents_shape, device=device)
        latent_mask = [[True] * length + [False] * (self.config.num_encoder_latents - length) for length in lengths]
        latent_mask = torch.tensor(latent_mask, dtype=torch.bool, device=device)
        self_cond = None

        for time, time_next in tqdm(time_pairs, total=self.config.sampling_timesteps):
            diffusion_output = self.forward(
                latent=latents,
                context=None,
                times=time,
                attention_mask=latent_mask,
                self_cond=self_cond,)
            pred_x_start = diffusion_output.pred_start
            pred_eps = diffusion_output.pred_noise

            alpha = utils.time_to_alpha(time, latents.ndim)
            alpha_next = utils.time_to_alpha(time_next, latents.ndim)
            alpha_now = alpha / alpha_next

            if time_next[0] <= 0:
                latents = pred_x_start
                continue

            noise = torch.randn_like(latents)
            latents = 1 / alpha_now.sqrt() * (latents - (1 - alpha_now) / (1 - alpha).sqrt() * pred_eps)
            latents = latents + (torch.sqrt(1 - alpha_now) * noise)

        return latents, latent_mask
