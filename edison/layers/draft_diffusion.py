from typing import Optional
import math
from random import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat

from edison.configs.base import Config
from edison.layers import register_module
from edison.layers.base import BaseDiffusion
from edison.layers.draft_encoder import Encoder, BaselineEncoder
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
        self.input_dim = config.lm_dim
        self.internal_dim = config.tx_dim
        self.output_dim = config.lm_dim
        self.encoder = Encoder(
            internal_dim=self.internal_dim,
            depth=config.tx_depth,
            num_heads=config.num_attn_heads,
            ff_mult=config.ff_mult,
            max_seq_len=config.max_seq_len,
            context_max_seq_len=config.num_encoder_latents,
            num_dense_connections=config.num_dense_connections,
        )

        # layers
        self.pos_emb = None
        self.context_input_proj = self._build_projection(config.latent_dim, self.internal_dim)
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
        if context:
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
        if self.self_condition and (random() < self.config.train_prob_self_cond):
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

    def _get_sampling_timesteps(self, batch, device, invert=False):
        times = torch.linspace(1., 0., self.sampling_timesteps + 1, device=device)
        if invert:
            times = times.flip(dims=(0,))
        times = times.unsqueeze(0).repeat(batch, 1)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        return times.unbind(dim=-1)

    @torch.no_grad()
    def _ddpm_sample(
        self,
        batch_size,
        lengths,
        invert=False,
        context_z_t=None,
        embedding_z_t=None,
    ):
        device = next(self.embedding_diffusion_model.parameters()).device
        time_pairs = self._get_sampling_timesteps(batch_size, device, invert)

        context_shape = (batch_size, self.config.num_encoder_latents, self.latent_dim)
        context_z_t = torch.randn(context_shape, device=device) if context_z_t is None else context_z_t
        context_mask = [[True] * length + [False] * (self.config.num_encoder_latents - length) for length in lengths]
        context_mask = torch.tensor(context_mask, dtype=torch.bool, device=device)

        embedding_shape = (batch_size, self.max_seq_len, self.config.tx_dim)
        embedding_z_t = torch.randn(embedding_shape, device=device) if embedding_z_t is None else embedding_z_t
        embedding_mask = [[True] * length + [False] * (self.max_seq_len - length) for length in lengths]
        embedding_mask = torch.tensor(embedding_mask, dtype=torch.bool, device=device)
        embedding_x_start = None

        from tqdm import tqdm
        for time, time_next in tqdm(time_pairs, desc='sampling loop time step', total=self.sampling_timesteps):
            embedding_output = self.diffusion_model_predictions(
                embedding_z_t,
                embedding_mask,
                time,
                main_self_cond=embedding_x_start,
                sub_latents=context_z_t,
                sub_latents_mask=context_mask,
                diffusion_model=self.embedding_diffusion_model,
            )
            embedding_x_start = embedding_output.pred_x_start
            embedding_eps = embedding_output.pred_noise

            alpha, alpha_next = self._get_alpha(embedding_z_t, time, True), self._get_alpha(embedding_z_t, time_next, True)
            alpha_now = alpha / alpha_next

            if time_next[0] <= 0:
                embedding_z_t = embedding_x_start
                continue
            embedding_noise = torch.randn_like(embedding_z_t)
            embedding_z_t = 1 / alpha_now.sqrt() * (embedding_z_t - (1 - alpha_now) / (1 - alpha).sqrt() * embedding_eps)
            embedding_z_t = embedding_z_t + (torch.sqrt(1 - alpha_now) * embedding_noise)

        return embedding_z_t, embedding_mask

    def sample(
        self,
        batch_size,
        lengths,
        invert=False,
        context_z_t=None,
        embedding_z_t=None,
    ):
        return self._ddpm_sample(
            batch_size,
            lengths,
            invert=invert,
            context_z_t=context_z_t,
            embedding_z_t=embedding_z_t,
        )


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
        self.input_dim = config.latent_dim
        self.internal_dim = config.tx_dim
        self.output_dim = config.latent_dim

        # layers
        self.encoder = BaselineEncoder(
            internal_dim=self.internal_dim,
            depth=config.tx_depth,
            num_heads=config.num_attn_heads,
            ff_mult=config.ff_mult,
            max_seq_len=config.num_encoder_latents,
            num_dense_connections=config.num_dense_connections,)
        self.pos_emb = AbsolutePositionalEmbedding(config.tx_dim, config.num_encoder_latents)
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
        if self.self_condition and (random() < self.config.train_prob_self_cond):
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

    def sample(self, x):
        raise NotImplementedError
