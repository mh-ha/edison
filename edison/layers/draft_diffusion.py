from typing import Optional
import math
from random import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat

from edison.configs.config import EdisonConfig
from edison.layers.base import BaseDiffusion
from edison.layers.draft_encoder import Encoder
from edison.layers.positional_embedding import SinusoidalPosEmb
from edison.schemas.model import DiffusionOutput
import edison.utils.utils as utils


class Diffusion(BaseDiffusion):
    def __init__(
        self,
        config: EdisonConfig,
    ) -> None:
        super().__init__()
        # init
        self.config = config
        self.self_condition = config.self_condition
        self.input_dim = config.lm_dim
        self.internal_dim = config.tx_dim
        self.output_dim = config.lm_dim

        # layers
        self.encoder = Encoder(
            internal_dim=self.internal_dim,
            depth=config.tx_depth,
            num_heads=config.num_attn_heads,
            ff_mult=config.ff_mult,
            max_seq_len=config.max_seq_len,
            context_max_seq_len=config.num_encoder_latents,
            num_dense_connections=config.num_dense_connections,
        )
        self.time_mlp = self._build_time_mlp(self.internal_dim, self.internal_dim * config.ff_mult)
        self.time_proj = self._build_time_projection(self.internal_dim * config.ff_mult, self.internal_dim)
        if self.self_condition:
            self.input_proj = self._build_projection(config.lm_dim * 2, self.internal_dim)
            self.learnable_self_cond = nn.Parameter(torch.randn(1, self.input_dim))
            nn.init.normal_(self.learnable_self_cond, mean=0, std=0.02)
        else:
            self.input_proj = self._build_projection(config.latent_dim, self.internal_dim)
            self.learnable_self_cond = None
        self.context_input_proj = self._build_projection(config.latent_dim, self.internal_dim)
        if config.num_dense_connections < 1:
            self.output_proj = self._build_projection(self.internal_dim, self.output_dim)
        else:
            self.output_proj = self._build_projection(self.internal_dim, self.output_dim)
        self.norm = nn.LayerNorm(self.internal_dim)

    def _build_time_mlp(
        self,
        internal_dim: int,
        time_emb_dim: int,
    ) -> nn.Module:
        sinu_pos_emb = SinusoidalPosEmb(internal_dim)
        return nn.Sequential(
            sinu_pos_emb,
            nn.Linear(internal_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

    def _build_time_projection(
        self,
        input_dim: int,
        output_dim: int,
    ) -> nn.Module:
        return nn.Sequential(
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def _build_projection(
        self,
        input_dim: int,
        output_dim: int,
    ) -> nn.Module:
        return nn.Linear(input_dim, output_dim)

    def forward(
        self,
        latent: Tensor,
        context: Tensor,
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
            self_cond=self_cond,
        )
        alpha = utils.time_to_alpha(times, encoded.ndim)
        pred_start = self._predict_start_from_v(latent, alpha, encoded)
        pred_noise = self._predict_noise_from_v(latent, alpha, encoded)
        pred_v = encoded
        return DiffusionOutput(
            pred_start=pred_start,
            pred_noise=pred_noise,
            pred_v=pred_v,
        )

    def _predict_start_from_v(
        self,
        latent: Tensor,
        alpha: Tensor,
        v: Tensor,
    ) -> Tensor:
        # TODO: 수식 이해하기
        return alpha.sqrt() * latent - (1 - alpha).sqrt() * v

    def _predict_noise_from_v(
        self,
        latent: Tensor,
        alpha: Tensor,
        v: Tensor,
    ) -> Tensor:
        return (1 - alpha).sqrt() * latent + alpha.sqrt() * v

    # def _predict_noise_from_start(self, latent, alpha, x0):
    #     return (latent - alpha.sqrt() * x0) / (1 - alpha).sqrt().clamp(min=1e-8)

    # def _predict_v_from_start_and_eps(self, latent, alpha, x, noise):
    #     return alpha.sqrt() * noise - x * (1 - alpha).sqrt()

    def encode(
        self,
        latent: Tensor,
        context: Tensor,
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

        # input projection
        latent = self.input_proj(latent)
        context = self.context_input_proj(context)

        # add time embedding
        # print(f"[Diffusion.encode] latent: {latent.shape}, alpha: {alpha.shape}")
        time_emb = self.time_mlp(alpha * 1000)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')
        # print(f"[Diffusion.encode] time_emb: {time_emb.shape}")
        latent = latent + self.time_proj(time_emb)

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
        context: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        # generate times, noise, alpha, target
        times = torch.zeros((latent.shape[0],)).uniform_(0, 1.).to(latent.device)
        noise = torch.randn_like(latent).to(latent.device)
        alpha = utils.time_to_alpha(times, latent.ndim)
        # print(f"[Diffusion.training_step] alpha: {alpha.shape}, noise: {noise.shape}, latent: {latent.shape}, latent.ndim: {latent.ndim}")
        target = alpha.sqrt() * noise - (1 - alpha).sqrt() * latent
        latent = alpha.sqrt() * latent + (1 - alpha).sqrt() * noise
        # TODO: add context process

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
            self_cond=self_cond,
        )

        # calculate loss using pred_v
        pred = predictions.pred_v
        loss = self.loss_fn(pred, target, reduction='mean')
        return loss

    def sample(self, x) -> Tensor:
        return torch.tensor(0.)
