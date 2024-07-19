from typing import Optional
import math
from random import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat

from edison.configs.config import EdisonConfig
from edison.layers.base import BaseDiffusion
from edison.layers.draft_encoder import BaseEncoder
from edison.layers.positional_embedding import SinusoidalPosEmb
from edison.schemas.model import DiffusionOutput
import edison.utils.utils as utils


class Diffusion(BaseDiffusion):
    def __init__(
        self,
        encoder: BaseEncoder,
        config: EdisonConfig,
    ) -> None:
        super().__init__(
            encoder=encoder
        )
        # init
        self.config = config
        self.self_condition = config.self_condition
        self.input_dim = config.lm_dim
        self.internal_dim = encoder.internal_dim
        self.output_dim = config.lm_dim

        # layers
        self.time_proj = self._build_time_mlp(self.internal_dim)
        self.input_proj = self._build_projection(config.lm_dim, self.internal_dim)
        self.context_input_proj = self._build_projection(config.latent_dim, self.internal_dim)
        if config.num_dense_connections < 1:
            self.output_proj = self._build_projection(self.internal_dim, self.output_dim)
        else:
            self.output_proj = self._build_projection(self.internal_dim * 2, self.output_dim)
        self.learnable_self_cond = nn.Parameter(torch.randn(1, self.input_dim)) if config.self_condition else None
        self.norm = nn.LayerNorm(self.internal_dim)

    def _build_time_mlp(self, internal_dim):
        sinu_pos_emb = SinusoidalPosEmb(internal_dim)
        time_emb_dim = internal_dim * 4
        return nn.Sequential(
            sinu_pos_emb,
            nn.Linear(internal_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, internal_dim),
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
        alpha = utils.time_to_alpha(times, latent.ndim)
        encoded = self.encode(
            latent=latent,
            context=context,
            alpha=alpha,
            attention_mask=attention_mask,
            self_cond=self_cond,
        )
        pred_start = self._predict_start_from_v(latent, alpha, encoded)
        pred_noise = self._predict_noise_from_v(latent, alpha, encoded)
        pred_v = encoded
        # TODO: check sampling process: how pred calculation is different from training
        if self.config.l2_normalize_latents and sampling:
            pred_start = F.normalize(pred_start, dim=-1) * math.sqrt(pred_start.shape[-1])
            pred_noise = self._predict_noise_from_start(latent, alpha, pred_start)
            pred_v = self._predict_v_from_start_and_eps(latent, alpha, pred_start, pred_noise)
        return DiffusionOutput(
            pred_start=pred_start,
            pred_noise=pred_noise,
            pred_v=pred_v,
        )

    def _predict_start_from_v(self, latent, alpha, v):
        return alpha.sqrt() * latent - (1 - alpha).sqrt() * v

    def _predict_noise_from_v(self, latent, alpha, v):
        return (1 - alpha).sqrt() * latent + alpha.sqrt() * v

    def _predict_noise_from_start(self, latent, alpha, x0):
        return (latent - alpha.sqrt() * x0) / (1 - alpha).sqrt().clamp(min=1e-8)

    def _predict_v_from_start_and_eps(self, latent, alpha, x, noise):
        return alpha.sqrt() * noise - x * (1 - alpha).sqrt()

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
        time_emb = self.time_proj(alpha * 1000)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')
        latent = latent + time_emb

        # encoding
        encoded = self.encoder(
            latent,
            context,
            attention_mask_words=attention_mask,
            time_emb=time_emb)

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
        embedding_latents,
        context_latents,
        embedding_latents_mask,
        class_id,
        context_latents_mask,
        times,
    ) -> Tensor:
        # generate noise, alpha, target
        noise = torch.randn_like(embedding_latents).to(embedding_latents.device)
        alpha = utils.time_to_alpha(times, embedding_latents.ndim)
        target = alpha.sqrt() * noise - (1 - alpha).sqrt() * embedding_latents
        embedding_latents = alpha.sqrt() * embedding_latents + (1 - alpha).sqrt() * noise

        # self-conditioning
        self_cond = None
        if self.self_condition and (random() < self.config.train_prob_self_cond):
            # generate self condition using diffusion model
            with torch.no_grad():
                model_output = self.diffusion_model_predictions(
                    embedding_latents,
                    embedding_latents_mask,
                    times,
                    class_id=class_id,
                    sub_latents=context_latents,
                    sub_latents_mask=context_latents_mask,
                )
                self_cond = model_output.pred_x_start.detach()
                if self.config.l2_normalize_latents:
                    self_cond = F.normalize(self_cond, dim=-1) * math.sqrt(self_cond.shape[-1])

        # predict
        predictions = self.diffusion_model_predictions(
            embedding_latents,
            embedding_latents_mask,
            times,
            main_self_cond=self_cond,
            class_id=class_id,
            sub_latents=context_latents,
            sub_latents_mask=context_latents_mask,
        )

        # calculate loss using pred_v
        pred = predictions.pred_v
        loss = self.loss_fn(pred, target, reduction='mean')
        return loss

    def sample(self, x) -> Tensor:
        return torch.tensor(0.)
