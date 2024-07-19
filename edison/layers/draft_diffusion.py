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


class Diffusion(BaseDiffusion):
    def __init__(
        self,
        encoder: BaseEncoder,
        config: EdisonConfig,
    ) -> None:
        super().__init__(
            encoder=encoder
        )
        self.config = config
        self.self_condition = config.self_condition
        self.input_dim = config.lm_dim
        self.internal_dim = encoder.internal_dim
        self.output_dim = config.lm_dim
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
        time: int,
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
        time_emb = self.time_proj(time * 1000)
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
        return DiffusionOutput(
            encoded=encoded,
        )

    def encode(
        self,
        latent: Tensor,
        context: Tensor,
        time: int,
        attention_mask: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
    ) -> DiffusionOutput:
        return self.forward(
            latent=latent,
            context=context,
            time=time,
            attention_mask=attention_mask,
            self_cond=self_cond
        )

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
        alpha = self._time_to_alpha(times, embedding_latents.ndim)
        # alpha = right_pad_dims_to(embedding_latents, alpha)
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

    def _cosine_schedule(
        self,
        t: Tensor,
        start: int = 0,
        end: int = 1,
        tau: int = 1,
        clip_min: float = 1e-9,
    ):
        power = 2 * tau
        v_start = math.cos(start * math.pi / 2) ** power
        v_end = math.cos(end * math.pi / 2) ** power
        output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
        output = (v_end - output) / (v_end - v_start)
        return output.clamp(min=clip_min)

    def _time_to_alpha(
        self,
        t: Tensor,
        latent_ndim: int,
        scale: float = 1.,
    ):
        alpha = self._cosine_schedule(t)
        shifted_log_snr = torch.log((alpha / (1 - alpha))).clamp(min=-15, max=15)
        shifted_log_snr = shifted_log_snr + (2 * math.log(scale))
        shifted_log_snr = torch.sigmoid(shifted_log_snr)
        if shifted_log_snr.ndim < latent_ndim:
            shifted_log_snr = shifted_log_snr.unsqueeze(-1)
        return shifted_log_snr

    def sample(self, x) -> Tensor:
        return torch.tensor(0.)
