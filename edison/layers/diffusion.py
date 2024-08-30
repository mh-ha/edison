from typing import Optional, List, Tuple
import math
from random import random

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat
from tqdm import tqdm

from edison.configs.base import Config
from edison.configs.discrete_diffusion import DiscreteDiffusionConfig
from edison.layers import register_module
from edison.layers.base import BaseDiffusion
from edison.layers.encoder import Encoder, BaselineEncoder, DiscreteDiffusionEncoder
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
        self.pos_emb = AbsolutePositionalEmbedding(config.internal_dim, config.num_encoder_latents)
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
        # latent_mask = [[True] * length + [False] * (self.config.num_encoder_latents - length) for length in lengths]
        # latent_mask = torch.tensor(latent_mask, dtype=torch.bool, device=device)
        latent_mask = torch.ones((batch_size, self.config.num_encoder_latents), dtype=torch.bool, device=device)
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
            self_cond = pred_x_start

        return latents, latent_mask


@register_module(name="discrete_diffusion_layer")
class DiscreteDiffusionLayer(BaseDiffusion):
    def __init__(
        self,
        config: DiscreteDiffusionConfig,
        word_embedding_layer: nn.Module,
    ) -> None:
        super().__init__()

        # init
        self.config = config
        self.self_condition = config.self_condition
        self.input_dim = config.embedding_dim
        self.internal_dim = config.internal_dim
        self.output_dim = config.embedding_dim
        self.use_mask = config.use_mask
        self.word_embedding_layer = word_embedding_layer

        # layers
        self.encoder = DiscreteDiffusionEncoder(
            internal_dim=self.internal_dim,
            depth=config.network_depth,
            num_heads=config.num_attn_heads,
            ff_mult=config.ff_mult,
            max_seq_len=config.max_seq_len,
            num_dense_connections=config.num_dense_connections,)
        self.pos_emb = AbsolutePositionalEmbedding(config.internal_dim, config.max_seq_len)
        self.time_mlp = self._build_time_mlp(self.internal_dim, self.internal_dim * config.ff_mult)
        self.time_proj = self._build_time_projection(self.internal_dim * config.ff_mult, self.internal_dim)

        self.input_proj = self._build_projection(self.input_dim * 2, self.internal_dim)
        self.learnable_self_cond = nn.Parameter(torch.randn(1, self.input_dim))
        nn.init.normal_(self.learnable_self_cond, mean=0., std=0.02)
        self.output_proj = self._build_projection(self.internal_dim, self.output_dim)
        self.norm = nn.LayerNorm(self.internal_dim)

        self.head = nn.Linear(self.output_dim, self.config.vocab_size)

    def forward(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        times: Tensor,
        attention_mask: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
    ) -> DiffusionOutput:
        encoded = self.encode(
            latent=latent,
            context=context,
            times=times,
            attention_mask=attention_mask,
            self_cond=self_cond,)
        pred_start = self.head(encoded)
        return DiffusionOutput(
            pred_start=pred_start,
            pred_noise=None,
            pred_v=None,)

    def encode(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        times: Tensor,
        attention_mask: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
    ) -> Tensor:
        alpha = utils.time_to_alpha(times)
        # concat self condition with latent
        if self_cond is None:
            latent_shape = latent.shape
            self_cond = repeat(self.learnable_self_cond, '1 d -> b l d', b=latent_shape[0], l=latent_shape[1])
            latent = torch.cat((latent, self_cond), dim=-1)
        else:
            latent = torch.cat((latent, self_cond), dim=-1)
        # print(f"[DiscreteDiffusion.encode] latent: {latent.shape}, context: {context}")

        # positional embedding
        pos_emb = self.pos_emb(latent)

        # input projection
        latent = self.input_proj(latent)

        # add time embedding
        time_emb = self.time_mlp(alpha * 1000)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')
        latent = latent + self.time_proj(time_emb) + pos_emb
        # print(f"[DiscreteDiffusion.encode] latent: {latent.shape}, time_emb: {time_emb.shape}, pos_emb: {pos_emb.shape}")

        # encoding
        encoded = self.encoder(
            latent=latent,
            context=context,
            attention_mask=attention_mask,
            time_emb=time_emb,)
        # print(f"[DiscreteDiffusion.encode] encoded: {encoded.shape}")

        # normalization and output projection
        encoded = self.norm(encoded)
        encoded = self.output_proj(encoded)
        # print(f"[DiscreteDiffusion.encode] encoded: {encoded.shape}")
        return encoded

    @property
    def loss_fn(self):
        loss_types = {'l1': F.l1_loss, 'l2': F.mse_loss, 'smooth_l1': F.smooth_l1_loss, 'ce': F.cross_entropy}
        if self.config.loss_type in loss_types:
            return loss_types[self.config.loss_type]
        else:
            raise ValueError(f'Invalid loss type: {self.config.loss_type}')

    def training_step(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        blank_token_id: int,
    ) -> Tensor:
        (
            x_t, new_attention_mask, target, times, alpha, kappa, gamma
        ) = self.forward_process(input_ids, attention_mask, blank_token_id)
        # token_ids to embedding
        latent = self.word_embedding_layer(x_t)
        context = None
        # print(f"[DiscreteDiffusion.training_step] latent: {latent.shape}, context: {context}")

        # self-conditioning
        self_cond = None
        # if self.self_condition and (random() < self.config.train_self_cond_prob):
        if self.self_condition and (0 < self.config.train_self_cond_prob):
            # generate self condition using diffusion model
            with torch.no_grad():
                model_output = self.forward(
                    latent=latent,
                    context=context,
                    times=times,
                    attention_mask=new_attention_mask,
                )
                pred_start = model_output.pred_start.detach()
                # print(f"[DiscreteDiffusion.training_step] pred_start: {pred_start.shape}, new_attention_mask: {new_attention_mask.shape}")
                pred_start = pred_start.softmax(dim=-1).view(-1, pred_start.shape[-1])
                pred_start = torch.multinomial(pred_start, 1)
                # print(f"[DiscreteDiffusion.training_step] pred_start: {pred_start.shape}")
                pred_start = pred_start.view(x_t.shape)
                # print(f"[DiscreteDiffusion.training_step] pred_start: {pred_start.shape}, x_t: {x_t.shape}")
                pred_sentence = x_t * new_attention_mask + pred_start * (~new_attention_mask)
                # print(f"[DiscreteDiffusion.training_step] pred_sentence: {pred_sentence}")
                self_cond = self.word_embedding_layer(pred_sentence)

                # self_cond = model_output.detach()
                # print(f"[DiscreteDiffusion.training_step] self_cond: {self_cond.shape if self_cond is not None else None}")
                if self.config.l2_normalize_latents:
                    self_cond = F.normalize(self_cond, dim=-1) * math.sqrt(self_cond.shape[-1])
        # print(f"[DiscreteDiffusion.training_step] self_cond: {self_cond.shape if self_cond is not None else None}")

        # predict
        predictions = self.forward(
            latent=latent,
            context=context,
            times=times,
            attention_mask=new_attention_mask,
            self_cond=self_cond,)

        # calculate loss using pred_start
        pred = predictions.pred_start
        pred = pred.softmax(dim=-1)
        # print(f"[DiscreteDiffusion.training_step] pred: {pred.shape}, target: {target.shape}")

        new_attention_mask = new_attention_mask.view(-1)
        pred = pred.view(-1, pred.shape[-1])
        # print(f"[DiscreteDiffusion.training_step] pred: {pred.shape}, new_attention_mask: {new_attention_mask.shape}")
        pred = pred[~new_attention_mask]
        # print(f"[DiscreteDiffusion.training_step] pred: {pred.shape}")
        target = target.view(-1)
        # print(f"[DiscreteDiffusion.training_step] target: {target.shape}")
        target = target[~new_attention_mask].long()
        # make target to one-hot
        # target = F.one_hot(target, num_classes=self.config.vocab_size).float()
        # print(f"[DiscreteDiffusion.training_step] target: {target.shape}")
        # print(f"pred.shape: {pred}, target.shape: {target}")
        loss = self.loss_fn(pred, target, reduction='mean')
        return loss

# LD4LG를 옵션으로 넣지 않으면 masked diffusion과 동일
# 성능이 masked diffusion과 비슷하게 나와야 함
    def forward_process(self, input_ids: Tensor, attention_mask: Tensor, blank_token_id: int):
        # attention_mask가 padding token을 가리키지만 여기서는 blank token을 가리키는 것으로 간주
        attention_mask = attention_mask.bool()
        # input_ids = input_ids * attention_mask + blank_token_id * (~attention_mask)
        # input_ids = input_ids * attention_mask + 1 * (~attention_mask)

        times = torch.zeros((input_ids.shape[0],)).uniform_(0, 1.).to(input_ids.device)
        sequence = torch.ones(input_ids.shape, device=input_ids.device)
        target = input_ids.clone()
        alpha = utils.time_to_alpha(times, input_ids.ndim)
        kappa = utils.time_to_alpha(times, input_ids.ndim) if not self.use_mask else torch.zeros_like(alpha)
        gamma = utils.time_to_alpha(times, input_ids.ndim) if not self.use_mask else torch.zeros_like(alpha)
        prob_word_blank = alpha
        prob_word_change = kappa
        prob_word_stay = 1-alpha-kappa
        prob_blank_stay = 1-gamma
        prob_blank_change = gamma
        # print(f"[DiscreteDiffusion.forward_process] prob_word_blank: {prob_word_blank.shape}, prob_word_change: {prob_word_change.shape}, prob_word_stay: {prob_word_stay.shape}")

        word_probs = torch.multinomial(
            torch.stack([prob_word_change, prob_word_stay, prob_word_blank], dim=-1).squeeze(), input_ids.shape[-1], replacement=True)
        blank_probs = torch.multinomial(
            torch.stack([prob_blank_change, prob_blank_stay], dim=-1).squeeze(), input_ids.shape[-1], replacement=True)
        # print(f"[DiscreteDiffusion.forward_process] word_probs: {word_probs.shape}, blank_probs: {blank_probs.shape}")

        change_word_to_word = (word_probs == 0).int()
        change_blank_to_word = (blank_probs == 0).int()
        stay_word = (word_probs == 1).int()
        stay_blank = (blank_probs == 1).int()
        change_word_to_blank = (word_probs == 2).int()
        # print(f"[DiscreteDiffusion.forward_process] change_word: {change_word_to_word.shape}, change_blank: {change_blank_to_word.shape}, stay_word: {stay_word.shape}, stay_blank: {stay_blank.shape}")
        # print(f"[DiscreteDiffusion.forward_process] blank: {change_word_to_blank.shape}, blank_token_id: {blank_token_id}")
        # print(f"[DiscreteDiffusion.forward_process] change_word_to_word: {change_word_to_word}\nchange_blank_to_word: {change_blank_to_word}\nstay_word: {stay_word}\nstay_blank: {stay_blank}\nchange_word_to_blank: {change_word_to_blank}")

        change_word_to_word = change_word_to_word * attention_mask
        change_blank_to_word = change_blank_to_word * (~attention_mask)

        stay_word = stay_word * attention_mask
        stay_blank = stay_blank * (~attention_mask)

        sampled = self._uniformly_sample_from_vocab(
            self.config.vocab_size, shape=change_word_to_word.shape, device=change_word_to_word.device
        )
        sampled_word_to_word = sampled * change_word_to_word
        sampled_blank_to_word = sampled * change_blank_to_word

        change_word_to_blank = change_word_to_blank * attention_mask
        # print(f"[DiscreteDiffusion.forward_process] attention_mask: {attention_mask}\n")
        # print(f"[DiscreteDiffusion.forward_process] change_word: {change_word}\n")
        # print(f"[DiscreteDiffusion.forward_process] change_blank: {change_blank}\n")
        # print(f"[DiscreteDiffusion.forward_process] sampled_word_to_word: {sampled_word_to_word}\n")
        # print(f"[DiscreteDiffusion.forward_process] sampled_blank_to_word: {sampled_blank_to_word}\n")
        # print(f"[DiscreteDiffusion.forward_process] stay_word: {stay_word}\n")
        # print(f"[DiscreteDiffusion.forward_process] stay_blank: {stay_blank}\n")
        # print(f"[DiscreteDiffusion.forward_process] change_word_to_blank: {change_word_to_blank}\n")
        # result_attention = (sequence * stayed) + (sequence * change_word_to_blank) + (sequence * sampled)
        result_attention = (
            (input_ids * stay_word * attention_mask) +
            (blank_token_id * stay_blank * ~attention_mask) +
            (blank_token_id * change_word_to_blank * attention_mask) +
            (sequence * sampled_word_to_word * attention_mask) +
            (sequence * sampled_blank_to_word * ~attention_mask)
        ).int()
        # print(f"[DiscreteDiffusion.forward_process] result_attention: {result_attention.shape}\n")
        # print(f"[DiscreteDiffusion.forward_process] stayed + change_word_to_blank + sampled: {result_attention}\n")

        x_t = result_attention
        # print(f"[DiscreteDiffusion.forward_process] x_t: {x_t}\n\n")
        new_attention_mask = (x_t != blank_token_id)
        # print(f"[DiscreteDiffusion.forward_process] x_t: {x_t.shape}, new_attention_mask: {new_attention_mask.shape}, target: {target.shape}")
        # print(f"[DiscreteDiffusion.forward_process] x_t: {x_t}\n\n")
        # print(f"[DiscreteDiffusion.forward_process] new_attention_mask: {new_attention_mask}\n\n")
        # print(f"[DiscreteDiffusion.forward_process] target: {target}\n\n")
        return x_t, new_attention_mask, target, times, alpha, kappa, gamma

    def _uniformly_sample_from_vocab(self, vocab_size: int, shape: Tuple[int], device: torch.device):
        return torch.randint(0, vocab_size, shape, device=device)

    def sample(self, batch_size: int, lengths: List[int]):
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
        latent_mask = torch.ones((batch_size, self.config.num_encoder_latents), dtype=torch.bool, device=device)
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
            self_cond = pred_x_start

        return latents, latent_mask


"""
        1. discrete diffusion은 probability array를 예측한다. softmax를 통해 각 토큰의 확률을 구하고, 이를 이용해 다음 토큰을 샘플링한다.
        2. 바뀌는 걸로 결정되면 sequence의 각 토큰이 어떤 단어로 바뀌는지 확률을 계산한다. (batch, seq_len, vocab_size)
        3. forward process에서 params 3개로 transition matrix(즉, probability)를 계산하여 sequence의 각 토큰에 적용한다.
            - 중요한 건, blank에서 word로 바뀔 확률도 있으나 t=1에 가까워질수록 미세하도록 구현해야 하는 것이다. -> 그래서 params가 여러 개 필요
        4. baseline diffusion은 LD4LG와 동일하게 작동하지만, context를 discrete diffusion에서 이용한다.
        """
