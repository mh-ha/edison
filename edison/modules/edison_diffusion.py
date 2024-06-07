import math
import random 
from functools import partial
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, reduce, repeat
from tqdm.auto import tqdm

from edison.config.config import Config
from .positional_embedding import AbsolutePositionalEmbedding
from .edison_diffusion_layer import Encoder
from .utils import time_to_alpha, cosine_schedule, right_pad_dims_to, init_zero_


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'pred_v'])


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class DiffusionTransformer(nn.Module):
    def __init__(
        self, tx_dim, tx_depth, heads, latent_dim=None, max_seq_len=64, self_condition=False, 
        dropout=0.1, scale_shift=False, class_conditional=False, num_classes=0, 
        class_unconditional_prob=0, seq2seq=False, seq2seq_context_dim=0, 
        dual_output=False, num_dense_connections=0, dense_output_connection=False
        ):
        super().__init__()

        self.latent_dim = latent_dim
        self.self_condition = self_condition
        self.scale_shift = scale_shift
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.class_unconditional_prob = class_unconditional_prob
        self.seq2seq = seq2seq
        self.dense_output_connection = dense_output_connection
        self.max_seq_len = max_seq_len

        self.time_mlp = self._build_time_mlp(tx_dim)
        self.time_pos_embed_mlp = nn.Sequential(nn.GELU(), nn.Linear(tx_dim * 4, tx_dim))
        self.pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len)
        self.encoder = self._build_encoder(tx_dim, tx_depth, heads)

        self.class_embedding = self._build_class_embedding(tx_dim, num_classes, class_conditional)
        self.null_embedding_seq2seq, self.seq2seq_proj = self._build_seq2seq(seq2seq, seq2seq_context_dim, tx_dim)

        self.input_proj = nn.Linear(latent_dim * 2 if self_condition else latent_dim, tx_dim)
        self.init_self_cond = nn.Parameter(torch.randn(1, latent_dim)) if self_condition else None
        if self_condition:
            nn.init.normal_(self.init_self_cond, std=0.02)

        self.norm = nn.LayerNorm(tx_dim)
        self.output_proj = nn.Linear(tx_dim * 2 if dense_output_connection else tx_dim, latent_dim * 2 if dual_output else latent_dim)

        init_zero_(self.output_proj)

    def _build_time_mlp(self, tx_dim):
        sinu_pos_emb = SinusoidalPosEmb(tx_dim)
        time_emb_dim = tx_dim * 4
        return nn.Sequential(
            sinu_pos_emb,
            nn.Linear(tx_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

    def _build_encoder(self, tx_dim, tx_depth, heads):
        return Encoder(
            dim=tx_dim,
            depth=tx_depth,
            num_heads=heads,
        )

    def _build_class_embedding(self, tx_dim, num_classes, class_conditional):
        if class_conditional:
            assert num_classes > 0
            return nn.Sequential(nn.Embedding(num_classes + 1, tx_dim), nn.Linear(tx_dim, tx_dim * 4))
        return None

    def _build_seq2seq(self, seq2seq, seq2seq_context_dim, tx_dim):
        if seq2seq:
            null_embedding_seq2seq = nn.Embedding(1, tx_dim)
            seq2seq_proj = nn.Linear(seq2seq_context_dim, tx_dim)
            return null_embedding_seq2seq, seq2seq_proj
        return None, None

    def forward(self, x, mask, time, x_self_cond=None, class_id=None, seq2seq_cond=None, seq2seq_mask=None):
        time_emb = self.time_mlp(time * 1000)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')

        if self.class_conditional:
            class_emb = self.class_embedding(class_id)
            class_emb = rearrange(class_emb, 'b d -> b 1 d')
            time_emb = time_emb + class_emb

        pos_emb = self.pos_emb(x)

        if self.self_condition:
            x = torch.cat((x, x_self_cond if x_self_cond is not None else repeat(self.init_self_cond, '1 d -> b l d', b=x.shape[0], l=x.shape[1])), dim=-1)

        x_input = self.input_proj(x)
        tx_input = x_input + pos_emb + self.time_pos_embed_mlp(time_emb)

        if self.seq2seq:
            context, context_mask = self._build_context(seq2seq_cond, seq2seq_mask, x.shape[0], x.device)
            x = self.encoder(tx_input, mask=mask, context=context, context_mask=context_mask, time_emb=time_emb)
        else:
            x = self.encoder(tx_input, mask=mask, time_emb=time_emb)

        x = self.norm(x)
        return self.output_proj(x)

    def _build_context(self, seq2seq_cond, seq2seq_mask, batch_size, device):
        context, context_mask = [], []
        if seq2seq_cond is None:
            null_context = repeat(self.null_embedding_seq2seq.weight, '1 d -> b 1 d', b=batch_size)
            context.append(null_context)
            context_mask.append(torch.tensor([[True] for _ in range(batch_size)], dtype=bool, device=device))
        else:
            context.append(self.seq2seq_proj(seq2seq_cond))
            context_mask.append(seq2seq_mask)
        context = torch.cat(context, dim=1)
        context_mask = torch.cat(context_mask, dim=1)
        return context, context_mask





class EdisonGaussianDiffusion(nn.Module):
    def __init__(self, config:Config, diffusion_for=None):
        super().__init__()
        self.diffusion_mode = None if diffusion_for is None else config.diffusion_mode
        self.diffusion_model = self._initialize_diffusion_model(config, diffusion_for)
        self._initialize_buffers(config)
        self._initialize_schedules(config)
        self.latent_dim = self.diffusion_model.latent_dim
        self.self_condition = self.diffusion_model.self_condition
        self.max_seq_len = config.num_encoder_latents
        self.l2_normalize = False
        self.objective = config.objective
        self.loss_type = config.loss_type
        assert self.objective in {'pred_noise', 'pred_x0', 'pred_v', 'pred_v_dual'}, 'objective must be one of pred_noise, pred_x0, pred_v, pred_v_dual'
        if self.diffusion_model.class_conditional and self.diffusion_model.class_unconditional_prob > 0:
            self.class_unconditional_bernoulli = torch.distributions.Bernoulli(probs=self.diffusion_model.class_unconditional_prob)

    def _initialize_diffusion_model(self, config, diffusion_for):
        if diffusion_for:
            assert diffusion_for in {'context', 'embedding'}, 'diffusion_for must be one of context, embedding'
            prefix = f"{diffusion_for}_"
            return DiffusionTransformer(
                tx_dim=getattr(config, f'{prefix}tx_dim'),
                tx_depth=getattr(config, f'{prefix}tx_depth'),
                heads=getattr(config, f'{prefix}tx_dim') // getattr(config, f'{prefix}attn_head_dim'),
                latent_dim=getattr(config, f'{prefix}latent_dim'),
                max_seq_len=config.num_encoder_latents,
                self_condition=getattr(config, f'{prefix}self_condition'),
                scale_shift=getattr(config, f'{prefix}scale_shift'),
                dropout=getattr(config, f'{prefix}dropout'),
                class_conditional=getattr(config, f'{prefix}class_conditional'),
                num_classes=getattr(config, f'{prefix}num_classes'),
                class_unconditional_prob=getattr(config, f'{prefix}class_unconditional_prob'),
                seq2seq=True,
                seq2seq_context_dim=getattr(config, f'{prefix}lm_dim'),
                num_dense_connections=getattr(config, f'{prefix}num_dense_connections'),
            )
        else:
            return DiffusionTransformer(
                tx_dim=config.tx_dim,
                tx_depth=config.tx_depth,
                heads=config.tx_dim // config.attn_head_dim,
                latent_dim=config.latent_dim,
                max_seq_len=config.num_encoder_latents,
                self_condition=config.self_condition,
                scale_shift=config.scale_shift,
                dropout=config.dropout,
                class_conditional=config.class_conditional,
                num_classes=config.num_classes,
                class_unconditional_prob=config.class_unconditional_prob,
                seq2seq=(config.dataset_name in {'xsum', 'qqp', 'qg', 'wmt14-de-en', 'wmt14-en-de'}),
                seq2seq_context_dim=config.lm_dim,
                num_dense_connections=config.num_dense_connections
            )

    def _initialize_buffers(self, config):
        self.register_buffer('latent_mean', torch.zeros(self.latent_dim, dtype=torch.float32))
        self.register_buffer('latent_scale', torch.tensor(1, dtype=torch.float32))

    def _initialize_schedules(self, config):
        self.train_schedule = partial(time_to_alpha, alpha_schedule=cosine_schedule, scale=config.scale)
        self.sampling_schedule = partial(time_to_alpha, alpha_schedule=cosine_schedule, scale=config.scale)
        self.sampling_timesteps = config.sampling_timesteps
        self.train_prob_self_cond = config.train_prob_self_cond

    def predict_start_from_noise(self, z_t, t, noise, sampling=False):
        alpha = self._get_alpha(z_t, t, sampling)
        return (z_t - (1 - alpha).sqrt() * noise) / alpha.sqrt().clamp(min=1e-8)

    def predict_noise_from_start(self, z_t, t, x0, sampling=False):
        alpha = self._get_alpha(z_t, t, sampling)
        return (z_t - alpha.sqrt() * x0) / (1 - alpha).sqrt().clamp(min=1e-8)

    def predict_start_from_v(self, z_t, t, v, sampling=False):
        alpha = self._get_alpha(z_t, t, sampling)
        return alpha.sqrt() * z_t - (1 - alpha).sqrt() * v

    def predict_noise_from_v(self, z_t, t, v, sampling=False):
        alpha = self._get_alpha(z_t, t, sampling)
        return (1 - alpha).sqrt() * z_t + alpha.sqrt() * v

    def predict_v_from_start_and_eps(self, z_t, t, x, noise, sampling=False):
        alpha = self._get_alpha(z_t, t, sampling)
        return alpha.sqrt() * noise - x * (1 - alpha).sqrt()

    def _get_alpha(self, z_t, t, sampling):
        schedule = self.sampling_schedule if sampling else self.train_schedule
        alpha = schedule(t)
        return right_pad_dims_to(z_t, alpha)

    def normalize_latent(self, x_start):
        return (x_start - self.latent_mean) / self.latent_scale.clamp(min=1e-5)

    def unnormalize_latent(self, x_start):
        return x_start * self.latent_scale.clamp(min=1e-5) + self.latent_mean

    def diffusion_model_predictions(self, z_t, mask, t, x_self_cond=None, class_id=None, seq2seq_cond=None, seq2seq_mask=None, sampling=False, cls_free_guidance=1.0, l2_normalize=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        time_cond = time_to_alpha(t)
        model_output = self.diffusion_model(z_t, mask, time_cond, x_self_cond, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)

        if cls_free_guidance != 1.0:
            unc_class_id = torch.full_like(class_id, fill_value=self.diffusion_model.num_classes) if class_id is not None else None
            unc_model_output = self.diffusion_model(z_t, mask, time_cond, x_self_cond, class_id=unc_class_id)
            model_output = model_output * cls_free_guidance + unc_model_output * (1 - cls_free_guidance)

        return self._process_model_output(z_t, t, model_output, sampling, l2_normalize)

    def _process_model_output(self, z_t, t, model_output, sampling, l2_normalize):
        pred_v = None
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(z_t, t, pred_noise, sampling=sampling)
        elif self.objective == 'pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(z_t, t, x_start, sampling=sampling)
            pred_v = self.predict_v_from_start_and_eps(z_t, t, x_start, pred_noise, sampling=sampling)
        elif self.objective == 'pred_v':
            pred_v = model_output
            x_start = self.predict_start_from_v(z_t, t, pred_v, sampling=sampling)
            pred_noise = self.predict_noise_from_v(z_t, t, pred_v, sampling=sampling)
        else:
            raise ValueError(f'Invalid objective {self.objective}')

        if l2_normalize:
            assert sampling
            x_start = F.normalize(x_start, dim=-1) * math.sqrt(x_start.shape[-1])
            pred_noise = self.predict_noise_from_start(z_t, t, x_start, sampling=sampling)
            pred_v = self.predict_v_from_start_and_eps(z_t, t, x_start, pred_noise, sampling=sampling)

        return ModelPrediction(pred_noise, x_start, pred_v)

    def get_sampling_timesteps(self, batch, device, invert=False):
        times = torch.linspace(1., 0., self.sampling_timesteps + 1, device=device)
        if invert:
            times = times.flip(dims=(0,))
        times = times.unsqueeze(0).repeat(batch, 1)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim=0)
        return times.unbind(dim=-1)

    @torch.no_grad()
    def ddpm_sample(self, shape, lengths, class_id, seq2seq_cond, seq2seq_mask, cls_free_guidance=1.0, l2_normalize=False, invert=False, z_t=None):
        batch, device = shape[0], next(self.diffusion_model.parameters()).device
        time_pairs = self.get_sampling_timesteps(batch, device, invert)
        z_t = torch.randn(shape, device=device) if z_t is None else z_t
        mask = self._create_mask(shape, lengths, device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step', total=self.sampling_timesteps):
            model_output = self.diffusion_model_predictions(
                z_t, mask, time, class_id=class_id, x_self_cond=x_start,
                seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask,
                sampling=True, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize
                )
            alpha, alpha_next = self._get_alpha(time, True), self._get_alpha(time_next, True)
            alpha_now = alpha / alpha_next
            x_start = model_output.pred_x_start
            eps = model_output.pred_noise
            
            if time_next[0] <= 0:
                z_t = x_start
                continue
            
            noise = torch.randn_like(z_t)
            z_t = (1 / alpha_now.sqrt() * (z_t - (1 - alpha_now) / (1 - alpha).sqrt() * eps) + torch.sqrt(1 - alpha_now) * noise)

        return z_t, mask

    @torch.no_grad()
    def sample(self, batch_size, length, class_id=None, seq2seq_cond=None, seq2seq_mask=None, cls_free_guidance=1.0, l2_normalize=False):
        return self.ddpm_sample((batch_size, self.max_seq_len, self.latent_dim), length, class_id, seq2seq_cond, seq2seq_mask, cls_free_guidance, l2_normalize)

    @property
    def loss_fn(self):
        loss_types = {'l1': F.l1_loss, 'l2': F.mse_loss, 'smooth_l1': F.smooth_l1_loss}
        if self.loss_type in loss_types:
            return loss_types[self.loss_type]
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')

    def forward(self, txt_latent, mask, class_id, seq2seq_cond=None, seq2seq_mask=None, return_x_start=False, *args, **kwargs):
        batch, l, d, device = *txt_latent.shape, txt_latent.device
        assert l == self.max_seq_len, f'length must be {self.max_seq_len}'

        times = torch.zeros((batch,), device=device).uniform_(0, 1.)
        noise = torch.randn_like(txt_latent)
        alpha = self.train_schedule(times)
        alpha = right_pad_dims_to(txt_latent, alpha)
        z_t = alpha.sqrt() * txt_latent + (1 - alpha).sqrt() * noise

        if self.diffusion_model.class_conditional and self.diffusion_model.class_unconditional_prob > 0:
            assert class_id is not None
            class_unconditional_mask = self.class_unconditional_bernoulli.sample(class_id.shape).bool()
            class_id[class_unconditional_mask] = self.diffusion_model.num_classes

        self_cond = None
        if self.self_condition and (random.random() < self.train_prob_self_cond):
            with torch.no_grad():
                model_output = self.diffusion_model_predictions(z_t, mask, times, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                self_cond = model_output.pred_x_start.detach()
                if self.l2_normalize:
                    self_cond = F.normalize(self_cond, dim=-1) * math.sqrt(self_cond.shape[-1])

        predictions = self.diffusion_model_predictions(z_t, mask, times, x_self_cond=self_cond, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)

        if self.objective == 'pred_x0':
            target = txt_latent
            pred = predictions.pred_x_start
        elif self.objective == 'pred_noise':
            target = noise
            pred = predictions.pred_noise
        elif self.objective == 'pred_v':
            target = alpha.sqrt() * noise - (1 - alpha).sqrt() * txt_latent
            assert predictions.pred_v is not None
            pred = predictions.pred_v

        loss = self.loss_fn(pred, target, reduction='none')
        loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(txt_latent.shape[0])], 'b 1 -> b 1')

        if return_x_start:
            return loss.mean(), predictions.pred_x_start
        return loss.mean()

    def _create_mask(self, shape, lengths, device):
        if self.using_latent_model:
            return torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:
            mask = [[True] * length + [False] * (self.max_seq_len - length) for length in lengths]
            return torch.tensor(mask, dtype=torch.bool, device=device)
