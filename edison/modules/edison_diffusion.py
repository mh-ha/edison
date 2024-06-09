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
from .edison_diffusion_layer import Encoder, ContextEncoder
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
        dual_output=False, num_dense_connections=0, dense_output_connection=False,
        is_context_diffusion=False,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.self_condition = self_condition
        self.scale_shift = scale_shift
        self.class_conditional = class_conditional
        self.num_classes = num_classes
        self.class_unconditional_prob = class_unconditional_prob
        self.seq2seq = seq2seq
        # TODO: Implement dense connection
        self.dense_output_connection = dense_output_connection
        self.max_seq_len = max_seq_len

        self.time_mlp = self._build_time_mlp(tx_dim)
        self.time_pos_embed_mlp = nn.Sequential(nn.GELU(), nn.Linear(tx_dim * 4, tx_dim))
        self.pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len)
        self.encoder = self._build_encoder(tx_dim, latent_dim, tx_depth, heads, is_context_diffusion)

        self.class_embedding = self._build_class_embedding(tx_dim, num_classes, class_conditional)
        self.null_embedding_seq2seq, self.seq2seq_proj = self._build_seq2seq(seq2seq, seq2seq_context_dim, tx_dim)

        self.input_proj = nn.Linear(tx_dim * 2 if self_condition else tx_dim, tx_dim)
        self.init_self_cond = nn.Parameter(torch.randn(1, tx_dim)) if self_condition else None
        if self_condition:
            nn.init.normal_(self.init_self_cond, std=0.02)

        self.norm = nn.LayerNorm(tx_dim)
        self.output_proj = nn.Linear(tx_dim * 2 if dense_output_connection else tx_dim, tx_dim * 2 if dual_output else tx_dim)

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

    def _build_encoder(self, tx_dim, latent_dim, tx_depth, heads, is_context_diffusion):
        if is_context_diffusion:
            return ContextEncoder(
                dim=tx_dim,
                cross_dim=latent_dim,
                depth=tx_depth,
                num_heads=heads,
            )
        else:
            return Encoder(
                dim=tx_dim,
                cross_dim=latent_dim,
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

    def forward(
        self,
        main_latents,
        time,
        main_self_cond=None,
        class_id=None,
        sub_latents=None,
        main_latents_mask=None,
        sub_latents_mask=None,
    ):
        z_t = main_latents
        time_emb = self.time_mlp(time * 1000)
        time_emb = rearrange(time_emb, 'b d -> b 1 d')

        if self.class_conditional:
            class_emb = self.class_embedding(class_id)
            class_emb = rearrange(class_emb, 'b d -> b 1 d')
            time_emb = time_emb + class_emb

        pos_emb = self.pos_emb(z_t)

        if self.self_condition:
            z_t = torch.cat((z_t, main_self_cond if main_self_cond is not None else repeat(self.init_self_cond, '1 d -> b l d', b=z_t.shape[0], l=z_t.shape[1])), dim=-1)

        x_input = self.input_proj(z_t)
        tx_input = x_input + pos_emb + self.time_pos_embed_mlp(time_emb)
        main_latents = tx_input

        # context, context_mask = self._build_context(embedding_latents, embedding_latents_mask, z_t.shape[0], z_t.device)
        z_t = self.encoder(main_latents, cross_kv=sub_latents, attention_mask=main_latents_mask, time_emb=time_emb)

        z_t = self.norm(z_t)
        z_t = self.output_proj(z_t)
        main_latents = z_t
        
        return main_latents

    def _build_context(self, seq2seq_cond, seq2seq_cond_mask, batch_size, device):
        context, context_mask = [], []
        if seq2seq_cond is None:
            null_context = repeat(self.null_embedding_seq2seq.weight, '1 d -> b 1 d', b=batch_size)
            context.append(null_context)
            context_mask.append(torch.tensor([[True] for _ in range(batch_size)], dtype=bool, device=device))
        else:
            context.append(self.seq2seq_proj(seq2seq_cond))
            context_mask.append(seq2seq_cond_mask)
        context = torch.cat(context, dim=1)
        context_mask = torch.cat(context_mask, dim=1)
        return context, context_mask





class EdisonGaussianDiffusion(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self._initialize_diffusion_model(config)
        self._initialize_buffers(config)
        self._initialize_schedules(config)
        self.latent_dim = config.latent_dim
        self.self_condition = self.embedding_diffusion_model.self_condition
        self.max_seq_len = config.max_seq_len
        self.l2_normalize = False
        self.loss_type = config.loss_type
        self.objective = config.objective
        if self.embedding_diffusion_model.class_conditional and self.embedding_diffusion_model.class_unconditional_prob > 0:
            self.class_unconditional_bernoulli = torch.distributions.Bernoulli(probs=self.embedding_diffusion_model.class_unconditional_prob)

    def _initialize_diffusion_model(self, config:Config) -> DiffusionTransformer:
        self.embedding_diffusion_model = DiffusionTransformer(
            tx_dim=config.tx_dim,
            tx_depth=config.tx_depth,
            heads=config.tx_dim // config.attn_head_dim,
            latent_dim=config.latent_dim,
            max_seq_len=config.max_seq_len,
            self_condition=config.self_condition,
            scale_shift=config.scale_shift,
            dropout=config.dropout,
            class_conditional=config.class_conditional,
            num_classes=config.num_classes,
            class_unconditional_prob=config.class_unconditional_prob,
            seq2seq=(config.dataset_name in {'xsum', 'qqp', 'qg', 'wmt14-de-en', 'wmt14-en-de'}),
            seq2seq_context_dim=config.lm_dim,
            num_dense_connections=config.num_dense_connections,
            is_context_diffusion=False,
        )
        self.context_diffusion_model = DiffusionTransformer(
            tx_dim=config.latent_dim,
            tx_depth=config.tx_depth,
            heads=config.latent_dim // config.attn_head_dim,
            latent_dim=config.tx_dim,
            max_seq_len=config.max_seq_len,
            self_condition=config.self_condition,
            scale_shift=config.scale_shift,
            dropout=config.dropout,
            class_conditional=config.class_conditional,
            num_classes=config.num_classes,
            class_unconditional_prob=config.class_unconditional_prob,
            seq2seq=(config.dataset_name in {'xsum', 'qqp', 'qg', 'wmt14-de-en', 'wmt14-en-de'}),
            seq2seq_context_dim=config.lm_dim,
            num_dense_connections=config.num_dense_connections,
            is_context_diffusion=True,
        )
        

    def _initialize_buffers(self, config:Config):
        self.register_buffer('latent_mean', torch.zeros(config.latent_dim, dtype=torch.float32))
        self.register_buffer('latent_scale', torch.tensor(1, dtype=torch.float32))

    def _initialize_schedules(self, config:Config):
        self.train_schedule = partial(time_to_alpha, alpha_schedule=cosine_schedule, scale=config.scale)
        self.sampling_schedule = partial(time_to_alpha, alpha_schedule=cosine_schedule, scale=config.scale)
        self.sampling_timesteps = config.sampling_timesteps
        self.train_prob_self_cond = config.train_prob_self_cond

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

    def diffusion_model_predictions(
        self,
        main_latents,
        main_latents_mask,
        times,
        main_self_cond=None,
        class_id=None,
        sub_latents=None,
        sub_latents_mask=None,
        sampling=False,
        cls_free_guidance=1.0,
        l2_normalize=False,
        diffusion_model=None,
    ):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        time_cond = time_to_alpha(times)
        model_output = diffusion_model(
            main_latents,
            time_cond,
            main_self_cond,
            class_id=class_id,
            sub_latents=sub_latents,
            main_latents_mask=main_latents_mask,
        )
        # if cls_free_guidance != 1.0:
        #     unc_class_id = torch.full_like(class_id, fill_value=self.diffusion_model.num_classes) if class_id is not None else None
        #     unc_model_output = self.diffusion_model(context_latents, mask, time_cond, context_self_cond, class_id=unc_class_id)
        #     model_output = model_output * cls_free_guidance + unc_model_output * (1 - cls_free_guidance)
        return self._process_model_output(main_latents, times, model_output, sampling, l2_normalize)

    def _process_model_output(self, z_t, t, model_output, sampling, l2_normalize):
        x_start = self.predict_start_from_v(z_t, t, model_output, sampling=sampling)
        pred_noise = self.predict_noise_from_v(z_t, t, model_output, sampling=sampling)
        pred_v = model_output
        if l2_normalize and sampling:
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
    def ddpm_sample(self, shape, lengths, class_id, seq2seq_cond, seq2seq_cond_mask, cls_free_guidance=1.0, l2_normalize=False, invert=False, z_t=None):
        batch, device = shape[0], next(self.embedding_diffusion_model.parameters()).device
        time_pairs = self.get_sampling_timesteps(batch, device, invert)
        z_t = torch.randn(shape, device=device) if z_t is None else z_t
        mask = self._create_mask(shape, lengths, device)

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step', total=self.sampling_timesteps):
            model_output = self.diffusion_model_predictions(
                z_t, mask, time, class_id=class_id, main_self_cond=x_start,
                sub_latents=seq2seq_cond, context_latents_mask=seq2seq_cond_mask,
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
    def sample(self, batch_size, length, class_id=None, seq2seq_cond=None, seq2seq_cond_mask=None, cls_free_guidance=1.0, l2_normalize=False):
        return self.ddpm_sample((batch_size, self.max_seq_len, self.latent_dim), length, class_id, seq2seq_cond, seq2seq_cond_mask, cls_free_guidance, l2_normalize)

    @property
    def loss_fn(self):
        loss_types = {'l1': F.l1_loss, 'l2': F.mse_loss, 'smooth_l1': F.smooth_l1_loss}
        if self.loss_type in loss_types:
            return loss_types[self.loss_type]
        else:
            raise ValueError(f'Invalid loss type {self.loss_type}')

    def context_forward(self, embedding_latents, context_latents, embedding_latents_mask, class_id, context_latents_mask, times):
        txt_latent = context_latents
        noise = torch.randn_like(txt_latent).to(txt_latent.device)
        alpha = self.train_schedule(times)
        alpha = right_pad_dims_to(txt_latent, alpha)
        # print(alpha.shape, txt_latent.shape, noise.shape)
        z_t = alpha.sqrt() * txt_latent + (1 - alpha).sqrt() * noise
        context_latents = z_t

        if self.context_diffusion_model.class_conditional and self.context_diffusion_model.class_unconditional_prob > 0:
            assert class_id is not None
            class_unconditional_mask = self.class_unconditional_bernoulli.sample(class_id.shape).bool()
            class_id[class_unconditional_mask] = self.context_diffusion_model.num_classes
        context_self_cond = None
        if self.self_condition and (random.random() < self.train_prob_self_cond):
            with torch.no_grad():
                model_output = self.diffusion_model_predictions(
                    context_latents,
                    context_latents_mask,
                    times,
                    class_id=class_id,
                    sub_latents=embedding_latents,
                    sub_latents_mask=embedding_latents_mask,
                    diffusion_model=self.context_diffusion_model,
                )
                context_self_cond = model_output.pred_x_start.detach()
                if self.l2_normalize:
                    context_self_cond = F.normalize(context_self_cond, dim=-1) * math.sqrt(context_self_cond.shape[-1])
        predictions = self.diffusion_model_predictions(
            context_latents,
            context_latents_mask,
            times,
            main_self_cond=context_self_cond,
            class_id=class_id,
            sub_latents=embedding_latents,
            sub_latents_mask=embedding_latents_mask,
            diffusion_model=self.context_diffusion_model,
        )
        target = alpha.sqrt() * noise - (1 - alpha).sqrt() * txt_latent
        pred = predictions.pred_v
        loss = self.loss_fn(pred, target, reduction='none')
        loss = rearrange([reduce(loss[i][:torch.sum(context_latents_mask[i])], 'l d -> 1', 'mean') for i in range(txt_latent.shape[0])], 'b 1 -> b 1')
        return loss.mean()

    def embedding_forward(self, embedding_latents, context_latents, embedding_latents_mask, class_id, context_latents_mask, times):
        txt_latent = embedding_latents
        noise = torch.randn_like(txt_latent).to(txt_latent.device)
        alpha = self.train_schedule(times)
        alpha = right_pad_dims_to(txt_latent, alpha)
        z_t = alpha.sqrt() * txt_latent + (1 - alpha).sqrt() * noise
        embedding_latents = z_t

        if self.embedding_diffusion_model.class_conditional and self.embedding_diffusion_model.class_unconditional_prob > 0:
            assert class_id is not None
            class_unconditional_mask = self.class_unconditional_bernoulli.sample(class_id.shape).bool()
            class_id[class_unconditional_mask] = self.embedding_diffusion_model.num_classes

        embedding_self_cond = None
        if self.self_condition and (random.random() < self.train_prob_self_cond):
            with torch.no_grad():
                model_output = self.diffusion_model_predictions(
                    embedding_latents,
                    embedding_latents_mask,
                    times,
                    class_id=class_id,
                    sub_latents=context_latents,
                    sub_latents_mask=context_latents_mask,
                    diffusion_model=self.embedding_diffusion_model,
                )
                embedding_self_cond = model_output.pred_x_start.detach()
                if self.l2_normalize:
                    embedding_self_cond = F.normalize(embedding_self_cond, dim=-1) * math.sqrt(embedding_self_cond.shape[-1])
        predictions = self.diffusion_model_predictions(
            embedding_latents,
            embedding_latents_mask,
            times,
            main_self_cond=embedding_self_cond,
            class_id=class_id,
            sub_latents=context_latents,
            sub_latents_mask=context_latents_mask,
            diffusion_model=self.embedding_diffusion_model,
        )
        target = alpha.sqrt() * noise - (1 - alpha).sqrt() * txt_latent
        pred = predictions.pred_v
        loss = self.loss_fn(pred, target, reduction='none')
        loss = rearrange([reduce(loss[i][:torch.sum(embedding_latents_mask[i])], 'l d -> 1', 'mean') for i in range(txt_latent.shape[0])], 'b 1 -> b 1')
        return loss.mean()

    def _create_mask(self, shape, lengths, device):
        if self.using_latent_model:
            return torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:
            mask = [[True] * length + [False] * (self.max_seq_len - length) for length in lengths]
            return torch.tensor(mask, dtype=torch.bool, device=device)

    def forward(self, embedding_latents, context_latents, embedding_latents_mask, class_id, context_latents_mask):
        embedding_times = torch.zeros((embedding_latents.shape[0],)).uniform_(0, 1.).to(embedding_latents.device)
        context_times = torch.clamp(embedding_times * self.config.time_difference_embedding_over_context, 0, 1)
        loss_context = self.context_forward(embedding_latents, context_latents, embedding_latents_mask, class_id, context_latents_mask, context_times)
        loss_embedding = self.embedding_forward(embedding_latents, context_latents, embedding_latents_mask, class_id, context_latents_mask, embedding_times)
        loss = loss_context + loss_embedding
        return loss