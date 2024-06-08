import math
import random 
from functools import partial
from collections import namedtuple
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import lightning as L

from einops import rearrange, reduce, repeat

from tqdm.auto import tqdm

from edison.config.config import Config
from .positional_embedding import AbsolutePositionalEmbedding
from .diffusion_layer import Encoder


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
        self,
        tx_dim,
        tx_depth,
        heads,
        latent_dim = None,
        max_seq_len=64,
        self_condition = False,
        dropout = 0.1,
        scale_shift = False,
        class_conditional=False,
        num_classes=0,
        class_unconditional_prob=0,
        seq2seq=False,
        seq2seq_context_dim=0,
        dual_output=False,
        num_dense_connections=0,
        dense_output_connection=False,
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
        self.cross_attend=self.seq2seq

        self.max_seq_len = max_seq_len

        # time embeddings

        sinu_pos_emb = SinusoidalPosEmb(tx_dim)
        fourier_dim = tx_dim

        time_emb_dim = tx_dim*4
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.time_pos_embed_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(time_emb_dim, tx_dim)
            )

        self.pos_emb = AbsolutePositionalEmbedding(tx_dim, max_seq_len)
        
        self.encoder = Encoder(
            dim=tx_dim,
            depth=tx_depth,
            heads=heads,
            attn_dropout = dropout,    # dropout post-attention
            ff_dropout = dropout,       # feedforward dropout
            rel_pos_bias=False,
            ff_glu=True,
            cross_attend=self.seq2seq,
            time_emb_dim=tx_dim*4 if self.scale_shift else None,
            num_dense_connections=num_dense_connections,
        )

        if self.class_conditional:
            assert num_classes > 0
            self.class_embedding = nn.Sequential(nn.Embedding(num_classes+1, tx_dim),
                                                    nn.Linear(tx_dim, time_emb_dim))
        if self.seq2seq:
            self.null_embedding_seq2seq = nn.Embedding(1, tx_dim)
            self.seq2seq_proj = nn.Linear(seq2seq_context_dim, tx_dim)
        
        if self.self_condition:
            self.input_proj = nn.Linear(latent_dim*2, tx_dim)
            self.init_self_cond = nn.Parameter(torch.randn(1, latent_dim))
            nn.init.normal_(self.init_self_cond, std = 0.02)
        else:
            self.input_proj = nn.Linear(latent_dim, tx_dim)
        self.norm = nn.LayerNorm(tx_dim)
        self.output_proj = nn.Linear(tx_dim*2 if dense_output_connection else tx_dim, latent_dim*2 if dual_output else latent_dim)

        init_zero_(self.output_proj)

    def forward(self, x, mask, time, x_self_cond = None, class_id = None, seq2seq_cond = None, seq2seq_mask = None):
        """
        x: input, [batch, length, latent_dim]
        mask: bool tensor where False indicates masked positions, [batch, length] 
        time: timestep, [batch]
        """

        time_emb = self.time_mlp(time*1000)

        time_emb = rearrange(time_emb, 'b d -> b 1 d')

        if self.class_conditional:
            assert exists(class_id)
            class_emb = self.class_embedding(class_id)
            class_emb = rearrange(class_emb, 'b d -> b 1 d')
            time_emb = time_emb + class_emb

        pos_emb = self.pos_emb(x)

        if self.self_condition:
            if exists(x_self_cond):
                x = torch.cat((x, x_self_cond), dim=-1)
            else:
                repeated_x_self_cond = repeat(self.init_self_cond, '1 d -> b l d', b=x.shape[0], l=x.shape[1])
                x = torch.cat((x, repeated_x_self_cond), dim=-1)

        x_input = self.input_proj(x)
        tx_input = x_input + pos_emb + self.time_pos_embed_mlp(time_emb)

        if self.cross_attend:
            context, context_mask = [], []
            if self.seq2seq:
                if seq2seq_cond is None:
                    null_context = repeat(self.null_embedding_seq2seq.weight, '1 d -> b 1 d', b=x.shape[0])
                    context.append(null_context)
                    context_mask.append(torch.tensor([[True] for _ in range(x.shape[0])], dtype=bool, device=x.device))
                else:
                    context.append(self.seq2seq_proj(seq2seq_cond))
                    context_mask.append(seq2seq_mask)
            context = torch.cat(context, dim=1)
            context_mask = torch.cat(context_mask, dim=1)
            
            x = self.encoder(tx_input, mask=mask, context=context, context_mask=context_mask, time_emb=time_emb)
        else:
            x = self.encoder(tx_input, mask=mask, time_emb=time_emb)

        x = self.norm(x)

        return self.output_proj(x)


ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start', 'pred_v'])

# helpers functions

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def l2norm(t):
    return F.normalize(t, dim = -1)

def log(t, eps = 1e-12):
    return torch.log(t.clamp(min = eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

# gaussian diffusion trainer class
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

# normalize variance of noised latent, if scale is not 1
def normalize_z_t_variance(z_t, mask, eps = 1e-5):
    std = rearrange([reduce(z_t[i][:torch.sum(mask[i])], 'l d -> 1 1', partial(torch.std, unbiased = False)) for i in range(z_t.shape[0])], 'b 1 1 -> b 1 1')
    return z_t / std.clamp(min = eps)
    

# noise schedules
def cosine_schedule(t, start = 0, end = 1, tau = 1, clip_min = 1e-9):
    power = 2 * tau
    v_start = math.cos(start * math.pi / 2) ** power
    v_end = math.cos(end * math.pi / 2) ** power
    output = torch.cos((t * (end - start) + start) * math.pi / 2) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min = clip_min)

# converting gamma to alpha, sigma or logsnr
def log_snr_to_alpha(log_snr):
    alpha = torch.sigmoid(log_snr)
    return alpha

def alpha_to_shifted_log_snr(alpha, scale = 1):
    return log((alpha / (1 - alpha))).clamp(min=-15, max=15) + 2*np.log(scale).item()

def time_to_alpha(t, alpha_schedule, scale):
    alpha = alpha_schedule(t)
    shifted_log_snr = alpha_to_shifted_log_snr(alpha, scale = scale)
    return log_snr_to_alpha(shifted_log_snr)

def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        config:Config,
        diffusion_for:str=None,
    ):
        super().__init__()
        if diffusion_for is not None:
            assert diffusion_for in {'context', 'embedding'}, 'diffusion_for must be one of context, embedding'
            self.diffusion_mode = config.diffusion_mode
            if diffusion_for == 'context':
                prefix = 'context_'
            elif diffusion_for == 'embedding':
                prefix = 'embedding_'
            self.diffusion_model = DiffusionTransformer(
                tx_dim = getattr(config, f'{prefix}tx_dim'),
                tx_depth = getattr(config, f'{prefix}tx_depth'),
                heads = getattr(config, f'{prefix}tx_dim') // getattr(config, f'{prefix}attn_head_dim'),
                latent_dim = getattr(config, f'{prefix}latent_dim'),
                max_seq_len = config.num_encoder_latents,
                self_condition = getattr(config, f'{prefix}self_condition'),
                scale_shift = getattr(config, f'{prefix}scale_shift'),
                dropout = getattr(config, f'{prefix}dropout'),
                class_conditional= getattr(config, f'{prefix}class_conditional'),
                num_classes= getattr(config, f'{prefix}num_classes'),    # the number of classes if class conditional else 0
                class_unconditional_prob= getattr(config, f'{prefix}class_unconditional_prob'),
                seq2seq=True,   # always True when using edison
                seq2seq_context_dim=getattr(config, f'{prefix}lm_dim'),
                num_dense_connections=getattr(config, f'{prefix}num_dense_connections'),
            )
        else:
            self.diffusion_mode = None
            self.diffusion_model = DiffusionTransformer(
                tx_dim = config.tx_dim,
                tx_depth = config.tx_depth,
                heads = config.tx_dim // config.attn_head_dim,
                latent_dim = config.latent_dim,
                max_seq_len = config.num_encoder_latents,
                self_condition = config.self_condition,
                scale_shift = config.scale_shift,
                dropout = config.dropout,
                class_conditional= config.class_conditional,
                num_classes= config.num_classes,    # the number of classes if class conditional else 0
                class_unconditional_prob= config.class_unconditional_prob,
                seq2seq=(config.dataset_name in {'xsum', 'qqp', 'qg', 'wmt14-de-en', 'wmt14-en-de'}),
                seq2seq_context_dim=config.lm_dim,
                num_dense_connections=config.num_dense_connections,)
            
        if self.diffusion_model.class_conditional:
            if self.diffusion_model.class_unconditional_prob > 0:
                self.class_unconditional_bernoulli = torch.distributions.Bernoulli(probs=self.diffusion_model.class_unconditional_prob)

        self.latent_dim = self.diffusion_model.latent_dim
        self.self_condition = self.diffusion_model.self_condition
        self.max_seq_len = config.num_encoder_latents
        self.l2_normalize = False
        self.objective = config.objective
        self.loss_type = config.loss_type
        assert self.objective in {'pred_noise', 'pred_x0', 'pred_v', 'pred_v_dual'}, 'objective must be one of pred_noise, pred_x0, pred_v, pred_v_dual'

        self.train_schedule = partial(time_to_alpha, alpha_schedule=cosine_schedule, scale=config.scale)
        # Sampling schedule
        self.sampling_schedule = partial(time_to_alpha, alpha_schedule=cosine_schedule, scale=config.scale)
        self.sampling_timesteps = config.sampling_timesteps

        # probability for self conditioning during training
        self.train_prob_self_cond = config.train_prob_self_cond

        # Buffers for latent mean and scale values
        self.register_buffer('latent_mean', torch.tensor([0]*self.latent_dim).to(torch.float32))
        self.register_buffer('latent_scale', torch.tensor(1).to(torch.float32))


    def predict_start_from_noise(self, z_t, t, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - (1-alpha).sqrt() * noise) / alpha.sqrt().clamp(min = 1e-8)
        
    def predict_noise_from_start(self, z_t, t, x0, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        return (z_t - alpha.sqrt() * x0) / (1-alpha).sqrt().clamp(min = 1e-8)

    def predict_start_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        x = alpha.sqrt() * z_t - (1-alpha).sqrt() * v

        return x
    
    def predict_noise_from_v(self, z_t, t, v, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        eps = (1-alpha).sqrt() * z_t + alpha.sqrt() * v

        return eps
    
    def predict_v_from_start_and_eps(self, z_t, t, x, noise, sampling=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        alpha = time_to_alpha(t)
        alpha = right_pad_dims_to(z_t, alpha)

        v = alpha.sqrt() * noise - x* (1-alpha).sqrt()

        return v

    def normalize_latent(self, x_start):
        eps = 1e-5 
                
        return (x_start-self.latent_mean)/(self.latent_scale).clamp(min=eps)
    
    def unnormalize_latent(self, x_start):
        eps = 1e-5 
        return x_start*(self.latent_scale.clamp(min=eps))+self.latent_mean

    def diffusion_model_predictions(self, z_t, mask, t, *, x_self_cond = None,  class_id=None, seq2seq_cond=None, seq2seq_mask=None, sampling=False, cls_free_guidance=1.0, l2_normalize=False):
        time_to_alpha = self.sampling_schedule if sampling else self.train_schedule
        time_cond = time_to_alpha(t)
        model_output = self.diffusion_model(z_t, mask, time_cond, x_self_cond, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
        if cls_free_guidance!=1.0:
            if exists(class_id):
                unc_class_id = torch.full_like(class_id, fill_value=self.diffusion_model.num_classes)
            else:
                unc_class_id = None
            unc_model_output = self.diffusion_model(z_t, mask, time_cond, x_self_cond, class_id=unc_class_id, seq2seq_cond=None, seq2seq_mask=None)
            model_output = model_output*cls_free_guidance + unc_model_output*(1-cls_free_guidance)

        pred_v = None
        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(z_t, t, pred_noise, sampling=sampling)
        elif self.objective =='pred_x0':
            x_start = model_output
            pred_noise = self.predict_noise_from_start(z_t, t, x_start, sampling=sampling)
            pred_v = self.predict_v_from_start_and_eps(z_t, t, x_start, pred_noise, sampling=sampling)
        elif self.objective == 'pred_v':
            pred_v = model_output
            x_start = self.predict_start_from_v(z_t, t, pred_v, sampling=sampling)
            pred_noise = self.predict_noise_from_v(z_t, t, pred_v, sampling=sampling)
        else:
            raise ValueError(f'invalid objective {self.objective}')
        if l2_normalize:
            assert sampling
            x_start = F.normalize(x_start, dim=-1) * math.sqrt(x_start.shape[-1])
            pred_noise = self.predict_noise_from_start(z_t, t, x_start, sampling=sampling)
            pred_v = self.predict_v_from_start_and_eps(z_t, t, x_start, pred_noise, sampling=sampling)

        return ModelPrediction(pred_noise, x_start, pred_v)

    def get_sampling_timesteps(self, batch, *, device, invert = False):
        times = torch.linspace(1., 0., self.sampling_timesteps + 1, device = device)
        if invert:
            times = times.flip(dims = (0,))
        times = repeat(times, 't -> b t', b = batch)
        times = torch.stack((times[:, :-1], times[:, 1:]), dim = 0)
        times = times.unbind(dim = -1)
        return times

    @torch.no_grad()
    def ddpm_sample(self, shape, lengths, class_id, seq2seq_cond, seq2seq_mask, cls_free_guidance=1.0, l2_normalize=False, invert=False, z_t=None):
        batch, device = shape[0], next(self.diffusion_model.parameters()).device

        time_pairs = self.get_sampling_timesteps(batch, device = device)

        if not exists(z_t):
            z_t = torch.randn(shape, device=device)

        x_start = None
        latent=None
        if self.using_latent_model:
            mask = torch.ones((shape[0], shape[1]), dtype=torch.bool, device=device)
        else:    
            mask = [[True]*length + [False]*(self.max_seq_len-length) for length in lengths]
            mask = torch.tensor(mask, dtype=torch.bool, device=device)

        for time, time_next in tqdm(time_pairs, desc = 'sampling loop time step', total = self.sampling_timesteps):
            # get predicted x0
            model_output = self.diffusion_model_predictions(z_t, mask, time, class_id=class_id, x_self_cond=x_start, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask, sampling=True, cls_free_guidance=cls_free_guidance, l2_normalize=l2_normalize)
            
            # get alpha sigma of time and next time
            alpha = self.sampling_schedule(time)
            alpha_next = self.sampling_schedule(time_next)
            alpha, alpha_next = map(partial(right_pad_dims_to, z_t), (alpha, alpha_next))

            alpha_now = alpha/alpha_next

            # # calculate x0 and noise
            x_start = model_output.pred_x_start

            eps = model_output.pred_noise
            
            if time_next[0] <= 0:
                z_t = x_start
                continue         
            
            # get noise
            noise = torch.randn_like(z_t)
            
            z_t = 1/alpha_now.sqrt() * (z_t - (1-alpha_now)/(1-alpha).sqrt() * eps) + torch.sqrt(1 - alpha_now) * noise
        return (z_t, mask)
    

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        length,
        class_id=None, seq2seq_cond=None, seq2seq_mask=None, cls_free_guidance=1.0, l2_normalize=False):
        max_seq_len, latent_dim = self.max_seq_len, self.latent_dim
        return self.ddpm_sample((batch_size, max_seq_len, latent_dim), length, class_id, seq2seq_cond, seq2seq_mask, cls_free_guidance, l2_normalize)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        elif self.loss_type == 'smooth_l1':
            return F.smooth_l1_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def forward(self, txt_latent, mask, class_id, seq2seq_cond=None, seq2seq_mask=None, return_x_start=False, *args, **kwargs):
        batch, l, d, device, max_seq_len, = *txt_latent.shape, txt_latent.device, self.max_seq_len
        assert l == max_seq_len, f'length must be {self.max_seq_len}'
        
        # sample random times
        times = torch.zeros((batch,), device = device).float().uniform_(0, 1.)  # shape = (batch_size,)

        # noise sample
        noise = torch.randn_like(txt_latent)  # shape = (batch_size, max_seq_len, latent_dim)

        alpha = self.train_schedule(times)  # shape = (batch_size,)
        alpha = right_pad_dims_to(txt_latent, alpha)  # shape = (batch_size, 1, 1)

        z_t = alpha.sqrt() * txt_latent + (1-alpha).sqrt() * noise  # shape = (batch_size, max_seq_len, latent_dim)

        # Perform unconditional generation with some probability
        if self.diffusion_model.class_conditional and self.diffusion_model.class_unconditional_prob > 0:
            assert exists(class_id)
            class_unconditional_mask = self.class_unconditional_bernoulli.sample(class_id.shape).bool()
            class_id[class_unconditional_mask] = self.diffusion_model.num_classes

        self_cond = None

        if self.self_condition and (random.random() < self.train_prob_self_cond):
            with torch.no_grad():
                model_output = self.diffusion_model_predictions(z_t, mask, times, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)
                self_cond = model_output.pred_x_start.detach()
                if self.l2_normalize:
                    self_cond = F.normalize(self_cond, dim=-1) * math.sqrt(self_cond.shape[-1])

        # predict and take gradient step
        predictions = self.diffusion_model_predictions(z_t, mask, times, x_self_cond=self_cond, class_id=class_id, seq2seq_cond=seq2seq_cond, seq2seq_mask=seq2seq_mask)          
        if self.objective == 'pred_x0':
            target = txt_latent
            pred = predictions.pred_x_start
        elif self.objective == 'pred_noise':
            target = noise
            pred = predictions.pred_noise
        elif self.objective == 'pred_v':
            target = alpha.sqrt() * noise - (1-alpha).sqrt() * txt_latent
            assert exists(predictions.pred_v)
            pred = predictions.pred_v
            
        loss = self.loss_fn(pred, target, reduction = 'none')
        loss = rearrange([reduce(loss[i][:torch.sum(mask[i])], 'l d -> 1', 'mean') for i in range(txt_latent.shape[0])], 'b 1 -> b 1')


        if return_x_start:
            return loss.mean(), predictions.pred_x_start
        return loss.mean()



