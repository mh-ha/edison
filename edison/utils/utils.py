import math
import random
from functools import partial, wraps
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce


def exists(x):
    return x is not None

def divisible_by(numer, denom):
    return (numer % denom) == 0

def default(val, d):
    return val if exists(val) else d() if callable(d) else d

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        return fn(x, *args, **kwargs) if exists(x) else x
    return inner

def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max

def l2norm(t, groups=1):
    t = rearrange(t, '... (g d) -> ... g d', g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, '... g d -> ... (g d)')

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)

# Keyword argument helpers
def group_dict_by_key(cond, d):
    return [{k: v for k, v in d.items() if cond(k)}, {k: v for k, v in d.items() if not cond(k)}]

def string_begins_with(prefix, s):
    return s.startswith(prefix)

def groupby_prefix_and_trim(prefix, d):
    with_prefix, without_prefix = group_dict_by_key(partial(string_begins_with, prefix), d)
    return {k[len(prefix):]: v for k, v in with_prefix.items()}, without_prefix

# Initializations
def deepnorm_init(transformer, beta, module_name_match_list=['.ff.', '.to_v', '.to_out']):
    for name, module in transformer.named_modules():
        if type(module) != nn.Linear:
            continue
        gain = beta if any(sub in name for sub in module_name_match_list) else 1
        nn.init.xavier_normal_(module.weight.data, gain=gain)
        if exists(module.bias):
            nn.init.constant_(module.bias.data, 0)


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
    return F.normalize(t, dim=-1)

def log(t, eps=1e-12):
    return torch.log(t.clamp(min=eps))

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def normalize_z_t_variance(z_t, mask, eps=1e-5):
    std = rearrange([reduce(z_t[i][:torch.sum(mask[i])], 'l d -> 1 1', partial(torch.std, unbiased=False)) for i in range(z_t.shape[0])], 'b 1 1 -> b 1 1')
    return z_t / std.clamp(min=eps)


def cosine_schedule(
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


def time_to_alpha(
    t: Tensor,
    latent_ndim: Optional[int] = None,
    scale: float = 1.,
):
    alpha = cosine_schedule(t)
    shifted_log_snr = torch.log((alpha / (1 - alpha))).clamp(min=-15, max=15)
    shifted_log_snr = shifted_log_snr + (2 * math.log(scale))
    shifted_log_snr = torch.sigmoid(shifted_log_snr)
    if latent_ndim:
        while shifted_log_snr.ndim < latent_ndim:
            shifted_log_snr = shifted_log_snr.unsqueeze(-1)
    return shifted_log_snr


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class RMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.gamma
