from collections import namedtuple

from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

from .utils import default, max_neg_value, init_zero_, groupby_prefix_and_trim


DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple('Intermediates', ['pre_softmax_attn', 'post_softmax_attn'])
LayerIntermediates = namedtuple('Intermediates', ['hiddens', 'attn_intermediates'])


# Attention layers and helpers
class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=DEFAULT_DIM_HEAD, dropout=0., **kwargs):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        inner_dim = dim_head * heads
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.attn_fn = F.softmax

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        mask_value = max_neg_value(dots)
        if mask is not None:
            mask = rearrange(mask, 'b n -> b 1 1 n')
            dots.masked_fill_(~mask, mask_value)
        attn = self.attn_fn(dots, dim=-1)
        attn = self.dropout(attn)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0., **kwargs):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out),
            nn.Dropout(dropout)
        )
        if kwargs.get('zero_init_output', False):
            init_zero_(self.ff[-1])

    def forward(self, x):
        return self.ff(x)

class Encoder(nn.Module):
    def __init__(self, dim, depth, heads=8, **kwargs):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, kwargs = groupby_prefix_and_trim('attn_', kwargs)

        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                Attention(dim, heads=heads, **attn_kwargs),
                nn.LayerNorm(dim),
                FeedForward(dim, **ff_kwargs)
            ]))

    def forward(self, x, mask=None):
        for norm1, attn, norm2, ff in self.layers:
            x = attn(norm1(x), mask=mask) + x
            x = ff(norm2(x)) + x
        return x

