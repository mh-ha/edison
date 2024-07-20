from typing import Tuple
import math

import torch
from torch import nn, einsum, Tensor
import torch.nn.functional as F
from einops import rearrange, repeat

from edison.layers.positional_embedding import AbsolutePositionalEmbedding
from edison.utils.utils import exists, divisible_by, RMSNorm


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.,
    ) -> None:
        super().__init__()
        hidden_dim = int(dim * mult)
        self.network = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        return self.network(x)


# Standard attention
class Attention(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_head: int = 64,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        num_heads = dim_input // dim_head
        assert divisible_by(dim_input, num_heads), 'dimension must be divisible by number of heads'

        self.scale = dim_head ** -0.5
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim_input)
        self.query_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.to_q = nn.Linear(dim_input, dim_input, bias=False)
        self.to_k = nn.Linear(dim_input, dim_input, bias=False)
        self.to_v = nn.Linear(dim_input, dim_input, bias=False)
        self.to_out = nn.Linear(dim_input, dim_input)

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        h = self.num_heads
        x = self.norm(x)
        qkv = (self.to_q(x), self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        sim = einsum('b h i d, b h j d -> b h i j', self.query_norm(q) * self.scale, self.key_norm(k))
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# edison attention
class EdisonPerceiverAttention(nn.Module):
    def __init__(
        self,
        dim_input: int,   # d_model
        dim_latent: int,  # dim_ae
        dim_head: int = 64,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.scale = dim_head ** -0.5
        self.inner_dim = max(dim_latent, dim_input)
        self.num_heads = self.inner_dim // dim_head
        self.norm = nn.LayerNorm(dim_input)
        self.norm_latents = nn.LayerNorm(dim_latent)
        self.query_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.to_q = nn.Linear(dim_latent, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_input, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim_latent)

    def forward(
        self,
        x: Tensor,
        latents_c1: Tensor,
        latents_c0: Tensor,
        consciousness_mask: Tensor = None,
    ) -> Tuple[Tensor, Tensor]:
        # init
        x = self.norm(x)
        latents_c1 = self.norm_latents(latents_c1)
        latents_c0 = self.norm_latents(latents_c0)
        h = self.num_heads

        # get q, k, v
        q_c1 = self.to_q(latents_c1)
        q_c0 = self.to_q(latents_c0)
        kv_input = self.to_kv(x)
        k, v = rearrange(kv_input, 'b n (split d) -> split b n d', split=2)
        q_c1 = rearrange(q_c1, 'b n (h d) -> b h n d', h=h)
        q_c0 = rearrange(q_c0, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        # attention
        attn_c1 = einsum('b h i d, b h j d  -> b h i j', self.query_norm(q_c1) * self.scale, self.key_norm(k))
        attn_c0 = einsum('b h i d, b h j d  -> b h i j', self.query_norm(q_c0) * self.scale, self.key_norm(k))
        if exists(consciousness_mask):
            max_neg_value = -torch.finfo(attn_c1.dtype).max
            consciousness_mask = rearrange(consciousness_mask, 'b j -> b 1 1 j')
            attn_c1 = attn_c1.masked_fill(~consciousness_mask, max_neg_value)
        attn_c1 = attn_c1.softmax(dim=-1, dtype=attn_c1.dtype)
        attn_c0 = attn_c0.softmax(dim=-1, dtype=attn_c0.dtype)
        out_c1 = einsum('b h i j, b h j d -> b h i d', attn_c1, v)
        out_c0 = einsum('b h i j, b h j d -> b h i d', attn_c0, v)
        out_c1 = rearrange(out_c1, 'b h n d -> b n (h d)', h=h)
        out_c0 = rearrange(out_c0, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out_c1), self.to_out(out_c0)


class EdisonPerceiverResampler(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_latent: int,
        num_layers: int,
        dim_head: int = 64,
        num_latents: int = 16,
        max_seq_len: int = 64,
        ff_mult: int = 4,
        l2_normalize_latents: bool = False,
    ) -> None:
        super().__init__()
        self.pos_emb = AbsolutePositionalEmbedding(dim_input, max_seq_len)
        self.latents_c1 = nn.Parameter(torch.randn(num_latents, dim_latent))
        self.latents_c0 = nn.Parameter(torch.randn(num_latents, dim_latent))
        nn.init.normal_(self.latents_c1, std=0.02)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(
                nn.ModuleList([
                    EdisonPerceiverAttention(dim_input, dim_latent, dim_head=dim_head),
                    FeedForward(dim_latent, ff_mult)
                ]))
        if l2_normalize_latents:
            self.l2_normalize_latents = lambda x: F.normalize(x, dim=-1) * math.sqrt(x.shape[-1])
        else:
            self.l2_normalize_latents = lambda x: x
        self.final_norm = nn.LayerNorm(dim_latent)

    def forward(self, x: Tensor, consciousness_mask: Tensor) -> dict[str, Tensor]:
        pos_emb = self.pos_emb(x)
        x = x + pos_emb
        latents_c1 = repeat(self.latents_c1, 'n d -> b n d', b=x.shape[0])
        latents_c0 = repeat(self.latents_c0, 'n d -> b n d', b=x.shape[0])

        for attn_layer, ff_layer in self.layers:
            in_c1 = latents_c1
            in_c0 = latents_c0
            latents_c1, latents_c0 = attn_layer(x, latents_c1, latents_c0, consciousness_mask)
            latents_c1 = latents_c1 + in_c1
            latents_c0 = latents_c0 + in_c0
            latents_c1 = ff_layer(latents_c1) + latents_c1
            latents_c0 = ff_layer(latents_c0) + latents_c0
        latents_c1 = self.final_norm(latents_c1)
        latents_c1 = self.l2_normalize_latents(latents_c1)
        latents_c0 = self.final_norm(latents_c0)
        latents_c0 = self.l2_normalize_latents(latents_c0)

        return {'latents_c1': latents_c1, 'latents_c0': latents_c0}


class Transformer(nn.Module):
    def __init__(
        self,
        dim_input: int,
        dim_tx: int,
        num_layers: int,
        dim_head: int = 64,
        max_seq_len: int = 64,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.pos_emb = AbsolutePositionalEmbedding(dim_tx, max_seq_len)
        self.input_proj = nn.Linear(dim_input, dim_tx)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Attention(dim_input=dim_tx, dim_head=dim_head),
                FeedForward(dim=dim_tx, mult=ff_mult)
            ]))
        self.final_norm = nn.LayerNorm(dim_tx)

    def forward(
        self,
        x: Tensor,
        mask: Tensor = None,
    ) -> Tensor:
        assert not exists(mask)
        x = self.input_proj(x)
        pos_emb = self.pos_emb(x)
        x = x + pos_emb
        for attn_layer, ff_layer in self.layers:
            x = attn_layer(x) + x
            x = ff_layer(x) + x
        return self.final_norm(x)


class EdisonPerceiverAutoEncoder(nn.Module):
    def __init__(
        self,
        dim_lm: int,
        dim_ae: int,
        num_layers: int,
        dim_head: int = 64,
        num_encoder_latents: int = 32,
        num_decoder_latents: int = 32,
        max_seq_len: int = 64,
        ff_mult: int = 4,
        l2_normalize_latents: int = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.perceiver_encoder = EdisonPerceiverResampler(
            dim_input=dim_lm, dim_latent=dim_ae, num_layers=num_layers, dim_head=dim_head,
            num_latents=num_encoder_latents, max_seq_len=max_seq_len, ff_mult=ff_mult,
            l2_normalize_latents=l2_normalize_latents)
        self.perceiver_decoder = Transformer(
            dim_input=dim_ae, dim_tx=dim_lm, num_layers=num_layers,
            dim_head=dim_head, max_seq_len=num_decoder_latents, ff_mult=ff_mult)

    def decode(
        self,
        ae_latent: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        ae_latent_c1 = self.perceiver_decoder(ae_latent['latents_c1'])
        ae_latent_c0 = self.perceiver_decoder(ae_latent['latents_c0'])
        return {'latents_c1': ae_latent_c1, 'latents_c0': ae_latent_c0}

    def encode(
        self,
        encoder_outputs: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        processed_encoder_outputs = self.perceiver_encoder.forward(encoder_outputs, attention_mask.bool())
        return processed_encoder_outputs

    def forward(
        self,
        encoder_outputs: Tensor,
        attention_mask: Tensor,
    ) -> dict[str, Tensor]:
        encoder_latents = self.encode(encoder_outputs, attention_mask)
        decoder_outputs = self.decode(encoder_latents)
        return decoder_outputs
