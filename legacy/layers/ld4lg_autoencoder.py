import math

import torch
from torch import nn, einsum
from torch.optim import AdamW
import torch.nn.functional as F
import lightning as L
from einops import rearrange, repeat

from legacy.etc.config import Config
from edison.layers.positional_embedding import AbsolutePositionalEmbedding
from edison.layers.lm import get_BART
from edison.utils.utils import exists, divisible_by


class Autoencoder(L.LightningModule):
    def __init__(self, config:Config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.lm, self.tokenizer = get_BART(config.lm_model_path)
        self.lm.freeze()
        self.lm_encoder = self.lm.get_encoder()
        self.lm_decoder = self.lm.get_decoder()
        self.ae = PerceiverAutoEncoder(
            dim_lm=config.dim_lm,
            num_encoder_latents=config.num_encoder_latents,
            num_decoder_latents=config.num_decoder_latents,
            dim_ae=config.dim_ae,
            num_layers=config.num_layers,
            transformer_decoder=True,
            l2_normalize_latents=config.l2_normalize_latents)
        self.loss = None
        
    def forward_lm_encoder(self, inputs):
        return self.lm_encoder(inputs['input_ids'], attention_mask=inputs['attention_mask'])
    
    def forward_lm_decoder(self, inputs):
        return self.lm_decoder(inputs)
    
    def forward_compressor(self, inputs):
        return self.ae.perceiver_encoder(inputs)
    
    def forward_reconstructor(self, inputs):
        return self.ae.perceiver_decoder(inputs)
    
    def forward(self, inputs):
        comp_output = self.forward_compressor(inputs)
        recon_output = self.forward_reconstructor(comp_output)
        return recon_output

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        lm_encoder_outputs = self.forward_lm_encoder(inputs)
        output = self.forward(lm_encoder_outputs['last_hidden_state'])
        output = self.forward_lm_decoder(output)
        loss = self.loss(output, target)
        return loss

    def configure_optimizers(self):
        return {
            'optimizer': AdamW(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay),
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.StepLR(self.optimizers(), step_size=1, gamma=0.1),
                'interval': 'epoch',
                'frequency': 1,}
            }
        


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.gamma


def FeedForward(dim, mult=4, dropout=0.):
    hidden_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, dim)
    )

# Standard attention
class Attention(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_head = 64,
        qk_norm=True,
    ):
        super().__init__()
        num_heads = dim_input // dim_head
        assert divisible_by(dim_input, num_heads), 'dimension must be divisible by number of heads'

        self.scale = dim_head ** -0.5
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(dim_input) 
        self.query_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.to_q = nn.Linear(dim_input, dim_input, bias = False)
        self.to_k = nn.Linear(dim_input, dim_input, bias = False)
        self.to_v = nn.Linear(dim_input, dim_input, bias = False)
        self.to_out = nn.Linear(dim_input, dim_input)

    def forward(
        self,
        x,
    ):
        h = self.num_heads
        x = self.norm(x)
        qkv = (self.to_q(x), self.to_k(x), self.to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        sim = einsum('b h i d, b h j d -> b h i j', self.query_norm(q)* self.scale, self.key_norm(k))
        attn = sim.softmax(dim = -1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# attention pooling
class PerceiverAttention(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_latent,
        *,
        dim_head=64,
        qk_norm=True,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.inner_dim = max(dim_latent, dim_input)
        self.num_heads = self.inner_dim // dim_head
        self.norm = nn.LayerNorm(dim_input)
        self.norm_latents = nn.LayerNorm(dim_latent)
        self.query_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.key_norm = RMSNorm(dim_head) if qk_norm else nn.Identity()
        self.to_q = nn.Linear(dim_latent, self.inner_dim, bias=False)
        self.latent_to_kv = nn.Linear(dim_latent, self.inner_dim * 2, bias=False) if dim_latent != dim_input else None
        self.to_kv = nn.Linear(dim_input, self.inner_dim * 2, bias=False)
        self.to_out = nn.Linear(self.inner_dim, dim_latent)

    def forward(self, x, latents, mask=None):
        # init
        x = self.norm(x)
        latents = self.norm_latents(latents)
        h = self.num_heads

        # get q, k, v
        q = self.to_q(latents)
        if exists(self.latent_to_kv):
            kv_input = torch.cat([self.to_kv(x), self.latent_to_kv(latents)], dim=1)
        else:
            kv_input = torch.cat([self.to_kv(x), self.to_kv(latents)], dim=1)
        k, v = rearrange(kv_input, 'b n (split d) -> split b n d', split=2)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # attention
        attn = einsum('b h i d, b h j d  -> b h i j', self.query_norm(q) * self.scale, self.key_norm(k))
        if exists(mask):
            max_neg_value = -torch.finfo(attn.dtype).max
            mask = F.pad(mask, (0, latents.shape[-2]), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            attn = attn.masked_fill(~mask, max_neg_value)
        attn = attn.softmax(dim=-1, dtype=attn.dtype)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_latent,
        num_layers,
        *,
        dim_head=64,
        num_latents=16,
        max_seq_len=64,
        ff_mult=4,
        l2_normalize_latents=False,
    ):
        super().__init__()
        self.pos_emb = AbsolutePositionalEmbedding(dim_input, max_seq_len)
        self.latents = nn.Parameter(torch.randn(num_latents, dim_latent))
        nn.init.normal_(self.latents, std=0.02)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim_input, dim_latent, dim_head=dim_head),
                FeedForward(dim_latent, ff_mult)
            ]))
        self.l2_normalize_latents = lambda x: F.normalize(x, dim=-1) * math.sqrt(x.shape[-1]) if l2_normalize_latents else lambda x: x
        self.final_norm = nn.LayerNorm(dim_latent)

    def forward(self, x, mask=None):
        pos_emb = self.pos_emb(x)
        x = x + pos_emb
        latents = repeat(self.latents, 'n d -> b n d', b=x.shape[0])

        for attn_layer, ff_layer in self.layers:
            latents = attn_layer(x, latents, mask=mask) + latents
            latents = ff_layer(latents) + latents
        latents = self.final_norm(latents)
        latents = self.l2_normalize_latents(latents)
        return latents


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim_input,
        dim_tx,
        num_layers,
        dim_head=64,
        max_seq_len=64,
        ff_mult=4,
    ):
        super().__init__()
        self.pos_emb = AbsolutePositionalEmbedding(dim_tx, max_seq_len)
        self.input_proj = nn.Linear(dim_input, dim_tx)
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.ModuleList([
                Attention(
                    dim_input=dim_tx, dim_head=dim_head),
                FeedForward(dim=dim_tx, mult=ff_mult)
            ]))
        self.final_norm = nn.LayerNorm(dim_tx)

    def forward(self, x, mask=None):
        assert not exists(mask)
        x = self.input_proj(x)
        pos_emb = self.pos_emb(x)
        x = x + pos_emb
        for attn_layer, ff_layer in self.layers:
            x = attn_layer(x) + x
            x = ff_layer(x) + x
        return self.final_norm(x)


class PerceiverAutoEncoder(nn.Module):
    def __init__(
        self,
        dim_lm,
        dim_ae,
        num_layers,
        *,
        dim_head=64,
        num_encoder_latents=32,
        num_decoder_latents=32,
        max_seq_len=64,
        ff_mult=4,
        encoder_only=False,
        transformer_decoder=False,
        l2_normalize_latents=False,
    ):
        super().__init__()
        self.encoder_only = encoder_only
        if self.encoder_only:
            assert dim_ae == dim_lm
        self.perceiver_encoder = PerceiverResampler(
            dim_input=dim_lm, dim_latent=dim_ae, num_layers=num_layers, dim_head=dim_head,
            num_latents=num_encoder_latents, max_seq_len=max_seq_len, ff_mult=ff_mult,
            l2_normalize_latents=l2_normalize_latents)
        if transformer_decoder:
            self.perceiver_decoder = Transformer(
                dim_input=dim_ae, dim_tx=dim_lm, num_layers=num_layers,
                dim_head=dim_head, max_seq_len=num_encoder_latents, ff_mult=ff_mult)
        else:
            self.perceiver_decoder = PerceiverResampler(
                dim_input=dim_ae, dim_latent=dim_lm, num_layers=num_layers, dim_head=dim_head,
                num_latents=num_decoder_latents, max_seq_len=num_encoder_latents, ff_mult=ff_mult)

    def decode(self, ae_latent):
        return self.perceiver_decoder(ae_latent)

    def encode(self, encoder_outputs, attention_mask):
        return self.perceiver_encoder(encoder_outputs, mask=attention_mask.bool())

    def forward(self, encoder_outputs, attention_mask):
        encoder_latents = self.encode(encoder_outputs, attention_mask.bool())
        decoder_outputs = self.decode(encoder_latents)
        return decoder_outputs
