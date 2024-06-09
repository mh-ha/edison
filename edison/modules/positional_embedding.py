import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange


def l2norm(t, groups = 1):
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = -1)
    return rearrange(t, '... g d -> ... (g d)')

def exists(x):
    return x is not None

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.max_seq_len = max_seq_len
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None):
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = x.device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return l2norm(pos_emb) if self.l2norm_embed else pos_emb

class ConsciousnessEmbedding(nn.Module):
    def __init__(self, dim, num_flag, l2norm_embed = False):
        super().__init__()
        self.scale = dim ** -0.5 if not l2norm_embed else 1.
        self.l2norm_embed = l2norm_embed
        self.emb = nn.Embedding(num_flag, dim)

    def forward(self, attention_mask:torch.Tensor):
        emb = self.emb.weight
        emb_1 = attention_mask.unsqueeze(-1) * emb[1].unsqueeze(0)
        # print(emb_1)
        emb_0 = (~attention_mask.bool()).long().unsqueeze(-1) * emb[0].unsqueeze(0)
        # print(emb_0)
        emb = emb_1 + emb_0
        emb = emb * self.scale
        return l2norm(emb) if self.l2norm_embed else emb


class RelativePositionEmbedding(nn.Module):
    def __init__(
            self,
            num_heads:int=12,
            num_head_dim:int=64,
            max_seq_len:int=512,
            hidden_dim:int=768,
            layer_norm_eps:float=1e-9,
            normalize_relative_embedding:bool=True,
            **kwargs
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_head_dim = num_head_dim
        self.max_seq_len = max_seq_len
        self.position_bucket = self.max_seq_len // 2
        self.hidden_dim = hidden_dim
        self.layer_norm_eps = layer_norm_eps
        self.normalize_relative_embedding = normalize_relative_embedding
        self.relative_position_embedding_layer = nn.Embedding(max_seq_len, hidden_dim)
        if normalize_relative_embedding:
            self.layernorm = nn.LayerNorm(hidden_dim, layer_norm_eps)
        self.relative_position_query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.relative_position_key_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, seq_len:int):
        relative_position_idx = self.generate_relative_position(seq_len)
        # relative_position: (1, seq_len(q), seq_len(k))
        relative_position_embedding = self.generate_relative_position_embedding()
        # relative_position_embedding: (1, max_seq_len, hidden_dim)
        relative_position_query = self.relative_position_query_layer(relative_position_embedding)
        relative_position_key = self.relative_position_key_layer(relative_position_embedding)
        relative_position_idx = torch.clamp(relative_position_idx + self.position_bucket, 0, self.position_bucket*2-1).squeeze(0)
        # relative_position_query: (max_seq_len, hidden_dim)
        # relative_position_key: (max_seq_len, hidden_dim)
        # relative_position_idx: (1, seq_len(q), seq_len(k))
        return (relative_position_query, relative_position_key, relative_position_idx)

    def generate_relative_position(self, seq_len:int):
        relative_position = self.build_relative_position(seq_len, seq_len)
        return relative_position

    def make_log_bucket_dict(self, bucket_size, max_position):
        relative_pos = torch.arange(-max_position, max_position)
        sign = torch.sign(relative_pos)
        mid = bucket_size//2
        abs_pos = torch.where((relative_pos<mid) & (relative_pos > -mid), torch.tensor(mid-1).to(relative_pos), torch.abs(relative_pos))
        log_pos = torch.ceil(torch.log(abs_pos/mid)/torch.log(torch.tensor(max_position-1)/mid) * (mid-1)) + mid
        bucket_pos = torch.where(abs_pos<=mid, relative_pos, (log_pos*sign).to(relative_pos)).to(torch.long)
        return bucket_pos
    
    def make_log_bucket_position(self, relative_pos, bucket_size, max_position):
        relative_pos = torch.clamp(relative_pos,-max_position+1, max_position-1) + max_position
        bucket_dict = self.make_log_bucket_dict(bucket_size, max_position)
        for d in range(relative_pos.dim()-1):
            bucket_dict = bucket_dict.unsqueeze(0)
            bucket_pos = torch.gather(bucket_dict.expand(list(relative_pos.size())[:-1] + [bucket_dict.size(-1)]), index=relative_pos.long(), dim=-1)
        return bucket_pos
    
    def build_relative_position(self, query_size, key_size, bucket_size=-1, max_position=-1):
        q_ids = torch.arange(0, query_size)
        k_ids = torch.arange(0, key_size)
        rel_pos_ids = q_ids.view(-1,1) - k_ids.view(1,-1)
        if bucket_size>0 and max_position > 0:
            rel_pos_ids = self.make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
        rel_pos_ids = rel_pos_ids[:query_size, :]
        rel_pos_ids = rel_pos_ids.unsqueeze(0)
        return rel_pos_ids
    
    def generate_relative_position_embedding(self):
        if self.normalize_relative_embedding:
            relative_position_embedding_weight = self.layernorm(self.relative_position_embedding_layer.weight)
        else:
            relative_position_embedding_weight = self.relative_position_embedding_layer.weight
        return relative_position_embedding_weight


"""
# Use case
time_mlp = nn.Sequential(
    sinu_pos_emb,
    nn.Linear(fourier_dim, time_emb_dim),
    nn.GELU(),
    nn.Linear(time_emb_dim, time_emb_dim)
)
"""
# 이게 기본, 나머지 트릭(norm)은 실험 후 관찰
# standardization moving average 방식
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class VariationalFourierFeatures(nn.Module):
    """ following https://arxiv.org/abs/2107.00630 """

    def __init__(self, n_min=0, n_max=8):
        super().__init__()
        assert n_min <= n_max
        self.n_min = n_min
        self.n_max = n_max

    def forward(self, x):
        fourier_features = []
        for n in range(self.n_min, self.n_max+1):
            freqs = x * (2**n) * math.pi
            fourier_features.extend([freqs.sin(), freqs.cos()])
        fouriered = rearrange(fourier_features, 'n b l d -> b l d n')
        return fouriered
