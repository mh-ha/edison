import torch
from torch import nn, Tensor
from einops import rearrange, einsum, reduce, repeat

from ..configs.config import Config
from .layer import RelativePositionEmbedding
from .utils import MaskedLayerNorm

class AttentionFeedForward(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.layernorm = nn.LayerNorm(config.hidden_dim, config.layernorm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_states, attention_mask=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states += input_states
        hidden_states = MaskedLayerNorm(self.layernorm, hidden_states)
        return hidden_states

class DisentangledSelfAttention(nn.Module):
    # option: pre-, post-layernorm
    def __init__(self, config:Config):
        super().__init__()
        # (batch, seq_len, num_hidden_dim) -> (batch, num_heads, seq_len, num_hidden_dim//num_heads)
        self.config = config
        self.query_layer = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.key_layer = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.value_layer = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.num_head_dim = config.num_head_dim
        self.num_heads = config.num_heads
        self.hidden_dim = config.hidden_dim
        self.scale = 1.0 / ((self.num_head_dim * 3) ** 0.5)

        self.feedforward = AttentionFeedForward(config)

    def forward(self, hidden_states:Tensor, q_hidden_states:Tensor=None, relative_position_embedding:nn.Module=None):
        # hidden_states: (batch, seq_len, hidden_dim)
        # q_hidden_states: (batch, seq_len, hidden_dim)

        if q_hidden_states is None:
            q_hidden_states = hidden_states
        q = self.query_layer(q_hidden_states)
        k = self.key_layer(hidden_states)
        v = self.value_layer(hidden_states)
        # q, k, v: (batch, seq_len, hidden_dim)

        rel_q, rel_k, rel_idx = relative_position_embedding(hidden_states, self.query_layer, self.key_layer)
        # print(rel_q.shape, rel_k.shape, rel_idx.shape)
        # rel_q, rel_k: (seq_len, hidden_dim)
        # rel_idx: (seq_len(q), seq_len(k))

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        rel_q = rearrange(rel_q, 'n (h d) -> h n d', h=self.num_heads)
        rel_k = rearrange(rel_k, 'n (h d) -> h n d', h=self.num_heads)
        rel_idx = repeat(rel_idx, 'i j -> b h i j', b=q.shape[0], h=self.num_heads)
        # print(f"q: {q.shape}, k: {k.shape}, v: {v.shape}, rel_q: {rel_q.shape}, rel_k: {rel_k.shape}, rel_idx: {rel_idx.shape}")
        # q, k, v: (batch, num_heads, seq_len, num_head_dim)
        # rel_q, rel_k: (num_heads, seq_len, num_head_dim)
        # rel_idx: (batch, num_heads, seq_len(q), seq_len(k))

        attention_score = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale
        con_q_rel_k = einsum(q, rel_k, 'b h i d, h j d -> b h i j') * self.scale
        # print(f"before con_q_rel_k: {con_q_rel_k.shape}")
        con_q_rel_k = torch.gather(con_q_rel_k, dim=-1, index=rel_idx)
        rel_q_con_k = einsum(rel_q, k, 'h i d, b h j d -> b h i j') * self.scale
        # print(f"before rel_q_con_k: {rel_q_con_k.shape}")
        rel_q_con_k = torch.gather(rel_q_con_k, dim=-2, index=rel_idx)
        # print(f"attention_score: {attention_score.shape}, con_q_rel_k: {con_q_rel_k.shape}, rel_q_con_k: {rel_q_con_k.shape}")
        attention_score += con_q_rel_k + rel_q_con_k

        attention_score = attention_score.softmax(dim=-1)
        attention_output = einsum(attention_score, v, 'b h i j, b h j d -> b h i d')
        attention_output = rearrange(attention_output, 'b h n d -> b n (h d)')
        attention_output = self.feedforward(attention_output, hidden_states)
        # attention_output: (batch, seq_len, hidden_dim)
        return attention_output
    

#TODO: implement PerceiverAttention
# class PerceiverAttention
# option: pre-, post-layernorm



