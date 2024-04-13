import torch
from torch import nn, Tensor
from einops import rearrange, einsum, reduce, repeat

from ...config.config import Config
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

        # relative position embedding: position generator, position embedding weights(or uses embedding forward)
        self.relative_position_embedding = RelativePositionEmbedding(**config.__dict__)

        self.feedforward = AttentionFeedForward(config)

    def forward(self, hidden_states:Tensor, q_hidden_states:Tensor=None):
        # hidden_states: (batch, seq_len, hidden_dim)
        # q_hidden_states: (batch, seq_len, hidden_dim)

        if q_hidden_states is None:
            q_hidden_states = hidden_states
        q = self.query_layer(q_hidden_states)
        k = self.key_layer(hidden_states)
        v = self.value_layer(hidden_states)
        # q, k, v: (batch, seq_len, hidden_dim)

        #TODO: relative position embedding의 input과 output 명확하게 정의
        rel_q, rel_k = self.relative_position_embedding(hidden_states)
        # rel_q, rel_k: (batch, seq_len, hidden_dim)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)
        rel_q = rearrange(rel_q, 'b n (h d) -> b h n d', h=self.num_heads)
        rel_k = rearrange(rel_k, 'b n (h d) -> b h n d', h=self.num_heads)
        # q, k, v, rel_q, rel_k: (batch, num_heads, seq_len, num_head_dim)

        attention_score = einsum(q, k, 'b h i d, b h j d -> b h i j') * self.scale
        attention_score += einsum(rel_q, k, 'b h i d, b h j d -> b h i j') * self.scale
        attention_score += einsum(q, rel_k, 'b h i d, b h j d -> b h i j') * self.scale
        attention_score = attention_score.softmax(dim=-1)
        attention_output = einsum(attention_score, v, 'b h i j, b h j d -> b h i d')
        attention_output = rearrange(attention_output, 'b h n d -> b n (h d)')
        attention_output = self.feedforward(attention_output, hidden_states)
        # attention_output: (batch, seq_len, hidden_dim)
        return attention_output
    

# class PerceiverAttention
# option: pre-, post-layernorm



