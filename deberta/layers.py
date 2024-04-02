import torch
from torch import nn, Tensor
from einops import rearrange, einsum, reduce, repeat

from .config import Config
from .utils import MaskedLayerNorm





# class FeedForward


# class BaseEmbedding
# required: word embedding, absolute position embedding


class RelativePositionEmbedding(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.num_head_dim = config.num_head_dim
        self.max_seq_len = config.max_seq_len
        self.relative_position_embedding_layer = nn.Embedding(config.max_seq_len, config.hidden_dim)
        self.relative_position_query_layer = nn.Linear(config.hidden_dim, config.num_heads*config.num_head_dim)
        self.relative_position_key_layer = nn.Linear(config.hidden_dim, config.num_heads*config.num_head_dim)

    def forward(self, hidden_states:Tensor, device='cuda'):
        # hidden_states: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        relative_position = self.generate_relative_position(seq_len).to(hidden_states.device)
        # relative_position: (seq_len,)
        relative_position_embedding = self.generate_relative_position_embedding(relative_position)
        # relative_position_embedding: (seq_len, hidden_dim)
        relative_position_query = self.relative_position_query_layer(relative_position_embedding)
        relative_position_query = repeat(relative_position_query, 'n d -> b n d', b=batch_size)
        relative_position_key = self.relative_position_key_layer(relative_position_embedding)
        relative_position_key = repeat(relative_position_key, 'n d -> b n d', b=batch_size)
        # relative_position_query: (batch, seq_len, hidden_dim)
        # relative_position_key: (batch, seq_len, hidden_dim)
        return (relative_position_query, relative_position_key)

    def generate_relative_position(self, seq_len:int):
        #TODO: 구체적인 relative position 생성 방법 구현
        relative_position = torch.arange(0, seq_len, 1)
        return relative_position
    
    def generate_relative_position_embedding(self, relative_position:Tensor):
        return self.relative_position_embedding_layer(relative_position)

