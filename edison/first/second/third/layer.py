import torch
from torch import nn, Tensor
from einops import rearrange, einsum, reduce, repeat

from ...config.config import Config
from .utils import MaskedLayerNorm


class InputEmbedding(nn.Module):
    def __init__(
            self,
            embedding_dim:int=768,
            hidden_dim:int=768,
            vocab_size:int=30522,
            max_seq_len:int=512,
            padding_idx:int=0,
            layernorm_eps:float=1e-9,
            absolute_position_biased_input:bool=True,
            **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.padding_idx = padding_idx
        self.absolute_position_biased_input = absolute_position_biased_input

        self.word_embedding_layer = nn.Embedding(
            vocab_size,
            embedding_dim,
            padding_idx,
        )
        self.absolute_position_embedding_layer = nn.Embedding(
            max_seq_len,
            embedding_dim,
        )
        self.layernorm = nn.LayerNorm(hidden_dim, eps=layernorm_eps)
        if embedding_dim != hidden_dim:
            self.word_projection = nn.Linear(embedding_dim, hidden_dim, bias=False)
            self.position_projection = nn.Linear(embedding_dim, hidden_dim, bias=False)

    def forward(
            self,
            input_ids:torch.Tensor,  # (batch, seq_len)
            attention_mask:torch.Tensor=None,  # (batch, seq_len)
            position_ids:torch.Tensor=None  # (batch, seq_len)
        ):
        device = input_ids.device
        input_ids = input_ids.long()
        if not position_ids:
            input_seq_len = input_ids.shape[-1]
            position_ids = torch.arange(0, input_seq_len, dtype=torch.long, device=device)
            position_ids = repeat(position_ids, 'n -> b n', b=input_ids.shape[0])
        word_embeddings = self.word_embedding_layer(input_ids)
        position_embeddings = self.absolute_position_embedding_layer(position_ids)
        
        if self.absolute_position_biased_input:
            word_embeddings = word_embeddings + position_embeddings
        if self.embedding_dim != self.hidden_dim:
            word_embeddings = self.word_projection(word_embeddings)
            position_embeddings = self.position_projection(position_embeddings)
        word_embeddings = MaskedLayerNorm(self.layernorm, word_embeddings, attention_mask)

        return {
        'embeddings': word_embeddings,  # (batch, seq_len, hidden_dim)
        'position_embeddings': position_embeddings  # (batch, seq_len, embedding_dim)
        }


class RelativePositionEmbedding(nn.Module):
    def __init__(
            self,
            num_heads:int=12,
            num_head_dim:int=64,
            max_seq_len:int=512,
            hidden_dim:int=768,
            **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.num_head_dim = num_head_dim
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.relative_position_embedding_layer = nn.Embedding(max_seq_len, hidden_dim)
        self.relative_position_query_layer = nn.Linear(hidden_dim, num_heads*num_head_dim)
        self.relative_position_key_layer = nn.Linear(hidden_dim, num_heads*num_head_dim)

    def forward(self, hidden_states:Tensor):
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

