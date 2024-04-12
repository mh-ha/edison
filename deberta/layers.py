import torch
from torch import nn, Tensor
from einops import rearrange, einsum, reduce, repeat

from .config import Config
from .utils import MaskedLayerNorm


class InputEmbedding(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.padding_idx = config.padding_idx
        self.absolute_position_biased_input = config.absolute_position_biased_input

        self.word_embedding_layer = nn.Embedding(
            config.vocab_size,
            config.embedding_dim,
            config.padding_idx,
        )
        self.absolute_position_embedding_layer = nn.Embedding(
            config.max_seq_len,
            config.embedding_dim,
        )
        self.layernorm = nn.LayerNorm(config.hidden_dim, eps=config.layernorm_eps)
        if config.embedding_dim != config.hidden_dim:
            self.word_projection = nn.Linear(config.embedding_dim, config.hidden_dim, bias=False)
            self.position_projection = nn.Linear(config.embedding_dim, config.hidden_dim, bias=False)

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

