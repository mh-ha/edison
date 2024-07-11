"""
    각 레이어는 서로 독립적으로 구성되어야 하며, 다른 레이어와의 의존성이 없어야 한다.
"""

import torch
from torch import nn, Tensor
from einops import rearrange, einsum, reduce, repeat

from ..configs.config import Config
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
            layer_norm_eps:float=1e-9,
            share_attention_weights:bool=True,
            normalize_relative_embedding:bool=True,
            **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.num_head_dim = num_head_dim
        self.max_seq_len = max_seq_len
        self.position_bucket = self.max_seq_len // 2
        self.hidden_dim = hidden_dim
        self.layer_norm_eps = layer_norm_eps
        self.share_attention_weights = share_attention_weights
        self.normalize_relative_embedding = normalize_relative_embedding
        self.relative_position_embedding_layer = nn.Embedding(max_seq_len, hidden_dim)
        if normalize_relative_embedding:
            self.layernorm = nn.LayerNorm(hidden_dim, layer_norm_eps)
        if not share_attention_weights:
            self.relative_position_query_layer = nn.Linear(hidden_dim, hidden_dim)
            self.relative_position_key_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, hidden_states:Tensor, query_layer:nn.Linear=None, key_layer:nn.Linear=None):
        # hidden_states: (batch, seq_len, hidden_dim)
        batch_size, seq_len, hidden_dim = hidden_states.shape
        relative_position_idx = self.generate_relative_position(seq_len)
        # relative_position: (1, seq_len(q), seq_len(k))
        relative_position_embedding = self.generate_relative_position_embedding()
        # relative_position_embedding: (1, max_seq_len, hidden_dim)
        if not self.share_attention_weights:
            relative_position_query = self.relative_position_query_layer(relative_position_embedding)
            relative_position_key = self.relative_position_key_layer(relative_position_embedding)
        else:
            relative_position_query = query_layer(relative_position_embedding)
            relative_position_key = key_layer(relative_position_embedding)
        relative_position_idx = torch.clamp(relative_position_idx + self.position_bucket, 0, self.position_bucket*2-1).squeeze(0).to(hidden_states.device)
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
            relative_position_embedding_weight = MaskedLayerNorm(self.layernorm, self.relative_position_embedding_layer.weight)
        else:
            relative_position_embedding_weight = self.relative_position_embedding_layer.weight
        # relative_position_embedding = relative_position_embedding_weight[relative_position]
        # return relative_position_embedding
        return relative_position_embedding_weight


class EnhancedMaskDecoder(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        
    def forward(
            self,
            last_encoder_layer:nn.Module,
            last_hidden_states:Tensor,
            absolute_position_embeddings:Tensor,
            relative_position_embedding:nn.Module,
        ):
        hidden_states = last_hidden_states + absolute_position_embeddings
        for _ in range(2):
            hidden_states = last_encoder_layer(last_hidden_states, q_hidden_states=hidden_states, relative_position_embedding=relative_position_embedding)
        return hidden_states


class MaskedLanguageModelHead(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(config.embedding_dim, eps=config.layernorm_eps)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
    
    def forward(self, hidden_states:Tensor, word_embedding_weights:Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        logits = einsum(hidden_states, word_embedding_weights, 'b n d, v d -> b n v') + self.bias
        return logits


class ReplacedTokenDiscriminatorHead(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(config.hidden_dim, eps=config.layernorm_eps)
        self.classifier = nn.Linear(config.hidden_dim, 1)
    
    def forward(self, hidden_states:Tensor):
        hidden_states = self.dense(hidden_states)  # (batch, seq_len, hidden_dim)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        logits = self.classifier(hidden_states)
        return logits  # (batch, seq_len, 1)