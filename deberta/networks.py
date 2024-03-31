"""
FLOW:
    1. mask된 input data 생성(train dataloader, collate_fn 이용)

    2. DeBERTa(disentangled attention, Enhanced Mask Decoder): Generator
        - token embedding + segment embedding
        - transformer block
            context -> V_c, K_c, Q_c
            relative PE -> Q_r, K_r
                torch.arange(0, q_size or k_size)
                -> bucket position
                -> rel position
                -> rel embedding
            QK^T = Q_r@K_c + Q_c@K_r + Q_c@K_c
            -> scale (= 1/math.sqrt(dim*scale_factor))
            -> softmax
            -> @V_c
            -> output
            -> linear (+ dropout)
            -> add
            -> layernorm
            -> output  # end attention
            -> linear + activation  # hidden dim -> intermediate dim
            -> linear (+ dropout)  # interm dim -> hidden dim
            -> add
            -> layernorm
            -> output
        - transformer block(-1:)
            output[-2] 사용: L-1개 transformers, 마지막 transformer는 enhanced mask decoder를 위한 것 (masked_language_model.py:59 참조)
            output[-2] + Absolute Position Encoding: query state
            output[-2]: key, value state (hidden state)
            query state, output[-2], rel embedding
            -> output
            -> repeat
            -> output
        - loss 계산
            mask 된 부분만 or 전체
    
    3. electra input data 생성
        - 이거 만드는 과정에서 generator 돌리고 generator loss도 계산
        - mask 된 부분 gen output으로 변경 or 전체 변경
        - input data에서 input_ids 변경 후 출력, 전체에 대한 loss 계산 위해 labels도 변경 해야 할 수 있음
    
    4. DeBERTa(disentangled attention, Enhanced Mask Decoder): Discriminator
        - GDES: Gradient-disentangled weight/embedding sharing
            generator와 embedding 공유
            + disc만의 parameter 더하기
        - 계산 방식은 generator와 같음
            prediction head만 바꿔서 사용 (+ task에 맞는 loss)
"""
import torch
from torch import nn, Tensor
from einops import rearrange, einsum, reduce, repeat

from .config import Config
from .attentions import TransformerBlock
from .utils import MaskedLayerNorm

class InputEmbedding(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        # embedding
            # embedding_dim, hidden_dim, padding_idx
        self.config = config
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.padding_idx = config.padding_idx
        self.position_biased_input = config.position_biased_input

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
            self.projection = nn.Linear(config.embedding_dim, config.hidden_dim, bias=False)

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
        
        if self.position_biased_input:
            word_embeddings = word_embeddings + position_embeddings
        
        if self.embedding_dim != self.hidden_dim:
            word_embeddings = self.projection(word_embeddings)

        word_embeddings = MaskedLayerNorm(self.layernorm, word_embeddings, attention_mask)

        return {
        'embeddings': word_embeddings,  # (batch, seq_len, hidden_dim)
        'position_embeddings': position_embeddings  # (batch, seq_len, embedding_dim)
        }

class BaseNetwork(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])

    def forward(self, hidden_states:Tensor, attention_mask:Tensor=None, returns_all_hidden_states:bool=True):
        all_hidden_states = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            if returns_all_hidden_states:
                all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states



# class Generator


# class Discriminator









