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
import lightning as L

from ...config.config import Config
from .third.transformer import TransformerBlock
from .third.layer import InputEmbedding


class BaseNetwork(nn.Module):
    def __init__(self, config:Config, is_generator:bool=False):
        super().__init__()
        self.config = config
        if is_generator:
            self.layers = nn.ModuleList([
                TransformerBlock(config) for _ in range(int(config.num_hidden_layers * config.gen_over_disc_ratio))])
        else:
            self.layers = nn.ModuleList([
                TransformerBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states:Tensor, attention_mask:Tensor=None, returns_all_hidden_states:bool=True):
        all_hidden_states = []
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
            if returns_all_hidden_states:
                all_hidden_states.append(hidden_states)
        return hidden_states, all_hidden_states


class EnhancedMaskDecoder(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        
    def forward(
            self,
            last_encoder_layer:nn.Module,
            last_hidden_states:Tensor,
            absolute_position_embeddings:Tensor,
        ):
        hidden_states = last_hidden_states + absolute_position_embeddings
        for _ in range(2):
            hidden_states = last_encoder_layer(last_hidden_states, q_hidden_states=hidden_states)
        return hidden_states


class MaskedLanguageModelHead(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.embedding_dim, eps=config.layernorm_eps)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
    
    def forward(self, hidden_states:Tensor, word_embedding_weights:Tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = einsum(hidden_states, word_embedding_weights, 'b n d, v d -> b n v') + self.bias
        return logits


class Generator(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.embedding = InputEmbedding(**config.__dict__)
        self.encoder = BaseNetwork(config, is_generator=True)
        self.enhanced_mask_decoder = EnhancedMaskDecoder(config)
        self.head = MaskedLanguageModelHead(config)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean')

    def forward(
            self,
            input_ids:Tensor,
            attention_mask:Tensor=None,
            labels:Tensor=None,
            returns_all_hidden_states:bool=True,
            **kwargs,
        ):
        input_embeddings = self.embedding(input_ids, attention_mask)
        hidden_states, all_hidden_states = self.encoder(input_embeddings['embeddings'], attention_mask, returns_all_hidden_states)
        hidden_states = self.enhanced_mask_decoder(self.encoder.layers[-1], all_hidden_states[-2], input_embeddings['position_embeddings'])
        output = self.head(hidden_states, self.embedding.word_embedding_layer.weight)
        if labels is not None:
            loss = self._loss_fn(output, labels)
            return output, loss
        else:
            return output

    def _loss_fn(self, logits:Tensor, labels:Tensor):
        logits = logits[labels>0].view(-1, self.config.vocab_size)
        labels = labels[labels>0].view(-1).to(torch.long)
        return self.loss_fn(logits, labels)


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


class Discriminator(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.embedding = InputEmbedding(**config.__dict__)
        self.encoder = BaseNetwork(config)
        self.enhanced_mask_decoder = EnhancedMaskDecoder(config)
        self.head = ReplacedTokenDiscriminatorHead(config)
        self.loss_fn = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(
            self,
            input_ids:Tensor,
            attention_mask:Tensor=None,
            labels:Tensor=None,
            returns_all_hidden_states:bool=True,
            **kwargs,
        ):
        input_embeddings = self.embedding(input_ids, attention_mask)
        hidden_states, all_hidden_states = self.encoder(input_embeddings['embeddings'], attention_mask, returns_all_hidden_states)
        hidden_states = self.enhanced_mask_decoder(self.encoder.layers[-1], all_hidden_states[-2], input_embeddings['position_embeddings'])
        output = self.head(hidden_states)
        if labels is not None:
            loss = self._loss_fn(output, labels)
            return output, loss
        else:
            return output

    def _loss_fn(self, logits:Tensor, labels:Tensor):
        logits = logits.view(-1)
        labels = labels.view(-1).to(logits)
        return self.loss_fn(logits, labels)
