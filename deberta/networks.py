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
import random
from functools import partial

import torch
from torch import nn, Tensor
from torch import functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange, einsum, reduce, repeat
import lightning as L
from datasets import concatenate_datasets

from .config import Config
from .attentions import TransformerBlock
from .utils import MaskedLayerNorm, NGramMaskGenerator
from .data import LMDataModule
from .fetch_dataset import fetch_dataset
from .prep_dataset import split_sentences, tokenize
from .layers import InputEmbedding


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
        self.embedding = InputEmbedding(config)
        self.encoder = BaseNetwork(config)
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
        self.embedding = InputEmbedding(config)
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



class LM(L.LightningModule):
    def __init__(self, config:Config):
        super().__init__()
        self.save_hyperparameters("config")
        self.config = config
        self.generator = Generator(config)
        self.discriminator = Discriminator(config)
        self.automatic_optimization = False
        self._register_discriminator_fw_hook()

    def forward_generator(self, input_ids:Tensor, attention_mask:Tensor=None, labels:Tensor=None, **kwargs):
        return self.generator(input_ids, attention_mask, labels, **kwargs)
    
    def forward_discriminator(self, input_ids:Tensor, attention_mask:Tensor=None, labels:Tensor=None, **kwargs):
        return self.discriminator(input_ids, attention_mask, labels, **kwargs)

    def training_step(self, batch, batch_idx):
        # forward
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        output_gen, loss_gen = self.forward_generator(input_ids, attention_mask, labels)
        batch = self._get_discriminator_inputs(batch, output_gen, is_stochastic=True)
        input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
        output_disc, loss_disc = self.forward_discriminator(input_ids, attention_mask, labels)

        # optimize
        opt_gen, opt_disc = self.optimizers()
        opt_gen.zero_grad()
        self.manual_backward(loss_gen)
        self.clip_gradients(
            opt_gen,
            gradient_clip_val=self.config.gradient_clip_val,
            gradient_clip_algorithm=self.config.gradient_clip_algorithm)
        opt_gen.step()
        opt_disc.zero_grad()
        self.manual_backward(loss_disc)
        self.clip_gradients(
            opt_disc,
            gradient_clip_val=self.config.gradient_clip_val,
            gradient_clip_algorithm=self.config.gradient_clip_algorithm)
        opt_disc.step()

        # log
        self.log('loss_gen', loss_gen, on_step=True, prog_bar=True)
        self.log('loss_disc', loss_disc, on_step=True, prog_bar=True)
        # return [loss_gen, loss_disc]
    
    def configure_optimizers(self):
        gen_optimizer = torch.optim.Adam(self.generator.parameters(), lr=self.config.learning_rate)
        disc_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.config.learning_rate)
        return [gen_optimizer, disc_optimizer], []
    
    def _get_discriminator_inputs(self, masked_data, logits, is_stochastic=True, **kwargs):
        replaced_input_ids = self._replace_masked_tokens(masked_data, logits, is_stochastic, **kwargs)
        masked_data['input_ids'] = replaced_input_ids
        new_labels = self._check_labels(masked_data)
        masked_data['labels'] = new_labels
        return masked_data
    
    def _replace_masked_tokens(self, masked_data, logits, is_stochastic=True, **kwargs):
        masked_input_ids = masked_data['input_ids']
        if is_stochastic:
            probs = torch.softmax(logits, dim=-1)
            sampled_ids = torch.distributions.multinomial.Multinomial(1, probs).sample().topk(1).indices.squeeze(-1)
        else:
            sampled_ids = torch.argmax(logits, dim=-1)
        replaced_input_ids = torch.where(masked_data['labels'] > 0, sampled_ids, masked_input_ids)
        return replaced_input_ids

    def _check_labels(self, masked_data):
        replaced_input_ids = masked_data['input_ids']
        labels = masked_data['labels']
        new_labels = torch.zeros_like(labels)
        new_labels = torch.where(labels > 0, labels != replaced_input_ids, new_labels)
        return new_labels
    
    def _register_discriminator_fw_hook(self, *kwargs):
        word_bias = torch.zeros_like(self.discriminator.embedding.word_embedding_layer.weight)
        word_bias = torch.nn.Parameter(word_bias)
        position_bias = torch.zeros_like(self.discriminator.embedding.absolute_position_embedding_layer.weight)
        position_bias = torch.nn.Parameter(position_bias)
        delattr(self.discriminator.embedding.word_embedding_layer, 'weight')
        self.discriminator.embedding.word_embedding_layer.register_parameter('_weight', word_bias)
        delattr(self.discriminator.embedding.absolute_position_embedding_layer, 'weight')
        self.discriminator.embedding.absolute_position_embedding_layer.register_parameter('_weight', position_bias)

        def fw_hook(module, *inputs):
            if self.config.share_embedding == 'gdes': # Gradient-disentangled weight/embedding sharing
                g_w_ebd = self.generator.embedding.word_embedding_layer
                d_w_ebd = self.discriminator.embedding.word_embedding_layer
                self._set_param(d_w_ebd, 'weight', g_w_ebd.weight.detach() + d_w_ebd._weight)
                g_p_ebd = self.generator.embedding.absolute_position_embedding_layer
                d_p_ebd = self.discriminator.embedding.absolute_position_embedding_layer
                self._set_param(d_p_ebd, 'weight', g_p_ebd.weight.detach() + d_p_ebd._weight)

            elif self.config.share_embedding == 'es': # vallina embedding sharing
                g_w_ebd = self.generator.embedding.word_embedding_layer
                d_w_ebd = self.discriminator.embedding.word_embedding_layer
                self._set_param(d_w_ebd, 'weight', g_w_ebd.weight)
                g_p_ebd = self.generator.embedding.absolute_position_embedding_layer
                d_p_ebd = self.discriminator.embedding.absolute_position_embedding_layer
                self._set_param(d_p_ebd, 'weight', g_p_ebd.weight)
            return None
        self.discriminator.register_forward_pre_hook(fw_hook)

    @staticmethod
    def _set_param(module, param_name, value):
        if hasattr(module, param_name):
            delattr(module, param_name)
        module.register_buffer(param_name, value)
    
