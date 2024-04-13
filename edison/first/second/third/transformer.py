from torch import nn, Tensor

from ...config.config import Config
from .attention import DisentangledSelfAttention

class TransformerFeedForward(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.intermediate_dim = config.hidden_dim * 4
        self.feedforward_1 = nn.Linear(config.hidden_dim, self.intermediate_dim)
        self.activation = nn.GELU()
        self.feedforward_2 = nn.Linear(self.intermediate_dim, config.hidden_dim)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layernorm = nn.LayerNorm(config.hidden_dim, eps=config.layernorm_eps)

    def forward(self, hidden_states:Tensor):
        x = self.feedforward_1(hidden_states)
        x = self.activation(x)
        x = self.feedforward_2(x)
        x = self.dropout(x)
        x += hidden_states
        output = self.layernorm(x)
        return output

class TransformerBlock(nn.Module):
    def __init__(self, config:Config):
        super().__init__()
        self.config = config
        self.attention = DisentangledSelfAttention(config)
        self.feedforward = TransformerFeedForward(config)

    def forward(self, hidden_states:Tensor, attention_mask:Tensor=None, q_hidden_states:Tensor=None):
        attention_output = self.attention(hidden_states, q_hidden_states=q_hidden_states)
        feedforward_output = self.feedforward(attention_output)
        return feedforward_output