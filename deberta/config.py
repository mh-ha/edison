from dataclasses import dataclass

@dataclass
class Config:
    hidden_dim:int
    embedding_dim:int
    max_seq_len:int
    padding_idx:int
    vocab_size:int
    position_biased_input:bool
    num_heads:int
    num_head_dim:int
    layernorm_eps:float
    hidden_dropout_prob:float
    num_hidden_layers:int