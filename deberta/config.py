from dataclasses import dataclass

@dataclass
class Config:
    hidden_dim:int
    embedding_dim:int
    max_seq_len:int
    padding_idx:int
    vocab_size:int
    position_biased_input:bool
    