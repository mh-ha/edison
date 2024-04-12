from dataclasses import dataclass

@dataclass
class Config:
    hidden_dim:int
    embedding_dim:int
    padding_idx:int
    vocab_size:int
    absolute_position_biased_input:bool
    num_heads:int
    num_head_dim:int
    layernorm_eps:float
    hidden_dropout_prob:float
    num_hidden_layers:int
    device:str
    max_seq_len:int = 512
    mask_lm_prob:float = 0.15
    max_preds_per_seq:int = None
    learning_rate:float = 1e-4
    batch_size:int = 4
    gradient_clip_val:float = 1.0
    gradient_clip_algorithm:str = 'norm'
    tokenizer_name:str = 'microsoft/deberta-v3-base'
    share_embedding:str = 'gdes'  # 'es' or 'gdes' or None