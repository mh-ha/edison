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
    share_attention_weights:bool = True
    normalize_relative_embedding:bool = True
    learning_rate:float = 1e-4
    #############################
    ## num_total_steps = num_train_steps // gradient_accumulation_steps
    ## total_batch_size = batch_size * gradient_accumulation_steps (default: 4*2048=8192)
    batch_size:int = 8
    gradient_accumulation_steps:int = 8
    #############################
    lambda_discriminator:float = 50.0
    gradient_clip_val:float = 1.0
    gradient_clip_algorithm:str = 'norm'
    tokenizer_name:str = 'microsoft/deberta-v3-base'
    share_embedding:str = 'gdes'  # 'es' or 'gdes' or None
    gen_over_disc_ratio:float = 0.5  # num_gen_encoder = num_disc_encoder * gen_over_disc_ratio
    num_trainloader_workers:int = 16
    load_pretrained_weights:bool = False
