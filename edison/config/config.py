from dataclasses import dataclass

@dataclass
class Config:
    #############################
    # Basic information
    model_name:str
    train_for:str
    train_data:str
    #############################
    # Perceiver AutoEncoder
    d_model:int = 768
    num_encoder_latents:int = 256
    num_decoder_latents:int = 256
    dim_ae:int = 256
    num_layers:int = 12
    transformer_decoder:bool = True
    l2_normalize_latents:bool = True
    #############################
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
    weight_decay:float = 0.01
    epsilon:float = 1e-7
    beta_1:float = 0.9
    beta_2:float = 0.98
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
    
    #############################
    # AutoEncoder (compression, reconstruction)
    learning_rate:float = 1e-4
    weight_decay:float = 0.01
    #############################
