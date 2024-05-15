from dataclasses import dataclass

@dataclass
class Config:
    #############################
    # Basic information
    model_name:str = 'LD4LG'
    train_for:str = 'Diffusion'
    dataset_name:str = 'roc'
    gradient_clip_val:float = 1.0
    gradient_clip_algorithm:str = 'norm'
    gradient_accumulation_steps:int = 1
    train_batch_size:int = 8  #TODO: 8은 되고 256은 안 되는 현상
    max_seq_len:int = 64
    learning_rate:float = 1e-4  #1e-4 for AE, 2e-4 for diffusion
    #############################
    # Perceiver AutoEncoder
    d_model:int = 768
    dim_ae:int = 64
    num_layers:int = 3
    num_encoder_latents:int = 32
    num_decoder_latents:int = 32
    transformer_decoder:bool = True
    l2_normalize_latents:bool = True
    #############################
    # LD4LG AE
    #############################
    # LD4LG Diffusion
    pretrained_ae_path:str = ''
    sampling_timesteps:int = 250
    loss_type:str = 'l2'
    objective:str = 'pred_v'
    scale:float = 1.
    train_prob_self_cond:float = 0.5
    tx_dim:int = 768
    tx_depth:int = 12
    attn_head_dim:int = 64
    latent_dim:int = 64     # must be equal to dim_ae
    lm_dim:int = 768        # must be equal to lm dim(=d_model)
    dropout:float = 0.1
    class_conditional:bool = False
    num_classes:int = 0     # depends on class_conditional and dataset_name
    class_unconditional_prob:float = 0.1
    num_samples:int = 1000
    self_condition:bool = True
    scale_shift:bool = True
    num_dense_connections:int = 3
    #############################
    # hidden_dim:int
    # embedding_dim:int
    # padding_idx:int
    # vocab_size:int
    # absolute_position_biased_input:bool
    # num_heads:int
    # num_head_dim:int
    # layernorm_eps:float
    # hidden_dropout_prob:float
    # num_hidden_layers:int
    # device:str
    # max_seq_len:int = 512
    # mask_lm_prob:float = 0.15
    # max_preds_per_seq:int = None
    # share_attention_weights:bool = True
    # normalize_relative_embedding:bool = True
    # learning_rate:float = 1e-4
    # weight_decay:float = 0.01
    # epsilon:float = 1e-7
    # beta_1:float = 0.9
    # beta_2:float = 0.98
    # #############################
    # ## num_total_steps = num_train_steps // gradient_accumulation_steps
    # ## total_batch_size = batch_size * gradient_accumulation_steps (default: 4*2048=8192)
    # batch_size:int = 8
    # gradient_accumulation_steps:int = 8
    # #############################
    # lambda_discriminator:float = 50.0
    # gradient_clip_val:float = 1.0
    # gradient_clip_algorithm:str = 'norm'
    # tokenizer_name:str = 'microsoft/deberta-v3-base'
    # share_embedding:str = 'gdes'  # 'es' or 'gdes' or None
    # gen_over_disc_ratio:float = 0.5  # num_gen_encoder = num_disc_encoder * gen_over_disc_ratio
    # num_trainloader_workers:int = 16
    # load_pretrained_weights:bool = False
    
    # #############################
    # # AutoEncoder (compression, reconstruction)
    # learning_rate:float = 1e-4
    # weight_decay:float = 0.01
    # #############################
