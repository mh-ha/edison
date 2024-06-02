from dataclasses import dataclass

@dataclass
class Config:
    #############################
    # General
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
    # LD4LG AE
    dim_lm:int = 768
    dim_ae:int = 64
    num_layers:int = 3
    num_encoder_latents:int = 32
    num_decoder_latents:int = 32
    transformer_decoder:bool = True
    l2_normalize_latents:bool = True
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
    # Edison general
    min_buffer_size:int = 5
    buffer_sampling_ratio:float = 0.7   # ratio -> batch, (1-ratio) -> vocab
    diffusion_mode:str = 'same' # 'same', 'context_first', 'alternately'
    #############################
    # Edison AE
    dim_lm:int = 768
    dim_ae:int = 64
    num_layers:int = 3
    num_encoder_latents:int = 32
    num_decoder_latents:int = 32
    transformer_decoder:bool = True
    l2_normalize_latents:bool = True
    encoding_mode:str = 'sentence_only'  # 'sentence_only', 'both_separately', 'both_together'
    #############################
    # Edison context Diffusion
    context_pretrained_ae_path:str = ''
    context_sampling_timesteps:int = 250
    context_loss_type:str = 'l2'
    context_objective:str = 'pred_v'
    context_scale:float = 1.
    context_train_prob_self_cond:float = 0.5
    context_tx_dim:int = 768
    context_tx_depth:int = 12
    context_attn_head_dim:int = 64
    context_latent_dim:int = 64     # must be equal to dim_ae
    context_lm_dim:int = 768        # must be equal to lm dim(=d_model)
    context_dropout:float = 0.1
    context_class_conditional:bool = False
    context_num_classes:int = 0     # depends on class_conditional and dataset_name
    context_class_unconditional_prob:float = 0.1
    context_num_samples:int = 1000
    context_self_condition:bool = True
    context_scale_shift:bool = True
    context_num_dense_connections:int = 3
    #############################
    # Edison embedding Diffusion
    embedding_pretrained_ae_path:str = ''
    embedding_sampling_timesteps:int = 250
    embedding_loss_type:str = 'l2'
    embedding_objective:str = 'pred_v'
    embedding_scale:float = 1.
    embedding_train_prob_self_cond:float = 0.5
    embedding_tx_dim:int = 768
    embedding_tx_depth:int = 12
    embedding_attn_head_dim:int = 64
    embedding_latent_dim:int = 768    # must be equal to lm dim(=d_model=embedding_dim)
    embedding_lm_dim:int = 768        # must be equal to lm dim(=d_model)
    embedding_dropout:float = 0.1
    embedding_class_conditional:bool = False
    embedding_num_classes:int = 0     # depends on class_conditional and dataset_name
    embedding_class_unconditional_prob:float = 0.1
    embedding_num_samples:int = 1000
    embedding_self_condition:bool = True
    embedding_scale_shift:bool = True
    embedding_num_dense_connections:int = 3
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
