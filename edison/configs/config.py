from dataclasses import dataclass


@dataclass
class Config:
    # ############################
    # # General
    # model_name:  str = 'LD4LG'
    # train_for:  str = 'Diffusion'
    # dataset_name:  str = 'roc'
    # gradient_clip_val:  float = 1.0
    # gradient_clip_algorithm:  str = 'norm'
    # gradient_accumulation_steps:  int = 1
    # train_batch_size:  int = 8
    # max_seq_len:  int = 64
    # learning_rate:  float = 1e-4  #1e-4 for AE, 2e-4 for diffusion
    # #############################
    # # LD4LG AE
    # dim_lm:  int = 768
    # dim_ae:  int = 64
    # num_layers:  int = 3
    # num_encoder_latents:  int = 32
    # num_decoder_latents:  int = 32
    # transformer_decoder:  bool = True
    # l2_normalize_latents:  bool = True
    # #############################
    # # LD4LG Diffusion
    # pretrained_ae_path:  str = ''
    # sampling_timesteps:  int = 250
    # loss_type:  str = 'l2'
    # objective:  str = 'pred_v'
    # scale:  float = 1.
    # train_prob_self_cond:  float = 0.5
    # tx_dim:  int = 768
    # tx_depth: int = 12
    # attn_head_dim: int = 64
    # latent_dim: int = 64     # must be equal to dim_ae
    # lm_dim: int = 768        # must be equal to lm dim(=d_model)
    # dropout: float = 0.1
    # class_conditional: bool = False
    # num_classes: int = 0     # depends on class_conditional and dataset_name
    # class_unconditional_prob: float = 0.1
    # num_samples: int = 1000
    # self_condition: bool = True
    # scale_shift: bool = True
    # num_dense_connections: int = 3
    # #############################
    # # Edison general
    # model_name: str = 'Edison'
    # train_for: str = 'AE'
    # dataset_name: str = 'roc'
    # gradient_clip_val: float = 1.0
    # gradient_clip_algorithm: str = 'norm'
    # gradient_accumulation_steps: int = 1
    # train_batch_size: int = 8  #TODO:  8은 되고 256은 안 되는 현상
    # max_seq_len: int = 64
    # learning_rate: float = 1e-4  #1e-4 for AE, 2e-4 for diffusion
    # """
    # # add extra padding tokens for buffer
    # text = text[: max_seq_len-min_buffer_size]
    # return tokenizer(
    #     text,
    #     padding="max_length",
    #     truncation=True,
    #     max_length=max_seq_len)
    # """
    # min_buffer_size: int = 5
    # buffer_sampling_ratio: float = 0.7   # ratio -> vocab, (1-ratio) -> batch
    # ############################
    # # Edison AE
    # dim_lm: int = 768
    # dim_ae: int = 64
    # num_layers: int = 3
    # num_encoder_latents: int = 32
    # num_decoder_latents: int = 32
    # transformer_decoder: bool = True
    # l2_normalize_latents: bool = True
    # encoding_mode: str = 'sentence_only'  # 'sentence_only', 'both_separately', 'both_together'
    # #############################
    # # Edison Diffusion
    # pretrained_ae_path: str = ''
    # sampling_timesteps: int = 250
    # loss_type: str = 'l2'
    # objective: str = 'pred_v'
    # scale: float = 1.
    # train_prob_self_cond: float = 0.5
    # tx_dim: int = 768
    # tx_depth: int = 12
    # attn_head_dim: int = 64
    # latent_dim: int = 64     # must be equal to dim_ae
    # lm_dim: int = 768        # must be equal to lm dim(=d_model)
    # dropout: float = 0.1
    # class_conditional: bool = False
    # num_classes: int = 0     # depends on class_conditional and dataset_name
    # class_unconditional_prob: float = 0.1
    # num_samples: int = 1000
    # self_condition: bool = True
    # scale_shift: bool = True
    # num_dense_connections: int = 3
    # feedforward_mult: int = 4
    # time_difference_embedding_over_context: int = 2
    # use_latents_c0: bool = False
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


@dataclass
class LD4LGConfig(Config):
    ############################
    # LD4LG General
    model_name: str = 'LD4LG'
    train_for: str = 'AE'
    dataset_name: str = 'roc'
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = 'norm'
    gradient_accumulation_steps: int = 1
    train_batch_size: int = 256
    max_steps: int = 50000
    max_seq_len: int = 64
    learning_rate: float = 1e-4  # 1e-4 for AE, 2e-4 for diffusion
    #############################
    # LD4LG AE
    dim_lm: int = 768
    dim_ae: int = 64
    num_layers: int = 3
    num_encoder_latents: int = 32
    num_decoder_latents: int = 32
    transformer_decoder: bool = True
    l2_normalize_latents: bool = True
    #############################
    # LD4LG Diffusion
    pretrained_ae_path: str = ''
    sampling_timesteps: int = 250
    loss_type: str = 'l2'
    objective: str = 'pred_v'
    scale: float = 1.
    train_prob_self_cond: float = 0.5
    tx_dim: int = 768
    tx_depth: int = 12
    attn_head_dim: int = 64
    latent_dim: int = 64     # must be equal to dim_ae
    lm_dim: int = 768        # must be equal to lm dim(=d_model)
    dropout: float = 0.1
    class_conditional: bool = False
    num_classes: int = 0     # depends on class_conditional and dataset_name
    class_unconditional_prob: float = 0.1
    num_samples: int = 1000
    self_condition: bool = True
    scale_shift: bool = True
    num_dense_connections: int = 3
    #############################


@dataclass
class EdisonConfig(Config):
    #############################
    # Edison general
    model_name: str = 'Edison'
    train_for: str = 'AE'
    dataset_name: str = 'roc'
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = 'norm'
    gradient_accumulation_steps: int = 1
    max_steps: int = 250000
    train_batch_size: int = 256
    max_seq_len: int = 64
    learning_rate: float = 1e-4  # 1e-4 for AE, 2e-4 for diffusion
    """
    # add extra padding tokens for buffer
    text = text[: max_seq_len-min_buffer_size]
    return tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_seq_len)
    """
    min_buffer_size: int = 5
    buffer_sampling_ratio: float = 0.7   # ratio -> vocab, (1-ratio) -> batch
    ############################
    # Edison AE
    dim_lm: int = 768
    dim_ae: int = 64
    num_layers: int = 3
    num_encoder_latents: int = 32
    num_decoder_latents: int = 32
    transformer_decoder: bool = True
    l2_normalize_latents: bool = True
    encoding_mode: str = 'sentence_only'  # 'sentence_only', 'both_separately', 'both_together'
    #############################
    # Edison Diffusion
    pretrained_ae_path: str = 'lightning_logs/edison_ae'
    sampling_timesteps: int = 250
    loss_type: str = 'l2'
    objective: str = 'pred_v'
    scale: float = 1.
    train_prob_self_cond: float = 0.5
    tx_dim: int = 768
    tx_depth: int = 12
    attn_head_dim: int = 64
    latent_dim: int = 64     # must be equal to dim_ae
    lm_dim: int = 768        # must be equal to lm dim(=d_model)
    dropout: float = 0.1
    class_conditional: bool = False
    num_classes: int = 0     # depends on class_conditional and dataset_name
    class_unconditional_prob: float = 0.1
    num_samples: int = 1000
    self_condition: bool = True
    scale_shift: bool = True
    num_dense_connections: int = 3
    feedforward_mult: int = 4
    time_difference_embedding_over_context: int = 2
    use_latents_c0: bool = False
