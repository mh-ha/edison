# from typing import Optional, Union
from dataclasses import dataclass


@dataclass
class Config:
    #############################
    # Edison general
    project_name: str = 'experiment_edison'
    model_name: str = 'Edison'
    train_for: str = 'AE'
    dataset_name: str = 'roc'
    dataloader_name: str = 'get_dataloader'
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = 'norm'
    gradient_accumulation_steps: int = 1
    max_steps_ae: int = 50000
    max_steps_diffusion: int = 250000
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
    buffer_sampling_ratio: float = 0.5   # ratio -> vocab, (1-ratio) -> batch
    ############################
    # Edison AE
    ae_module_name: str = 'edison_ae'
    dim_lm: int = 768
    dim_ae: int = 64
    num_layers: int = 3
    num_encoder_latents: int = 32
    num_decoder_latents: int = 32
    transformer_decoder: bool = True
    l2_normalize_latents: bool = True
    #############################
    # Edison Diffusion
    diffusion_module_name: str = 'baseline_diffusion'
    # pretrained_ae_path: str = 'lightning_logs/edison_ae_100k/checkpoints/epoch=275-step=100000.ckpt'
    pretrained_ae_path: str = None
    sampling_timesteps: int = 250
    loss_type: str = 'l2'
    objective: str = 'pred_v'
    scale: float = 1.
    train_prob_self_cond: float = 0.5
    tx_dim: int = 768
    tx_depth: int = 12
    num_attn_heads: int = 12
    ff_mult: int = 4
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
