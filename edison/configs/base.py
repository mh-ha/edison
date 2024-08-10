from dataclasses import dataclass


@dataclass
class Config:
    ############################

    # Trainer
    # strategy: str = 'ddp'
    # strategy: str = ''
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = 'norm'
    gradient_accumulation_steps: int = 1

    #############################

    # Edison general
    project_name: str = 'experiment_edison'
    dataset_name: str = 'roc'
    dataloader_name: str = 'get_dataloader'
    max_steps_ae: int = 50000
    max_steps_diffusion: int = 250000
    # train_batch_size: int = 32
    # train_batch_size_ae: int = 32
    # train_batch_size_diffusion: int = 16
    train_batch_size: int = 256
    train_batch_size_ae: int = 256
    train_batch_size_diffusion: int = 128
    max_seq_len: int = 64
    dropout: float = 0.1

    # Optimizer
    learning_rate_peak_ae: float = 1e-4
    learning_rate_peak_diffusion: float = 2e-4
    learning_rate_final: float = 0
    warmup_steps: int = 1000
    lr_schedule: str = 'cosine'

    ############################

    # Edison AE
    ae_module_name: str = 'edison_ae'
    freeze_lm: bool = True
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
    pretrained_ae_path: str = None  # path to pretrained AE. If None, AE must be passed as an argument.

    # Training
    loss_type: str = 'l2'
    train_self_cond_prob: float = 0.5
    internal_dim: int = 768
    network_depth: int = 12
    num_attn_heads: int = 12
    ff_mult: int = 4
    self_condition: bool = True
    num_dense_connections: int = 3

    # Evaluation
    eval_epoch_interval: int = 20
    eval_samples: int = 128
    eval_batch_size: int = 128
    eval_seed: int = 42

    # Generation
    sampling_timesteps: int = 250
