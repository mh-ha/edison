from dataclasses import dataclass


@dataclass
class DiscreteDiffusionConfig:
    ############################

    # Trainer
    gradient_clip_val: float = 1.0
    gradient_clip_algorithm: str = 'norm'
    gradient_accumulation_steps: int = 1

    #############################

    # Edison general
    project_name: str = 'experiment_edison'
    dataset_name: str = 'roc'
    dataloader_name: str = 'get_dataloader'
    max_steps: int = 250000
    train_batch_size: int = 32
    max_seq_len: int = 64
    dropout: float = 0.1

    # Optimizer
    learning_rate_peak: float = 2e-4
    learning_rate_final: float = 0
    warmup_steps: int = 1000
    lr_schedule: str = 'cosine'

    ############################
    # Embedding
    lm_model_name: str = 'bart-base'
    vocab_size: int = 50257
    embedding_dim: int = 768

    # Diffusion
    diffusion_module_name: str = 'discrete_diffusion'
    l2_normalize_latents: bool = True
    use_mask: bool = True
    loss_type: str = 'ce'
    train_self_cond_prob: float = 0.5
    internal_dim: int = 768
    network_depth: int = 12
    num_attn_heads: int = 12
    ff_mult: int = 4
    self_condition: bool = True
    num_dense_connections: int = 3

    # Evaluation
    eval_epoch_interval: int = 10
    eval_samples: int = 1000
    eval_batch_size: int = 125
    eval_seed: int = 42

    # Generation
    sampling_timesteps: int = 250
