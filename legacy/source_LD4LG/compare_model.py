import warnings
import os

import torch
from torchinfo import summary

# import legacy.source_LD4LG.CONSTANTS as CONSTANTS
# from legacy.source_LD4LG.diffusion.text_denoising_diffusion import GaussianDiffusion
# from legacy.source_LD4LG.model.diffusion_transformer import DiffusionTransformer
import CONSTANTS
from diffusion.text_denoising_diffusion import GaussianDiffusion
from model.diffusion_transformer import DiffusionTransformer

os.environ['CURL_CA_BUNDLE'] = ''
warnings.filterwarnings("ignore")

ATTN_HEAD_DIM = 64
dataset_name = 'roc'
learning_rate = 2e-4
num_train_steps = 250000
train_batch_size = 128
tx_dim = 768
tx_depth = 12
objective = 'pred_v'
enc_dec_model = 'facebook/bart-base'
num_samples = 1000
self_condition = 'scale_shift'
loss_type = 'l2'
train_schedule = 'cosine'
wandb_name = 'roc_latent_v'
sampling_timesteps = 250
save_and_sample_every = 5000
num_dense_connections = 3
optimizer = 'adamw'
train_prob_self_cond = 0.5
max_seq_len = 32
scale_shift = False
disable_dropout = False
class_conditional = False
class_unconditional_prob = 0.1
sampler = 'ddpm'
sampling_schedule = 'cosine'
seq2seq_unconditional_prob = 0.1
scale = 1.0


latent_dim, lm_dim = 64, 768

model = DiffusionTransformer(
    tx_dim=tx_dim,
    tx_depth=tx_depth,
    heads=tx_dim//ATTN_HEAD_DIM,
    latent_dim=latent_dim,
    max_seq_len=max_seq_len,
    self_condition=self_condition,
    scale_shift=scale_shift,
    dropout=0 if disable_dropout else 0.1,
    class_conditional=class_conditional,
    num_classes=(CONSTANTS.NUM_CLASSES[dataset_name] if class_conditional else 0),
    class_unconditional_prob=class_unconditional_prob,
    seq2seq=(dataset_name in {'xsum', 'qqp', 'qg', 'wmt14-de-en', 'wmt14-en-de'}),
    seq2seq_context_dim=lm_dim,
    num_dense_connections=num_dense_connections,
)

diffusion = GaussianDiffusion(
    model,
    max_seq_len=model.max_seq_len,
    sampling_timesteps=sampling_timesteps,     # number of sampling steps
    sampler=sampler,
    train_schedule=train_schedule,
    sampling_schedule=sampling_schedule,
    loss_type=loss_type,            # L1 or L2
    objective=objective,
    train_prob_self_cond=train_prob_self_cond,
    seq2seq_unconditional_prob=seq2seq_unconditional_prob,
    scale=scale,
)
print(diffusion)

txt_latent = torch.randn(4, 32, 64)
mask = torch.ones(4, 32, dtype=torch.bool)
print(summary(
    diffusion,
    input_data=[txt_latent, mask],
    depth=10,
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
))
