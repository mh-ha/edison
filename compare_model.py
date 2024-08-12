import warnings
import os

import torch
from torchinfo import summary

from edison.configs.base import Config
from edison.modules import get_module

os.environ['CURL_CA_BUNDLE'] = ''
warnings.filterwarnings("ignore")


config = Config(train_batch_size=4)
model = get_module(module_name='baseline_diffusion')(config)

txt_latent = torch.randn(4, 32, 64)
mask = torch.ones(4, 32, dtype=torch.bool)
print(summary(
    model,
    input_data=[txt_latent],
    depth=10,
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
))
