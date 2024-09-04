import warnings
import os

import torch
from torchinfo import summary

from edison.configs.base import Config
from edison.modules import get_module

os.environ['CURL_CA_BUNDLE'] = ''
warnings.filterwarnings("ignore")


config = Config(train_batch_size=4)
# model = get_module(module_name='baseline_diffusion')(config)
model = get_module(module_name='edison_ae')(config)

# txt_latent = torch.randn(4, 32, 64)
txt_latent = torch.randint(0, 50000, size=(256, 64))
mask = torch.ones(256, 64, dtype=torch.bool)
inputs = {
    "input_ids": txt_latent,
    "attention_mask": mask,
}
print(summary(
    model,
    # input_data=[txt_latent],
    input_data=[inputs],
    depth=10,
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
))
