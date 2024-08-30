#!/bin/bash

wandb login 6b6d10185e469255f5e09e847bd5b9087da1ae73
spacy download en_core_web_sm

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
python run.py --run_name "latent_diffusion_1gpu_$current_time" --batch_size 256

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
python run.py --run_name "latent_diffusion_8gpu_$current_time" --batch_size 32

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
python run_discrete_diffusion.py --run_name "word_diffusion_1gpu_$current_time" --batch_size 256

current_time=$(date +"%Y-%m-%d_%H-%M-%S")
python run_discrete_diffusion.py --run_name "word_diffusion_8gpu_$current_time" --batch_size 32