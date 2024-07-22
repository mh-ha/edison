import argparse

import torch

from edison.configs.config import LD4LGConfig
from edison.pipes.generate import generate_from_model
from edison.modules.lightning_modules import (
    LD4LGAE,
    LD4LGDiffusion,
    EdisonDiffusion
)
from edison.layers.lm import get_BART
from edison.layers.ld4lg_autoencoder import PerceiverAutoEncoder
from edison.modules.lightning_data_module import get_dataset
from edison.metrics.evaluation import evaluate_model

# init wandb
import wandb
wandb.login()
wandb.init(project='evaluation_baseline', config={'eval_name': 'baseline'})

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint', required=True)
parser.add_argument('--edison', action='store_true', help='Generate from Edison model', default=False)
parser.add_argument('--num_samples', type=int, default=100, help='number of samples to generate', required=False)
parser.add_argument('--batch_size', type=int, default=32, help='batch size for generation', required=False)
parser.add_argument('--seq_len', type=int, default=64, help='sequence length for generation', required=False)
parser.add_argument(
    '--saved_file_name', type=str, default='generated_samples.csv', help='file name to save generated samples', required=False
)
args = parser.parse_args()


if __name__ == '__main__':
    if args.edison:
        model = EdisonDiffusion.load_from_checkpoint(args.model_path)
    else:
        config = LD4LGConfig(train_for='Diffusion')
        lm, tokenizer = get_BART()
        ae = PerceiverAutoEncoder(
            dim_lm=config.dim_lm,
            dim_ae=config.dim_ae,
            num_layers=config.num_layers,
            num_encoder_latents=config.num_encoder_latents,
            num_decoder_latents=config.num_decoder_latents,
            transformer_decoder=config.transformer_decoder,
            l2_normalize_latents=config.l2_normalize_latents)
        autoencoder = LD4LGAE.load_from_checkpoint(
                config.pretrained_ae_path,
                map_location='cuda' if torch.cuda.is_available() else 'cpu',
                strict=False,
                config=config,
                lm=lm,
                ae=ae,
            )
        model = LD4LGDiffusion.load_from_checkpoint(args.model_path, autoencoder=autoencoder, tokenizer=tokenizer)

    dataset = get_dataset('roc')
    reference_data = dataset['valid']['text'] + dataset['test']['text']
    generated_data = generate_from_model(
        model=model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        saved_file_name=args.saved_file_name
    )
    generated_data = generated_data['text'].tolist()

    result = evaluate_model(generated_data, reference_data)
    wandb.log(result)
