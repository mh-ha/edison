import os
import argparse
import warnings
from typing import Optional

from lightning.pytorch.loggers import WandbLogger

from edison.modules import get_module
from edison.modules.base import BaseEdisonDiffusion
from edison.configs.base import Config
from edison.pipes.generate import generate_from_model
from edison.modules.lightning_data_modules import get_dataset
from edison.metrics.evaluation import evaluate_model

os.environ['CURL_CA_BUNDLE'] = ''
warnings.filterwarnings("ignore")


def evaluate_trained_model(
    model: BaseEdisonDiffusion,
    saved_file_name='generated_samples.csv',
    wandb_logger: Optional[WandbLogger] = None
):
    model.eval()
    # model.cuda()
    dataset = get_dataset('roc')
    reference_data = dataset['valid']['text'] + dataset['test']['text']
    generated_data = generate_from_model(
        model=model,
        num_samples=5000,
        batch_size=128,
        seq_len=64,
        saved_file_name=saved_file_name
    )
    generated_data = generated_data['text'].tolist()

    for i in range(5):
        gen = generated_data[i*1000:(i+1)*1000]
        ref = reference_data[i*1000:(i+1)*1000]
        result = evaluate_model(gen, ref)
        if wandb_logger:
            wandb_logger.log_metrics(result, step=i)
            wandb_logger.log_table(key=f"text_generated_{i}", columns=['text'], data=[[text] for text in gen])
            wandb_logger.log_table(key=f"text_reference_{i}", columns=['text'], data=[[text] for text in ref])


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None, help='wandb run name', required=False)
    parser.add_argument('--batch_size', type=int, default=None, help='batch size', required=False)
    parser.add_argument('--checkpoint_path', type=str, default="weights/model.ckpt", help='path to model checkpoint', required=False)
    args = parser.parse_args()

    if args.batch_size:
        config = Config(train_batch_size=args.batch_size)
    else:
        config = Config()
    wandb_logger = WandbLogger(
        project=config.project_name,
        config=config.__dict__,
        name=args.run_name,
        log_model=True,)
    print(f"[evaluation.main] config: {config}")
    model: BaseEdisonDiffusion = get_module(module_name=config.diffusion_module_name).load_from_checkpoint(
        # checkpoint_path="experiment_edison/w1ratceh/checkpoints/epoch=688-step=250000.ckpt",
        checkpoint_path=args.checkpoint_path,
        # map_location='cpu',
        config=config,
        autoencoder=None
    )
    print(f"[evaluation.main] Model loaded from {args.checkpoint_path}.")


if __name__ == '__main__':
    main()
