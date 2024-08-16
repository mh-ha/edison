import os
import argparse
import warnings

from lightning.pytorch.loggers import WandbLogger

from edison.configs.base import Config
from edison.pipes.train import train
from edison.pipes.evaluate import evaluate_trained_model

os.environ['CURL_CA_BUNDLE'] = ''
warnings.filterwarnings("ignore")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None, help='wandb run name', required=False)
    parser.add_argument('--batch_size', type=int, default=None, help='batch size', required=False)
    args = parser.parse_args()

    if args.batch_size:
        config = Config(
            train_batch_size=args.batch_size,
            train_batch_size_ae=args.batch_size,
            train_batch_size_diffusion=args.batch_size//2,
        )
    else:
        config = Config()
    wandb_logger = WandbLogger(
        project=config.project_name,
        config=config.__dict__,
        name=args.run_name,
        log_model=True,)
    print(config)
    model = train(config, wandb_logger=wandb_logger)
    evaluate_trained_model(model, wandb_logger=wandb_logger)


if __name__ == '__main__':
    main()
