import argparse
import warnings

from lightning.pytorch.loggers import WandbLogger

from edison.configs.base import Config
from edison.pipes.train import train
from edison.pipes.evaluate import evaluate_trained_model

warnings.filterwarnings("ignore")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None, help='wandb run name', required=False)
    args = parser.parse_args()

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
