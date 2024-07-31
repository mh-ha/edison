import argparse
import warnings

from lightning.pytorch.loggers import WandbLogger

from edison.modules import get_module
from edison.modules.base import BaseEdisonDiffusion
from edison.configs.base import Config
from edison.pipes.evaluate import evaluate_trained_model

warnings.filterwarnings("ignore")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str, default=None, help='wandb run name', required=False)
    parser.add_argument('--batch_size', type=int, default=None, help='batch size', required=False)
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
    print(config)
    model: BaseEdisonDiffusion = get_module(module_name=config.diffusion_module_name).load_from_checkpoint(
        checkpoint_path="experiment_edison/5f7yiqy3/checkpoints/epoch=688-step=250000.ckpt",
        map_location='cuda',
        config=config,
        autoencoder=None
    )
    evaluate_trained_model(model, wandb_logger=wandb_logger)


if __name__ == '__main__':
    main()
