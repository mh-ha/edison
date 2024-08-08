import os
import argparse
import warnings

from lightning.pytorch.loggers import WandbLogger

from edison.modules import get_module
from edison.modules.base import BaseEdisonDiffusion
from edison.configs.base import Config
from edison.pipes.evaluate import evaluate_trained_model

os.environ['CURL_CA_BUNDLE'] = ''
warnings.filterwarnings("ignore")


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
    evaluate_trained_model(model, wandb_logger=wandb_logger)


if __name__ == '__main__':
    main()
