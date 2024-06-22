import argparse
import warnings

from edison.config.config import EdisonConfig, LD4LGConfig
from edison.train import main as train_main

warnings.filterwarnings("ignore")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ld4lg_ae', action='store_true', help='Train LD4LG AutoEncoder model', default=False)
    parser.add_argument('--ld4lg_diffusion', action='store_true', help='Train LD4LG Diffusion model', default=False)
    parser.add_argument('--edison_ae', action='store_true', help='Train Edison AutoEncoder model', default=False)
    parser.add_argument('--edison_diffusion', action='store_true', help='Train Edison Diffusion model', default=False)
    args = parser.parse_args()
    if not any([args.ld4lg_ae, args.ld4lg_diffusion, args.edison_ae, args.edison_diffusion]):
        print('Please specify a model to train')
        return
    if sum([args.ld4lg_ae, args.ld4lg_diffusion, args.edison_ae, args.edison_diffusion]) > 1:
        print('Please specify only one model to train')
        return

    if args.ld4lg_ae:
        config = LD4LGConfig(train_for='AE')
    elif args.ld4lg_diffusion:
        config = LD4LGConfig(train_for='Diffusion')
    elif args.edison_ae:
        config = EdisonConfig(train_for='AE')
    elif args.edison_diffusion:
        config = EdisonConfig(train_for='Diffusion')
    else:
        raise ValueError('Invalid model')
    print(config)

    train_main(config)


if __name__ == '__main__':
    main()
