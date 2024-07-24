import argparse
import warnings

from edison.configs.config import EdisonConfig, LD4LGConfig
from edison.pipes.train import main as train_main
from edison.pipes.test import evaluate_trained_model

warnings.filterwarnings("ignore")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ld4lg_ae', action='store_true', help='Train LD4LG AutoEncoder model', default=False)
    parser.add_argument('--ld4lg_diffusion', action='store_true', help='Train LD4LG Diffusion model', default=False)
    parser.add_argument('--ld4lg', action='store_true', help='Train LD4LG Diffusion model', default=False)
    parser.add_argument('--edison_ae', action='store_true', help='Train Edison AutoEncoder model', default=False)
    parser.add_argument('--edison_diffusion', action='store_true', help='Train Edison Diffusion model', default=False)
    parser.add_argument('--edison', action='store_true', help='Train Edison Diffusion model', default=False)
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training', required=False)
    parser.add_argument('--max_steps', type=int, default=50000, help='Max steps for training', required=False)
    parser.add_argument('--ae_checkpoint_path', type=str, default=None, help='Path to AE checkpoint', required=False)
    parser.add_argument('--saved_file_name', type=str, default='generated_samples.csv', help='Path to generated sentences.', required=False)
    parser.add_argument('--debug', type=str, default=None, help='Debug mode', required=False)
    args = parser.parse_args()
    if not any([args.ld4lg_ae, args.ld4lg_diffusion, args.edison_ae, args.edison_diffusion, args.ld4lg, args.edison]):
        print('Please specify a model to train')
        return
    if sum([args.ld4lg_ae, args.ld4lg_diffusion, args.edison_ae, args.edison_diffusion, args.ld4lg, args.edison]) > 1:
        print('Please specify only one model to train')
        return

    if args.ld4lg_ae:
        config = LD4LGConfig(train_for='AE', train_batch_size=args.batch_size, max_steps=args.max_steps)
    elif args.ld4lg_diffusion:
        config = LD4LGConfig(train_for='Diffusion', train_batch_size=args.batch_size, max_steps=args.max_steps)
    elif args.edison_ae:
        config = EdisonConfig(train_for='AE', train_batch_size=args.batch_size, max_steps=args.max_steps)
    elif args.edison_diffusion:
        config = EdisonConfig(train_for='Diffusion', train_batch_size=args.batch_size, max_steps=args.max_steps)
    elif args.ld4lg:
        config = LD4LGConfig(train_for='AE', train_batch_size=args.batch_size, max_steps=50000)
        autoencoder = train_main(config, **args.__dict__)
        config = LD4LGConfig(train_for='Diffusion', train_batch_size=args.batch_size, max_steps=250000)
        model = train_main(config, autoencoder, **args.__dict__)
        evaluate_trained_model(model, args.saved_file_name)
        return
    elif args.edison:
        config = EdisonConfig(train_for='AE', train_batch_size=args.batch_size, max_steps=args.max_steps)
        autoencoder = train_main(config, **args.__dict__)
        config = EdisonConfig(train_for='Diffusion', train_batch_size=args.batch_size, max_steps=args.max_steps)
        model = train_main(config, autoencoder, **args.__dict__)
        evaluate_trained_model(model, args.saved_file_name)
        return
    else:
        raise ValueError('Invalid model')
    print(config)

    model = train_main(config, **args.__dict__)
    if args.ld4lg_diffusion or args.edison_diffusion:
        evaluate_trained_model(model, args.saved_file_name)


if __name__ == '__main__':
    main()
