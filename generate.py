import argparse

from edison.pipes.generate import generate_from_model
from edison.modules.lightning_modules import LD4LGDiffusion, EdisonDiffusion


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
        model = LD4LGDiffusion.load_from_checkpoint(args.model_path)
    generate_from_model(
        model=model,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        saved_file_name=args.saved_file_name
    )
