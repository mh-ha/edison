import argparse

from edison.pipes.generate import generate_from_model
from edison.modules.lightning_modules import (
    EdisonDiffusion
)
from legacy.modules.lightning_data_module import get_dataset
from edison.metrics.evaluation import evaluate_model

# init wandb
import wandb
wandb.login()
wandb.init(project='experiment_edison', config={'eval_name': 'baseline'})

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default=None, help='path to model checkpoint', required=True)
parser.add_argument('--num_samples', type=int, default=100, help='number of samples to generate', required=False)
parser.add_argument('--batch_size', type=int, default=32, help='batch size for generation', required=False)
parser.add_argument('--seq_len', type=int, default=64, help='sequence length for generation', required=False)
parser.add_argument(
    '--saved_file_name', type=str, default='generated_samples.csv', help='file name to save generated samples', required=False
)
args = parser.parse_args()


if __name__ == '__main__':
    model = EdisonDiffusion.load_from_checkpoint(args.model_path)

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

    for i in range(5):
        gen = generated_data[i*1000:(i+1)*1000]
        ref = reference_data[i*1000:(i+1)*1000]
        result = evaluate_model(gen, ref)
        wandb.log(result)
