import argparse

import pandas as pd

from edison.modules.lightning_data_module import get_dataset
from edison.metrics.evaluation import evaluate_model

# init wandb
import wandb
wandb.login()
wandb.init(project='experiment_edison', config={'eval_name': 'baseline'})

parser = argparse.ArgumentParser()
parser.add_argument('--gen_path', type=str, default=None, help='path to generated sentences.', required=True)
# parser.add_argument('--ref_path', type=str, default=None, help='path to reference sentences.', required=True)

args = parser.parse_args()


if __name__ == '__main__':
    dataset = get_dataset('roc')
    reference_data = dataset['valid']['text'] + dataset['test']['text']
    # with open(args.gen_path, 'r') as f:
    #     generated_data = f.readlines()
    generated_data = pd.read_csv(args.gen_path)['text'].tolist()

    for i in range(5):
        gen = generated_data[i*1000:(i+1)*1000]
        ref = reference_data[i*1000:(i+1)*1000]
        result = evaluate_model(gen, ref)
        wandb.log(result)
