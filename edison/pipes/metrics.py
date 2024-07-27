import argparse

import pandas as pd
import wandb

from edison.metrics.evaluation import (
    compute_mauve,
    compute_perplexity,
    compute_diversity,
    compute_memorization,
)


# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gen_path', type=str, default=None, help='path of generated texts', required=True)
parser.add_argument('--ref_path', type=str, default=None, help='path of human references', required=True)
parser.add_argument('--model_id', type=str, default='gpt2-large', help='model id', required=False)
parser.add_argument('--wandb_log', action='store_true', help='log to wandb', default=False)
parser.add_argument('--evaluation_name', type=str, default='evaluation', help='evaluation name', required=False)
args = parser.parse_args()


def evaluate_model(all_texts_list, human_references, model_id='gpt2-large'):
    metrics = {}
    metrics['perplexity'] = compute_perplexity(all_texts_list, model_id)
    metrics.update(compute_diversity(all_texts_list))
    metrics['memorization'] = compute_memorization(all_texts_list, human_references)
    metrics['mauve'], metrics['divergence_curve'] = compute_mauve(all_texts_list, human_references, model_id)
    return metrics


def evaluate(all_texts_list, human_references, model_id='gpt2-large', wandb_log=False, evaluation_name='evaluation'):
    metrics = evaluate_model(all_texts_list, human_references, model_id)
    # Log metrics to wandb
    if wandb_log:
        wandb.login()
        wandb.init(
            project='experiment_edison',
            config={
                "eval_name": evaluation_name,
            },
        )
        wandb.log(metrics)
    # Log metrics to csv
    else:
        metrics_df = pd.DataFrame(metrics, index=[0])
        metrics_df.to_csv(f'{evaluation_name}.csv', index=False)


if __name__ == '__main__':
    evaluate(
        all_texts_list=pd.read_csv(args.gen_path)['text'].tolist(),
        human_references=pd.read_csv(args.ref_path)['text'].tolist(),
        model_id=args.model_id,
        wandb_log=args.wandb_log,
        evaluation_name=args.evaluation_name,
    )
