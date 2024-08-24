from typing import Optional

import torch
from lightning.pytorch.loggers import WandbLogger

from edison.modules.base import BaseEdisonDiffusion
from edison.pipes.generate import generate_from_model
from edison.modules.lightning_data_modules import get_dataset
from edison.metrics.evaluation import evaluate_model


def evaluate_trained_model(
    model: BaseEdisonDiffusion,
    saved_file_name='generated_samples.csv',
    wandb_logger: Optional[WandbLogger] = None
):
    torch.cuda.empty_cache()
    model.eval()
    model.cuda()
    dataset = get_dataset('roc')
    reference_data = dataset['valid']['text'] + dataset['test']['text']
    generated_data = generate_from_model(
        model=model,
        num_samples=5000,
        batch_size=125,
        # seq_len=64,
        saved_file_name=saved_file_name
    )
    generated_data = generated_data['text'].tolist()

    results = []
    for i in range(5):
        gen = generated_data[i*1000:(i+1)*1000]
        ref = reference_data[i*1000:(i+1)*1000]
        result = evaluate_model(gen, ref)
        results.append(result)
        if wandb_logger:
            wandb_logger.log_metrics(result, step=i)
            wandb_logger.log_table(key=f"text_generated_{i}", columns=['text'], data=[[text] for text in gen])
            wandb_logger.log_table(key=f"text_reference_{i}", columns=['text'], data=[[text] for text in ref])
    for key in results[0].keys():
        wandb_logger.log_metrics({key+"_mean": sum([result[key] for result in results])/5})
