import wandb

from edison.pipes.generate import generate_from_model
from edison.modules.lightning_data_module import get_dataset
from edison.metrics.evaluation import evaluate_model


def evaluate_trained_model(model, saved_file_name='generated_samples.csv'):
    dataset = get_dataset('roc')
    reference_data = dataset['valid']['text'] + dataset['test']['text']
    generated_data = generate_from_model(
        model=model,
        num_samples=5000,
        batch_size=128,
        seq_len=64,
        saved_file_name=saved_file_name
    )
    generated_data = generated_data['text'].tolist()

    for i in range(5):
        gen = generated_data[i*1000:(i+1)*1000]
        ref = reference_data[i*1000:(i+1)*1000]
        result = evaluate_model(gen, ref)
        wandb.log(result)
