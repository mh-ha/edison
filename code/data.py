from dataclasses import dataclass
import dataset_utils.text_dataset as text_dataset

@dataclass
class Args:
    dataset_name: str
    enc_dec_model: str
    train_batch_size: int


def get_dataset(
        tokenizer,
        max_seq_len,
        dataset_name,
        enc_dec_model,
        train_batch_size,
        model_config,
        eval=False,
        ):
    dataset = text_dataset.get_dataset(
        dataset_name,
    )
    args = Args(dataset_name, enc_dec_model, train_batch_size)
    
    if eval:
        dataset['train'] = dataset['train'].select(range(1000))
    dataloader = text_dataset.get_dataloader(args, dataset['train'], model_config, tokenizer, max_seq_len, context_tokenizer=tokenizer)
    val_dataloader = text_dataset.get_dataloader(args, dataset['valid'], model_config, tokenizer, max_seq_len, shuffle=False, context_tokenizer=tokenizer)

    return dataloader, val_dataloader