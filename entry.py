"""
entry point for the application
parsing arguments:
    hidden_dim:int = 768
    embedding_dim:int = 768
    padding_idx:int = 0
    vocab_size:int = 128001
    absolute_position_biased_input:bool = True
    num_heads:int = 12
    num_head_dim:int = 64
    layernorm_eps:float = 1e-9
    hidden_dropout_prob:float = 0.1
    num_hidden_layers:int = 12
    device:str = 'cuda'
    max_seq_len:int = 512
    mask_lm_prob:float = 0.15
    max_preds_per_seq:int = None
    learning_rate:float = 1e-4
    batch_size:int = 4
    gradient_clip_val:float = 1.0
    gradient_clip_algorithm:str = 'norm'`
"""

import argparse
import yaml

import lightning as L
from transformers import AutoTokenizer
from edison.config.config import Config
from edison.first.module import LM
from edison.first.datamodule import LMDataModule


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--trainer_config', type=str, default='edison/config/trainer_config.yaml')
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--embedding_dim', type=int, default=768)
    parser.add_argument('--padding_idx', type=int, default=0)
    parser.add_argument('--vocab_size', type=int, default=128001)
    parser.add_argument('--absolute_position_biased_input', type=bool, default=True)
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--num_head_dim', type=int, default=64)
    parser.add_argument('--layernorm_eps', type=float, default=1e-9)
    parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)
    parser.add_argument('--num_hidden_layers', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--mask_lm_prob', type=float, default=0.15)
    parser.add_argument('--max_preds_per_seq', type=int, default=None)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)
    parser.add_argument('--gradient_clip_algorithm', type=str, default='norm')
    parser.add_argument('--tokenizer_name', type=str, default='microsoft/deberta-v3-base')
    args = parser.parse_args()

    trainer_config = yaml.safe_load(open(args.trainer_config, 'r'))
    if args.config is not None:
        config = yaml.safe_load(open(args.config, 'r'))
        args = argparse.Namespace(**config)

    config = Config(
        hidden_dim=args.hidden_dim,
        embedding_dim=args.embedding_dim,
        padding_idx=args.padding_idx,
        vocab_size=args.vocab_size,
        absolute_position_biased_input=args.absolute_position_biased_input,
        num_heads=args.num_heads,
        num_head_dim=args.num_head_dim,
        layernorm_eps=args.layernorm_eps,
        hidden_dropout_prob=args.hidden_dropout_prob,
        num_hidden_layers=args.num_hidden_layers,
        device=args.device,
        max_seq_len=args.max_seq_len,
        mask_lm_prob=args.mask_lm_prob,
        max_preds_per_seq=args.max_preds_per_seq,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        tokenizer_name=args.tokenizer_name
    )
    print(trainer_config)
    print(config)
    model = LM(config)
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    data_module = LMDataModule(config, tokenizer)
    trainer = L.Trainer(**trainer_config)
    trainer.fit(model, datamodule=data_module)

if __name__ == '__main__':
    main()