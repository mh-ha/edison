import argparse
import warnings

from edison.configs.base import Config
from edison.pipes.draft_train import train
from edison.pipes.test import evaluate_trained_model

warnings.filterwarnings("ignore")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training', required=False)
    parser.add_argument('--saved_file_name', type=str, default='generated_samples.csv', help='Path to generated sentences.', required=False)
    args = parser.parse_args()

    config = Config(train_batch_size=args.batch_size)
    print(config)
    model = train(config)
    evaluate_trained_model(model, args.saved_file_name)


if __name__ == '__main__':
    main()
