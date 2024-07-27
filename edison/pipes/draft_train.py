import os
from typing import Optional

from lightning.pytorch.loggers import WandbLogger

from edison.modules.base import BaseEdisonAE, BaseEdisonDiffusion
from edison.modules.lightning_data_module import get_dataset, get_dataloader, get_xtdataloader
from edison.pipes.trainer import get_trainer
from edison.configs.base import Config

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainFunction:
    def __init__(self, config: Config, **kwargs):
        self.config = config
        debug = kwargs.get('debug', None)
        wandb_logger = WandbLogger(project="experiment_edison")
        self.trainer = get_trainer(config, debug=debug, logger=wandb_logger)
        self.dataset = get_dataset(config.dataset_name)

    def train_AE(self, **kwargs):
        model = EdisonAE(self.config)

        self.dataloader = get_xtdataloader(
            self.config,
            self.dataset['train'],
            model.lm._get_decoder_start_token_id(),
            model.tokenizer,
            self.config.max_seq_len,
            self.config.min_buffer_size,
            mode='ae',)

        self.trainer.fit(model, train_dataloaders=self.dataloader)
        return model

    def train_diffusion(self, model: Optional[BaseEdisonAE] = None, **kwargs):
        checkpoint_path = kwargs.get('ae_checkpoint_path', None)
        ae_path = checkpoint_path if checkpoint_path else self.config.pretrained_ae_path

        diffusion = EdisonDiffusion(self.config, ae_path, model)

        self.dataloader = get_xtdataloader(
            self.config,
            self.dataset['train'],
            diffusion.autoencoder.lm._get_decoder_start_token_id(),
            diffusion.tokenizer,
            self.config.max_seq_len,
            self.config.min_buffer_size,
            mode='diffusion',
        )

        self.trainer.fit(diffusion, train_dataloaders=self.dataloader)
        return diffusion
