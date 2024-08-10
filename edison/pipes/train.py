
import os
from typing import Optional, Dict

from lightning.pytorch.loggers import WandbLogger

from edison.modules import get_module
from edison.modules.base import BaseEdisonAE, BaseEdisonDiffusion
from edison.modules.lightning_data_modules import get_dataset, get_dataloader_from_name
from edison.pipes.trainer import get_trainer
from edison.configs.base import Config

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainFunction:
    """
    abstract level.
    """
    def __init__(self, config: Config, wandb_logger: Dict[str, Optional[WandbLogger]], **kwargs):
        self.config = config
        self.wandb_logger_ae = wandb_logger.get('ae')
        self.wandb_logger_diffusion = wandb_logger.get('diffusion')
        self.dataset = get_dataset(config.dataset_name)

    def train_AE(self, **kwargs):
        # model
        model: BaseEdisonAE = get_module(module_name=self.config.ae_module_name)(self.config)

        # dataloader
        self.config.train_batch_size = self.config.train_batch_size_ae
        self.dataloader = get_dataloader_from_name(self.config.dataloader_name)(
            self.config,
            self.dataset['train'],
            model,
            self.config.max_seq_len,
            mode='ae',)

        # train
        self.trainer = get_trainer(self.config, logger=self.wandb_logger_ae, max_steps=self.config.max_steps_ae)
        self.trainer.fit(model, train_dataloaders=self.dataloader)
        return model

    def train_diffusion(self, autoencoder: Optional[BaseEdisonAE] = None, **kwargs):
        # model
        diffusion: BaseEdisonDiffusion = get_module(module_name=self.config.diffusion_module_name)(self.config, autoencoder)

        # dataloader
        self.config.train_batch_size = self.config.train_batch_size_ae
        self.dataloader = get_dataloader_from_name(self.config.dataloader_name)(
            self.config,
            self.dataset['train'],
            diffusion.autoencoder,
            self.config.max_seq_len,
            mode='diffusion',)

        # train
        self.trainer = get_trainer(self.config, logger=self.wandb_logger_diffusion, max_steps=self.config.max_steps_diffusion)
        self.trainer.fit(diffusion, train_dataloaders=self.dataloader)
        return diffusion


def train(config: Config, wandb_logger: Dict[str, Optional[WandbLogger]] = None):
    """
    high-level function.
    """
    trainer = TrainFunction(config, wandb_logger=wandb_logger)
    model = trainer.train_AE()
    # model: BaseEdisonAE = get_module(module_name=config.ae_module_name)(config)
    model.freeze()
    model = trainer.train_diffusion(autoencoder=model)
    return model
