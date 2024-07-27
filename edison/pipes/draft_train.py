
import os
from typing import Optional

from lightning.pytorch.loggers import WandbLogger

from edison.modules import get_module
from edison.modules.base import BaseEdisonAE, BaseEdisonDiffusion
from edison.modules.draft_lightning_data_modules import get_dataset, get_dataloader_from_name
from edison.pipes.trainer import get_trainer
from edison.configs.base import Config

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainFunction:
    """
    abstract level.
    """
    def __init__(self, config: Config, **kwargs):
        self.config = config
        self.wandb_logger = WandbLogger(project=self.config.project_name)
        # self.trainer = get_trainer(config, logger=self.wandb_logger)
        self.dataset = get_dataset(config.dataset_name)

    def train_AE(self, **kwargs):
        # model
        model: BaseEdisonAE = get_module(module_name=self.config.ae_module_name)(self.config)

        # dataloader
        self.dataloader = get_dataloader_from_name(self.config.dataloader_name)(
            self.config,
            self.dataset['train'],
            model,
            self.config.max_seq_len,
            mode='ae',)

        # train
        self.trainer = get_trainer(self.config, logger=self.wandb_logger, max_steps=self.config.max_steps_ae)
        self.trainer.fit(model, train_dataloaders=self.dataloader)
        return model

    def train_diffusion(self, model: Optional[BaseEdisonAE] = None, **kwargs):
        # model
        diffusion: BaseEdisonDiffusion = get_module(module_name=self.config.diffusion_module_name)(self.config, model)

        # dataloader
        self.dataloader = get_dataloader_from_name(self.config.dataloader_name)(
            self.config,
            self.dataset['train'],
            diffusion.autoencoder,
            self.config.max_seq_len,
            mode='diffusion',)

        # train
        self.trainer = get_trainer(self.config, logger=self.wandb_logger, max_steps=self.config.max_steps_diffusion)
        self.trainer.fit(diffusion, train_dataloaders=self.dataloader)
        return diffusion


def train(config: Config):
    """
    high-level function.
    """
    trainer = TrainFunction(config)
    # model = trainer.train_AE()
    model: BaseEdisonAE = get_module(module_name=config.ae_module_name)(config)
    model = trainer.train_diffusion(model=model)
    return model
