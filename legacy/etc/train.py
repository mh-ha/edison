import os
from typing import Optional

from lightning.pytorch.loggers import WandbLogger

from edison.modules.lightning_modules import EdisonAE, EdisonDiffusion
from legacy.modules.lightning_data_module import get_dataset, get_dataloader, get_xtdataloader
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

    def train_LD4LG_AE(self, **kwargs):
        model = EdisonAE(self.config)

        self.dataloader = get_dataloader(
            self.config,
            self.dataset['train'],
            model.lm._get_decoder_start_token_id(),
            model.tokenizer,
            self.config.max_seq_len,
            mode='ae',)

        self.trainer.fit(model, train_dataloaders=self.dataloader)
        return model

    def train_LD4LG_Diffusion(self, model: Optional[EdisonAE] = None, **kwargs):
        checkpoint_path = kwargs.get('ae_checkpoint_path', None)
        ae_path = checkpoint_path if checkpoint_path else self.config.pretrained_ae_path

        diffusion = EdisonDiffusion(self.config, ae_path, model)

        self.dataloader = get_dataloader(
            self.config,
            self.dataset['train'],
            diffusion.autoencoder.lm._get_decoder_start_token_id(),
            diffusion.autoencoder.tokenizer,
            self.config.max_seq_len,
            mode='diffusion',)

        self.trainer.fit(diffusion, train_dataloaders=self.dataloader)
        return diffusion

    def train_edison_AE(self, **kwargs):
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

    def train_edison_Diffusion(self, model: Optional[EdisonAE] = None, **kwargs):
        """
        1. init LM
        2. init pretrained AE
        3. init Diffusions
        4. init lightning module using LM, AE, Diffusions
        5. init data loader
        6. train
        """
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


def main(config: Config, model: Optional[EdisonAE] = None, **kwargs):
    debug = kwargs.get('debug', None)
    if debug:
        print("##### Debug Mode #####")
    train_function = TrainFunction(config, debug=debug)
    if config.model_name == 'LD4LG':
        if config.train_for == 'AE':
            return train_function.train_LD4LG_AE(**kwargs)
        elif config.train_for == 'Diffusion':
            return train_function.train_LD4LG_Diffusion(model, **kwargs)
        else:
            raise ValueError(f'{config.train_for} is not supported')
    elif config.model_name == 'Edison':
        if config.train_for == 'AE':
            return train_function.train_edison_AE(**kwargs)
        elif config.train_for == 'Diffusion':
            return train_function.train_edison_Diffusion(model, **kwargs)
        else:
            raise ValueError(f'{config.train_for} is not supported')
    else:
        raise ValueError(f'{config.model_name} is not supported')
