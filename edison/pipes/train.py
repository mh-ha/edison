# train 진입점
"""
1. config는 parsing 했다고 가정 -> config 있음
2. config에 따라서 train을 진행
    1) LD4LG - AE
    2) LD4LG - Diffusion
    3) edison - AE
    4) edison - Diffusion
"""
import os
from typing import Optional, Union

import torch

from edison.layers.lm import get_BART
from edison.layers.ld4lg_autoencoder import PerceiverAutoEncoder
# from edison.modules.lightning_modules import LD4LGAE, LD4LGDiffusion, EdisonAE, EdisonDiffusion
from edison.modules.lightning_modules import LD4LGAE, LD4LGDiffusion
from edison.modules.draft_lightning_modules import EdisonAE, EdisonDiffusion
from edison.modules.lightning_data_module import get_dataset, get_dataloader, get_xtdataloader
from edison.pipes.trainer import get_trainer
from edison.configs.config import Config

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TrainFunction:
    def __init__(self, config: Config, **kwargs):
        self.config = config
        """
        0. init lightning trainer
        1. load data along with config.train_data
        2. init lightning data module
        """
        debug = kwargs.get('debug', None)
        self.trainer = get_trainer(config, debug=debug)
        self.dataset = get_dataset(config.dataset_name)

    # def train_LD4LG_AE(self, **kwargs):
    #     """
    #     1. init LM
    #     2. init AE
    #     3. init lightning module using LM and AE
    #     4. init data loader
    #     5. train
    #     """
    #     # 1. init LM
    #     lm, tokenizer = get_BART()

    #     # 2. init AE
    #     ae = PerceiverAutoEncoder(
    #         dim_lm=self.config.dim_lm,
    #         dim_ae=self.config.dim_ae,
    #         num_layers=self.config.num_layers,
    #         num_encoder_latents=self.config.num_encoder_latents,
    #         num_decoder_latents=self.config.num_decoder_latents,
    #         transformer_decoder=self.config.transformer_decoder,
    #         l2_normalize_latents=self.config.l2_normalize_latents)

    #     # 3. init lightning module using LM and AE
    #     # training_step: inputs['input_ids', 'attention_mask'], target -> loss
    #     # forward: inputs['input_ids', 'attention_mask'] -> encoder_outputs
    #     model = LD4LGAE(self.config, lm, ae)

    #     # 4. init data loader
    #     self.dataloader = get_dataloader(
    #         self.config,
    #         self.dataset['train'],
    #         lm._get_decoder_start_token_id(),
    #         tokenizer,
    #         self.config.max_seq_len,
    #         mode='ae',)

    #     # 5. train
    #     self.trainer.fit(model, train_dataloaders=self.dataloader)
    #     return model

    # def train_LD4LG_Diffusion(self, model: Optional[LD4LGAE] = None, **kwargs):
    #     """
    #     1. init LM
    #     2. init pretrained AE
    #     3. init Diffusion
    #     4. init lightning module using LM, AE, Diffusion
    #     5. init data loader
    #     6. train
    #     """
    #     # 1-2. load pretrained LM and AE
    #     if model is None:
    #         checkpoint_path = kwargs.get('ae_checkpoint_path', None)
    #         lm, tokenizer = get_BART()
    #         ae = PerceiverAutoEncoder(
    #             dim_lm=self.config.dim_lm,
    #             dim_ae=self.config.dim_ae,
    #             num_layers=self.config.num_layers,
    #             num_encoder_latents=self.config.num_encoder_latents,
    #             num_decoder_latents=self.config.num_decoder_latents,
    #             transformer_decoder=self.config.transformer_decoder,
    #             l2_normalize_latents=self.config.l2_normalize_latents)
    #         # print(f"debug - ae_params : {list(ae.parameters())[0]}")
    #         if checkpoint_path or self.config.pretrained_ae_path:
    #             path = checkpoint_path if checkpoint_path else self.config.pretrained_ae_path
    #             model = LD4LGAE.load_from_checkpoint(
    #                 path,
    #                 map_location='cuda' if torch.cuda.is_available() else 'cpu',
    #                 strict=False,
    #                 config=self.config,
    #                 lm=lm,
    #                 ae=ae,
    #             )
    #             print("Model loaded from checkpoint.")
    #         else:
    #             model = LD4LGAE(self.config, lm, ae)
    #             print("Model initialized.")
    #         # print(f"debug - model_params : {list(model.ae.parameters())[0]}")

    #     # 3-4. init lightning module using LM, AE, Diffusion
    #     diffusion = LD4LGDiffusion(self.config, model, tokenizer)

    #     # 5. init data loader
    #     self.dataloader = get_dataloader(
    #         self.config,
    #         self.dataset['train'],
    #         lm._get_decoder_start_token_id(),
    #         tokenizer,
    #         self.config.max_seq_len,
    #         mode='diffusion',)

    #     # 6. train
    #     # print(diffusion)
    #     self.trainer.fit(diffusion, train_dataloaders=self.dataloader)
    #     return diffusion

    def train_LD4LG_AE(self, **kwargs):
        """
        1. init LM
        2. init AE
        3. init lightning module using LM and AE
        4. init data loader
        5. train
        """
        model = EdisonAE(self.config)

        # 4. init data loader
        self.dataloader = get_dataloader(
            self.config,
            self.dataset['train'],
            model.lm._get_decoder_start_token_id(),
            model.tokenizer,
            self.config.max_seq_len,
            mode='ae',)

        # 5. train
        self.trainer.fit(model, train_dataloaders=self.dataloader)
        return model

    def train_LD4LG_Diffusion(self, model: Union[LD4LGAE, EdisonAE, None] = None, **kwargs):
        # 1-2. load pretrained LM and AE
        checkpoint_path = kwargs.get('ae_checkpoint_path', None)
        ae_path = checkpoint_path if checkpoint_path else self.config.pretrained_ae_path

        # 3-4. init lightning module using LM, AE, Diffusion
        diffusion = EdisonDiffusion(self.config, ae_path, model)

        # 5. init data loader
        self.dataloader = get_dataloader(
            self.config,
            self.dataset['train'],
            diffusion.autoencoder.lm._get_decoder_start_token_id(),
            diffusion.autoencoder.tokenizer,
            self.config.max_seq_len,
            mode='diffusion',)

        # 6. train
        # print(diffusion)
        self.trainer.fit(diffusion, train_dataloaders=self.dataloader)
        return diffusion

    def train_edison_AE(self, **kwargs):
        """
        1. init LM
        2. init AE
        3. init lightning module using LM and AE
        4. init data loader
        5. mapping data to xt_data
        6. train
        """
        # training_step: inputs['input_ids', 'attention_mask'], target -> loss
        # forward: inputs['input_ids', 'attention_mask'] -> encoder_outputs
        model = EdisonAE(self.config)

        # 4. init data loader
        self.dataloader = get_xtdataloader(
            self.config,
            self.dataset['train'],
            model.lm._get_decoder_start_token_id(),
            model.tokenizer,
            self.config.max_seq_len,
            self.config.min_buffer_size,
            mode='ae',)

        # 5. train
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
        # 1-2. load pretrained LM and AE
        checkpoint_path = kwargs.get('ae_checkpoint_path', None)
        ae_path = checkpoint_path if checkpoint_path else self.config.pretrained_ae_path

        # 3-4. init lightning module using LM, AE, Diffusion
        diffusion = EdisonDiffusion(self.config, ae_path, model)

        # 5. init data loader
        self.dataloader = get_xtdataloader(
            self.config,
            self.dataset['train'],
            diffusion.autoencoder.lm._get_decoder_start_token_id(),
            diffusion.tokenizer,
            self.config.max_seq_len,
            self.config.min_buffer_size,
            mode='diffusion',
        )

        # 6. train
        self.trainer.fit(diffusion, train_dataloaders=self.dataloader)
        return diffusion


def main(config: Config, model: Union[LD4LGAE, EdisonAE, None] = None, **kwargs):
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
