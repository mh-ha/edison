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
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from .modules.lm import get_BART
from .modules.ae import PerceiverAutoEncoder
from .modules.lightning_modules import LD4LGAE, LD4LGDiffusion
from .modules.lightning_data_module import get_dataset, get_dataloader
from .trainer import get_trainer
from .config.config import Config

class TrainFunction:
    def __init__(self, config:Config):
        self.config = config
        """
        0. init lightning trainer
        1. load data along with config.train_data
        2. init lightning data module
        """
        self.trainer = get_trainer(config)
        self.dataset = get_dataset(config.dataset_name)

    def train_LD4LG_AE(self):
        """
        1. init LM
        2. init AE
        3. init lightning module using LM and AE
        4. init data loader
        5. train
        """
        # 1. init LM
        lm, tokenizer = get_BART()
        
        # 2. init AE
        ae = PerceiverAutoEncoder(
            dim_lm=self.config.d_model,
            dim_ae=self.config.dim_ae,
            depth=self.config.num_layers,
            num_encoder_latents=self.config.num_encoder_latents,
            num_decoder_latents=self.config.num_decoder_latents,
            transformer_decoder=self.config.transformer_decoder,
            l2_normalize_latents=self.config.l2_normalize_latents)

        # 3. init lightning module using LM and AE
        # training_step: inputs['input_ids', 'attention_mask'], target -> loss
        # forward: inputs['input_ids', 'attention_mask'] -> encoder_outputs
        model = LD4LGAE(self.config, lm, ae)
        
        # 4. init data loader
        self.dataloader = get_dataloader(
            self.config,
            self.dataset['train'],
            lm._get_decoder_start_token_id(),
            tokenizer,
            self.config.max_seq_len,
            mode='ae',)
        
        # 5. train
        self.trainer.fit(model, train_dataloaders=self.dataloader)
    
    def train_LD4LG_Diffusion(self):
        """
        1. init LM
        2. init pretrained AE
        3. init Diffusion
        4. init lightning module using LM, AE, Diffusion
        5. init data loader
        6. train
        """
        # # 1-2. load pretrained LM and AE
        # ae = LD4LGAE.load_from_checkpoint(self.config.pretrained_ae_path)
        
        # 1-2. init LM and AE
        lm, tokenizer = get_BART()
        ae = PerceiverAutoEncoder(
            dim_lm=self.config.d_model,
            dim_ae=self.config.dim_ae,
            depth=self.config.num_layers,
            num_encoder_latents=self.config.num_encoder_latents,
            num_decoder_latents=self.config.num_decoder_latents,
            transformer_decoder=self.config.transformer_decoder,
            l2_normalize_latents=self.config.l2_normalize_latents)
        model = LD4LGAE(self.config, lm, ae)
        
        # 3-4. init lightning module using LM, AE, Diffusion
        diffusion = LD4LGDiffusion(self.config, model)
        
        # 5. init data loader
        self.dataloader = get_dataloader(
            self.config,
            self.dataset['train'],
            lm._get_decoder_start_token_id(),
            tokenizer,
            self.config.max_seq_len,)
        
        # 6. train
        self.trainer.fit(diffusion, train_dataloaders=self.dataloader)
    
    def train_edison_AE(self):
        """
        1. init LM
        2. init AE
        3. init lightning module using LM and AE
        4. init data loader
        5. train
        """
        
    def train_edison_Diffusion(self):
        """
        1. init LM
        2. init pretrained AE
        3. init Diffusion + ??
        """



def main(config:Config):
    train_function = TrainFunction(config)
    if config.model_name == 'LD4LG':
        if config.train_for == 'AE':
            train_function.train_LD4LG_AE()
        elif config.train_for == 'Diffusion':
            train_function.train_LD4LG_Diffusion()
        else:
            raise ValueError(f'{config.train_for} is not supported')
    elif config.model_name == 'edison':
        if config.train_for == 'AE':
            train_function.train_edison_AE()
        elif config.train_for == 'Diffusion':
            train_function.train_edison_Diffusion()
        else:
            raise ValueError(f'{config.train_for} is not supported')
    else:
        raise ValueError(f'{config.model_name} is not supported')
