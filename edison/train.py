# train 진입점
"""
1. config는 parsing 했다고 가정 -> config 있음
2. config에 따라서 train을 진행
    1) LD4LG - AE
    2) LD4LG - Diffusion
    3) edison - AE
    4) edison - Diffusion
"""

from modules.lm import get_BART
from modules.ae import PerceiverAutoEncoder
from modules.lightning_modules import LD4LGAE
from trainer import get_trainer
from config.config import Config

class TrainFunction:
    def __init__(self, config:Config):
        self.config = config
        """
        0. init lightning trainer
        1. load data along with config.train_data
        2. init lightning data module
        """
        #TODO
        self.trainer = get_trainer(config)
        self.data_module = None
    
    def train_LD4LG_AE(self, config:Config):
        """
        1. init LM
        2. init AE
        3. init lightning module using LM and AE
        5. train
        """
        # 1. init LM
        lm, tokenizer = get_BART()
        
        # 2. init AE
        ae = PerceiverAutoEncoder(
            dim_lm=config.d_model,
            num_encoder_latents=config.num_encoder_latents,
            num_decoder_latents=config.num_decoder_latents,
            dim_ae=config.dim_ae,
            depth=config.num_layers,
            transformer_decoder=config.transformer_decoder,
            l2_normalize_latents=config.l2_normalize_latents)

        # 3. init lightning module using LM and AE
        model = LD4LGAE(config, lm, ae)
        
        # 4. train
        self.trainer.fit(model, datamodule=self.data_module)
    
    def train_LD4LG_Diffusion(self, config:Config):
        """
        1. init LM
        2. init pretrained AE
        3. init Diffusion
        4. init lightning module using LM, AE, Diffusion
        6. train
        """
    
    def train_edison_AE(self, config:Config):
        """
        1. init LM
        2. init AE
        3. init lightning module using LM and AE
        5. train
        """
        
    def train_edison_Diffusion(self, config:Config):
        """
        1. init LM
        2. init pretrained AE
        3. init Diffusion + ??
        """



def main(config:Config):
    train_function = TrainFunction()
    if config.model_name == 'LD4LG':
        if config.train_for == 'AE':
            train_function.train_LD4LG_AE(config)
        elif config.train_for == 'Diffusion':
            train_function.train_LD4LG_Diffusion(config)
        else:
            raise ValueError(f'{config.train_for} is not supported')
    elif config.model_name == 'edison':
        if config.train_for == 'AE':
            train_function.train_edison_AE(config)
        elif config.train_for == 'Diffusion':
            train_function.train_edison_Diffusion(config)
        else:
            raise ValueError(f'{config.train_for} is not supported')
    else:
        raise ValueError(f'{config.model_name} is not supported')
