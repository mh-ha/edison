from abc import ABC, abstractmethod
from typing import Optional, List

from torch import Tensor
from lightning import LightningModule

from edison.configs.base import Config


class BaseEdisonAE(LightningModule, ABC):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self):
        raise NotImplementedError

    @abstractmethod
    def encode(self, input_ids: Tensor, attention_masks: Optional[Tensor]):
        raise NotImplementedError

    @abstractmethod
    def decode(self, encoder_outputs: Tensor):
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError


class BaseEdisonDiffusion(LightningModule, ABC):
    def __init__(self, config: Config, autoencoder: Optional[BaseEdisonAE]):
        super().__init__()
        self.config = config
        self.autoencoder = autoencoder
        # assert (self.autoencoder is not None) or (self.config.pretrained_ae_path is not None), \
        #     "autoencoder and pretrained_ae_path cannot be None at the same time"

    @abstractmethod
    def forward(
        self,
        embedding_latents: Tensor,
        context_latents: Optional[Tensor],
        attention_mask: Optional[Tensor],
    ):
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        num_samples: int,
        seq_len: int,
        batch_size: int,
        seed: int,
    ) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError
