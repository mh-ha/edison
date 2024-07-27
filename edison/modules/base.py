from abc import ABC, abstractmethod
from typing import Optional

from torch import Tensor
from lightning import LightningModule


class BaseEdisonAE(LightningModule, ABC):
    def __init__(self):
        super().__init__()

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
    def __init__(self):
        super().__init__()

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
    def generate(self):
        raise NotImplementedError

    @abstractmethod
    def configure_optimizers(self):
        raise NotImplementedError
