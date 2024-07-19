from abc import ABC, abstractmethod

from torch import nn, Tensor
from edison.layers.draft_encoder import BaseEncoder


class BaseDiffusion(nn.Module, ABC):
    def __init__(
        self,
        encoder: BaseEncoder
    ) -> None:
        super().__init__()
        self.encoder = encoder

    @abstractmethod
    def forward(self, x) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode(self, x) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def loss_fn(self):
        raise NotImplementedError

    @abstractmethod
    def training_step(self, x) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample(self, x) -> Tensor:
        raise NotImplementedError
