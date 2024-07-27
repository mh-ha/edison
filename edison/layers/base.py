from abc import ABC, abstractmethod
from typing import Optional, Union

from torch import nn, Tensor


class BaseDiffusion(nn.Module, ABC):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        alpha: Tensor,
        attention_mask: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        alpha: Tensor,
        attention_mask: Optional[Tensor] = None,
        self_cond: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def loss_fn(self):
        raise NotImplementedError

    @abstractmethod
    def training_step(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        attention_mask: Optional[Tensor],
        times: Union[Tensor, float],
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample(self, x) -> Tensor:
        raise NotImplementedError


class BaseEncoder(nn.Module, ABC):
    def __init__(
        self,
        internal_dim: Tensor,
        depth: int,
    ) -> None:
        super().__init__()
        self.internal_dim = internal_dim
        self.depth = depth

    @abstractmethod
    def forward(
        self,
        latent: Tensor,
        context: Optional[Tensor],
        attention_mask: Optional[Tensor] = None,
        time_emb: Optional[Tensor] = None,
    ) -> Tensor:
        raise NotImplementedError


class BaseAutoEncoder(nn.Module, ABC):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        encoder_outputs: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def encode(
        self,
        encoder_outputs: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode(
        self,
        encoder_outputs: Tensor
    ) -> Tensor:
        raise NotImplementedError
