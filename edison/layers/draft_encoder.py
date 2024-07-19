from abc import ABC
from abc import abstractmethod

from torch import nn, Tensor


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
    def forward(self) -> Tensor:
        raise NotImplementedError
