from abc import ABCMeta, abstractmethod
from .utils import layers
import torch.nn as nn
import torch.nn.functional as F


class Network(metaclass=ABCMeta):
    pass

class PytorchBasedNetwork(Network):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass


class Network1(PytorchBasedNetwork):
    pass

class Network2(PytorchBasedNetwork):
    pass

class Network3(PytorchBasedNetwork):
    pass