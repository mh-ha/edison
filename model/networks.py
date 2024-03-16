from abc import ABC, abstractmethod
import layers
import torch.nn as nn
import torch.nn.functional as F


class PytorchBasedNetwork(ABC, nn.Module):
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