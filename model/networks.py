from abc import ABC, abstractmethod
import layers
import torch.nn as nn
import torch.nn.functional as F


class Network(ABC, nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x):
        pass


class Network1(nn.Module):
    pass

class Network2(nn.Module):
    pass

class Network3(nn.Module):
    pass