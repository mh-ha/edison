from abc import ABC, abstractmethod


class Config(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def set_loss_fn(self, loss_fn):
        pass

    @abstractmethod
    def set_optimizer(self, optimizer):
        pass

    @abstractmethod
    def set_hparams(self, hparams):
        pass