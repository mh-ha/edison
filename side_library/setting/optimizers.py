from abc import ABCMeta, abstractmethod
from facade.entrypoint import get_facade
from setting.configs import Config

class Optimizer(metaclass=ABCMeta):
    def __init__(self, base_framework_name:str) -> None:
        self.lib = get_facade(base_framework_name)

    @abstractmethod
    def step(self, *args, **kwargs):
        pass

class OptimizerPreset1(Optimizer):
    def __init__(
            self,
            config:Config,
            **kwargs
            ) -> None:
        super().__init__(config.base_framework_name)
        self.adam = self.lib.adam()

    def step(self, loss):
        self.adam.step(loss)