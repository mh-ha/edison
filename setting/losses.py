from abc import ABCMeta, abstractmethod
from facade.entrypoint import get_facade
from setting.configs import Config

class Loss(metaclass=ABCMeta):
    def __init__(self, base_framework_name:str) -> None:
        self.lib = get_facade(base_framework_name)
    
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @staticmethod
    def gradient(*args, **kwargs):
        pass

class LossPreset1(Loss):
    def __init__(
            self,
            config:Config,
            **kwargs
            ) -> None:
        super().__init__(config.base_framework_name)
        self.cross_entropy = self.lib.cross_entropy()
