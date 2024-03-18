from abc import ABCMeta, abstractmethod
from facade.entrypoint import get_facade
from setting.configs import Config

class Data(metaclass=ABCMeta):
    def __init__(self, base_framework_name:str) -> None:
        self.lib = get_facade(base_framework_name)

    @abstractmethod
    def take(self, *args, **kwargs):
        pass

class DataPreset1(Data):
    def __init__(
            self,
            config:Config,
            **kwargs
            ) -> None:
        super().__init__(config.base_framework_name)
        self.train = self.lib.dataset()
        self.validation = self.lib.dataset()
        self.test = self.lib.dataset()