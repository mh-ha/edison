from abc import ABCMeta, abstractmethod
from facade.entrypoint import get_facade
from setting.configs import Config

class Network(metaclass=ABCMeta):
    def __init__(self, base_framework_name:str) -> None:
        self.lib = get_facade(base_framework_name)
    
    @abstractmethod
    def forward(self, x): pass

class NetworkPreset1(Network):
    def __init__(
            self,
            config:Config,
            **kwargs
            ) -> None:
        super().__init__(config.base_framework_name)
        
    def forward(self, x):
        pass

    def encode(self, x):
        pass
    
    def decode(self, x):
        pass