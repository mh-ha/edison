from abc import ABC, abstractmethod

class Adapter(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def set_model(self, model):
        pass

    @abstractmethod
    def set_callback(self, callback):
        pass

    @abstractmethod
    def set_config(self, config):
        pass
