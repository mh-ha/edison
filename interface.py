from abc import ABC, abstractmethod

class Adapter(ABC):
    @abstractmethod
    def init_network(self):
        pass

    @abstractmethod
    def init_callback(self):
        pass

    @abstractmethod
    def init_config(self):
        pass

    @abstractmethod
    def init_training_library(self):
        pass

