from abc import ABCMeta, abstractmethod

class Data(metaclass=ABCMeta):
    @abstractmethod
    def take(self, *args, **kwargs):
        pass

