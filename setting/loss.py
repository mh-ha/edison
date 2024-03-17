from abc import ABCMeta, abstractmethod

class Loss(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass