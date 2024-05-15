from abc import ABC, abstractmethod




class Wrapper(ABC):
    @abstractmethod
    def train(self, proof_ready):
        pass

    @abstractmethod
    def validation(self, proof_ready):
        pass