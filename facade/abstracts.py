from abc import ABC, abstractmethod

class Facade(ABC):
    """
    내부 코드에서 사용할 모든 메서드를 여기에 선언합니다.
    각 메서드를 여기서 먼저 정의한 후, 베이스 프레임워크에 맞춰 구현합니다.
    """
    ########################################
    # Layer

    @staticmethod
    @abstractmethod
    def feedforward():
        pass

    @staticmethod
    @abstractmethod
    def attention():
        pass

    @staticmethod
    @abstractmethod
    def absolute_positional_encoding():
        pass

    @staticmethod
    @abstractmethod
    def multi_head_attention():
        pass

    @staticmethod
    @abstractmethod
    def transformer():
        pass

    ########################################
    # Optimizer

    @staticmethod
    @abstractmethod
    def opt_adam():
        pass