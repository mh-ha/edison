from abc import ABCMeta, abstractmethod
from setting.callbacks import Callback
from setting.networks import Network
from setting.optimizers import Optimizer
from setting.configs import Config
from setting.data import Data
from setting.loss import Loss

class DLProofReady(metaclass=ABCMeta):
    """
    목적:
        Deep Learning 실험을 위한 모든 객체 묶음
        다양한 프레임워크 래퍼에서 사용 가능한 단일 인터페이스

    You should subclass this class and implement the following methods:
        - init_network: model, layers, etc.
        - init_loss_function
        - init_optimizer
        - init_callback: logging, early stopping, scheduler, etc.
        - init_data: train, validation, test, etc.
        - init_config: hyperparameters, etc.
        - forward: forward pass of the network
    """
    def __init__(self):
        self.network: Network
        self.loss: Loss
        self.optimizer: Optimizer
        self.callback: Callback
        self.data: Data
        self.config: Config

        # Check the network process
        self.test_network_process()

    @abstractmethod
    def forward(self, x): pass

    @abstractmethod
    def get_loss_fn(self): pass

    @abstractmethod
    def get_optimizer(self): pass

    @abstractmethod
    def get_callback_list(self): pass

    @abstractmethod
    def get_dataset(self): pass

    @abstractmethod
    def get_config(self): pass

    def test_network_process(self):
        print("Testing the network process...")
        # 1. Forward pass
        x = self.data.take(1)
        y = self.forward(x)
        # 2. Calculate loss
        loss = self.loss(x, y)
        # 3. Calculate gradient
        gradient = self.loss.gradient(loss)
        # 4. Update weights
        self.optimizer.step(self.network, gradient)
        print("Network process is working correctly.")

    def summary(self):
        pass
