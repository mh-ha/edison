from abc import ABCMeta, abstractmethod
from typing import Callable
from setting.callbacks import Callback
from setting.networks import Network
from setting.optimizers import Optimizer
from setting.configs import Config
from setting.data import Data
from setting.loss import Loss

class Setting(metaclass=ABCMeta):
    """
    This class is designed to be used as a base class for setting up a neural network.
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
        self.network: Network = self.init_network()
        self.loss_fn: Loss = self.init_loss_function()
        self.optimizer: Optimizer = self.init_optimizer()
        self.callback: Callback = self.init_callback()
        self.data: Data = self.init_data()
        self.config: Config = self.init_config()

    @abstractmethod
    def init_network(self) -> Network: pass

    @abstractmethod
    def init_loss_function(self) -> Callable: pass

    @abstractmethod
    def init_optimizer(self) -> Optimizer: pass

    @abstractmethod
    def init_callback(self) -> Callback: pass

    @abstractmethod
    def init_data(self) -> Callable: pass

    @abstractmethod
    def init_config(self) -> dict: pass

    @abstractmethod
    def forward(self, x): pass

    def test_network_process(self):
        pass

    def summary(self):
        pass


class DistributedTrainingWrapper(metaclass=ABCMeta):
    def __init__(self, setting):
        self.setting = setting
    
    @abstractmethod
    def train(self):
        pass

    def validation(self):
        pass
        
class PyTorchLightningWrapper(DistributedTrainingWrapper):
    def __init__(self, setting):
        super().__init__(setting)
        # PyTorch Lightning-specific initialization

    def train(self):
        # Use PyTorch Lightning's Trainer to train the model
        from pytorch_lightning import Trainer
        trainer = Trainer(...)
        trainer.fit(self.setting.model, self.setting.data)

class FlaxWrapper(DistributedTrainingWrapper):
    def __init__(self, setting):
        super().__init__(setting)
        # Flax-specific initialization

    def train(self):
        # Implement training logic using Flax/JAX
        # This might involve setting up a JAX pmap for distributed training, for example
        pass