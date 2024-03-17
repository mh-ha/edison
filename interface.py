from abc import ABCMeta, abstractmethod
from setting.callbacks import Callback
from setting.networks import Network
from setting.optimizers import Optimizer

class Setting(metaclass=ABCMeta):
    """
    This class is designed to be used as a base class for setting up a neural network.
    You should subclass this class and implement the following methods:
        - init_network: model, layers, etc.
        - init_loss_function
        - init_optimizer
        - init_callback: logging, early stopping, scheduler, etc.
        - init_data: train, validation, test, etc.
        - init_logging_function: tensorboard, wandb, etc.
        - init_config: hyperparameters, etc.
        - forward: forward pass of the network
    """
    def __init__(self):
        self.network = self.init_network()
        self.loss_fn = self.init_loss_function()
        self.optimizer = self.init_optimizer()
        self.callback = self.init_callback()
        self.data = self.init_data()
        self.logging_fn = self.init_logging_function()
        self.config = self.init_config()

    @abstractmethod
    def init_network(self) -> Network: pass

    @abstractmethod
    def init_loss_function(self) -> function: pass

    @abstractmethod
    def init_optimizer(self) -> Optimizer: pass

    @abstractmethod
    def init_callback(self) -> Callback: pass

    @abstractmethod
    def init_data(self) -> function: pass

    @abstractmethod
    def init_logging_function(self) -> function: pass

    @abstractmethod
    def init_config(self) -> dict: pass

    @abstractmethod
    def forward(self, x): pass

    def test_network_process(self):
        pass

    def summary(self):
        pass


class PyTorchSetting(Setting):
    def __init__(self):
        super().__init__()

    def init_network(self):
        pass

    def forward(self, x):
        pass

    def init_loss_function(self):
        pass

    def init_optimizer(self):
        pass

    def init_callback(self):
        pass

    def init_data(self):
        pass

    def init_logging(self):
        pass

    def init_config(self):
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