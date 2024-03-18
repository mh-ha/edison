from abc import ABCMeta, abstractmethod
from setting.callbacks import Callback
from setting.networks import Network
from setting.optimizers import Optimizer
from setting.configs import Config
from setting.data import Data
from setting.losses import Loss

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
    @abstractmethod
    def forward(self, x): pass

    @property
    def config(self): return self._config
    @config.setter
    def config(self, config):
        assert isinstance(config, Config), "config must be an instance of Config class."
        self._config = config
        self.__dict__.update(config.__dict__)
    @config.deleter
    def config(self): del self._config

    @property
    def network(self): return self._network
    @network.setter
    def network(self, network):
        assert isinstance(network, Network), "network must be an instance of Network class."
        self._network = network
    @network.deleter
    def network(self): del self._network

    @property
    def loss(self): return self._loss
    @loss.setter
    def loss(self, loss):
        assert isinstance(loss, Loss), "loss must be an instance of Loss class."
        self._loss = loss
    @loss.deleter
    def loss(self): del self._loss

    @property
    def optimizer(self): return self._optimizer
    @optimizer.setter
    def optimizer(self, optimizer):
        assert isinstance(optimizer, Optimizer), "optimizer must be an instance of Optimizer class."
        self._optimizer = optimizer
    @optimizer.deleter
    def optimizer(self): del self._optimizer

    @property
    def data(self): return self._data
    @data.setter
    def data(self, data):
        assert isinstance(data, Data), "data must be an instance of Data class."
        self._data = data
    @data.deleter
    def data(self): del self._data
    
    @property
    def callback(self): return self._callback
    @callback.setter
    def callback(self, callback):
        assert isinstance(callback, Callback), "callback must be an instance of Callback class."
        self._callback = callback
    @callback.deleter
    def callback(self): del self._callback

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
