from abstract.abstracts import Setting
from setting.callbacks import Callback
from setting.networks import Network
from setting.optimizers import Optimizer

class PyTorchSetting(Setting):
    def __init__(self):
        super().__init__()

    def init_network(self):
        pass

    def init_loss_function(self):
        pass

    def init_optimizer(self):
        pass

    def init_callback(self):
        pass

    def init_data(self):
        pass

    def init_config(self):
        pass

    def forward(self, x):
        pass