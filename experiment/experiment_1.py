from abstract.abstracts import DLSetting
from setting.callbacks import Callback
from setting.networks import Network
from setting.optimizers import Optimizer

class PyTorchDLSetting(DLSetting):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        pass

    def calculate_loss(self, x, y):
        pass

    def get_optimizer(self):
        pass

    def get_callback_list(self):
        pass

    def get_dataset(self):
        pass

    def get_config(self):
        pass