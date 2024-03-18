"""
1. Config
2. Network
3. Loss
4. Optimizer
5. Data
6. Callback
"""

from proof_ready.abstracts import DLProofReady
from setting.configs import Config
from setting.networks import Network
from setting.losses import Loss
from setting.optimizers import Optimizer
from setting.data import Data
from setting.callbacks import Callback

class MyExperiment(DLProofReady):
    def __init__(self):
        super(MyExperiment, self).__init__()

    def forward(self, x):
        pass

exp = MyExperiment()

#TODO
config_path = 'config.path'
config = Config()
config.read_config_from_path(config_path)
exp.config = config

network = Network(config)
exp.network = network

loss = Loss(config)
exp.loss = loss

optimizer = Optimizer(config)
exp.optimizer = optimizer

data = Data(config)
exp.data = data

callback = Callback(config)
exp.callback = callback

exp.test_network_process()