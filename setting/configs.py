from dataclasses import dataclass
import yaml

@dataclass
class Config:
    base_framework_name: str
    input_shape: tuple
    output_shape: tuple
    batch_size: int
    learning_rate: float
    epochs: int
    device: str
    seed: int

    def read_config_from_path(self, config_path):
        config = yaml.safe_load(open(config_path, 'r'))
        self.__dict__.update(config)