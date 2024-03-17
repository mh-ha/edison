from dataclasses import dataclass

@dataclass
class Config:
    input_shape: tuple
    output_shape: tuple
    batch_size: int
    learning_rate: float
    epochs: int
    device: str
    seed: int