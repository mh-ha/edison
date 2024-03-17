from dataclasses import dataclass

@dataclass
class Config:
    batch_size: int
    learning_rate: float
    epochs: int
    device: str
    seed: int