# Description: This file contains the dataclasses for the input and output of the models.
from typing import Optional
from dataclasses import dataclass

from torch import Tensor


@dataclass
class DiffusionOutput:
    encoded: Tensor


@dataclass
class AEOutput:
    pass
