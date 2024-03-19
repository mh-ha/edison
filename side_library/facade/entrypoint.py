from .abstracts import Facade
from .pytorch import PyTorchFacade

framework_name_to_class = {
    'torch': PyTorchFacade,
    'tf': 'tensorflow',
    'jax': 'jax',
}

def get_facade(framework_name:str) -> Facade:
    return framework_name_to_class[framework_name]