from .pytorch import PyTorchFacade

framework_name_to_class = {
    'torch': PyTorchFacade,
    'tf': 'tensorflow',
    'jax': 'jax',
}

def get_facade(framework_name):
    return framework_name_to_class[framework_name]