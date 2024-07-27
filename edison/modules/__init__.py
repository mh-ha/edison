from typing import Callable, Type
from pathlib import Path

from lightning import LightningModule

from edison.utils.file_utils import import_all_modules


_supported_modules = {}


def register_module(name: str) -> Callable[[Type[LightningModule]], Type[LightningModule]]:
    def register_module_cls(module_cls: Type[LightningModule]) -> Type[LightningModule]:
        assert name not in _supported_modules, f"Cannot register duplicate module: {name}"
        _supported_modules[name] = module_cls
        return module_cls
    return register_module_cls


def get_module(module_name: str) -> Type[LightningModule]:
    assert module_name in _supported_modules, f"There is no module: {module_name}"
    return _supported_modules[module_name]


FILE_ROOT = Path(__file__).parent
import_all_modules(FILE_ROOT, "edison.modules")
