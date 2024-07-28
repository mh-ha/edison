import lightning as L
from lightning.pytorch.strategies import DDPStrategy
# from lightning.pytorch.profilers import PyTorchProfiler

from edison.configs.base import Config


# TODO: 상세하게 구현
def get_trainer(config: Config, logger=None, max_steps=250000):
    trainer = L.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True) if config.strategy == 'ddp' else "auto",
        logger=logger,
        max_steps=max_steps,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
    )
    return trainer
