import lightning as L
from lightning.pytorch.strategies import DDPStrategy

from .config.config import Config


# TODO: 상세하게 구현
def get_trainer(config: Config, debug=None):
    trainer = L.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=1,
        # max_steps=500000,
        precision=16,
        gradient_clip_val=config.gradient_clip_val,
        gradient_clip_algorithm=config.gradient_clip_algorithm,
        accumulate_grad_batches=config.gradient_accumulation_steps,
        num_sanity_val_steps=0,
        log_every_n_steps=50,
        profiler=debug,
        # callbacks=[checkpoint_callback],
        # resume_from_checkpoint='path/to/checkpoint.ckpt',
        # auto_lr_find=True,
        # auto_scale_batch_size='power',
        # auto_scale_batch_size='binsearch',
        # auto_scale_batch_size='power
    )
    return trainer
