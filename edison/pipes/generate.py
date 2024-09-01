import pandas as pd

from edison.modules.base import BaseEdisonDiffusion


def generate_from_model(
    model: BaseEdisonDiffusion,
    num_samples: int = 100,
    batch_size: int = 32,
    seq_len: int = 64,
    seed=1004,
) -> pd.DataFrame:
    model.eval()
    samples = model.generate(num_samples, seq_len=seq_len, batch_size=batch_size, seed=seed)
    df = pd.DataFrame(samples, columns=['text'])
    return df
