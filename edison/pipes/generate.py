from typing import Union

import pandas as pd

from edison.modules.draft_lightning_modules import LD4LGDiffusion, EdisonDiffusion


def generate_from_model(
    model: Union[LD4LGDiffusion, EdisonDiffusion],
    num_samples: int = 100,
    batch_size: int = 32,
    seq_len: int = 64,
    saved_file_name: str = 'generated_samples.csv'
) -> pd.DataFrame:
    model.eval()
    samples = model.generate(num_samples, seq_len=seq_len, batch_size=batch_size, seed=1004)
    df = pd.DataFrame(samples, columns=['text'])
    df.to_csv(saved_file_name, index=False)
    return df
