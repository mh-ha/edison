"""
senario:
    - LM
        1. sentence embedding + buffer embedding
        2. sentence embedding
    - AE
        1. extracted sentence only
        1. sentence + buffer (together)
        3. sentence + buffer (seperately)
        
    - Diffusions
        1. XT + latent (at the same time)
        2. XT + latent (alternately)
        ...
"""

#TODO: input 형태를 여기서 정하고, 이에 맞게 datamodule에서 처리할 수 있도록 수정 (모든 단계에서)