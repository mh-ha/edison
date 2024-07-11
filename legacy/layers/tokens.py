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

import numpy as np
import torch

class TokenConverter:
    def __init__(self, num_buffer_tokens=5):
        self.num_buffer_tokens = num_buffer_tokens
    """
    batch token_ids → batch tokens(c=1), batch tokens(c=0), position, c
    """
    def batch_to_tokens(self, batch):
        """
        batch: {'input_ids', 'attention_mask', 'labels'}
        """
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        return input_ids, attention_mask, labels
    
    def tokens_to_batch(self, input_ids, attention_mask, labels):
        return {'input_ids':input_ids, 'attention_mask':attention_mask, 'labels':labels}
    
    def tokens_to_xtokens(self, input_ids, attention_mask):
        """
        input_ids: [batch, seq_len]
        attention_mask: [batch, seq_len]
        
        패딩(max_seq_len)에 맞춰서 버퍼 크기
        배치 + tokenizer 내의 단어
        """
        # init
        result_buffer = []
        result_conscious = []
        result_position = []
        
        buffers = input_ids.view(-1)[attention_mask.view(-1)==1]
        buffers = set(buffers.tolist())
        # extract tokens from current batches
        for i in range(len(input_ids)):
            current_input_ids = input_ids[i].view(-1)[attention_mask[i].view(-1)==1]
            # get buffer tokens
            set_current_input_ids = set(current_input_ids.tolist())
            buffer_candidates = buffers - set_current_input_ids
            result_buffer.append(self.choice_buffer_tokens(buffer_candidates))
            # assigns conscious
            result_conscious.append(
                [1 for _ in range(len(current_input_ids))] +
                [0 for _ in range(len(result_buffer[-1]))])
            # assigns position
            result_position.append(
                [pos for pos in range(len(current_input_ids))] +
                [-1 for _ in range(len(result_buffer[-1]))])
        # convert to torch.Tensor then return
        result_buffer = torch.tensor(result_buffer)
        result_conscious = torch.tensor(result_conscious)
        result_position = torch.tensor(result_position)
        return result_buffer, result_conscious, result_position
    
    def choice_buffer_tokens(self, buffer_candidates):
        return np.random.choice(buffer_candidates, size=self.num_buffer_tokens)