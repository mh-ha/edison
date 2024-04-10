"""
model, tokenizer:
    'deberta-v3-small': PretrainedModel('deberta-v3-small', 'spm.model', 'spm'),
    'deberta-v3-base': PretrainedModel('deberta-v3-base', 'spm.model', 'spm'),
    'deberta-v3-large': PretrainedModel('deberta-v3-large', 'spm.model', 'spm'),
    'mdeberta-v3-base': PretrainedModel('mdeberta-v3-base', 'spm.model', 'spm'),
    'deberta-v3-xsmall': PretrainedModel('deberta-v3-xsmall', 'spm.model', 'spm'),
"""

from bisect import bisect
import math
import numpy as np
import random

from utils import NGramMaskGenerator


class MaskTaskData:
    def __init__(self, config, tokenizer, mask_lm_prob=0.15, max_seq_len=512, max_preds_per_seq=None, **kwargs):
        self.config = config
        self.tokenizer = tokenizer
        self.mask_generator = NGramMaskGenerator(tokenizer, mask_lm_prob, max_seq_len, max_preds_per_seq, **kwargs)

    def generate_masked_tokens(self, tokens, rng=random, **kwargs):
        return self.mask_generator.mask_tokens(tokens, rng, **kwargs)
    
    def from_text_to_inputs(self, text, rng=random, **kwargs):
        tokens = self.tokenizer.tokenize(text)
        masked_tokens, target_labels = self.generate_masked_tokens(tokens, rng, **kwargs)
        original_inputs_ids = self.tokenizer(text, add_special_tokens=True)['input_ids']
        masked_input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        # add special tokens
        self._add_ids_at_first_and_last(masked_input_ids, add_eos_bos=True)
        self._add_ids_at_first_and_last(target_labels)
        target_labels
        output = {
            'input_ids': masked_input_ids,
            'attention_mask': [1]*len(masked_input_ids),
            'position_ids': list(range(len(masked_input_ids))),
            'labels': target_labels,
            'original_input_ids': original_inputs_ids,
        }
        return output
    
    def _add_ids_at_first_and_last(self, ids:list, add_eos_bos=False, fill=0):
        if add_eos_bos:
            ids.insert(0, self.tokenizer.bos_token_id)
            ids.append(self.tokenizer.eos_token_id)
        else:
            ids.insert(0, fill)
            ids.append(fill)




# class ReplaceTaskData

