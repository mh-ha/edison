from bisect import bisect
import math
import numpy as np
import random


def MaskedLayerNorm(layerNorm, input, mask=None):
    """ Masked LayerNorm which will apply mask over the output of LayerNorm to avoid inaccurate updatings to the LayerNorm module.
    
    Args:
        layernorm (:obj:`~DeBERTa.deberta.LayerNorm`): LayerNorm module or function
        input (:obj:`torch.tensor`): The input tensor
        mask (:obj:`torch.IntTensor`): The mask to applied on the output of LayerNorm where `0` indicate the output of that element will be ignored, i.e. set to `0`

    Example::

        # Create a tensor b x n x d
        x = torch.randn([1,10,100])
        m = torch.tensor([[1,1,1,0,0,0,0,0,0,0]], dtype=torch.int)
        LayerNorm = DeBERTa.deberta.LayerNorm(100)
        y = MaskedLayerNorm(LayerNorm, x, m)

    """
    output = layerNorm(input).to(input)
    if mask is None:
        return output
    if mask.dim()!=input.dim():
        if mask.dim()==4:
            mask=mask.squeeze(1).squeeze(1)
        mask = mask.unsqueeze(2)
    mask = mask.to(output.dtype)
    return output*mask


class NGramMaskGenerator:
    """
    Mask ngram tokens
    https://github.com/zihangdai/xlnet/blob/0b642d14dd8aec7f1e1ecbf7d6942d5faa6be1f0/data_utils.py
    """
    def __init__(self, tokenizer, mask_lm_prob=0.15, max_seq_len=512, max_preds_per_seq=None, max_gram = 1, keep_prob = 0.1, mask_prob=0.8, **kwargs):
        self.tokenizer = tokenizer
        self.mask_lm_prob = mask_lm_prob
        self.keep_prob = keep_prob
        self.mask_prob = mask_prob
        assert self.mask_prob+self.keep_prob<=1, f'The prob of using [MASK]({mask_prob}) and the prob of using original token({keep_prob}) should between [0,1]'
        self.max_preds_per_seq = max_preds_per_seq
        if max_preds_per_seq is None:
            self.max_preds_per_seq = math.ceil(max_seq_len*mask_lm_prob /10)*10
        self.max_gram = max(max_gram, 1)
        self.mask_window = int(1/mask_lm_prob) # make ngrams per window sized context
        self.vocab_words = list(tokenizer.vocab.keys())

    def mask_tokens(self, tokens, rng=random, **kwargs):
        # import pdb
        special_tokens = ['[MASK]', '[CLS]', '[SEP]', '[PAD]', '[UNK]'] # + self.tokenizer.tokenize(' ')
        indices = [i for i in range(len(tokens)) if tokens[i] not in special_tokens]
        ngrams = np.arange(1, self.max_gram + 1, dtype=np.int64)
        pvals = 1. / np.arange(1, self.max_gram + 1)
        pvals /= pvals.sum(keepdims=True)
        unigrams = []
        for id in indices:
            if self.max_gram>1 and len(unigrams)>=1 and self.tokenizer.part_of_whole_word(tokens[id]):
                unigrams[-1].append(id)
            else:
                unigrams.append([id])

        num_to_predict = min(self.max_preds_per_seq, max(1, int(round(len(tokens) * self.mask_lm_prob))))
        mask_len = 0
        offset = 0
        mask_grams = np.array([False]*len(unigrams))
        while offset < len(unigrams):
            n = self._choice(rng, ngrams, p=pvals)
            ctx_size = min(n*self.mask_window, len(unigrams)-offset)
            m = rng.randint(0, ctx_size-1)
            s = offset + m
            e = min(offset+m+n, len(unigrams))
            offset = max(offset+ctx_size, e)
            mask_grams[s:e] = True

        target_labels = [None]*len(tokens)
        w_cnt = 0
        for m,word in zip(mask_grams, unigrams):
            if m:
                for idx in word:
                    label = self._mask_token(idx, tokens, rng, self.mask_prob, self.keep_prob)
                    target_labels[idx] = label
                    w_cnt += 1
                if w_cnt >= num_to_predict:
                    break

        target_labels = [self.tokenizer.vocab[x] if x else 0 for x in target_labels]
        # pdb.set_trace()
        return tokens, target_labels

    def _choice(self, rng, data, p):
        cul = np.cumsum(p)
        x = rng.random()*cul[-1]
        id = bisect(cul, x)
        return data[id]

    def _mask_token(self, idx, tokens, rng, mask_prob, keep_prob):
        label = tokens[idx]
        mask = '[MASK]'
        rand = rng.random()
        if rand < mask_prob:
            new_label = mask
        elif rand < mask_prob+keep_prob:
            new_label = label
        else:
            new_label = rng.choice(self.vocab_words)
        tokens[idx] = new_label
        return label