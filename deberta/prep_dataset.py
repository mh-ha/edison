"""
preprocess datasets.
    - split sentences
    - tokenize
    - truncate
    - pad
    - collate
"""
from transformers import AutoTokenizer

SPLIT_IDX = 768
TOKENIZER = AutoTokenizer.from_pretrained('microsoft/deberta-base')

def split_sentences(samples):
    sentences = []
    for sample in samples['text']:
        sentences += [sample[i:i+SPLIT_IDX] for i in range(0, len(sample), SPLIT_IDX)]
    return {'text': sentences}

def tokenize(sentences, max_seq_len):
    tokenized = TOKENIZER(
        sentences['text'],
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=False,
        truncation=True,
        max_length=max_seq_len,
        )
    return tokenized