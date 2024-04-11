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
MAX_SEQ_LEN = 512
TOKENIZER = AutoTokenizer.from_pretrained('microsoft/deberta-base')

def split_sentences(samples):
    sentences = []
    for sample in samples['text']:
        sentences += [sample[i:i+SPLIT_IDX] for i in range(0, len(sample), SPLIT_IDX)]
    return {'text': sentences}

def tokenize(sentences):
    tokenized = TOKENIZER(sentences['text'], max_length=MAX_SEQ_LEN, truncation=True, padding=True, return_tensors='pt')
    return tokenized