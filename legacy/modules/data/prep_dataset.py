"""
preprocess datasets.
    - split sentences
    - tokenize
    - truncate
    - pad
    - collate
"""
from transformers import AutoTokenizer

SPLIT_IDX = 512
TOKENIZER = AutoTokenizer.from_pretrained('microsoft/deberta-v3-xsmall')

def split_sentences(samples):
    sentences = []
    for sample in samples['text']:
        sentences += [sample[i:i+SPLIT_IDX] for i in range(0, len(sample), SPLIT_IDX)]
    return {'text': sentences}

def tokenize(sentences:list[str], max_seq_len):
    tokenized = TOKENIZER(
        sentences,
        add_special_tokens=True,
        return_token_type_ids=False,
        return_attention_mask=False,
        truncation=True,
        max_length=128,
        )
    return tokenized