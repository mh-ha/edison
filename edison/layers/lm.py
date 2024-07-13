from typing import Tuple

from transformers import AutoTokenizer
from transformers.models.bart.modeling_bart import (
    BartForConditionalGeneration,
)


def get_BART(
    model_path: str = 'facebook/bart-base'
) -> Tuple[BartForConditionalGeneration, AutoTokenizer]:
    model = BartForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer
