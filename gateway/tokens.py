"""Rough token counting.

Real systems use the model's tokenizer (e.g. `transformers.AutoTokenizer`).
~4 characters per token is a common rule of thumb for English.
LEARN: swap this for tiktoken / HF tokenizer and compare the counts.
"""


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4)
