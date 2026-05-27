"""Pilot-dataset loader.

The full 9-dataset adapter contract (docs/notes/02-project-synthesis.md §4)
lands in iter 7. This module only exposes the one-batch helper iter 0 needs.
"""

import torch
from datasets import load_dataset


NUM_LABELS_TWEETEVAL_SENTIMENT = 3


def load_tweeteval_sentiment_batch(
    tokenizer,
    batch_size: int = 4,
    max_length: int = 128,
    split: str = "train",
) -> dict[str, torch.Tensor]:
    """Return one batch from TweetEval-sentiment as CPU tensors.

    Returns {input_ids, attention_mask, token_type_ids, labels}. First three have
    shape [batch_size, max_length]; labels has shape [batch_size]. The caller
    moves tensors to device.
    """
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split=split)
    examples = ds.select(range(batch_size))
    encoded = tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(examples["label"], dtype=torch.long)
    return dict(encoded)
