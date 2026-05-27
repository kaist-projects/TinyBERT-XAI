"""Internal dataset loading — not part of the public API.

KDPair (kdpair.py) is the user-facing entry point.
The full 9-dataset adapter contract (docs/notes/02-project-synthesis.md §4) lands in iter 7.
"""

import torch
from datasets import Dataset, load_dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase


def _load_batch(
    hf_path: str,
    hf_config: str | None,
    text_column: str,
    label_column: str,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    max_length: int,
    split: str,
) -> BatchEncoding:
    """Load `batch_size` examples from a HuggingFace dataset as a BatchEncoding.

    Returns a BatchEncoding with keys {input_ids, attention_mask, token_type_ids, labels}.
    Shapes: first three are [batch_size, max_length]; labels is [batch_size].
    Call .to(device) on the result to move to GPU.
    """
    ds = load_dataset(hf_path, hf_config, split=split)
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected a Dataset for split={split!r}, got {type(ds).__name__}")

    examples = ds.select(range(batch_size))
    encoding = tokenizer(
        examples[text_column],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoding["labels"] = torch.tensor(examples[label_column], dtype=torch.long)
    return encoding
