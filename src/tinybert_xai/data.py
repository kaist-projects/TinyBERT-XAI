import torch
from datasets import Dataset, load_dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase

from tinybert_xai.datasets import DatasetSpec


def load_batch(
    spec: DatasetSpec,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    max_length: int,
    split: str | None = None,
    device: str | None = None,
) -> BatchEncoding:
    chosen_split = split or spec.default_split
    ds = load_dataset(spec.hf_path, spec.hf_config, split=chosen_split)
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected Dataset for split={chosen_split!r}, got {type(ds).__name__}")

    examples = ds.select(range(batch_size))
    encoding = tokenizer(
        examples[spec.text_column],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoding["labels"] = torch.tensor(examples[spec.label_column], dtype=torch.long)

    if device is not None:
        encoding = encoding.to(device)
    return encoding
