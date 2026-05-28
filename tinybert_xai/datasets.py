from dataclasses import dataclass
from enum import IntEnum

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch.utils.data import DataLoader
from transformers import BatchEncoding, PreTrainedTokenizerBase


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    hf_path: str
    hf_config: str | None
    num_labels: int


class SentimentLabel(IntEnum):
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2


DATASET_TWEETEVAL_SENTIMENT = DatasetSpec(
    name="tweet_eval-sentiment",
    hf_path="cardiffnlp/tweet_eval",
    hf_config="sentiment",
    num_labels=len(SentimentLabel),
)


def load_split(spec: DatasetSpec, split: str) -> Dataset:
    splits = load_dataset(spec.hf_path, spec.hf_config)
    if not isinstance(splits, DatasetDict):
        raise TypeError(f"Expected DatasetDict, got {type(splits).__name__}")
    ds = splits[split]
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected Dataset for split={split!r}, got {type(ds).__name__}")
    return ds


def encode_batch(
    tokenizer: PreTrainedTokenizerBase,
    ds: Dataset,
    *,
    max_length: int,
    device: str | None = None,
) -> BatchEncoding:
    encoding = tokenizer(
        ds["text"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoding["labels"] = torch.tensor(ds["label"], dtype=torch.long)
    if device is not None:
        encoding = encoding.to(device)
    return encoding


def build_loader(
    spec: DatasetSpec,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    batch_size: int,
    shuffle: bool = False,
    seed: int | None = None,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """Tokenize a split once (HF cache) and wrap it in a DataLoader.

    The returned loader yields dicts of CPU tensors {input_ids, attention_mask,
    labels} ready for `model(**batch)`. Move to device at iteration time.

    For per-epoch reshuffling, pass `shuffle=True, seed=<base>` and call
    `loader.generator.manual_seed(base + epoch)` before each epoch's loop.
    """
    ds = load_split(spec, split)
    ds = ds.map(
        lambda batch: tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
        remove_columns=[c for c in ds.column_names if c != "label"],
    )
    ds = ds.rename_column("label", "labels")
    ds = ds.with_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    generator = torch.Generator().manual_seed(seed) if (shuffle and seed is not None) else None

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )
