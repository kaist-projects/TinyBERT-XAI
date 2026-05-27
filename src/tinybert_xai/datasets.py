from dataclasses import dataclass
from enum import IntEnum

import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase


@dataclass(frozen=True)
class DatasetSpec:
    hf_path: str
    hf_config: str | None
    num_labels: int


class SentimentLabel(IntEnum):
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2


DATASET_TWEETEVAL_SENTIMENT = DatasetSpec(
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
