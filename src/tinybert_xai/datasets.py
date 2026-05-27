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


class DatasetLoader:
    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self.splits = load_dataset(spec.hf_path, spec.hf_config)
        if not isinstance(self.splits, DatasetDict):
            raise TypeError(f"Expected DatasetDict, got {type(self.splits).__name__}")

    def get_split(self, split: str) -> Dataset:
        ds = self.splits[split]
        if not isinstance(ds, Dataset):
            raise TypeError(f"Expected Dataset for split={split!r}, got {type(ds).__name__}")
        return ds

    def get_batch(self, split: str, indices: range | list[int]) -> Dataset:
        return self.get_split(split).select(indices)


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
