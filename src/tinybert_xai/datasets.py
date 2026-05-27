from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar

from datasets import Dataset, DatasetDict, load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    hf_path: str
    hf_config: str | None
    num_labels: int
    data_cls: type


class SentimentLabel(IntEnum):
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2


@dataclass(frozen=True)
class TweetEvalSentimentData:
    text: str
    label: SentimentLabel

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "TweetEvalSentimentData":
        return cls(
            text=row["text"],
            label=SentimentLabel(row["label"]),
        )


DATASET_TWEETEVAL_SENTIMENT = DatasetSpec(
    hf_path="cardiffnlp/tweet_eval",
    hf_config="sentiment",
    num_labels=3,
    data_cls=TweetEvalSentimentData,
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
