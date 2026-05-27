from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar

from datasets import Dataset, DatasetDict, load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    hf_path: str
    hf_config: str | None
    data_cls: type

    @property
    def num_labels(self) -> int:
        return len(self.data_cls.Label)


class SentimentLabel(IntEnum):
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2


@dataclass(frozen=True)
class TweetEvalSentimentData:
    Label: ClassVar[type[SentimentLabel]] = SentimentLabel

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
    data_cls=TweetEvalSentimentData,
)


class DatasetLoader:
    def __init__(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self.splits = load_dataset(spec.hf_path, spec.hf_config)
        if not isinstance(self.splits, DatasetDict):
            raise TypeError(f"Expected DatasetDict, got {type(self.splits).__name__}")

    def get(self, split: str) -> Dataset:
        ds = self.splits[split]
        if not isinstance(ds, Dataset):
            raise TypeError(f"Expected Dataset for split={split!r}, got {type(ds).__name__}")
        return ds
