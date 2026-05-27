from dataclasses import dataclass
from enum import IntEnum
from typing import Any, ClassVar


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
