"""Project-wide config shared across all 8 conditions and all 9 datasets."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    """Immutable global config.

    Per-condition configs (8 conditions, iter 2+) and per-dataset configs
    (9 datasets, iter 7) layer on top of this with their own dataclasses.
    """

    seed: int = 42
    device: str = "cuda"
    max_seq_length: int = 128
    teacher_model_name: str = "bert-base-uncased"
    student_model_name: str = "huawei-noah/TinyBERT_General_4L_312D"
    pilot_dataset: str = "cardiffnlp/tweet_eval"
    pilot_dataset_config: str = "sentiment"
