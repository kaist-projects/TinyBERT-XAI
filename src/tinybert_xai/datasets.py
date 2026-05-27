from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    hf_path: str
    hf_config: str | None
    label_names: tuple[str, ...]
    text_column: str = "text"
    label_column: str = "label"
    default_split: str = "train"

    @property
    def num_labels(self) -> int:
        return len(self.label_names)


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "tweet_eval/sentiment": DatasetSpec(
        hf_path="cardiffnlp/tweet_eval",
        hf_config="sentiment",
        label_names=("negative", "neutral", "positive"),
    ),
}


def get_dataset_spec(key: str) -> DatasetSpec:
    if key not in DATASET_REGISTRY:
        available = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unknown dataset {key!r}. Available: {available}")
    return DATASET_REGISTRY[key]
