from dataclasses import dataclass, field
from enum import IntEnum

import torch
from datasets import ClassLabel, Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import BatchEncoding, PreTrainedTokenizerBase

_DEV_SPLIT_SEED = 42


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    hf_path: str
    hf_config: str | None
    num_labels: int
    label_names: list[str]
    text_keys: tuple[str, ...] = ("text",)
    label_key: str = "label"
    split_sources: dict[str, str | tuple[str, ...]] = field(
        default_factory=lambda: {"train": "train", "validation": "validation", "test": "test"}
    )
    dev_split: float | None = None

    @property
    def input_type(self) -> str:
        return "sentence_pair" if len(self.text_keys) == 2 else "single_text"

    @property
    def split_scheme(self) -> str:
        if self.dev_split is None:
            return "official"
        return f"stratified_dev_{self.dev_split}_seed{_DEV_SPLIT_SEED}"


class SentimentLabel(IntEnum):
    NEGATIVE = 0
    NEUTRAL = 1
    POSITIVE = 2


DATASET_TWEETEVAL_SENTIMENT = DatasetSpec(
    name="tweet_eval-sentiment",
    family="sentiment",
    hf_path="cardiffnlp/tweet_eval",
    hf_config="sentiment",
    num_labels=len(SentimentLabel),
    label_names=["negative", "neutral", "positive"],
)


DATASET_IMDB = DatasetSpec(
    name="imdb",
    family="sentiment",
    hf_path="stanfordnlp/imdb",
    hf_config=None,
    num_labels=2,
    label_names=["negative", "positive"],
    text_keys=("text",),
    split_sources={"train": "train", "test": "test"},
    dev_split=0.1,
)


DATASET_ANLI = DatasetSpec(
    name="anli",
    family="nli",
    hf_path="facebook/anli",
    hf_config=None,
    num_labels=3,
    label_names=["entailment", "neutral", "contradiction"],
    text_keys=("premise", "hypothesis"),
    split_sources={
        "train": ("train_r1", "train_r2", "train_r3"),
        "validation": ("dev_r1", "dev_r2", "dev_r3"),
        "test": ("test_r1", "test_r2", "test_r3"),
    },
)


ALL_DATASETS = (DATASET_TWEETEVAL_SENTIMENT, DATASET_IMDB, DATASET_ANLI)
DATASETS_BY_NAME = {spec.name: spec for spec in ALL_DATASETS}


def dataset_by_name(name: str) -> DatasetSpec:
    try:
        return DATASETS_BY_NAME[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}; known: {sorted(DATASETS_BY_NAME)}") from None


def load_split(spec: DatasetSpec, split: str) -> Dataset:
    """Resolve a canonical split (``train``/``validation``/``test``) to a Dataset.

    Hides per-dataset layout from the pipeline: concatenates multi-source splits
    (e.g. ANLI rounds) and synthesizes a seed-42 stratified dev set from train
    when the dataset ships no official validation split (e.g. IMDB).
    """
    if spec.dev_split is not None and split in ("train", "validation"):
        return _synthesized_split(spec, split)
    return _load_sources(spec, spec.split_sources[split])


def encode_batch(
    tokenizer: PreTrainedTokenizerBase,
    ds: Dataset,
    spec: DatasetSpec,
    *,
    max_length: int,
    device: str | None = None,
) -> BatchEncoding:
    encoding = tokenizer(
        *[ds[key] for key in spec.text_keys],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoding["labels"] = torch.tensor(ds[spec.label_key], dtype=torch.long)
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
    token_type_ids, labels} ready for `model(**batch)`. Move to device at
    iteration time.

    For per-epoch reshuffling, pass `shuffle=True, seed=<base>` and call
    `loader.generator.manual_seed(base + epoch)` before each epoch's loop.
    """
    ds = _tokenize_split(load_split(spec, split), spec, tokenizer, max_length)

    generator = torch.Generator().manual_seed(seed) if (shuffle and seed is not None) else None

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        generator=generator,
    )


def _load_sources(spec: DatasetSpec, sources: str | tuple[str, ...]) -> Dataset:
    names = (sources,) if isinstance(sources, str) else tuple(sources)
    parts = [_load_one(spec, name) for name in names]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def _load_one(spec: DatasetSpec, hf_split: str) -> Dataset:
    ds = load_dataset(spec.hf_path, spec.hf_config, split=hf_split)
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected Dataset for split={hf_split!r}, got {type(ds).__name__}")
    return ds


def _synthesized_split(spec: DatasetSpec, split: str) -> Dataset:
    train = _ensure_classlabel(_load_sources(spec, spec.split_sources["train"]), spec)
    train_part, dev_part = stratified_dev_partition(train, spec.dev_split, spec.label_key)
    return train_part if split == "train" else dev_part


def stratified_dev_partition(ds: Dataset, dev_split: float, label_key: str) -> tuple[Dataset, Dataset]:
    """Deterministically carve a seed-42 stratified dev set out of ``ds``.

    Returns ``(train_remainder, dev)``. The two partitions are disjoint and the
    split is reproducible across calls (fixed seed), so asking for ``train`` and
    ``validation`` separately yields a consistent, leak-free partition.
    """
    parts = ds.train_test_split(test_size=dev_split, seed=_DEV_SPLIT_SEED, stratify_by_column=label_key)
    return parts["train"], parts["test"]


def _ensure_classlabel(ds: Dataset, spec: DatasetSpec) -> Dataset:
    if isinstance(ds.features[spec.label_key], ClassLabel):
        return ds
    return ds.cast_column(spec.label_key, ClassLabel(num_classes=spec.num_labels))


def _tokenize_split(ds: Dataset, spec: DatasetSpec, tokenizer: PreTrainedTokenizerBase, max_length: int) -> Dataset:
    ds = ds.map(
        lambda batch: tokenizer(
            *[batch[key] for key in spec.text_keys],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        ),
        batched=True,
        remove_columns=[c for c in ds.column_names if c != spec.label_key],
    )
    if spec.label_key != "labels":
        ds = ds.rename_column(spec.label_key, "labels")
    return ds.with_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
