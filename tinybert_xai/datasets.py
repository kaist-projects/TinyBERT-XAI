import hashlib
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

import torch
from datasets import ClassLabel, Dataset, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from transformers import BatchEncoding, PreTrainedTokenizerBase

_DEV_SPLIT_SEED = 42


@dataclass(frozen=True)
class LocalCsvSource:
    """A dataset distributed as a single local CSV with an in-file split column."""

    path: str
    split_column: str
    label_map: dict[str, int]
    download_url: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    family: str
    hf_path: str | None
    hf_config: str | None
    num_labels: int
    label_names: list[str]
    text_keys: tuple[str, ...] = ("text",)
    label_key: str = "label"
    split_sources: dict[str, str | tuple[str, ...]] = field(
        default_factory=lambda: {"train": "train", "validation": "validation", "test": "test"}
    )
    dev_split: float | None = None
    test_split: float | None = None
    train_subsample: int | None = None
    hf_label_map: dict[str, int] | None = None
    local_csv: LocalCsvSource | None = None

    @property
    def input_type(self) -> str:
        return "sentence_pair" if len(self.text_keys) == 2 else "single_text"

    @property
    def split_scheme(self) -> str:
        if self.local_csv is not None:
            return "official_csv"
        tags = []
        if self.dev_split is not None:
            tags.append(f"dev_{self.dev_split}")
        if self.test_split is not None:
            tags.append(f"test_{self.test_split}")
        if self.train_subsample is not None:
            tags.append(f"sub{self.train_subsample}")
        if not tags:
            return "official"
        return "stratified_" + "_".join(tags) + f"_seed{_DEV_SPLIT_SEED}"


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


DATASET_DAVIDSON = DatasetSpec(
    name="davidson",
    family="hate",
    hf_path="tdavidson/hate_speech_offensive",
    hf_config=None,
    num_labels=3,
    label_names=["hate", "offensive", "neither"],
    text_keys=("tweet",),
    label_key="class",
    split_sources={"train": "train"},
    dev_split=0.1,
    test_split=0.1,
)


DATASET_DYNAHATE = DatasetSpec(
    name="dynahate",
    family="hate",
    hf_path=None,
    hf_config=None,
    num_labels=2,
    label_names=["nothate", "hate"],
    text_keys=("text",),
    label_key="label",
    split_sources={"train": "train", "validation": "dev", "test": "test"},
    local_csv=LocalCsvSource(
        path="data/dynahate/dynahate_v0.2.3.csv",
        split_column="split",
        label_map={"nothate": 0, "hate": 1},
        download_url="https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset",
    ),
)


DATASET_FEVER = DatasetSpec(
    name="fever",
    family="nli",
    hf_path="pietrolesci/nli_fever",
    hf_config=None,
    num_labels=3,
    label_names=["entailment", "neutral", "contradiction"],
    text_keys=("premise", "hypothesis"),
    label_key="label",
    # Official test split ships unlabeled, so use dev for validation and carve a
    # seed-42 stratified test out of train. Train is large (~208K); cap it to 50K
    # to keep the 3-epoch budget comparable to the other datasets.
    split_sources={"train": "train", "validation": "dev"},
    test_split=0.1,
    train_subsample=50_000,
)


DATASET_HATEVAL = DatasetSpec(
    name="hateval",
    family="hate",
    hf_path="valeriobasile/HatEval",
    # HF repo exposes one multilingual config and names the validation split "dev".
    hf_config="default",
    num_labels=2,
    label_names=["not_hate", "hate"],
    text_keys=("text",),
    label_key="HS",
    split_sources={"train": "train", "validation": "dev", "test": "test"},
)


DATASET_VARDIAL = DatasetSpec(
    name="vardial",
    family="dialect",
    hf_path="statworx/swiss-dialects",
    hf_config=None,
    num_labels=4,
    label_names=["BE", "BS", "LU", "ZH"],
    text_keys=("sentence",),
    label_key="label",
    # ArchiMob/GDI Swiss-German dialect ID. Labels ship as strings; map them to
    # ints, and carve a seed-42 stratified dev/test from the single train split.
    hf_label_map={"BE": 0, "BS": 1, "LU": 2, "ZH": 3},
    split_sources={"train": "train"},
    dev_split=0.1,
    test_split=0.1,
)


DATASET_MULTIVALUE = DatasetSpec(
    name="multivalue",
    family="dialect",
    hf_path=None,
    hf_config=None,
    num_labels=2,
    label_names=["sae", "dialect"],
    text_keys=("text",),
    label_key="label",
    # Binary SAE-vs-dialect detection over text generated by the Multi-VALUE
    # toolkit. Build the CSV with scripts/build_multivalue.py before running.
    split_sources={"train": "train", "validation": "dev", "test": "test"},
    local_csv=LocalCsvSource(
        path="data/multivalue/multivalue.csv",
        split_column="split",
        label_map={"sae": 0, "dialect": 1},
        download_url="generate it locally with scripts/build_multivalue.py",
    ),
)


ALL_DATASETS = (
    DATASET_TWEETEVAL_SENTIMENT,
    DATASET_IMDB,
    DATASET_ANLI,
    DATASET_DAVIDSON,
    DATASET_DYNAHATE,
    DATASET_FEVER,
    DATASET_HATEVAL,
    DATASET_VARDIAL,
    DATASET_MULTIVALUE,
)
DATASETS_BY_NAME = {spec.name: spec for spec in ALL_DATASETS}


def dataset_by_name(name: str) -> DatasetSpec:
    try:
        return DATASETS_BY_NAME[name]
    except KeyError:
        raise KeyError(f"Unknown dataset {name!r}; known: {sorted(DATASETS_BY_NAME)}") from None


def load_split(spec: DatasetSpec, split: str) -> Dataset:
    """Resolve a canonical split (``train``/``validation``/``test``) to a Dataset.

    Hides per-dataset layout from the pipeline: reads a local CSV with an in-file
    split column (e.g. DynaHate), concatenates multi-source splits (e.g. ANLI
    rounds), and synthesizes seed-42 stratified dev/test sets from train when the
    dataset ships no official split (dev for IMDB; dev+test for Davidson).
    """
    if spec.local_csv is not None:
        return _load_local_csv_split(spec, split)
    if _is_synthesized(spec, split):
        return _synthesized_split(spec, split)
    ds = _load_sources(spec, spec.split_sources[split])
    return _maybe_subsample_train(ds, spec) if split == "train" else ds


def source_fingerprint(spec: DatasetSpec) -> dict | None:
    """Raw-file fingerprint for local-source datasets; ``None`` for HF datasets."""
    if spec.local_csv is None:
        return None
    path = Path(spec.local_csv.path)
    if not path.exists():
        return None
    return {"raw_file": str(path), "sha256": _sha256(path)}


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
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    return _apply_label_map(ds, spec)


def _apply_label_map(ds: Dataset, spec: DatasetSpec) -> Dataset:
    """Map string labels to ints for HF datasets that ship non-integer labels.

    No-op unless ``spec.hf_label_map`` is set. Mirrors ``LocalCsvSource.label_map``
    so the rest of the pipeline (stratified splits, tokenization) sees integer
    labels uniformly. Used by dialect-ID datasets like ``statworx/swiss-dialects``.
    """
    mapping = spec.hf_label_map
    if mapping is None:
        return ds
    return ds.map(lambda row: {spec.label_key: mapping[row[spec.label_key]]})


def _load_one(spec: DatasetSpec, hf_split: str) -> Dataset:
    ds = load_dataset(spec.hf_path, spec.hf_config, split=hf_split)
    if not isinstance(ds, Dataset):
        raise TypeError(f"Expected Dataset for split={hf_split!r}, got {type(ds).__name__}")
    return ds


def _is_synthesized(spec: DatasetSpec, split: str) -> bool:
    needs_dev = "validation" not in spec.split_sources and spec.dev_split is not None
    needs_test = "test" not in spec.split_sources and spec.test_split is not None
    return {
        "validation": needs_dev,
        "test": needs_test,
        "train": needs_dev or needs_test,
    }.get(split, False)


def _synthesized_split(spec: DatasetSpec, split: str) -> Dataset:
    pool = _ensure_classlabel(_load_sources(spec, spec.split_sources["train"]), spec)
    pool = _maybe_subsample_train(pool, spec)
    parts = stratified_split_partition(
        pool, dev_split=spec.dev_split, test_split=spec.test_split, label_key=spec.label_key
    )
    return parts[split]


def stratified_split_partition(
    ds: Dataset,
    *,
    dev_split: float | None,
    test_split: float | None,
    label_key: str,
) -> dict[str, Dataset]:
    """Deterministically carve seed-42 stratified dev/test sets out of ``ds``.

    Returns a dict keyed by canonical split name (``train`` always present,
    ``validation``/``test`` present when their fraction is given). Test is carved
    from the pool, then dev from the remainder (relative fraction so its size is
    ``dev_split`` of the *pool*), then train is the rest. The split is reproducible
    across calls (fixed seed), so asking for each split separately yields a
    consistent, leak-free partition.
    """
    parts: dict[str, Dataset] = {}
    remaining = ds
    if test_split is not None:
        carved = remaining.train_test_split(test_size=test_split, seed=_DEV_SPLIT_SEED, stratify_by_column=label_key)
        parts["test"], remaining = carved["test"], carved["train"]
    if dev_split is not None:
        relative = dev_split / (1 - (test_split or 0))
        carved = remaining.train_test_split(test_size=relative, seed=_DEV_SPLIT_SEED, stratify_by_column=label_key)
        parts["validation"], remaining = carved["test"], carved["train"]
    parts["train"] = remaining
    return parts


def _maybe_subsample_train(ds: Dataset, spec: DatasetSpec) -> Dataset:
    """Cap an oversized train pool to ``spec.train_subsample`` via a seed-42 stratified draw.

    No-op when the cap is unset or the pool already fits, so the train/test partition
    carved afterwards is deterministic and class-balanced. Used to keep very large
    datasets (e.g. FEVER) within the 3-epoch wall-clock budget.
    """
    cap = spec.train_subsample
    if cap is None or len(ds) <= cap:
        return ds
    ds = _ensure_classlabel(ds, spec)
    carved = ds.train_test_split(train_size=cap, seed=_DEV_SPLIT_SEED, stratify_by_column=spec.label_key)
    return carved["train"]


def _load_local_csv_split(spec: DatasetSpec, split: str) -> Dataset:
    src = spec.local_csv
    path = Path(src.path)
    if not path.exists():
        raise FileNotFoundError(
            f"{spec.name}: expected the dataset CSV at {src.path!r}. "
            f"Download it from {src.download_url} and save it there."
        )
    ds = load_dataset("csv", data_files=str(path), split="train")
    official = spec.split_sources[split]
    ds = ds.filter(lambda row: row[src.split_column] == official)
    return ds.map(lambda row: {spec.label_key: src.label_map[row[spec.label_key]]})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
