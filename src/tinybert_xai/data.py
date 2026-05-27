import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase

from tinybert_xai.datasets import DatasetSpec


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

    def load_batch(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        batch_size: int,
        max_length: int,
        split: str,
        device: str | None = None,
    ) -> BatchEncoding:
        ds = self.get_split(split)
        return batch_from_dataset(
            self.spec,
            ds,
            tokenizer,
            batch_size=batch_size,
            max_length=max_length,
            device=device,
        )


def batch_from_dataset(
    spec: DatasetSpec,
    ds: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    max_length: int,
    device: str | None = None,
) -> BatchEncoding:
    examples = ds.select(range(batch_size))
    rows = [spec.data_cls.from_row(row) for row in examples]
    encoding = tokenizer(
        [row.text for row in rows],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    encoding["labels"] = torch.tensor([row.label.value for row in rows], dtype=torch.long)

    if device is not None:
        encoding = encoding.to(device)
    return encoding
