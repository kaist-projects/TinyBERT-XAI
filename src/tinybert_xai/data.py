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


class BatchEncoder:
    def __init__(
        self,
        spec: DatasetSpec,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int,
        device: str | None = None,
    ) -> None:
        self.spec = spec
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def encode(self, ds: Dataset, *, batch_size: int) -> BatchEncoding:
        examples = ds.select(range(batch_size))
        rows = [self.spec.data_cls.from_row(row) for row in examples]
        encoding = self.tokenizer(
            [row.text for row in rows],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding["labels"] = torch.tensor([row.label.value for row in rows], dtype=torch.long)

        if self.device is not None:
            encoding = encoding.to(self.device)
        return encoding
