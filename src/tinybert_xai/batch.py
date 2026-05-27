import torch
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase


class TweetEvalSentimentBatchEncoder:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        device: str | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.device = device

    def encode(self, ds: Dataset) -> BatchEncoding:
        encoding = self.tokenizer(
            ds["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoding["labels"] = torch.tensor(ds["label"], dtype=torch.long)

        if self.device is not None:
            encoding = encoding.to(self.device)
        return encoding
