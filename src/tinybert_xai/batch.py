import torch
from datasets import Dataset
from transformers import BatchEncoding, PreTrainedTokenizerBase

from tinybert_xai.datasets import TweetEvalSentimentData


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
        rows = [TweetEvalSentimentData.from_row(row) for row in ds]
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
