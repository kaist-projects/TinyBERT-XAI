from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    seed: int = 42
    device: str | None = None
    max_seq_length: int = 128
    teacher_checkpoint: str = "bert-base-uncased"
    student_checkpoint: str = "huawei-noah/TinyBERT_General_4L_312D"
    tokenizer_checkpoint: str = "bert-base-uncased"
