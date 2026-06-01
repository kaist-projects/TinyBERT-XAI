from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ── iter-0: identity & checkpoints ──
    seed: int = 42
    device: str | None = None
    max_seq_length: int = 128
    teacher_checkpoint: str = "bert-base-uncased"
    student_checkpoint: str = "huawei-noah/TinyBERT_General_4L_312D"
    tokenizer_checkpoint: str = "bert-base-uncased"
    precision: str = "bf16"
    # ── iter-1: training hyperparameters ──
    learning_rate: float = 2e-5
    train_batch_size: int = 16
    eval_batch_size: int = 32
    num_epochs: int = 3
    patience: int = 2
    # ── loss weights: L = ce·CE + logit·L_logit + hidden·L_hidden + attn·L_attn ──
    # All 1.0 reproduces the unweighted sum (locked design-doc behavior).
    ce_weight: float = 1.0
    logit_weight: float = 1.0
    hidden_weight: float = 1.0
    attn_weight: float = 1.0
