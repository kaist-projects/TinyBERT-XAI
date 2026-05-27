from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import BatchEncoding, PreTrainedModel, PreTrainedTokenizerBase


@dataclass
class KDOutputs:
    teacher: Any
    student: Any
    num_labels: int
    batch_size: int
    seq_len: int

    def summary(self) -> str:
        t, s = self.teacher, self.student
        lines = [
            f"  logits            : teacher {tuple(t.logits.shape)}  student {tuple(s.logits.shape)}",
            f"  hidden_states     : teacher ×{len(t.hidden_states)} ({t.hidden_states[0].shape[-1]}-d)"
            f"  student ×{len(s.hidden_states)} ({s.hidden_states[0].shape[-1]}-d)",
            f"  attentions        : teacher ×{len(t.attentions)}  student ×{len(s.attentions)}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class KDPair:
    teacher: PreTrainedModel
    student: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase

    def forward(self, batch: BatchEncoding, *, train_mode: bool = False) -> KDOutputs:
        inputs = {k: v for k, v in batch.items() if k != "labels"}
        if train_mode:
            t_out = self.teacher(**inputs)
            s_out = self.student(**inputs)
        else:
            with torch.no_grad():
                t_out = self.teacher(**inputs)
                s_out = self.student(**inputs)

        return KDOutputs(
            teacher=t_out,
            student=s_out,
            num_labels=t_out.logits.shape[-1],
            batch_size=batch["input_ids"].shape[0],
            seq_len=batch["input_ids"].shape[1],
        )
