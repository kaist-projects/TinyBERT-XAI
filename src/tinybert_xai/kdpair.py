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

    def assert_shapes_consistent(self) -> None:
        t, s = self.teacher, self.student
        bs, sl, nl = self.batch_size, self.seq_len, self.num_labels

        assert t.logits.shape == (bs, nl), f"teacher logits {t.logits.shape}"
        assert s.logits.shape == (bs, nl), f"student logits {s.logits.shape}"

        t_layers = len(t.hidden_states) - 1
        s_layers = len(s.hidden_states) - 1
        t_hidden = t.hidden_states[0].shape[-1]
        s_hidden = s.hidden_states[0].shape[-1]
        for i, h in enumerate(t.hidden_states):
            assert h.shape == (bs, sl, t_hidden), f"teacher h[{i}] {h.shape}"
        for i, h in enumerate(s.hidden_states):
            assert h.shape == (bs, sl, s_hidden), f"student h[{i}] {h.shape}"

        assert len(t.attentions) == t_layers, f"teacher attentions len={len(t.attentions)}"
        assert len(s.attentions) == s_layers, f"student attentions len={len(s.attentions)}"
        t_heads = t.attentions[0].shape[1]
        s_heads = s.attentions[0].shape[1]
        for i, a in enumerate(t.attentions):
            assert a.shape == (bs, t_heads, sl, sl), f"teacher a[{i}] {a.shape}"
        for i, a in enumerate(s.attentions):
            assert a.shape == (bs, s_heads, sl, sl), f"student a[{i}] {a.shape}"

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
