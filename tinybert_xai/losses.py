"""Student loss composition contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers.modeling_outputs import SequenceClassifierOutput

    from tinybert_xai.conditions import ConditionSpec


def compute_student_losses(
    student_out: "SequenceClassifierOutput",
    teacher_out: "SequenceClassifierOutput | None",
    cond: "ConditionSpec",
) -> tuple[torch.Tensor, dict[str, float]]:
    if cond.uses_teacher and teacher_out is None:
        raise RuntimeError(f"Condition {cond.name!r} requires teacher outputs")
    if student_out.loss is None:
        raise RuntimeError("Student batch must include labels so the model returns CE loss")

    losses = {"ce": student_out.loss}
    total = sum(losses.values())
    return total, {name: value.item() for name, value in losses.items()}
