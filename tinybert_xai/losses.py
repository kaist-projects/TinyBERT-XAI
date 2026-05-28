"""Student loss composition contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers.modeling_outputs import SequenceClassifierOutput

    from tinybert_xai.conditions import ConditionSpec


def logit_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    T: float = 1.0,
) -> torch.Tensor:
    """T^2 * KL(softmax(teacher/T) || softmax(student/T))."""
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / T, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T**2)


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
    if cond.logit:
        losses["logit"] = logit_kd_loss(student_out.logits, teacher_out.logits)
    total = sum(losses.values())
    return total, {name: value.item() for name, value in losses.items()}
