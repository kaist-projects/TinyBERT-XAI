"""Teacher-student comparison metrics for post-hoc evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TeacherStudentAnalysis:
    top1_agreement: float
    teacher_student_kl: float
    teacher_correct_student_wrong: int
    teacher_wrong_student_correct: int
    error_copying: float


def compute_teacher_student_analysis(
    teacher_probs: torch.Tensor,
    student_probs: torch.Tensor,
    labels: torch.Tensor,
) -> TeacherStudentAnalysis:
    if teacher_probs.shape != student_probs.shape:
        raise ValueError("teacher_probs and student_probs must have the same shape")
    if teacher_probs.ndim != 2:
        raise ValueError("teacher_probs and student_probs must be rank-2 tensors")
    if labels.shape != (teacher_probs.shape[0],):
        raise ValueError("labels must have shape [N]")

    teacher_preds = teacher_probs.argmax(dim=-1)
    student_preds = student_probs.argmax(dim=-1)
    teacher_correct = teacher_preds == labels
    student_correct = student_preds == labels

    both_wrong = ~teacher_correct & ~student_correct
    copied_wrong = both_wrong & (teacher_preds == student_preds)
    double_wrong_count = both_wrong.sum().item()
    error_copying = 0.0 if double_wrong_count == 0 else copied_wrong.sum().item() / double_wrong_count

    eps = torch.finfo(teacher_probs.dtype).eps
    teacher_clamped = teacher_probs.clamp_min(eps)
    student_clamped = student_probs.clamp_min(eps)
    kl = (teacher_clamped * (teacher_clamped.log() - student_clamped.log())).sum(dim=-1).mean()

    return TeacherStudentAnalysis(
        top1_agreement=(teacher_preds == student_preds).float().mean().item(),
        teacher_student_kl=kl.item(),
        teacher_correct_student_wrong=(teacher_correct & ~student_correct).sum().item(),
        teacher_wrong_student_correct=(~teacher_correct & student_correct).sum().item(),
        error_copying=error_copying,
    )
