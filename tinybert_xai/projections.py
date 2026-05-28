"""Projection modules for representation-level distillation."""

from __future__ import annotations

import torch
from torch import nn


STUDENT_HIDDEN_LAYERS = (1, 2, 3, 4)
TEACHER_HIDDEN_LAYERS = (3, 6, 9, 12)
STUDENT_TO_TEACHER_LAYER = dict(zip(STUDENT_HIDDEN_LAYERS, TEACHER_HIDDEN_LAYERS, strict=True))


class HiddenProjection(nn.Module):
    """Independent student-hidden to teacher-hidden projections per mapped layer."""

    def __init__(
        self,
        *,
        student_hidden_dim: int = 312,
        teacher_hidden_dim: int = 768,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        self.student_hidden_dim = student_hidden_dim
        self.teacher_hidden_dim = teacher_hidden_dim
        self.num_layers = num_layers
        self.projections = nn.ModuleList(
            [nn.Linear(student_hidden_dim, teacher_hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        return self.projections[layer_idx](hidden)
