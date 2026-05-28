import math
from types import SimpleNamespace

import pytest
import torch

from tinybert_xai import KD_LOGIT
from tinybert_xai.losses import compute_student_losses, logit_kd_loss


def test_logit_kd_loss_identical_logits_is_zero():
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.2, 0.3, 0.4]])

    assert logit_kd_loss(logits, logits).item() == pytest.approx(0.0, abs=1e-7)


def test_logit_kd_loss_uniform_teacher_matches_analytic_value():
    student_logits = torch.tensor([[2.0, 0.0]])
    teacher_logits = torch.tensor([[0.0, 0.0]])

    p0 = math.exp(2.0) / (math.exp(2.0) + 1.0)
    p1 = 1.0 / (math.exp(2.0) + 1.0)
    expected = 0.5 * math.log(0.5 / p0) + 0.5 * math.log(0.5 / p1)

    assert logit_kd_loss(student_logits, teacher_logits).item() == pytest.approx(expected)


def test_logit_kd_loss_temperature_changes_value():
    student_logits = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    teacher_logits = torch.tensor([[0.0, 2.0], [1.0, 0.0]])

    assert logit_kd_loss(student_logits, teacher_logits, T=2.0).item() != pytest.approx(
        logit_kd_loss(student_logits, teacher_logits, T=1.0).item()
    )


def test_compute_student_losses_adds_logit_component_for_kd_logit():
    student_out = SimpleNamespace(
        loss=torch.tensor(0.5),
        logits=torch.tensor([[2.0, 0.0]]),
    )
    teacher_out = SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]))

    total, losses = compute_student_losses(student_out, teacher_out, KD_LOGIT)

    assert losses.keys() == {"ce", "logit"}
    assert total.item() == pytest.approx(losses["ce"] + losses["logit"])
