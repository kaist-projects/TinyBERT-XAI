import math
from types import SimpleNamespace

import pytest
import torch

from tinybert_xai import KD_ATTN, KD_FULL, KD_HIDDEN, KD_LOGIT, KD_LOGIT_HIDDEN
from tinybert_xai.losses import attention_kd_loss, compute_student_losses, hidden_kd_loss, logit_kd_loss
from tinybert_xai.projections import HiddenProjection


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


def test_hidden_kd_loss_is_zero_when_projected_student_matches_teacher():
    projections = HiddenProjection(student_hidden_dim=2, teacher_hidden_dim=3)
    for projection in projections.projections:
        projection.weight.data.zero_()
        projection.bias.data.zero_()
    student_hidden = tuple(torch.randn(2, 3, 2) for _ in range(5))
    teacher_hidden = tuple(torch.zeros(2, 3, 3) for _ in range(13))
    attention_mask = torch.ones(2, 3)

    assert hidden_kd_loss(student_hidden, teacher_hidden, projections, attention_mask).item() == pytest.approx(0.0)


def test_hidden_kd_loss_ignores_padded_tokens():
    projections = HiddenProjection(student_hidden_dim=2, teacher_hidden_dim=3)
    for projection in projections.projections:
        projection.weight.data.zero_()
        projection.bias.data.zero_()
    student_hidden = tuple(torch.zeros(1, 3, 2) for _ in range(5))
    teacher_hidden = [torch.zeros(1, 3, 3) for _ in range(13)]
    for layer_idx in (3, 6, 9, 12):
        teacher_hidden[layer_idx][:, 1:] = 1000.0
    attention_mask = torch.tensor([[1, 0, 0]])

    assert hidden_kd_loss(student_hidden, tuple(teacher_hidden), projections, attention_mask).item() == pytest.approx(0.0)


def test_hidden_kd_loss_averages_mapped_layers():
    projections = HiddenProjection(student_hidden_dim=2, teacher_hidden_dim=2)
    for projection in projections.projections:
        projection.weight.data.zero_()
        projection.bias.data.zero_()
    student_hidden = tuple(torch.zeros(1, 2, 2) for _ in range(5))
    teacher_hidden = [torch.zeros(1, 2, 2) for _ in range(13)]
    for teacher_layer_idx, value in zip((3, 6, 9, 12), (1.0, 2.0, 3.0, 4.0), strict=True):
        teacher_hidden[teacher_layer_idx].fill_(value)
    attention_mask = torch.ones(1, 2)

    assert hidden_kd_loss(student_hidden, tuple(teacher_hidden), projections, attention_mask).item() == pytest.approx(
        (1.0 + 4.0 + 9.0 + 16.0) / 4.0
    )


def test_compute_student_losses_adds_hidden_component_for_kd_hidden():
    projections = HiddenProjection(student_hidden_dim=2, teacher_hidden_dim=3)
    for projection in projections.projections:
        projection.weight.data.zero_()
        projection.bias.data.zero_()
    student_out = SimpleNamespace(
        loss=torch.tensor(0.5),
        logits=torch.tensor([[2.0, 0.0]]),
        hidden_states=tuple(torch.randn(1, 2, 2) for _ in range(5)),
    )
    teacher_out = SimpleNamespace(
        logits=torch.tensor([[0.0, 2.0]]),
        hidden_states=tuple(torch.zeros(1, 2, 3) for _ in range(13)),
    )

    total, losses = compute_student_losses(
        student_out,
        teacher_out,
        KD_HIDDEN,
        projections=projections,
        attention_mask=torch.ones(1, 2),
    )

    assert losses.keys() == {"ce", "hidden"}
    assert total.item() == pytest.approx(losses["ce"] + losses["hidden"])


def test_compute_student_losses_supports_logit_hidden_condition():
    projections = HiddenProjection(student_hidden_dim=2, teacher_hidden_dim=3)
    for projection in projections.projections:
        projection.weight.data.zero_()
        projection.bias.data.zero_()
    student_out = SimpleNamespace(
        loss=torch.tensor(0.5),
        logits=torch.tensor([[2.0, 0.0]]),
        hidden_states=tuple(torch.randn(1, 2, 2) for _ in range(5)),
    )
    teacher_out = SimpleNamespace(
        logits=torch.tensor([[0.0, 2.0]]),
        hidden_states=tuple(torch.zeros(1, 2, 3) for _ in range(13)),
    )

    total, losses = compute_student_losses(
        student_out,
        teacher_out,
        KD_LOGIT_HIDDEN,
        projections=projections,
        attention_mask=torch.ones(1, 2),
    )

    assert losses.keys() == {"ce", "logit", "hidden"}
    assert total.item() == pytest.approx(sum(losses.values()))


def test_attention_kd_loss_is_zero_when_attention_matches():
    student_attn = tuple(torch.rand(2, 2, 3, 3) for _ in range(4))
    teacher_attn = [torch.rand(2, 2, 3, 3) for _ in range(12)]
    for student_idx, teacher_idx in enumerate((2, 5, 8, 11)):
        teacher_attn[teacher_idx] = student_attn[student_idx].clone()

    assert attention_kd_loss(student_attn, tuple(teacher_attn), torch.ones(2, 3)).item() == pytest.approx(0.0)


def test_attention_kd_loss_ignores_padded_token_pairs():
    student_attn = tuple(torch.zeros(1, 2, 3, 3) for _ in range(4))
    teacher_attn = [torch.zeros(1, 2, 3, 3) for _ in range(12)]
    for layer_idx in (2, 5, 8, 11):
        teacher_attn[layer_idx][:, :, 1:, :] = 1000.0
        teacher_attn[layer_idx][:, :, :, 1:] = 1000.0

    assert attention_kd_loss(student_attn, tuple(teacher_attn), torch.tensor([[1, 0, 0]])).item() == pytest.approx(0.0)


def test_attention_kd_loss_averages_mapped_layers():
    student_attn = tuple(torch.zeros(1, 2, 2, 2) for _ in range(4))
    teacher_attn = [torch.zeros(1, 2, 2, 2) for _ in range(12)]
    for teacher_idx, value in zip((2, 5, 8, 11), (1.0, 2.0, 3.0, 4.0), strict=True):
        teacher_attn[teacher_idx].fill_(value)

    assert attention_kd_loss(student_attn, tuple(teacher_attn), torch.ones(1, 2)).item() == pytest.approx(
        (1.0 + 4.0 + 9.0 + 16.0) / 4.0
    )


def test_attention_kd_loss_averages_heads_when_head_counts_mismatch():
    student_layer = torch.stack(
        [torch.zeros(1, 2, 2), torch.full((1, 2, 2), 2.0)],
        dim=1,
    )
    teacher_layer = torch.full((1, 4, 2, 2), 3.0)
    student_attn = tuple(student_layer.clone() for _ in range(4))
    teacher_attn = [torch.zeros(1, 4, 2, 2) for _ in range(12)]
    for teacher_idx in (2, 5, 8, 11):
        teacher_attn[teacher_idx] = teacher_layer.clone()

    assert attention_kd_loss(student_attn, tuple(teacher_attn), torch.ones(1, 2)).item() == pytest.approx(4.0)


def test_compute_student_losses_adds_attention_component_for_kd_attn():
    student_out = SimpleNamespace(
        loss=torch.tensor(0.5),
        logits=torch.tensor([[2.0, 0.0]]),
        attentions=tuple(torch.zeros(1, 2, 2, 2) for _ in range(4)),
    )
    teacher_out = SimpleNamespace(
        logits=torch.tensor([[0.0, 2.0]]),
        attentions=tuple(torch.zeros(1, 2, 2, 2) for _ in range(12)),
    )

    total, losses = compute_student_losses(
        student_out,
        teacher_out,
        KD_ATTN,
        attention_mask=torch.ones(1, 2),
    )

    assert losses.keys() == {"ce", "attention"}
    assert total.item() == pytest.approx(sum(losses.values()))


def test_compute_student_losses_supports_full_condition():
    projections = HiddenProjection(student_hidden_dim=2, teacher_hidden_dim=3)
    for projection in projections.projections:
        projection.weight.data.zero_()
        projection.bias.data.zero_()
    student_out = SimpleNamespace(
        loss=torch.tensor(0.5),
        logits=torch.tensor([[2.0, 0.0]]),
        hidden_states=tuple(torch.randn(1, 2, 2) for _ in range(5)),
        attentions=tuple(torch.zeros(1, 2, 2, 2) for _ in range(4)),
    )
    teacher_out = SimpleNamespace(
        logits=torch.tensor([[0.0, 2.0]]),
        hidden_states=tuple(torch.zeros(1, 2, 3) for _ in range(13)),
        attentions=tuple(torch.zeros(1, 2, 2, 2) for _ in range(12)),
    )

    total, losses = compute_student_losses(
        student_out,
        teacher_out,
        KD_FULL,
        projections=projections,
        attention_mask=torch.ones(1, 2),
    )

    assert losses.keys() == {"ce", "logit", "hidden", "attention"}
    assert total.item() == pytest.approx(sum(losses.values()))
