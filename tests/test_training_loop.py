from types import SimpleNamespace

import pytest
import torch

from tinybert_xai import CE_ONLY
from tinybert_xai.student import train_student_epoch
from tinybert_xai.teacher import train_teacher_epoch
from tinybert_xai.training import TrainStats


class TinyLossModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask=None, labels=None):
        return SimpleNamespace(loss=self.weight * input_ids.float().sum())


def _loader():
    return [
        {
            "input_ids": torch.tensor([2.0]),
            "attention_mask": torch.tensor([1]),
            "labels": torch.tensor([0]),
        },
        {
            "input_ids": torch.tensor([float("nan")]),
            "attention_mask": torch.tensor([1]),
            "labels": torch.tensor([0]),
        },
        {
            "input_ids": torch.tensor([3.0]),
            "attention_mask": torch.tensor([1]),
            "labels": torch.tensor([0]),
        },
    ]


def test_train_student_epoch_averages_valid_batches_and_skips_nonfinite():
    model = TinyLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    stats = train_student_epoch(
        model,
        _loader(),
        optimizer,
        CE_ONLY,
        projections=None,
        teacher_model=None,
        device="cpu",
        seed=42,
        epoch=0,
        global_step=7,
        precision="fp32",
    )

    assert stats.loss_total_mean == pytest.approx(2.2)
    assert stats.loss_means == {"ce": pytest.approx(2.2)}
    assert stats.grad_norm_mean == pytest.approx(2.5)
    assert stats.global_step == 9
    assert model.weight.item() == pytest.approx(0.5)


def test_train_teacher_epoch_averages_valid_batches_and_skips_nonfinite():
    model = TinyLossModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    stats = train_teacher_epoch(
        model,
        _loader(),
        optimizer,
        device="cpu",
        seed=42,
        epoch=0,
        global_step=7,
        precision="fp32",
    )

    assert stats.loss_total_mean == pytest.approx(2.2)
    assert stats.loss_ce_mean == pytest.approx(2.2)
    assert stats.grad_norm_mean == pytest.approx(2.5)
    assert stats.global_step == 9
    assert model.weight.item() == pytest.approx(0.5)


def test_train_stats_rejects_unknown_loss_components():
    stats = TrainStats()

    with pytest.raises(ValueError, match="unknown loss component"):
        stats.add_batch(loss=1.0, grad_norm=2.0, components={"typo": 3.0})


def test_train_stats_exposes_named_loss_means_and_rejects_missing_required_component():
    stats = TrainStats()
    stats.add_batch(loss=3.0, grad_norm=2.0, components={"ce": 1.0, "logit": 2.0})
    stats.add_batch(loss=5.0, grad_norm=4.0, components={"ce": 3.0, "logit": 4.0})

    assert stats.loss.total == pytest.approx(4.0)
    assert stats.loss.ce == pytest.approx(2.0)
    assert stats.loss.logit == pytest.approx(3.0)
    assert stats.loss.component_means() == {"ce": pytest.approx(2.0), "logit": pytest.approx(3.0)}

    with pytest.raises(KeyError, match="hidden"):
        _ = stats.loss.hidden
