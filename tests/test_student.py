from types import SimpleNamespace

import torch

from tinybert_xai import condition_from_flags
from tinybert_xai.pipeline.student import prepare_student_model


class DummyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = torch.nn.Linear(2, 2)


def test_prepare_student_model_adds_hidden_projections_to_optimizer(monkeypatch):
    def fake_load_classifier(checkpoint, num_labels, device):
        return DummyClassifier().to(torch.device(device))

    monkeypatch.setattr("tinybert_xai.pipeline.student.load_classifier", fake_load_classifier)
    cfg = SimpleNamespace(student_checkpoint="student", learning_rate=2e-5)
    spec = SimpleNamespace(num_labels=2)

    student = prepare_student_model(cfg, spec, condition_from_flags(False, True, False), "cpu")

    assert student.projections is not None
    optimizer_param_ids = {id(param) for group in student.optimizer.param_groups for param in group["params"]}
    projection_param_ids = {id(param) for param in student.projections.parameters()}
    assert projection_param_ids <= optimizer_param_ids
    assert student.projection_parameter_count == 961536
    assert student.parameter_count == 6 + 961536
