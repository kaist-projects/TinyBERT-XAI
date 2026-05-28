"""Checkpoint + results path conventions and state-dict IO."""

from __future__ import annotations

from pathlib import Path

import torch

from tinybert_xai.utils import clone_state_dict_cpu


def teacher_dir(dataset_name: str) -> Path:
    return Path("checkpoints") / "teachers" / dataset_name


def student_dir(dataset_name: str, condition: str) -> Path:
    return Path("checkpoints") / "students" / dataset_name / condition


def results_dir(stage: str, dataset_name: str, condition: str | None = None) -> Path:
    base = Path("results") / f"{stage}s" / dataset_name
    return base / condition if condition is not None else base


def save_state_dict(model: torch.nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clone_state_dict_cpu(model), path)


def load_state_dict(model: torch.nn.Module, path: Path, device: str) -> None:
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
