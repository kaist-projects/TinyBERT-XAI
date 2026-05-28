"""Checkpoint + results path conventions and state-dict IO."""

from __future__ import annotations

import json
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


def validate_run_artifacts(
    ckpt_path: Path,
    metadata_path: Path,
    *,
    regenerate_hint: str,
    expected_condition: str | None = None,
) -> None:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}\n{regenerate_hint}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"run_metadata.json not found: {metadata_path}\n{regenerate_hint}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    if metadata.get("schema_version") != "2":
        raise RuntimeError(f"run_metadata.json is not schema v2: {metadata_path}\n{regenerate_hint}")
    if expected_condition is not None and metadata.get("run", {}).get("condition") != expected_condition:
        raise RuntimeError(
            f"run_metadata.json condition does not match {expected_condition!r}: {metadata_path}\n"
            f"{regenerate_hint}"
        )
