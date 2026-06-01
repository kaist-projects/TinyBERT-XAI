"""Checkpoint + results path conventions and state-dict IO."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from src.utils import clone_state_dict_cpu


# All generated artifacts live under a single dataset-first results/ root:
#   results/checkpoints/<dataset>/{teacher,student/<condition>}/...
#   results/metadata/<dataset>/{teacher,student/<condition>}/run_metadata.json
#   results/analysis/<dataset>/... and results/analysis/cross_dataset/...
RESULTS_ROOT = Path("results")
CHECKPOINTS_ROOT = RESULTS_ROOT / "checkpoints"
METADATA_ROOT = RESULTS_ROOT / "metadata"
ANALYSIS_ROOT = RESULTS_ROOT / "analysis"


def teacher_dir(dataset_name: str) -> Path:
    return CHECKPOINTS_ROOT / dataset_name / "teacher"


def student_dir(dataset_name: str, condition: str) -> Path:
    return CHECKPOINTS_ROOT / dataset_name / "student" / condition


def metadata_dir(dataset_name: str, stage: str, condition: str | None = None) -> Path:
    """Metadata directory for a run. ``stage`` is ``"teacher"`` or ``"student"``."""
    base = METADATA_ROOT / dataset_name / stage
    return base / condition if condition is not None else base


def analysis_dir(dataset_name: str) -> Path:
    return ANALYSIS_ROOT / dataset_name


def cross_dataset_dir() -> Path:
    return ANALYSIS_ROOT / "cross_dataset"


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
