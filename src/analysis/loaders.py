"""Load schema-v2 run metadata into tidy analysis frames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.distill.conditions import ConditionSpec, all_conditions
from src.storage.checkpoints import METADATA_ROOT
METRIC_COLUMNS = (
    "test_macro_f1",
    "test_micro_f1",
    "test_accuracy",
    "test_ece",
    "test_nll",
    "test_brier",
    "dev_macro_f1",
    "top1_agreement",
    "teacher_student_kl",
    "teacher_correct_student_wrong",
    "teacher_wrong_student_correct",
    "error_copying",
    "loss_ce",
    "loss_logit",
    "loss_hidden",
    "loss_attention",
)


def load_runs(dataset_name: str, metadata_root: Path | str = METADATA_ROOT) -> pd.DataFrame:
    """Load one tidy row per student condition for a dataset.

    Missing or corrupt run metadata is represented as ``valid=False`` with
    numeric fields left as ``NaN`` so the validity gate can report all issues.
    """
    root = Path(metadata_root)
    rows = []
    for condition in all_conditions():
        metadata_path = root / dataset_name / "student" / condition.name / "run_metadata.json"
        try:
            with open(metadata_path) as f:
                payload = json.load(f)
            rows.append(_student_row(dataset_name, condition, payload, metadata_path))
        except Exception as exc:  # noqa: BLE001 - keep validation report complete.
            rows.append(_invalid_row(dataset_name, condition, metadata_path, str(exc)))
    return pd.DataFrame(rows)


def load_all_runs(metadata_root: Path | str = METADATA_ROOT) -> pd.DataFrame:
    """Load every dataset that has student metadata under the metadata root."""
    root = Path(metadata_root)
    if not root.exists():
        return pd.DataFrame()
    datasets = [p.name for p in sorted(root.iterdir()) if (p / "student").is_dir()]
    frames = [load_runs(name, metadata_root) for name in datasets]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_teacher(dataset_name: str, metadata_root: Path | str = METADATA_ROOT) -> pd.Series:
    """Load the teacher reference row for a dataset."""
    metadata_path = Path(metadata_root) / dataset_name / "teacher" / "run_metadata.json"
    with open(metadata_path) as f:
        payload = json.load(f)
    test = payload.get("metrics", {}).get("test", {})
    dev = payload.get("metrics", {}).get("dev", {})
    return pd.Series(
        {
            "dataset": dataset_name,
            "condition": "teacher",
            "valid": True,
            "path": str(metadata_path),
            "test_macro_f1": _get(test, "macro_f1"),
            "test_micro_f1": _get(test, "micro_f1"),
            "test_accuracy": _get(test, "accuracy"),
            "test_ece": _get(test, "ECE"),
            "test_nll": _get(test, "NLL"),
            "test_brier": _get(test, "Brier"),
            "dev_macro_f1": _get(dev, "macro_f1"),
            "num_labels": _get(payload, "dataset", "num_labels"),
            "epochs_completed": _get(payload, "training", "epochs_completed"),
            "num_epochs": _get(payload, "optimization", "num_epochs"),
            "early_stopped": _get(payload, "checkpoint_selection", "early_stopped"),
        }
    )


def _student_row(dataset_name: str, condition: ConditionSpec, payload: dict, path: Path) -> dict:
    test = payload.get("metrics", {}).get("test", {})
    dev = payload.get("metrics", {}).get("dev", {})
    analysis = test.get("teacher_student_analysis", {})
    final_losses = _final_losses(payload)
    metadata_condition = payload.get("run", {}).get("condition")
    valid = metadata_condition == condition.name
    error = None if valid else f"metadata condition is {metadata_condition!r}"

    return {
        "dataset": dataset_name,
        "condition": condition.name,
        "logit": condition.logit,
        "hidden": condition.hidden,
        "attention": condition.attention,
        "valid": valid,
        "error": error,
        "path": str(path),
        "test_macro_f1": _get(test, "macro_f1"),
        "test_micro_f1": _get(test, "micro_f1"),
        "test_accuracy": _get(test, "accuracy"),
        "test_ece": _get(test, "ECE"),
        "test_nll": _get(test, "NLL"),
        "test_brier": _get(test, "Brier"),
        "dev_macro_f1": _get(dev, "macro_f1"),
        "top1_agreement": _get(analysis, "top1_agreement"),
        "teacher_student_kl": _get(analysis, "teacher_student_kl"),
        "teacher_correct_student_wrong": _get(analysis, "teacher_correct_student_wrong"),
        "teacher_wrong_student_correct": _get(analysis, "teacher_wrong_student_correct"),
        "error_copying": _get(analysis, "error_copying"),
        "loss_ce": _get(final_losses, "ce"),
        "loss_logit": _get(final_losses, "logit"),
        "loss_hidden": _get(final_losses, "hidden"),
        "loss_attention": _get(final_losses, "attention"),
        "epochs_completed": _get(payload, "training", "epochs_completed"),
        "num_epochs": _get(payload, "optimization", "num_epochs"),
        "early_stopped": _get(payload, "checkpoint_selection", "early_stopped"),
        "num_labels": _get(payload, "dataset", "num_labels"),
    }


def _invalid_row(dataset_name: str, condition: ConditionSpec, path: Path, error: str) -> dict:
    row = {
        "dataset": dataset_name,
        "condition": condition.name,
        "logit": condition.logit,
        "hidden": condition.hidden,
        "attention": condition.attention,
        "valid": False,
        "error": error,
        "path": str(path),
        "epochs_completed": pd.NA,
        "num_epochs": pd.NA,
        "early_stopped": pd.NA,
        "num_labels": pd.NA,
    }
    row.update({column: pd.NA for column in METRIC_COLUMNS})
    return row


def _final_losses(payload: dict) -> dict:
    history = payload.get("training", {}).get("history", [])
    if not history:
        return {}
    return history[-1].get("losses", {})


def _get(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return pd.NA
        value = value.get(key, pd.NA)
    return value
