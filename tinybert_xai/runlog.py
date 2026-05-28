"""Run-metadata helpers — design doc §6/§7/§8 top-level schema."""

from __future__ import annotations

import datetime as _dt
import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch


@dataclass
class TrainEpochEntry:
    """Per-epoch training history entry. Design doc §6 schema.

    KD-loss fields are None for the teacher (CE-only) and float for student
    runs that activate the corresponding signal.
    """

    epoch: int
    train_loss_total: float
    train_loss_ce: float
    train_raw_loss_ce: float
    train_loss_logit: float | None
    train_raw_loss_logit: float | None
    train_loss_hidden: float | None
    train_raw_loss_hidden: float | None
    train_loss_attention: float | None
    train_raw_loss_attention: float | None
    grad_norm_mean: float
    learning_rate: float
    global_step: int
    epoch_time_seconds: float
    dev_macro_f1: float
    dev_micro_f1: float
    dev_accuracy: float
    dev_ECE: float
    dev_NLL: float
    dev_Brier: float


@dataclass
class RunMetadata:
    """Top-level run_metadata.json schema.

    `condition` is None for teacher runs; one of the 8 factorial codes for students.
    Nested fields stay as dicts — their shape varies across stages.
    """

    run_id: str
    stage: str
    dataset: str
    dataset_family: str
    condition: str | None
    seed: int
    config: dict
    package_versions: dict
    hardware: dict
    splits: dict = field(default_factory=dict)
    training: dict | None = None
    dev_metrics: dict | None = None
    test_metrics: dict | None = None
    efficiency: dict | None = None


def make_run_id(stage: str, dataset_name: str) -> str:
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{stage}-{dataset_name}-{ts}"


def collect_package_versions() -> dict:
    import datasets as hf_datasets
    import sklearn
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": hf_datasets.__version__,
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
    }


def collect_hardware(device: str) -> dict:
    if device.startswith("cuda"):
        gpu_model = torch.cuda.get_device_name(device)
        gpu_total_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
    else:
        gpu_model = None
        gpu_total_mb = None
    return {
        "gpu_model": gpu_model,
        "gpu_memory_total_mb": gpu_total_mb,
    }


def write_run_metadata(meta: RunMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(meta), f, indent=2)
