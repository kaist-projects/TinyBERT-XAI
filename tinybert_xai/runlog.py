"""Run-metadata helpers for the schema-v2 per-run JSON artifact."""

from __future__ import annotations

import datetime as _dt
import json
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


@dataclass
class TrainEpochEntry:
    """Per-epoch training history entry.

    `losses` contains only active loss components. For teacher runs this is
    `{"ce": ...}`; student runs add active KD components.
    """

    epoch: int
    global_step: int
    epoch_time_seconds: float
    loss_total: float
    losses: dict
    grad_norm_mean: float
    dev: dict


@dataclass
class RunMetadata:
    """Top-level run_metadata.json schema v2."""

    schema_version: str
    run: dict
    dataset: dict
    model: dict
    optimization: dict
    checkpoint_selection: dict
    reproducibility: dict
    environment: dict
    training: dict | None = None
    metrics: dict = field(default_factory=dict)
    efficiency: dict | None = None
    metric_definitions: dict = field(default_factory=dict)


def make_run_id(stage: str, dataset_name: str) -> str:
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return f"{stage}-{dataset_name}-{ts}"


def collect_package_versions() -> dict:
    import datasets as hf_datasets
    import numpy as np
    import sklearn
    import tokenizers
    import transformers

    return {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": hf_datasets.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "python": platform.python_version(),
    }


def collect_hardware(device: str) -> dict:
    if device.startswith("cuda"):
        device_index = torch.device(device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        actual_device = f"cuda:{device_index}"
        gpu_model = torch.cuda.get_device_name(actual_device)
        gpu_total_mb = torch.cuda.get_device_properties(actual_device).total_memory / (1024 ** 2)
    else:
        actual_device = device
        gpu_model = None
        gpu_total_mb = None
    return {
        "device": actual_device,
        "gpu_model": gpu_model,
        "gpu_memory_total_mb": gpu_total_mb,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
    }


def collect_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def metric_definitions() -> dict:
    return {
        "ECE": "10 equal-width bins, max-confidence",
        "NLL": "mean negative log probability of the true class",
        "Brier": "mean multiclass squared error against one-hot labels",
        "confusion_matrix": "rows=true, cols=pred",
        "per_class_f1": "ordered by label id",
    }


def write_run_metadata(meta: RunMetadata, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(dumps_run_metadata(meta))
        f.write("\n")


def dumps_run_metadata(meta: RunMetadata) -> str:
    return dumps_metadata_payload(asdict(meta))


def dumps_metadata_payload(payload: dict) -> str:
    payload = _rounded(payload)
    text = json.dumps(payload, indent=2)
    return _compact_numeric_lists(text)


def _rounded(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _rounded(v, k) for k, v in value.items()}
    if isinstance(value, list):
        if key in {"betas"}:
            return value
        return [_rounded(v, key) for v in value]
    if isinstance(value, float):
        if key in {"learning_rate", "weight_decay", "eps"}:
            return value
        if key in {"latency_p50_ms", "latency_p95_ms"}:
            return round(value, 2)
        if key == "throughput_samples_per_sec":
            return round(value, 1)
        if key and key.endswith("_seconds"):
            return round(value, 1)
        if key and (key.endswith("_mb") or key == "model_size_mb"):
            return round(value, 1)
        return round(value, 4)
    return value


def _compact_numeric_lists(text: str) -> str:
    number = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
    pattern = re.compile(r"\[\n((?:\s+" + number + r",\n)*\s+" + number + r"\n)\s+\]", re.I)

    def repl(match: re.Match[str]) -> str:
        values = re.findall(number, match.group(1), re.I)
        return "[" + ", ".join(values) + "]"

    previous = None
    while previous != text:
        previous = text
        text = pattern.sub(repl, text)
    return text
