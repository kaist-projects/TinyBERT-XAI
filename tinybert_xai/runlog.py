"""Run-metadata helpers for the schema-v2 per-run JSON artifact."""

from __future__ import annotations

import datetime as _dt
import json
import re
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


def make_run_id(stage: str, dataset_name: str, condition: str | None = None) -> str:
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    condition_part = f"-{condition}" if condition is not None else ""
    return f"{stage}{condition_part}-{dataset_name}-{ts}"


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
        "torch_cuda": torch.version.cuda,
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


_EXACT_FLOAT_KEYS = {"learning_rate", "weight_decay", "eps"}


def _rounded(value: Any, key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _rounded(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return value if key == "betas" else [_rounded(v, key) for v in value]
    if isinstance(value, float):
        return value if key in _EXACT_FLOAT_KEYS else round(value, 5)
    return value


def _compact_numeric_lists(text: str) -> str:
    number = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
    pattern = re.compile(r"\[\n((?:\s+" + number + r",\n)*\s+" + number + r"\n)\s+\]", re.I)

    def repl(match: re.Match[str]) -> str:
        values = re.findall(number, match.group(1), re.I)
        return "[" + ", ".join(values) + "]"

    return pattern.sub(repl, text)
