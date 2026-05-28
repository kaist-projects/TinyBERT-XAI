"""Synthetic-batch efficiency measurement (latency, throughput, size, memory).

Uses an all-ones input batch so results are reproducible and decoupled
from any real dataset.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from tinybert_xai.utils import count_params

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


@dataclass(frozen=True)
class EfficiencyMetrics:
    """Synthetic-batch efficiency. Field names map 1:1 to run_metadata.json keys."""

    latency_p50_ms: float
    latency_p95_ms: float
    throughput_samples_per_sec: float
    model_size_mb: float
    parameter_count: int
    gpu_memory_mb: float | None


def compute_efficiency(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",  # noqa: ARG001 — kept for API symmetry / future use
    *,
    device: str,
    max_length: int,
    batch_size: int = 32,
    n_warmup: int = 3,
    n_measure: int = 10,
) -> EfficiencyMetrics:
    """Measure latency, throughput, model size, parameter count, peak GPU memory."""
    model.eval()
    batch = {
        "input_ids": torch.ones(batch_size, max_length, dtype=torch.long, device=device),
        "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long, device=device),
    }
    latencies = _measure_latency(
        model, batch, device=device, n_warmup=n_warmup, n_measure=n_measure,
    )
    gpu_mem = (
        torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        if device.startswith("cuda") else None
    )
    return EfficiencyMetrics(
        latency_p50_ms=float(np.percentile(latencies, 50)),
        latency_p95_ms=float(np.percentile(latencies, 95)),
        throughput_samples_per_sec=float(batch_size / (latencies.mean() / 1000)),
        model_size_mb=_model_size_mb(model),
        parameter_count=count_params(model),
        gpu_memory_mb=float(gpu_mem) if gpu_mem is not None else None,
    )


def _measure_latency(
    model: "PreTrainedModel",
    batch: dict[str, torch.Tensor],
    *,
    device: str,
    n_warmup: int,
    n_measure: int,
) -> np.ndarray:
    """Warmup + N timed forward passes. Returns ms-per-batch array (length n_measure)."""
    with torch.no_grad():
        for _ in range(n_warmup):
            model(**batch)

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    latencies: list[float] = []
    with torch.no_grad():
        for _ in range(n_measure):
            if device.startswith("cuda"):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                model(**batch)
                end.record()
                torch.cuda.synchronize()
                latencies.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                model(**batch)
                latencies.append((time.perf_counter() - t0) * 1000)
    return np.array(latencies)


def _model_size_mb(model: "PreTrainedModel") -> float:
    """Serialised state-dict size in megabytes."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / (1024 ** 2)
