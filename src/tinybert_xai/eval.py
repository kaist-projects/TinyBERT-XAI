"""Evaluation helpers — metric functions and inference runner.

All classification metrics are thin wrappers around sklearn.
Calibration metrics (ECE, NLL, Brier) are implemented directly.
Efficiency metrics (latency, throughput, model size) use torch timing.

Public API
----------
macro_f1, micro_f1, accuracy, per_class_f1, confusion_matrix
calibration_metrics   → {"ECE": float, "NLL": float, "Brier": float}
compute_efficiency    → {"latency_p50_ms", "latency_p95_ms", "throughput_samples_per_sec",
                         "model_size_mb", "parameter_count", "gpu_memory_mb"}
evaluate              → dict with all performance + calibration metrics
"""

from __future__ import annotations

import io
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
from datasets import Dataset

import sklearn.metrics as skm

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

from tinybert_xai.datasets import encode_batch


# ─────────────────────────────────────────────
# Classification metrics (sklearn wrappers)
# ─────────────────────────────────────────────

def macro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Macro-averaged F1 — primary metric per design doc §7."""
    return float(skm.f1_score(labels, preds, average="macro", zero_division=0))


def micro_f1(preds: np.ndarray, labels: np.ndarray) -> float:
    """Micro-averaged (instance-level) F1 — secondary metric per design doc §7."""
    return float(skm.f1_score(labels, preds, average="micro", zero_division=0))


def accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Overall accuracy — secondary metric per design doc §7."""
    return float(skm.accuracy_score(labels, preds))


def per_class_f1(preds: np.ndarray, labels: np.ndarray, *, num_classes: int) -> list[float]:
    """Per-class F1 scores, one float per class (indexed 0..num_classes-1)."""
    scores = skm.f1_score(labels, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    return [float(s) for s in scores]


def confusion_matrix(preds: np.ndarray, labels: np.ndarray, *, num_classes: int) -> list[list[int]]:
    """Confusion matrix as nested list[list[int]]."""
    cm = skm.confusion_matrix(labels, preds, labels=list(range(num_classes)))
    return cm.tolist()


# ─────────────────────────────────────────────
# Calibration metrics (ECE, NLL, Brier)
# ─────────────────────────────────────────────

def ece(probs: np.ndarray, labels: np.ndarray, *, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins).

    probs: (N, C) float array of softmax probabilities
    labels: (N,) int array of ground-truth class indices
    """
    confidences = probs.max(axis=1)          # (N,) highest softmax value
    preds = probs.argmax(axis=1)             # (N,)
    correct = (preds == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece_val += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece_val)


def nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """Mean negative log-likelihood of the true class.

    probs: (N, C) float array of softmax probabilities
    labels: (N,) int array of ground-truth class indices
    """
    n = len(labels)
    true_probs = probs[np.arange(n), labels]
    return float(-np.mean(np.log(np.clip(true_probs, 1e-12, 1.0))))


def brier(probs: np.ndarray, labels: np.ndarray, *, num_classes: int) -> float:
    """Multiclass Brier score — mean squared error between probability vectors
    and one-hot targets.

    probs: (N, C) float array of softmax probabilities
    labels: (N,) int array of ground-truth class indices
    """
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    n_bins: int = 10,
) -> dict:
    """Return ECE, NLL, and Brier score as a dict."""
    return {
        "ECE": ece(probs, labels, n_bins=n_bins),
        "NLL": nll(probs, labels),
        "Brier": brier(probs, labels, num_classes=num_classes),
    }


# ─────────────────────────────────────────────
# Efficiency metrics
# ─────────────────────────────────────────────

def compute_efficiency(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    *,
    device: str,
    max_length: int,
    batch_size: int = 32,
    n_warmup: int = 3,
    n_measure: int = 10,
) -> dict:
    """Measure latency, throughput, model size, and peak GPU memory.

    Uses a fixed synthetic batch (all-ones input_ids) so results are
    hardware-reproducible and independent of any real dataset.

    Returns
    -------
    dict with keys:
        latency_p50_ms, latency_p95_ms, throughput_samples_per_sec,
        model_size_mb, parameter_count, gpu_memory_mb
    """
    model.eval()

    # ── serialised checkpoint size ──
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    model_size_mb = buf.tell() / (1024 ** 2)

    # ── parameter count ──
    parameter_count = sum(p.numel() for p in model.parameters())

    # ── synthetic batch ──
    vocab_size = tokenizer.vocab_size
    input_ids = torch.ones(batch_size, max_length, dtype=torch.long, device=device)
    attention_mask = torch.ones(batch_size, max_length, dtype=torch.long, device=device)
    batch = {"input_ids": input_ids, "attention_mask": attention_mask}

    # ── warmup ──
    with torch.no_grad():
        for _ in range(n_warmup):
            model(**batch)

    # ── reset GPU memory stats before measurement ──
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(device)

    # ── timing ──
    latencies = []
    with torch.no_grad():
        for _ in range(n_measure):
            if device.startswith("cuda"):
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                model(**batch)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))
            else:
                t0 = time.perf_counter()
                model(**batch)
                latencies.append((time.perf_counter() - t0) * 1000)

    latencies_arr = np.array(latencies)  # milliseconds per batch

    # ── GPU peak memory ──
    gpu_memory_mb: float | None
    if device.startswith("cuda"):
        gpu_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    else:
        gpu_memory_mb = None

    throughput = batch_size / (latencies_arr.mean() / 1000)  # samples/sec

    return {
        "latency_p50_ms": float(np.percentile(latencies_arr, 50)),
        "latency_p95_ms": float(np.percentile(latencies_arr, 95)),
        "throughput_samples_per_sec": float(throughput),
        "model_size_mb": float(model_size_mb),
        "parameter_count": int(parameter_count),
        "gpu_memory_mb": float(gpu_memory_mb) if gpu_memory_mb is not None else None,
    }


# ─────────────────────────────────────────────
# Umbrella evaluate()
# ─────────────────────────────────────────────

def evaluate(
    model: "PreTrainedModel",
    ds: Dataset,
    tokenizer: "PreTrainedTokenizerBase",
    *,
    max_length: int,
    device: str,
    batch_size: int = 32,
    num_classes: int,
) -> dict:
    """Run inference over `ds` and return all performance + calibration metrics.

    Does NOT measure latency/throughput (call compute_efficiency separately).

    Returns
    -------
    dict with keys:
        macro_f1, micro_f1, accuracy, per_class_f1, confusion_matrix,
        ECE, NLL, Brier
    """
    model.eval()
    all_preds: list[int] = []
    all_labels: list[int] = []
    all_probs: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, len(ds), batch_size):
            chunk = ds.select(range(start, min(start + batch_size, len(ds))))
            batch = encode_batch(tokenizer, chunk, max_length=max_length, device=device)
            labels_tensor = batch.pop("labels")

            logits = model(**batch).logits          # (B, C)
            probs_batch = torch.softmax(logits, dim=-1).cpu().numpy()
            preds_batch = logits.argmax(dim=-1).cpu().numpy()

            all_probs.append(probs_batch)
            all_preds.extend(preds_batch.tolist())
            all_labels.extend(labels_tensor.cpu().numpy().tolist())

    preds_arr = np.array(all_preds)
    labels_arr = np.array(all_labels)
    probs_arr = np.concatenate(all_probs, axis=0)  # (N, C)

    return {
        "macro_f1": macro_f1(preds_arr, labels_arr),
        "micro_f1": micro_f1(preds_arr, labels_arr),
        "accuracy": accuracy(preds_arr, labels_arr),
        "per_class_f1": per_class_f1(preds_arr, labels_arr, num_classes=num_classes),
        "confusion_matrix": confusion_matrix(preds_arr, labels_arr, num_classes=num_classes),
        **calibration_metrics(probs_arr, labels_arr, num_classes=num_classes),
    }
