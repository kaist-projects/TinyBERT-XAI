"""Inference + classification + calibration metrics. Design doc §7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import sklearn.metrics as skm
import torch

from src.utils import move_batch_to_device

if TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from transformers import PreTrainedModel


@dataclass(frozen=True)
class EvaluationResult:
    """Per-split evaluation. Field names map 1:1 to run_metadata.json keys."""

    macro_f1: float
    micro_f1: float
    accuracy: float
    per_class_f1: list[float]
    confusion_matrix: list[list[int]]
    ECE: float
    NLL: float
    Brier: float


def evaluate(
    model: "PreTrainedModel",
    loader: "DataLoader",
    *,
    device: str,
    num_classes: int,
) -> EvaluationResult:
    """Run inference over `loader`, return classification + calibration metrics."""
    preds, labels, probs = _run_inference(model, loader, device=device)
    return EvaluationResult(
        **_classification_metrics(preds, labels, num_classes=num_classes),
        **_calibration_metrics(probs, labels, num_classes=num_classes),
    )


def collect_probabilities(
    model: "PreTrainedModel",
    loader: "DataLoader",
    *,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run inference over `loader`, return (probs, labels) as CPU tensors."""
    _, labels, probs = _run_inference(model, loader, device=device)
    return torch.from_numpy(probs), torch.from_numpy(labels)


def _run_inference(
    model: "PreTrainedModel",
    loader: "DataLoader",
    *,
    device: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Batched forward pass. Returns (preds, labels, probs) as numpy arrays."""
    model.eval()
    chunks_preds, chunks_labels, chunks_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            labels = batch.pop("labels")
            logits = model(**batch).logits
            chunks_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
            chunks_preds.append(logits.argmax(dim=-1).cpu().numpy())
            chunks_labels.append(labels.cpu().numpy())
    return (
        np.concatenate(chunks_preds),
        np.concatenate(chunks_labels),
        np.concatenate(chunks_probs, axis=0),
    )


def _classification_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
) -> dict:
    lbl = list(range(num_classes))
    return {
        "macro_f1": float(skm.f1_score(labels, preds, average="macro", zero_division=0)),
        "micro_f1": float(skm.f1_score(labels, preds, average="micro", zero_division=0)),
        "accuracy": float(skm.accuracy_score(labels, preds)),
        "per_class_f1": [
            float(s) for s in
            skm.f1_score(labels, preds, average=None, labels=lbl, zero_division=0)
        ],
        "confusion_matrix": skm.confusion_matrix(labels, preds, labels=lbl).tolist(),
    }


def _calibration_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    *,
    num_classes: int,
    n_bins: int = 10,
) -> dict:
    n = len(labels)
    idx = np.arange(n)

    # ECE — equal-width binning of max-confidence vs accuracy
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == labels).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece_val = 0.0
    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        ece_val += mask.sum() / n * abs(correct[mask].mean() - confidences[mask].mean())

    # NLL — mean -log(p_true)
    nll_val = skm.log_loss(labels, probs, labels=list(range(num_classes)))

    # Brier — mean squared error vs one-hot
    one_hot = np.zeros_like(probs)
    one_hot[idx, labels] = 1.0
    brier_val = np.mean(np.sum((probs - one_hot) ** 2, axis=1))

    return {"ECE": float(ece_val), "NLL": float(nll_val), "Brier": float(brier_val)}
