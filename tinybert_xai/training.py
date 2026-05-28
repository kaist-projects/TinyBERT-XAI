"""Shared training-loop mechanics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class RunningMeans:
    sums: dict[str, float] = field(default_factory=dict)
    count: int = 0

    def add(self, values: dict[str, float]) -> None:
        for name, value in values.items():
            self.sums[name] = self.sums.get(name, 0.0) + value
        self.count += 1

    def mean(self, name: str) -> float:
        return self.sums.get(name, 0.0) / max(self.count, 1)

    def means(self) -> dict[str, float]:
        return {name: value / max(self.count, 1) for name, value in self.sums.items()}


def seed_loader(loader: Any, seed: int, epoch: int) -> None:
    generator = getattr(loader, "generator", None)
    if generator is not None:
        generator.manual_seed(seed + epoch)


def warn_non_finite(pbar: Any, epoch: int, step: int, loss: torch.Tensor) -> None:
    pbar.write(
        f"  [WARN] non-finite loss at epoch {epoch} "
        f"step {step}: {loss.item():.6f} - skipping batch"
    )


def log_epoch(epoch: int, loss: float, macro_f1: float, accuracy: float, seconds: float) -> None:
    print(
        f"  epoch {epoch}  "
        f"train_loss={loss:.4f}  "
        f"dev_macro_f1={macro_f1:.4f}  "
        f"dev_acc={accuracy:.4f}  "
        f"({seconds:.1f}s)"
    )
