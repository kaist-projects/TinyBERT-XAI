"""Shared training-loop mechanics."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any

import torch

from src.utils import move_batch_to_device

LOSS_COMPONENTS = ("ce", "logit", "hidden", "attention")


@dataclass
class _Mean:
    total: float = 0.0
    count: int = 0

    def add(self, value: float) -> None:
        self.total += value
        self.count += 1

    @property
    def value(self) -> float:
        return self.total / max(self.count, 1)


@dataclass
class LossStats:
    _total: _Mean = field(default_factory=_Mean)
    _components: dict[str, _Mean] = field(default_factory=dict)

    def add_batch(self, *, total: float, components: dict[str, float]) -> None:
        self._total.add(total)
        for name, value in components.items():
            if name not in LOSS_COMPONENTS:
                expected = ", ".join(LOSS_COMPONENTS)
                raise ValueError(f"unknown loss component {name!r}; expected one of: {expected}")
            self._components.setdefault(name, _Mean()).add(value)

    @property
    def total(self) -> float:
        return self._total.value

    @property
    def ce(self) -> float:
        return self._required_component("ce")

    @property
    def logit(self) -> float:
        return self._required_component("logit")

    @property
    def hidden(self) -> float:
        return self._required_component("hidden")

    @property
    def attention(self) -> float:
        return self._required_component("attention")

    def component_means(self) -> dict[str, float]:
        return {name: mean.value for name, mean in self._components.items()}

    def _required_component(self, name: str) -> float:
        if name not in self._components:
            raise KeyError(f"loss component {name!r} was not recorded")
        return self._components[name].value


@dataclass
class TrainStats:
    loss: LossStats = field(default_factory=LossStats)
    grad_norm: _Mean = field(default_factory=_Mean)
    count: int = 0

    def add_batch(self, *, loss: float, grad_norm: float, components: dict[str, float]) -> None:
        self.loss.add_batch(total=loss, components=components)
        self.grad_norm.add(grad_norm)
        self.count += 1

    @property
    def grad_norm_mean(self) -> float:
        return self.grad_norm.value


def measure_grad_norm(parameters) -> float:
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm=float("inf")).item()


@dataclass(frozen=True)
class TrainingEpochResult:
    stats: TrainStats
    global_step: int
    epoch_time_seconds: float


def run_training_epoch(
    loader: Iterable[Mapping[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    batch_loss_fn: Callable[[dict[str, torch.Tensor]], tuple[torch.Tensor, dict[str, float | torch.Tensor]]],
    parameters,
    device: str,
    seed: int,
    epoch: int,
    global_step: int,
    progress_factory,
) -> TrainingEpochResult:
    epoch_start = time.perf_counter()
    seed_loader(loader, seed, epoch)
    stats = TrainStats()

    pbar = progress_factory(loader, total=len(loader), desc=f"epoch {epoch}", unit="batch")
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        loss, components = batch_loss_fn(batch)

        if not torch.isfinite(loss):
            warn_non_finite(pbar, epoch, global_step, loss)
            optimizer.zero_grad()
            continue

        loss.backward()
        grad_norm = measure_grad_norm(parameters)
        optimizer.step()
        optimizer.zero_grad()

        component_values = {
            name: value.item() if isinstance(value, torch.Tensor) else value
            for name, value in components.items()
        }
        stats.add_batch(loss=loss.item(), grad_norm=grad_norm, components=component_values)
        global_step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return TrainingEpochResult(
        stats=stats,
        global_step=global_step,
        epoch_time_seconds=time.perf_counter() - epoch_start,
    )


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
