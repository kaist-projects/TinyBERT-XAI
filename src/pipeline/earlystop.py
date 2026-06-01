"""Early-stopping state machine. No torch dependency."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EarlyStopper:
    patience: int
    mode: str = "max"
    best_value: float = field(init=False)
    best_step: int = field(init=False, default=-1)
    no_improve: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode must be 'max' or 'min', got {self.mode!r}")
        self.best_value = float("-inf") if self.mode == "max" else float("inf")

    def update(self, value: float, step: int) -> tuple[bool, bool]:
        """Record an observation. Returns (is_best, should_stop)."""
        improved = value > self.best_value if self.mode == "max" else value < self.best_value
        if improved:
            self.best_value = value
            self.best_step = step
            self.no_improve = 0
            return True, False
        self.no_improve += 1
        return False, self.no_improve >= self.patience
