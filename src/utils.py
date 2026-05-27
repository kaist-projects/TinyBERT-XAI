"""Tiny helpers used across the project."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and torch (CPU + CUDA) RNG seeds.

    Does NOT set torch.use_deterministic_algorithms(True) — that goes on in iter 1
    when we actually train, since it carries throughput and op-availability costs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    """Return 'cuda' if a CUDA GPU is available, else 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"
