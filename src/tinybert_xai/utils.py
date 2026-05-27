import os
import random
from typing import Iterator

import numpy as np
import torch
from datasets import Dataset


def set_seed(seed: int) -> None:
    # Required for deterministic matmul on CUDA >= 10.2; cuBLAS reads it on
    # first call, so set before any model forward.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def clone_state_dict_cpu(model: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def iter_batches(ds: Dataset, batch_size: int) -> Iterator[Dataset]:
    """Yield HF Dataset slices of size `batch_size` (last slice may be shorter)."""
    for start in range(0, len(ds), batch_size):
        yield ds.select(range(start, min(start + batch_size, len(ds))))
