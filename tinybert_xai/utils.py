import os

import torch
from transformers import set_seed as _hf_set_seed


def set_seed(seed: int) -> None:
    # Required for deterministic matmul on CUDA >= 10.2; cuBLAS reads it on
    # first call, so set before any model forward.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _hf_set_seed(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def clone_state_dict_cpu(model: torch.nn.Module) -> dict:
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}
