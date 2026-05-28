import os
from collections.abc import Mapping

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


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def clone_state_dict_cpu(model: torch.nn.Module) -> dict:
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}
