from collections.abc import Mapping
from contextlib import nullcontext

import torch


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def training_autocast(device: str, precision: str):
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"precision must be 'bf16' or 'fp32', got {precision!r}")
    if precision == "bf16" and device.startswith("cuda"):
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def clone_state_dict_cpu(model: torch.nn.Module) -> dict:
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}
