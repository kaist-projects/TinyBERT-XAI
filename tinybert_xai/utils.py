from collections.abc import Mapping

import torch


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def move_batch_to_device(
    batch: Mapping[str, torch.Tensor],
    device: str,
) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def clone_state_dict_cpu(model: torch.nn.Module) -> dict:
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}
