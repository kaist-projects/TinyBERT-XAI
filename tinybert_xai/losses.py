"""Student loss composition contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from tinybert_xai.projections import TEACHER_HIDDEN_LAYERS, HiddenProjection

if TYPE_CHECKING:
    from transformers.modeling_outputs import SequenceClassifierOutput

    from tinybert_xai.conditions import ConditionSpec


# Attention tuples have no embedding entry, unlike hidden-state tuples.
TEACHER_ATTENTION_LAYERS = (2, 5, 8, 11)


def logit_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    T: float = 1.0,
) -> torch.Tensor:
    """T^2 * KL(softmax(teacher/T) || softmax(student/T))."""
    if T <= 0:
        raise ValueError(f"T must be positive, got {T}")
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / T, dim=-1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T**2)


def hidden_kd_loss(
    student_hidden: tuple[torch.Tensor, ...],
    teacher_hidden: tuple[torch.Tensor, ...],
    projections: HiddenProjection,
    attention_mask: torch.Tensor,
    *,
    layer_map: tuple[int, ...] = TEACHER_HIDDEN_LAYERS,
) -> torch.Tensor:
    """Masked MSE between projected student layers 1..N and mapped teacher layers."""
    if len(student_hidden) <= len(layer_map):
        raise RuntimeError(f"Student hidden states must include layers 1..{len(layer_map)}")
    if len(teacher_hidden) <= max(layer_map):
        raise RuntimeError(f"Teacher hidden states must include layer {max(layer_map)}")
    if projections.num_layers != len(layer_map):
        raise RuntimeError(f"Expected {len(layer_map)} hidden projections, got {projections.num_layers}")
    if attention_mask is None:
        raise RuntimeError("Hidden KD requires attention_mask for padding-token masking")

    layer_losses = []
    for projection_idx, teacher_layer_idx in enumerate(layer_map):
        student_layer_idx = projection_idx + 1
        projection = projections.projections[projection_idx]
        student_state = student_hidden[student_layer_idx].to(dtype=projection.weight.dtype)
        teacher_state = teacher_hidden[teacher_layer_idx].detach().to(dtype=projection.weight.dtype)

        projected = projections(projection_idx, student_state)
        mask = attention_mask.to(device=projected.device, dtype=projected.dtype).unsqueeze(-1)
        hidden_dim = projected.shape[-1]
        # Count valid tokens from the integer mask so the divisor is exact regardless of tensor dtype.
        denominator = (attention_mask.sum() * hidden_dim).clamp_min(1)
        layer_losses.append((mask * (projected - teacher_state).pow(2)).sum() / denominator)

    return torch.stack(layer_losses).mean()


def attention_kd_loss(
    student_attn: tuple[torch.Tensor, ...],
    teacher_attn: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    *,
    layer_map: tuple[int, ...] = TEACHER_ATTENTION_LAYERS,
) -> torch.Tensor:
    """Masked MSE between student and mapped teacher attention probabilities."""
    if len(student_attn) < len(layer_map):
        raise RuntimeError(f"Student attentions must include {len(layer_map)} layers")
    if len(teacher_attn) <= max(layer_map):
        raise RuntimeError(f"Teacher attentions must include layer {max(layer_map)}")
    if attention_mask is None:
        raise RuntimeError("Attention KD requires attention_mask for token-pair masking")

    pair_mask = _valid_token_pair_mask(attention_mask)
    valid_pairs = pair_mask.sum()  # exact integer count, independent of tensor dtype
    layer_losses = []
    for student_layer_idx, teacher_layer_idx in enumerate(layer_map):
        student_layer_attn = student_attn[student_layer_idx]
        teacher_layer_attn = teacher_attn[teacher_layer_idx].detach().to(dtype=student_layer_attn.dtype)

        if student_layer_attn.shape[1] != teacher_layer_attn.shape[1]:
            student_layer_attn = student_layer_attn.mean(dim=1, keepdim=True)
            teacher_layer_attn = teacher_layer_attn.mean(dim=1, keepdim=True)

        mask = pair_mask.to(device=student_layer_attn.device, dtype=student_layer_attn.dtype).unsqueeze(1)
        num_heads = student_layer_attn.shape[1]
        denominator = (valid_pairs * num_heads).clamp_min(1)
        layer_losses.append((mask * (student_layer_attn - teacher_layer_attn).pow(2)).sum() / denominator)

    return torch.stack(layer_losses).mean()


def compute_student_losses(
    student_out: "SequenceClassifierOutput",
    teacher_out: "SequenceClassifierOutput | None",
    cond: "ConditionSpec",
    *,
    projections: HiddenProjection | None = None,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if cond.uses_teacher and teacher_out is None:
        raise RuntimeError(f"Condition {cond.name!r} requires teacher outputs")
    if student_out.loss is None:
        raise RuntimeError("Student batch must include labels so the model returns CE loss")

    losses = {"ce": student_out.loss}
    if cond.logit:
        losses["logit"] = logit_kd_loss(student_out.logits, teacher_out.logits)
    if cond.hidden:
        if projections is None:
            raise RuntimeError(f"Condition {cond.name!r} requires hidden projections")
        losses["hidden"] = hidden_kd_loss(
            student_out.hidden_states,
            teacher_out.hidden_states,
            projections,
            attention_mask,
        )
    if cond.attention:
        losses["attention"] = attention_kd_loss(
            student_out.attentions,
            teacher_out.attentions,
            attention_mask,
        )
    total = sum(losses.values())
    return total, {name: value.item() for name, value in losses.items()}


def _valid_token_pair_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.bool()
    return mask.unsqueeze(1) & mask.unsqueeze(2)
