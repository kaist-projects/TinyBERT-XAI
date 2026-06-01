"""Checkpoint-forward representation analysis for the KD ablation.

These helpers reload the saved teacher and student classifiers and run forward
passes to produce the artifacts that cannot be read from ``run_metadata.json``:

- **Layer similarity via linear CKA.** The trained ``HiddenProjection`` weights
  were never checkpointed (only the student classifier is saved), so the design
  doc's "projected cosine" is not reproducible. Linear CKA compares the 312-d
  student and 768-d teacher representations directly, needs no learned
  projection, and is the standard representation-similarity measure.
- **Attention-distribution KL** per mapped layer (head-averaged), the realized
  form of the design doc's "layer KL divergence".
- **Attention heatmaps** for representative examples.
- **Efficiency**: teacher-vs-student parameter count and forward latency.

The mapped pairs follow the locked layer mapping ``student m -> teacher 3m`` for
``m in {1, 2, 3, 4}``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch

#: Student layer -> teacher layer, the locked uniform g(m) = 3m mapping.
LAYER_MAP = {1: 3, 2: 6, 3: 9, 4: 12}
_KL_EPS = 1e-8


@dataclass(frozen=True)
class ForwardOutputs:
    """Hidden states, attentions, and predictions from one model on one batch set."""

    hidden_states: list[list[torch.Tensor]]  # per batch: tuple of (B, S, H) tensors
    attentions: list[list[torch.Tensor]]  # per batch: tuple of (B, heads, S, S) tensors
    masks: list[torch.Tensor]  # per batch: (B, S) attention masks
    input_ids: list[torch.Tensor]  # per batch: (B, S) token ids
    predictions: torch.Tensor  # (N,) argmax labels
    labels: torch.Tensor  # (N,)


def collect_forward_outputs(model, loader, device: str) -> ForwardOutputs:
    """Run ``model`` over ``loader`` once, keeping hidden states and attentions.

    Everything is moved to CPU so several models' outputs fit in host memory.
    """
    hidden_states: list[list[torch.Tensor]] = []
    attentions: list[list[torch.Tensor]] = []
    masks: list[torch.Tensor] = []
    input_ids: list[torch.Tensor] = []
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out = model(**inputs)
            hidden_states.append([h.cpu() for h in out.hidden_states])
            attentions.append([a.cpu() for a in out.attentions])
            masks.append(inputs["attention_mask"].cpu())
            input_ids.append(batch["input_ids"])
            predictions.append(out.logits.argmax(dim=-1).cpu())
            labels.append(batch["labels"])

    return ForwardOutputs(
        hidden_states=hidden_states,
        attentions=attentions,
        masks=masks,
        input_ids=input_ids,
        predictions=torch.cat(predictions),
        labels=torch.cat(labels),
    )


def layer_cka(student: ForwardOutputs, teacher: ForwardOutputs) -> dict[int, float]:
    """Linear CKA between student and teacher hidden states, per mapped pair.

    Returns ``{student_layer: cka}``. Valid (non-padding) token vectors from every
    batch are stacked, then compared with linear CKA.
    """
    result: dict[int, float] = {}
    for student_layer, teacher_layer in LAYER_MAP.items():
        student_tokens = _valid_tokens(student, student_layer)
        teacher_tokens = _valid_tokens(teacher, teacher_layer)
        result[student_layer] = _linear_cka(student_tokens, teacher_tokens)
    return result


def attention_kl(student: ForwardOutputs, teacher: ForwardOutputs) -> dict[int, float]:
    """Head-averaged KL(teacher || student) of attention maps, per mapped pair.

    Averaged over valid query positions across the whole sample.
    """
    result: dict[int, float] = {}
    for student_layer, teacher_layer in LAYER_MAP.items():
        total, count = 0.0, 0
        for batch_idx in range(len(student.masks)):
            mask = student.masks[batch_idx].bool()  # (B, S)
            teacher_map = _head_mean(teacher.attentions[batch_idx][teacher_layer - 1])
            student_map = _head_mean(student.attentions[batch_idx][student_layer - 1])
            batch_total, batch_count = _masked_row_kl(teacher_map, student_map, mask)
            total += batch_total
            count += batch_count
        result[student_layer] = total / count if count else float("nan")
    return result


@dataclass(frozen=True)
class EfficiencyResult:
    teacher_parameters: int
    student_parameters: int
    teacher_latency_ms: float
    student_latency_ms: float
    parameter_ratio: float
    speedup: float


def measure_efficiency(
    teacher, student, sample_batch: dict, device: str, *, warmup: int = 3, iters: int = 20
) -> EfficiencyResult:
    """Benchmark a single fixed batch through teacher vs student."""
    inputs = {k: v.to(device) for k, v in sample_batch.items() if k != "labels"}
    teacher_params = _parameter_count(teacher)
    student_params = _parameter_count(student)
    teacher_ms = _time_forward(teacher, inputs, device, warmup, iters)
    student_ms = _time_forward(student, inputs, device, warmup, iters)
    return EfficiencyResult(
        teacher_parameters=teacher_params,
        student_parameters=student_params,
        teacher_latency_ms=teacher_ms,
        student_latency_ms=student_ms,
        parameter_ratio=teacher_params / student_params,
        speedup=teacher_ms / student_ms if student_ms else float("nan"),
    )


def select_example_indices(student: ForwardOutputs, teacher: ForwardOutputs) -> dict[str, int]:
    """Pick one representative example per teacher/student correctness category."""
    labels = student.labels
    teacher_ok = teacher.predictions == labels
    student_ok = student.predictions == labels
    categories = {
        "both_correct": teacher_ok & student_ok,
        "teacher_correct_student_wrong": teacher_ok & ~student_ok,
        "teacher_wrong_student_correct": ~teacher_ok & student_ok,
        "both_wrong": ~teacher_ok & ~student_ok,
    }
    chosen: dict[str, int] = {}
    for name, mask in categories.items():
        hits = torch.nonzero(mask, as_tuple=False)
        if len(hits):
            chosen[name] = int(hits[0].item())
    return chosen


def attention_map_for_example(
    outputs: ForwardOutputs, global_index: int, layer: int
) -> tuple[torch.Tensor, int]:
    """Return the head-averaged attention map and valid length for one example.

    ``layer`` is 1-indexed (student 1..4, teacher 1..12).
    """
    batch_idx, local_idx = _locate_example(outputs, global_index)
    attn = _head_mean(outputs.attentions[batch_idx][layer - 1])[local_idx]
    valid = int(outputs.masks[batch_idx][local_idx].sum().item())
    return attn[:valid, :valid], valid


def input_ids_for_example(outputs: ForwardOutputs, global_index: int, length: int) -> torch.Tensor:
    """Return the first ``length`` token ids of one example, for heatmap labels."""
    batch_idx, local_idx = _locate_example(outputs, global_index)
    return outputs.input_ids[batch_idx][local_idx][:length]


def _valid_tokens(outputs: ForwardOutputs, layer: int) -> torch.Tensor:
    chunks = []
    for batch_idx, hidden in enumerate(outputs.hidden_states):
        mask = outputs.masks[batch_idx].bool()  # (B, S)
        chunks.append(hidden[layer][mask])  # (valid_tokens, H)
    return torch.cat(chunks, dim=0)


def _linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = (x - x.mean(dim=0, keepdim=True)).double()
    y = (y - y.mean(dim=0, keepdim=True)).double()
    cross = (x.T @ y).pow(2).sum()
    norm_x = (x.T @ x).pow(2).sum().sqrt()
    norm_y = (y.T @ y).pow(2).sum().sqrt()
    denom = norm_x * norm_y
    return float(cross / denom) if denom > 0 else float("nan")


def _head_mean(attn: torch.Tensor) -> torch.Tensor:
    return attn.mean(dim=1)  # (B, heads, S, S) -> (B, S, S)


def _masked_row_kl(
    teacher_map: torch.Tensor, student_map: torch.Tensor, mask: torch.Tensor
) -> tuple[float, int]:
    key_mask = mask.unsqueeze(1)  # (B, 1, S)
    p = (teacher_map * key_mask).clamp_min(_KL_EPS)
    q = (student_map * key_mask).clamp_min(_KL_EPS)
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    row_kl = (p * (p / q).log()).sum(dim=-1)  # (B, S)
    query_mask = mask.bool()
    total = float(row_kl[query_mask].sum().item())
    count = int(query_mask.sum().item())
    return total, count


def _parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _time_forward(model, inputs: dict, device: str, warmup: int, iters: int) -> float:
    is_cuda = torch.device(device).type == "cuda"
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(**inputs)
        if is_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            model(**inputs)
        if is_cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return 1000.0 * elapsed / iters


def _locate_example(outputs: ForwardOutputs, global_index: int) -> tuple[int, int]:
    offset = 0
    for batch_idx, mask in enumerate(outputs.masks):
        size = mask.shape[0]
        if global_index < offset + size:
            return batch_idx, global_index - offset
        offset += size
    raise IndexError(f"example index {global_index} out of range ({offset} total)")
