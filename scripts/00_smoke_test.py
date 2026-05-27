"""Iteration 0 smoke test.

Verifies that teacher, student, and pilot dataset all load on GPU and produce
the hidden-state / attention shapes iters 1–9 will rely on. Exits 0 on success.
"""

import torch

from config import Config
from data import NUM_LABELS_TWEETEVAL_SENTIMENT, load_tweeteval_sentiment_batch
from models import load_student_for_classification, load_teacher_for_classification
from utils import get_device, set_seed


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    set_seed(42)
    device = get_device()
    assert device == "cuda", f"Expected CUDA device, got {device!r}"

    cfg = Config()
    batch_size = 4
    expected_seq = cfg.max_seq_length

    teacher, tok = load_teacher_for_classification(
        cfg.teacher_model_name,
        num_labels=NUM_LABELS_TWEETEVAL_SENTIMENT,
        device=device,
    )
    student, _ = load_student_for_classification(
        cfg.student_model_name,
        num_labels=NUM_LABELS_TWEETEVAL_SENTIMENT,
        device=device,
    )

    batch = load_tweeteval_sentiment_batch(
        tok, batch_size=batch_size, max_length=expected_seq
    )
    batch = {k: v.to(device) for k, v in batch.items()}

    with torch.no_grad():
        t_out = teacher(**batch)
        s_out = student(**batch)

    # ---- Logit shapes ----
    assert t_out.logits.shape == (batch_size, 3), f"teacher logits {t_out.logits.shape}"
    assert s_out.logits.shape == (batch_size, 3), f"student logits {s_out.logits.shape}"

    # ---- Hidden states (embedding + N transformer layers) ----
    assert len(t_out.hidden_states) == 13, f"teacher hidden_states len={len(t_out.hidden_states)}"
    for i, h in enumerate(t_out.hidden_states):
        assert h.shape == (batch_size, expected_seq, 768), f"teacher h[{i}] shape {h.shape}"

    assert len(s_out.hidden_states) == 5, f"student hidden_states len={len(s_out.hidden_states)}"
    for i, h in enumerate(s_out.hidden_states):
        assert h.shape == (batch_size, expected_seq, 312), f"student h[{i}] shape {h.shape}"

    # ---- Attention probabilities (one tensor per transformer layer) ----
    assert len(t_out.attentions) == 12, f"teacher attentions len={len(t_out.attentions)}"
    for i, a in enumerate(t_out.attentions):
        assert a.shape == (batch_size, 12, expected_seq, expected_seq), f"teacher a[{i}] shape {a.shape}"

    assert len(s_out.attentions) == 4, f"student attentions len={len(s_out.attentions)}"
    for i, a in enumerate(s_out.attentions):
        assert a.shape == (batch_size, 12, expected_seq, expected_seq), f"student a[{i}] shape {a.shape}"

    # ---- Parameter-count sanity ----
    teacher_params_m = _count_params(teacher) / 1e6
    student_params_m = _count_params(student) / 1e6
    assert 100 < teacher_params_m < 120, f"teacher ~110M expected, got {teacher_params_m:.1f}M"
    assert 13 < student_params_m < 16, f"student ~14.5M expected, got {student_params_m:.1f}M"

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[OK] Smoke test passed. "
        f"Teacher {teacher_params_m:.1f}M params, student {student_params_m:.1f}M params. "
        f"Peak VRAM: {peak_vram_gb:.2f} GB."
    )


if __name__ == "__main__":
    main()
