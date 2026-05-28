"""Iteration 0 smoke test — explicit wiring proves DI/SRP layout works on GPU.

Usage
-----
    conda activate tinybert-xai
    # from repo root
    python scripts/00_smoke_test.py
"""
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tinybert_xai import (  # noqa: E402
    Config,
    DATASET_TWEETEVAL_SENTIMENT,
    configure_reproducibility,
    count_params,
    encode_batch,
    load_classifier,
    load_split,
    load_tokenizer,
    resolve_device,
)


def assert_shapes_consistent(teacher_out, student_out, *, batch_size: int, seq_len: int, num_labels: int) -> None:
    assert teacher_out.logits.shape == (batch_size, num_labels), f"teacher logits {teacher_out.logits.shape}"
    assert student_out.logits.shape == (batch_size, num_labels), f"student logits {student_out.logits.shape}"

    t_hidden = teacher_out.hidden_states[0].shape[-1]
    s_hidden = student_out.hidden_states[0].shape[-1]
    for i, h in enumerate(teacher_out.hidden_states):
        assert h.shape == (batch_size, seq_len, t_hidden), f"teacher h[{i}] {h.shape}"
    for i, h in enumerate(student_out.hidden_states):
        assert h.shape == (batch_size, seq_len, s_hidden), f"student h[{i}] {h.shape}"

    t_layers = len(teacher_out.hidden_states) - 1
    s_layers = len(student_out.hidden_states) - 1
    assert len(teacher_out.attentions) == t_layers, f"teacher attentions len={len(teacher_out.attentions)}"
    assert len(student_out.attentions) == s_layers, f"student attentions len={len(student_out.attentions)}"
    t_heads = teacher_out.attentions[0].shape[1]
    s_heads = student_out.attentions[0].shape[1]
    for i, a in enumerate(teacher_out.attentions):
        assert a.shape == (batch_size, t_heads, seq_len, seq_len), f"teacher a[{i}] {a.shape}"
    for i, a in enumerate(student_out.attentions):
        assert a.shape == (batch_size, s_heads, seq_len, seq_len), f"student a[{i}] {a.shape}"


def summarise(teacher_out, student_out) -> str:
    t, s = teacher_out, student_out
    return (
        f"  logits            : teacher {tuple(t.logits.shape)}  student {tuple(s.logits.shape)}\n"
        f"  hidden_states     : teacher ×{len(t.hidden_states)} ({t.hidden_states[0].shape[-1]}-d)"
        f"  student ×{len(s.hidden_states)} ({s.hidden_states[0].shape[-1]}-d)\n"
        f"  attentions        : teacher ×{len(t.attentions)}  student ×{len(s.attentions)}"
    )


def main() -> None:
    cfg = Config()
    configure_reproducibility(cfg.seed)
    device = resolve_device(cfg)

    spec = DATASET_TWEETEVAL_SENTIMENT
    tokenizer = load_tokenizer(cfg.tokenizer_checkpoint)
    teacher = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    student = load_classifier(cfg.student_checkpoint, spec.num_labels, device)

    raw = load_split(spec, "train").select(range(16))
    batch = encode_batch(tokenizer, raw, max_length=cfg.max_seq_length, device=device)
    inputs = {k: v for k, v in batch.items() if k != "labels"}

    with torch.no_grad():
        teacher_out = teacher(**inputs)
        student_out = student(**inputs)

    assert_shapes_consistent(
        teacher_out,
        student_out,
        batch_size=batch["input_ids"].shape[0],
        seq_len=batch["input_ids"].shape[1],
        num_labels=spec.num_labels,
    )

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[OK] Smoke test passed.\n"
        f"{summarise(teacher_out, student_out)}\n"
        f"  teacher params    : {count_params(teacher) / 1e6:.1f}M\n"
        f"  student params    : {count_params(student) / 1e6:.1f}M\n"
        f"  peak VRAM         : {peak_vram_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()
