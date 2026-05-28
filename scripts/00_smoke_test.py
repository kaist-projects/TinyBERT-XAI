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
    KDOutputs,
    KDPair,
    configure_reproducibility,
    count_params,
    encode_batch,
    load_classifier,
    load_split,
    load_tokenizer,
    resolve_device,
)


def assert_shapes_consistent(out: KDOutputs) -> None:
    t, s = out.teacher, out.student
    bs, sl, nl = out.batch_size, out.seq_len, out.num_labels

    assert t.logits.shape == (bs, nl), f"teacher logits {t.logits.shape}"
    assert s.logits.shape == (bs, nl), f"student logits {s.logits.shape}"

    t_layers = len(t.hidden_states) - 1
    s_layers = len(s.hidden_states) - 1
    t_hidden = t.hidden_states[0].shape[-1]
    s_hidden = s.hidden_states[0].shape[-1]
    for i, h in enumerate(t.hidden_states):
        assert h.shape == (bs, sl, t_hidden), f"teacher h[{i}] {h.shape}"
    for i, h in enumerate(s.hidden_states):
        assert h.shape == (bs, sl, s_hidden), f"student h[{i}] {h.shape}"

    assert len(t.attentions) == t_layers, f"teacher attentions len={len(t.attentions)}"
    assert len(s.attentions) == s_layers, f"student attentions len={len(s.attentions)}"
    t_heads = t.attentions[0].shape[1]
    s_heads = s.attentions[0].shape[1]
    for i, a in enumerate(t.attentions):
        assert a.shape == (bs, t_heads, sl, sl), f"teacher a[{i}] {a.shape}"
    for i, a in enumerate(s.attentions):
        assert a.shape == (bs, s_heads, sl, sl), f"student a[{i}] {a.shape}"


def main() -> None:
    cfg = Config()
    configure_reproducibility(cfg.seed)
    device = resolve_device(cfg)

    spec = DATASET_TWEETEVAL_SENTIMENT
    tokenizer = load_tokenizer(cfg.tokenizer_checkpoint)
    teacher = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    student = load_classifier(cfg.student_checkpoint, spec.num_labels, device)
    pair = KDPair(teacher, student)

    raw = load_split(spec, "train").select(range(16))
    batch = encode_batch(tokenizer, raw, max_length=cfg.max_seq_length, device=device)

    out = pair.forward(batch)
    assert_shapes_consistent(out)

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    print(
        f"[OK] Smoke test passed.\n"
        f"{out.summary()}\n"
        f"  teacher params    : {count_params(teacher) / 1e6:.1f}M\n"
        f"  student params    : {count_params(student) / 1e6:.1f}M\n"
        f"  peak VRAM         : {peak_vram_gb:.2f} GB"
    )


if __name__ == "__main__":
    main()
