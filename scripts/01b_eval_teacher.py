"""scripts/01b_eval_teacher.py — Test-set evaluation of the saved teacher.

Loads best.pt into a *fresh* model (separate process verification per design
doc), evaluates on the test split, measures efficiency, then patches the
teacher's run_metadata.json with dev_metrics, test_metrics, and efficiency.

Run AFTER 01_train_teacher.py has completed.

Usage
-----
    conda activate tinybert-xai
    python scripts/01b_eval_teacher.py

Writes
------
    results/teachers/tweet_eval-sentiment/run_metadata.json  (patched in-place)
"""

from __future__ import annotations

import json
from dataclasses import asdict

import torch

from tinybert_xai import (
    DATASET_TWEETEVAL_SENTIMENT,
    Config,
    compute_efficiency,
    evaluate,
    get_device,
    load_classifier,
    load_split,
    load_state_dict,
    load_tokenizer,
    results_dir,
    set_seed,
    teacher_dir,
)

# ── determinism ──────────────────────────────────────────────────────────────
torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    cfg = Config()
    spec = DATASET_TWEETEVAL_SENTIMENT

    set_seed(cfg.seed)

    device = cfg.device or get_device()
    print(f"Device: {device}")

    # ── locate artefacts from training run ───────────────────────────────────
    ckpt_path = teacher_dir(spec.name) / "best.pt"
    metadata_path = results_dir("teacher", spec.name) / "run_metadata.json"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\n"
            "Run scripts/01_train_teacher.py first."
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"run_metadata.json not found: {metadata_path}\n"
            "Run scripts/01_train_teacher.py first."
        )

    # ── fresh model load (design-doc DoD: separate process verification) ─────
    print(f"Loading checkpoint: {ckpt_path}")
    tokenizer = load_tokenizer(cfg.tokenizer_checkpoint)
    model = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    load_state_dict(model, ckpt_path, device)
    print("Checkpoint loaded successfully.")

    # ── load dev + test splits ───────────────────────────────────────────────
    print("Loading datasets …")
    dev_ds  = load_split(spec, "validation")
    test_ds = load_split(spec, "test")
    print(f"  dev={len(dev_ds)}  test={len(test_ds)}")

    # ── dev evaluation (best-epoch dev metrics) ──────────────────────────────
    print("\nEvaluating on dev split …")
    dev_result = evaluate(
        model, dev_ds, tokenizer,
        max_length=cfg.max_seq_length,
        device=device,
        batch_size=cfg.eval_batch_size,
        num_classes=spec.num_labels,
    )
    print(f"  dev macro-F1 : {dev_result.macro_f1:.4f}")
    print(f"  dev accuracy : {dev_result.accuracy:.4f}")
    print(f"  dev ECE      : {dev_result.ECE:.4f}")

    # ── test evaluation (use ONCE) ────────────────────────────────────────────
    print("\nEvaluating on test split …")
    test_result = evaluate(
        model, test_ds, tokenizer,
        max_length=cfg.max_seq_length,
        device=device,
        batch_size=cfg.eval_batch_size,
        num_classes=spec.num_labels,
    )
    # teacher-student analysis fields are N/A for the teacher stage
    test_metrics = asdict(test_result)
    test_metrics["top1_agreement"]                = None
    test_metrics["teacher_student_kl"]            = None
    test_metrics["teacher_correct_student_wrong"] = None
    test_metrics["teacher_wrong_student_correct"] = None
    test_metrics["error_copying"]                 = None

    print(f"  test macro-F1 : {test_result.macro_f1:.4f}")
    print(f"  test accuracy : {test_result.accuracy:.4f}")
    print(f"  test ECE      : {test_result.ECE:.4f}")
    print(f"  per-class F1  : {[f'{v:.3f}' for v in test_result.per_class_f1]}")

    # ── efficiency metrics ────────────────────────────────────────────────────
    print("\nMeasuring efficiency …")
    efficiency = compute_efficiency(
        model, tokenizer,
        device=device,
        max_length=cfg.max_seq_length,
        batch_size=cfg.eval_batch_size,
    )
    print(f"  latency p50   : {efficiency.latency_p50_ms:.1f} ms/batch")
    print(f"  latency p95   : {efficiency.latency_p95_ms:.1f} ms/batch")
    print(f"  throughput    : {efficiency.throughput_samples_per_sec:.0f} samples/s")
    print(f"  model size    : {efficiency.model_size_mb:.1f} MB")
    print(f"  param count   : {efficiency.parameter_count:,}")
    if efficiency.gpu_memory_mb is not None:
        print(f"  peak GPU mem  : {efficiency.gpu_memory_mb:.1f} MB")

    # ── update run_metadata.json ─────────────────────────────────────────────
    with open(metadata_path) as f:
        metadata = json.load(f)

    metadata["splits"]["test"] = len(test_ds)
    metadata["dev_metrics"]   = asdict(dev_result)
    metadata["test_metrics"]  = test_metrics
    metadata["efficiency"]    = asdict(efficiency)

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nrun_metadata.json updated: {metadata_path}")
    print("\n[OK] Teacher evaluation complete.")

    # ── summary ─────────────────────────────────────────────────────────────
    print("\n── Summary ─────────────────────────────────────────")
    print(f"  test macro-F1  : {test_result.macro_f1:.4f}  (target ≥ 0.62)")
    pass_fail = "✓ PASS" if test_result.macro_f1 >= 0.62 else "✗ FAIL"
    print(f"  DoD check      : {pass_fail}")


if __name__ == "__main__":
    main()
