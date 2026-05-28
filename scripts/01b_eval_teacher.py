"""scripts/01b_eval_teacher.py — Test-set evaluation of the saved teacher.

Loads best.pt into a *fresh* model (separate process verification per design
doc), evaluates on the test split, measures efficiency, then patches the
teacher's run_metadata.json with schema-v2 metrics and efficiency.

Run AFTER 01_train_teacher.py has completed.

Usage
-----
    conda activate tinybert-xai
    # from repo root
    python scripts/01b_eval_teacher.py

Writes
------
    results/teachers/tweet_eval-sentiment/run_metadata.json  (patched in-place)
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tinybert_xai import (  # noqa: E402
    DATASET_TWEETEVAL_SENTIMENT,
    Config,
    configure_reproducibility,
    evaluate_saved_teacher,
    resolve_device,
    save_teacher_evaluation_result,
)


def main() -> None:
    cfg = Config()
    spec = DATASET_TWEETEVAL_SENTIMENT

    configure_reproducibility(cfg.seed)
    device = resolve_device(cfg)
    print(f"Device: {device}")

    print("Loading teacher checkpoint and evaluation data …")
    result = evaluate_saved_teacher(cfg, spec, device=device)
    print("Evaluation complete.")

    print(f"  dev={result.dev_size}  test={result.test_size}")
    print(f"  dev macro-F1  : {result.dev_result.macro_f1:.4f}")
    print(f"  dev accuracy  : {result.dev_result.accuracy:.4f}")
    print(f"  dev ECE       : {result.dev_result.ECE:.4f}")
    print(f"  test macro-F1 : {result.test_result.macro_f1:.4f}")
    print(f"  test accuracy : {result.test_result.accuracy:.4f}")
    print(f"  test ECE      : {result.test_result.ECE:.4f}")
    print(f"  per-class F1  : {[f'{v:.3f}' for v in result.test_result.per_class_f1]}")
    print(f"  latency p50   : {result.efficiency.latency_p50_ms:.1f} ms/batch")
    print(f"  latency p95   : {result.efficiency.latency_p95_ms:.1f} ms/batch")
    print(f"  throughput    : {result.efficiency.throughput_samples_per_sec:.0f} samples/s")
    print(f"  model size    : {result.efficiency.model_size_mb:.1f} MB")
    print(f"  param count   : {result.efficiency.parameter_count:,}")
    if result.efficiency.gpu_memory_mb is not None:
        print(f"  peak GPU mem  : {result.efficiency.gpu_memory_mb:.1f} MB")

    save_teacher_evaluation_result(result)
    print(f"\nrun_metadata.json updated: {result.metadata_path}")
    print("\n[OK] Teacher evaluation complete.")

    print("\n── Summary ─────────────────────────────────────────")
    print(f"  test macro-F1  : {result.test_result.macro_f1:.4f}  (target ≥ 0.62)")
    pass_fail = "✓ PASS" if result.test_result.macro_f1 >= 0.62 else "✗ FAIL"
    print(f"  DoD check      : {pass_fail}")


if __name__ == "__main__":
    main()
