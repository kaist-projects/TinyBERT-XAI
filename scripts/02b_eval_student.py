"""scripts/02b_eval_student.py - Test-set evaluation of the CE-only student.

Loads best.pt into a fresh TinyBERT student, evaluates dev/test, measures
efficiency, and patches the student's schema-v2 run_metadata.json.

Run AFTER 02_train_student.py has completed.

Usage
-----
    conda activate tinybert-xai
    # from repo root
    python scripts/02b_eval_student.py

Writes
------
    results/students/tweet_eval-sentiment/ce_only/run_metadata.json
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tinybert_xai import (  # noqa: E402
    CE_ONLY,
    DATASET_TWEETEVAL_SENTIMENT,
    Config,
    configure_reproducibility,
    evaluate_saved_student,
    resolve_device,
    save_student_evaluation_result,
)


def main() -> None:
    cfg = Config()
    spec = DATASET_TWEETEVAL_SENTIMENT
    cond = CE_ONLY

    configure_reproducibility(cfg.seed)
    device = resolve_device(cfg)
    print(f"Device: {device}")
    print(f"Condition: {cond.name}")

    print("Loading student checkpoint and evaluation data ...")
    result = evaluate_saved_student(cfg, spec, cond, device=device)
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

    save_student_evaluation_result(result)
    print(f"\nrun_metadata.json updated: {result.metadata_path}")
    print("\n[OK] Student evaluation complete.")

    print("\n-- Summary ----------------------------------------")
    print(f"  test macro-F1  : {result.test_result.macro_f1:.4f}  (target >= 0.33)")
    pass_fail = "PASS" if result.test_result.macro_f1 >= 0.33 else "FAIL"
    print(f"  DoD check      : {pass_fail}")


if __name__ == "__main__":
    main()
