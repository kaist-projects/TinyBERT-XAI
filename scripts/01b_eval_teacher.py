"""scripts/01b_eval_teacher.py — Test-set evaluation of the saved teacher.

Loads best.pt into a *fresh* model (separate process verification per design
doc), evaluates on the test split, then patches the teacher's
run_metadata.json with schema-v2 metrics.

Run AFTER 01_train_teacher.py has completed.

Usage
-----
    conda activate tinybert-xai
    # from repo root
    python scripts/01b_eval_teacher.py --dataset tweet_eval-sentiment

Writes
------
    results/metadata/<dataset>/teacher/run_metadata.json  (patched in-place)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _config_cli import add_config_flag, add_dataset_override, resolve_run_spec  # noqa: E402

from tinybert_xai import (  # noqa: E402
    configure_reproducibility,
    dataset_by_name,
    evaluate_saved_teacher,
    format_teacher_eval_summary,
    resolve_device,
    save_teacher_evaluation_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the saved teacher on one dataset's test split.")
    add_config_flag(parser)
    add_dataset_override(parser)
    return parser.parse_args()


def main() -> None:
    run = resolve_run_spec(parse_args())
    cfg = run.config
    spec = dataset_by_name(run.dataset)

    configure_reproducibility(cfg.seed)
    device = resolve_device(cfg)
    print(f"Device: {device}")

    print("Loading teacher checkpoint and evaluation data …")
    result = evaluate_saved_teacher(cfg, spec, device=device)
    print("Evaluation complete.")

    print(format_teacher_eval_summary(result))

    save_teacher_evaluation_result(result)
    print(f"\nrun_metadata.json updated: {result.metadata_path}")
    print("\n[OK] Teacher evaluation complete.")

    print("\n── Summary ─────────────────────────────────────────")
    print(f"  test macro-F1  : {result.test_result.macro_f1:.4f}  (target ≥ 0.62)")
    pass_fail = "✓ PASS" if result.test_result.macro_f1 >= 0.62 else "✗ FAIL"
    print(f"  DoD check      : {pass_fail}")


if __name__ == "__main__":
    main()
