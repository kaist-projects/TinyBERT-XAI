"""scripts/02b_eval_student.py - Test-set evaluation of a TinyBERT student.

Loads best.pt into a fresh TinyBERT student, evaluates dev/test, computes
teacher-student analysis, and patches the student's schema-v2 run_metadata.json.

Run AFTER 02_train_student.py has completed.

Usage
-----
    conda activate tinybert-xai
    # from repo root
    python scripts/02b_eval_student.py --logit --attention   # condition kd_logit_attn
    python scripts/02b_eval_student.py                        # no flags -> ce_only

Writes
------
    results/students/tweet_eval-sentiment/<condition>/run_metadata.json
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _config_cli import add_config_flag, add_dataset_override, add_signal_overrides, resolve_run_spec  # noqa: E402

from tinybert_xai import (  # noqa: E402
    condition_from_flags,
    configure_reproducibility,
    dataset_by_name,
    evaluate_saved_student,
    format_student_eval_summary,
    load_trained_teacher,
    resolve_device,
    save_student_evaluation_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained TinyBERT student for one distillation condition.")
    add_config_flag(parser)
    add_dataset_override(parser)
    add_signal_overrides(parser)
    return parser.parse_args()


def main() -> None:
    run = resolve_run_spec(parse_args())
    cfg = run.config
    spec = dataset_by_name(run.dataset)
    cond = condition_from_flags(run.logit, run.hidden, run.attention)

    configure_reproducibility(cfg.seed)
    device = resolve_device(cfg)
    print(f"Device: {device}")
    print(f"Condition: {cond.name}")

    print("Loading teacher checkpoint ...")
    teacher_model = load_trained_teacher(cfg, spec, device)

    print("Loading student checkpoint and evaluation data ...")
    result = evaluate_saved_student(cfg, spec, cond, device=device, teacher_model=teacher_model)
    save_student_evaluation_result(result)

    print(format_student_eval_summary(result))
    print(f"\nrun_metadata.json updated: {result.metadata_path}")
    print("[OK] Student evaluation complete.")


if __name__ == "__main__":
    main()
