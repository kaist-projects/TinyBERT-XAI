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

from tinybert_xai import (  # noqa: E402
    DATASET_TWEETEVAL_SENTIMENT,
    Config,
    condition_from_flags,
    configure_reproducibility,
    evaluate_saved_student,
    load_classifier,
    load_state_dict,
    resolve_device,
    save_student_evaluation_result,
    teacher_dir,
)


def parse_condition() -> "ConditionSpec":
    parser = argparse.ArgumentParser(description="Evaluate a trained TinyBERT student for one distillation condition.")
    parser.add_argument("--logit", action="store_true", help="enable logit distillation")
    parser.add_argument("--hidden", action="store_true", help="enable hidden-state distillation")
    parser.add_argument("--attention", action="store_true", help="enable attention distillation")
    args = parser.parse_args()
    return condition_from_flags(args.logit, args.hidden, args.attention)


def main() -> None:
    cfg = Config()
    spec = DATASET_TWEETEVAL_SENTIMENT
    cond = parse_condition()

    configure_reproducibility(cfg.seed)
    device = resolve_device(cfg)
    print(f"Device: {device}")
    print(f"Condition: {cond.name}")

    print("Loading teacher checkpoint ...")
    teacher_model = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    load_state_dict(teacher_model, teacher_dir(spec.name) / "best.pt", device)
    teacher_model.eval()

    print("Loading student checkpoint and evaluation data ...")
    result = evaluate_saved_student(cfg, spec, cond, device=device, teacher_model=teacher_model)
    print("Evaluation complete.")

    print(f"  dev={result.dev_size}  test={result.test_size}")
    print(f"  dev macro-F1  : {result.dev_result.macro_f1:.4f}")
    print(f"  dev accuracy  : {result.dev_result.accuracy:.4f}")
    print(f"  dev ECE       : {result.dev_result.ECE:.4f}")
    print(f"  test macro-F1 : {result.test_result.macro_f1:.4f}")
    print(f"  test accuracy : {result.test_result.accuracy:.4f}")
    print(f"  test ECE      : {result.test_result.ECE:.4f}")
    print(f"  per-class F1  : {[f'{v:.3f}' for v in result.test_result.per_class_f1]}")
    if result.teacher_student_analysis is not None:
        analysis = result.teacher_student_analysis
        print(f"  top1 agreement: {analysis.top1_agreement:.4f}")
        print(f"  teacher->student KL: {analysis.teacher_student_kl:.4f}")

    save_student_evaluation_result(result)
    print(f"\nrun_metadata.json updated: {result.metadata_path}")
    print("\n[OK] Student evaluation complete.")

    print("\n-- Summary ----------------------------------------")
    print(f"  test macro-F1  : {result.test_result.macro_f1:.4f}  (target >= 0.33)")
    pass_fail = "PASS" if result.test_result.macro_f1 >= 0.33 else "FAIL"
    print(f"  DoD check      : {pass_fail}")


if __name__ == "__main__":
    main()
