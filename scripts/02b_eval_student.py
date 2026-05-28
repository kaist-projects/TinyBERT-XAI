"""scripts/02b_eval_student.py - Test-set evaluation of the hidden-KD student.

Loads best.pt into a fresh TinyBERT student, evaluates dev/test, computes
teacher-student analysis, and patches the student's schema-v2 run_metadata.json.

Run AFTER 02_train_student.py has completed.

Usage
-----
    conda activate tinybert-xai
    # from repo root
    python scripts/02b_eval_student.py

Writes
------
    results/students/tweet_eval-sentiment/kd_hidden/run_metadata.json
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tinybert_xai import (  # noqa: E402
    DATASET_TWEETEVAL_SENTIMENT,
    KD_HIDDEN, KD_LOGIT_HIDDEN,
    Config,
    configure_reproducibility,
    evaluate_saved_student,
    load_classifier,
    load_state_dict,
    resolve_device,
    save_student_evaluation_result,
    teacher_dir,
)


def main() -> None:
    cfg = Config()
    spec = DATASET_TWEETEVAL_SENTIMENT
    cond = KD_LOGIT_HIDDEN

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
