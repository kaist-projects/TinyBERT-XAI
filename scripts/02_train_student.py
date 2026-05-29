"""scripts/02_train_student.py - TinyBERT student training.

Trains huawei-noah/TinyBERT_General_4L_312D on TweetEval-sentiment for the
selected condition, writes a best checkpoint, and records schema-v2 metadata.

Usage
-----
    conda activate tinybert-xai
    # from repo root
    python scripts/02_train_student.py --logit --attention   # condition kd_logit_attn
    python scripts/02_train_student.py                        # no flags -> ce_only
    python scripts/02_train_student.py --logit --eval         # train, then evaluate

Output
------
    checkpoints/students/tweet_eval-sentiment/<condition>/
        epoch_0.pt, epoch_1.pt, ..., best.pt
    results/students/tweet_eval-sentiment/<condition>/
        run_metadata.json

With --eval, the run_metadata.json is patched with dev/test metrics in the same
pass (equivalent to running scripts/02b_eval_student.py afterwards).
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
    fine_tune_student,
    load_classifier,
    load_state_dict,
    load_student_data,
    prepare_student_model,
    resolve_device,
    save_student_evaluation_result,
    save_student_training_result,
    start_student_metadata,
    teacher_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a TinyBERT student for one distillation condition.")
    parser.add_argument("--logit", action="store_true", help="enable logit distillation")
    parser.add_argument("--hidden", action="store_true", help="enable hidden-state distillation")
    parser.add_argument("--attention", action="store_true", help="enable attention distillation")
    parser.add_argument("--eval", action="store_true", help="evaluate on dev/test after training completes")
    return parser.parse_args()


def load_teacher(cfg, spec, device):
    teacher_model = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
    load_state_dict(teacher_model, teacher_dir(spec.name) / "best.pt", device)
    teacher_model.eval()
    return teacher_model


def train_student(cfg, spec, cond, device, teacher_model):
    meta = start_student_metadata(cfg, spec, cond, device)

    print("Loading datasets ...")
    data = load_student_data(cfg, spec)
    meta.dataset["splits"] = {"train": data.train_size, "validation": data.dev_size}
    print(f"  train={data.train_size}  dev={data.dev_size}")

    student = prepare_student_model(cfg, spec, cond, device)
    meta.model["parameter_count"] = student.parameter_count
    if student.projection_parameter_count is not None:
        meta.model["projection_parameter_count"] = student.projection_parameter_count

    result = fine_tune_student(cfg, spec, cond, data, student, device=device, teacher_model=teacher_model)
    best_ckpt_path, metadata_path = save_student_training_result(meta, result, spec, cond)

    print(f"\nBest checkpoint: epoch {result.best_epoch}")
    print(f"Saved to: {best_ckpt_path}")
    print(f"Metadata written to: {metadata_path}")
    print("\n[OK] Student training complete.")


def evaluate_student(cfg, spec, cond, device, teacher_model):
    print("\nEvaluating saved student on dev/test ...")
    result = evaluate_saved_student(cfg, spec, cond, device=device, teacher_model=teacher_model)
    save_student_evaluation_result(result)

    print(f"  dev macro-F1  : {result.dev_result.macro_f1:.4f}")
    print(f"  test macro-F1 : {result.test_result.macro_f1:.4f}")
    print(f"  test accuracy : {result.test_result.accuracy:.4f}")
    print(f"  test ECE      : {result.test_result.ECE:.4f}")
    pass_fail = "PASS" if result.test_result.macro_f1 >= 0.33 else "FAIL"
    print(f"  DoD check (test macro-F1 >= 0.33): {pass_fail}")
    print(f"\nrun_metadata.json updated: {result.metadata_path}")
    print("[OK] Student evaluation complete.")


def main() -> None:
    cfg = Config()
    spec = DATASET_TWEETEVAL_SENTIMENT
    args = parse_args()
    cond = condition_from_flags(args.logit, args.hidden, args.attention)

    configure_reproducibility(cfg.seed)
    device = resolve_device(cfg)
    print(f"Device: {device}")
    print(f"Condition: {cond.name}")

    # The teacher is needed for KD training and, when --eval is set, for the
    # teacher-student analysis on every condition (matching 02b_eval_student.py).
    teacher_model = None
    if cond.uses_teacher or args.eval:
        print("Loading teacher checkpoint ...")
        teacher_model = load_teacher(cfg, spec, device)

    train_student(cfg, spec, cond, device, teacher_model if cond.uses_teacher else None)

    if args.eval:
        evaluate_student(cfg, spec, cond, device, teacher_model)


if __name__ == "__main__":
    main()
