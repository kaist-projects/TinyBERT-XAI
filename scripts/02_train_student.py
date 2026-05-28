"""scripts/02_train_student.py - TinyBERT student training.

Trains huawei-noah/TinyBERT_General_4L_312D on TweetEval-sentiment for the
selected condition, writes a best checkpoint, and records schema-v2 metadata.

Usage
-----
    conda activate tinybert-xai
    # from repo root
    python scripts/02_train_student.py kd_attn

Output
------
    checkpoints/students/tweet_eval-sentiment/<condition>/
        epoch_0.pt, epoch_1.pt, ..., best.pt
    results/students/tweet_eval-sentiment/<condition>/
        run_metadata.json
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tinybert_xai import (  # noqa: E402
    CONDITIONS_BY_NAME,
    DATASET_TWEETEVAL_SENTIMENT,
    Config,
    configure_reproducibility,
    fine_tune_student,
    load_classifier,
    load_state_dict,
    load_student_data,
    prepare_student_model,
    resolve_device,
    save_student_training_result,
    start_student_metadata,
    teacher_dir,
)


def main() -> None:
    cfg = Config()
    spec = DATASET_TWEETEVAL_SENTIMENT
    cond = CONDITIONS_BY_NAME[sys.argv[1] if len(sys.argv) > 1 else "kd_attn"]

    configure_reproducibility(cfg.seed)
    device = resolve_device(cfg)
    print(f"Device: {device}")
    print(f"Condition: {cond.name}")

    meta = start_student_metadata(cfg, spec, cond, device)

    print("Loading datasets ...")
    data = load_student_data(cfg, spec)
    meta.dataset["splits"] = {"train": data.train_size, "validation": data.dev_size}
    print(f"  train={data.train_size}  dev={data.dev_size}")

    student = prepare_student_model(cfg, spec, cond, device)
    meta.model["parameter_count"] = student.parameter_count
    if student.projection_parameter_count is not None:
        meta.model["projection_parameter_count"] = student.projection_parameter_count
    teacher_model = None
    if cond.uses_teacher:
        print("Loading teacher checkpoint ...")
        teacher_model = load_classifier(cfg.teacher_checkpoint, spec.num_labels, device)
        load_state_dict(teacher_model, teacher_dir(spec.name) / "best.pt", device)
        teacher_model.eval()

    result = fine_tune_student(cfg, spec, cond, data, student, device=device, teacher_model=teacher_model)
    best_ckpt_path, metadata_path = save_student_training_result(meta, result, spec, cond)

    print(f"\nBest checkpoint: epoch {result.best_epoch}")
    print(f"Saved to: {best_ckpt_path}")
    print(f"Metadata written to: {metadata_path}")

    print("\n[OK] Student training complete.")


if __name__ == "__main__":
    main()
