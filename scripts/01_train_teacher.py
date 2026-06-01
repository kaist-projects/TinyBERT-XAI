"""scripts/01_train_teacher.py — Teacher fine-tuning on TweetEval-sentiment.

Trains bert-base-uncased on the chosen dataset, applies early stopping on dev
macro-F1 (patience=2), saves one checkpoint per epoch plus best.pt, then
evaluates on dev/test and writes
results/metadata/<dataset>/teacher/run_metadata.json. Evaluation always runs at
the end of training.

Usage
-----
    conda activate tinybert-xai
    # from repo root
    python scripts/01_train_teacher.py --dataset imdb          # default: tweet_eval-sentiment

Output
------
    results/checkpoints/<dataset>/teacher/
        epoch_0.pt, epoch_1.pt, ..., best.pt
    results/metadata/<dataset>/teacher/
        run_metadata.json   (with dev/test metrics)
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _config_cli import add_config_flag, add_dataset_override, resolve_run_spec  # noqa: E402

from src import (  # noqa: E402
    configure_reproducibility,
    dataset_by_name,
    evaluate_saved_teacher,
    fine_tune_teacher,
    format_teacher_eval_summary,
    load_teacher_data,
    prepare_teacher_model,
    resolve_device,
    save_teacher_evaluation_result,
    save_teacher_training_result,
    start_teacher_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the teacher (bert-base-uncased) on one dataset.")
    add_config_flag(parser)
    add_dataset_override(parser)
    return parser.parse_args()


def train_teacher(cfg, spec, device):
    meta = start_teacher_metadata(cfg, spec, device)

    print("Loading datasets …")
    data = load_teacher_data(cfg, spec)
    meta.dataset["splits"] = {"train": data.train_size, "validation": data.dev_size}
    print(f"  train={data.train_size}  dev={data.dev_size}")

    teacher = prepare_teacher_model(cfg, spec, device)
    meta.model["parameter_count"] = teacher.parameter_count
    result = fine_tune_teacher(cfg, spec, data, teacher, device=device)
    best_ckpt_path, metadata_path = save_teacher_training_result(meta, result, spec)

    print(f"\nBest checkpoint: epoch {result.best_epoch}")
    print(f"Saved to: {best_ckpt_path}")
    print(f"Metadata written to: {metadata_path}")
    print("\n[OK] Teacher training complete.")


def evaluate_teacher(cfg, spec, device):
    print("\nEvaluating saved teacher on dev/test ...")
    result = evaluate_saved_teacher(cfg, spec, device=device)
    save_teacher_evaluation_result(result)

    print(format_teacher_eval_summary(result))
    print(f"\nrun_metadata.json updated: {result.metadata_path}")
    print("[OK] Teacher evaluation complete.")


def main() -> None:
    run = resolve_run_spec(parse_args())
    cfg = run.config
    spec = dataset_by_name(run.dataset)

    configure_reproducibility(cfg.seed)
    device = resolve_device(cfg)
    print(f"Device: {device}")

    train_teacher(cfg, spec, device)
    evaluate_teacher(cfg, spec, device)


if __name__ == "__main__":
    main()
