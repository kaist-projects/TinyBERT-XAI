"""scripts/01_train_teacher.py — Teacher fine-tuning on TweetEval-sentiment.

Iteration 1 deliverable.  Trains bert-base-uncased on TweetEval-sentiment,
applies early stopping on dev macro-F1 (patience=2), saves one checkpoint per
epoch plus best.pt, and writes results/teachers/tweet_eval-sentiment/run_metadata.json.

Usage
-----
    conda activate tinybert-xai
    # from repo root
    python scripts/01_train_teacher.py --dataset imdb          # default: tweet_eval-sentiment
    python scripts/01_train_teacher.py --dataset imdb --eval   # train, then evaluate

Output
------
    checkpoints/teachers/<dataset>/
        epoch_0.pt, epoch_1.pt, ..., best.pt
    results/teachers/<dataset>/
        run_metadata.json

With --eval, the run_metadata.json is patched with dev/test metrics in the same
pass (equivalent to running scripts/01b_eval_teacher.py afterwards).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from _dataset_cli import add_dataset_flag, dataset_from_args  # noqa: E402

from tinybert_xai import (  # noqa: E402
    Config,
    configure_reproducibility,
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
    add_dataset_flag(parser)
    parser.add_argument("--eval", action="store_true", help="evaluate on dev/test after training completes")
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
    cfg = Config()
    args = parse_args()
    spec = dataset_from_args(args)

    configure_reproducibility(cfg.seed)
    device = resolve_device(cfg)
    print(f"Device: {device}")

    train_teacher(cfg, spec, device)

    if args.eval:
        evaluate_teacher(cfg, spec, device)


if __name__ == "__main__":
    main()
