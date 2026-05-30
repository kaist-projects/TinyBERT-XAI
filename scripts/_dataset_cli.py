"""Shared --dataset CLI glue for the teacher/student scripts.

Imported by scripts/01_train_teacher.py, 02_train_student.py, and
02b_eval_student.py, all of which insert the repo root onto sys.path before
importing this module, so the ``from tinybert_xai import ...`` below resolves.
"""

from __future__ import annotations

import argparse

from tinybert_xai import ALL_DATASETS, DatasetSpec, dataset_by_name


def add_dataset_flag(parser: argparse.ArgumentParser, default: str = "tweet_eval-sentiment") -> None:
    """Add a --dataset flag selecting one registered DatasetSpec by name."""
    choices = [spec.name for spec in ALL_DATASETS]
    parser.add_argument(
        "--dataset",
        default=default,
        choices=choices,
        help=f"dataset to run (default: {default})",
    )


def dataset_from_args(args: argparse.Namespace) -> DatasetSpec:
    """Resolve the parsed --dataset name to its DatasetSpec."""
    return dataset_by_name(args.dataset)
