"""Shared CLI glue for the student train/eval scripts.

Imported by scripts/02_train_student.py and scripts/02b_eval_student.py, both of
which insert the repo root onto sys.path before importing this module, so the
``from src import ...`` below resolves.
"""

from __future__ import annotations

import argparse

from src import ConditionSpec, condition_from_flags


def add_signal_flags(parser: argparse.ArgumentParser) -> None:
    """Add the three per-signal distillation flags to an argument parser."""
    parser.add_argument("--logit", action="store_true", help="enable logit distillation")
    parser.add_argument("--hidden", action="store_true", help="enable hidden-state distillation")
    parser.add_argument("--attention", action="store_true", help="enable attention distillation")


def condition_from_args(args: argparse.Namespace) -> ConditionSpec:
    """Build the distillation condition from parsed signal flags."""
    return condition_from_flags(args.logit, args.hidden, args.attention)


def condition_to_flags(cond: ConditionSpec) -> list[str]:
    """Inverse of the signal flags: the --flags that reproduce this condition."""
    flags = []
    if cond.logit:
        flags.append("--logit")
    if cond.hidden:
        flags.append("--hidden")
    if cond.attention:
        flags.append("--attention")
    return flags
