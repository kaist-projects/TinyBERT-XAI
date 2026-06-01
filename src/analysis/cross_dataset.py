"""Cross-dataset roll-ups over every completed student factorial sweep.

These functions consume the tidy frame produced by
:func:`src.analysis.loaders.load_all_runs` (one row per
``(dataset, condition)``) and reshape it into the per-(dataset, condition)
tables and dataset x condition matrices the iteration-8 artifacts need. They are
pure pandas; no checkpoints or GPU are touched here.
"""

from __future__ import annotations

import pandas as pd

from src.analysis.factorial import effects_table
from src.distill.conditions import all_conditions

#: Baseline condition that every ``delta`` is measured against.
DELTA_BASELINE = "ce_only"

#: Canonical condition order (matches the per-dataset figures).
CONDITION_ORDER = [condition.name for condition in all_conditions()]

#: Dataset display order, grouped by task family. Only datasets that are
#: actually present in the frame are kept, so partial sweeps render cleanly.
DATASET_ORDER = [
    "davidson",
    "dynahate",
    "hateval",
    "anli",
    "fever",
    "imdb",
    "tweet_eval-sentiment",
    "vardial",
]

CALIBRATION_COLUMNS = ["test_ece", "test_nll", "test_brier"]
TEACHER_STUDENT_COLUMNS = [
    "top1_agreement",
    "error_copying",
    "teacher_student_kl",
    "teacher_correct_student_wrong",
    "teacher_wrong_student_correct",
]


def cross_task_matrix(df: pd.DataFrame, metric: str = "test_macro_f1") -> pd.DataFrame:
    """Pivot valid student runs into a datasets x conditions matrix of ``metric``."""
    frame = _valid_runs(df)
    matrix = frame.pivot_table(index="dataset", columns="condition", values=metric, aggfunc="first")
    return _order_axes(matrix)


def delta_matrix(df: pd.DataFrame, metric: str = "test_macro_f1") -> pd.DataFrame:
    """Return ``cross_task_matrix`` minus each dataset's ``ce_only`` value.

    Datasets without a ``ce_only`` baseline are dropped, since their deltas are
    undefined.
    """
    matrix = cross_task_matrix(df, metric)
    if DELTA_BASELINE not in matrix.columns:
        raise ValueError(f"no {DELTA_BASELINE!r} column to compute deltas against")
    has_baseline = matrix[DELTA_BASELINE].notna()
    matrix = matrix.loc[has_baseline]
    return matrix.sub(matrix[DELTA_BASELINE], axis=0)


def aggregate_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy per-(dataset, condition) calibration table (ECE / NLL / Brier)."""
    return _tidy_table(df, CALIBRATION_COLUMNS)


def aggregate_teacher_student(df: pd.DataFrame) -> pd.DataFrame:
    """Tidy per-(dataset, condition) teacher-student agreement table."""
    return _tidy_table(df, TEACHER_STUDENT_COLUMNS)


def aggregate_main_effects(df: pd.DataFrame, metric: str = "test_macro_f1") -> pd.DataFrame:
    """Stack per-dataset factorial effects for every dataset with a full sweep.

    Datasets that are missing conditions (and so cannot form the 2^3 design) are
    skipped rather than raising, so a partial results tree still produces a table.
    """
    frames = []
    for dataset in _present_datasets(df):
        sub = df.loc[df["dataset"] == dataset]
        try:
            effects = effects_table(sub, metric)
        except ValueError:
            continue
        effects.insert(0, "dataset", dataset)
        frames.append(effects)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _tidy_table(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    frame = _valid_runs(df)
    columns = ["dataset", "condition", *value_columns]
    table = frame.loc[:, [c for c in columns if c in frame.columns]].copy()
    table["dataset"] = pd.Categorical(table["dataset"], _present_datasets(df), ordered=True)
    table["condition"] = pd.Categorical(table["condition"], CONDITION_ORDER, ordered=True)
    return table.sort_values(["dataset", "condition"]).reset_index(drop=True)


def _valid_runs(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[df["valid"].fillna(False).astype(bool)].copy()


def _present_datasets(df: pd.DataFrame) -> list[str]:
    present = set(df["dataset"]) if not df.empty else set()
    ordered = [name for name in DATASET_ORDER if name in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def _order_axes(matrix: pd.DataFrame) -> pd.DataFrame:
    rows = [name for name in DATASET_ORDER if name in matrix.index]
    rows += [name for name in matrix.index if name not in rows]
    # Always expose the full condition set so missing runs read as NaN cells.
    columns = list(CONDITION_ORDER) + [name for name in matrix.columns if name not in CONDITION_ORDER]
    return matrix.reindex(index=rows, columns=columns)
