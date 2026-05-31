"""Roll up every completed factorial sweep into cross-dataset artifacts.

Reads every ``results/students/<dataset>/<condition>/run_metadata.json`` that is
present and writes the iteration-8 cross-dataset assets under
``results/analysis/cross_dataset/``:

- ``figures/cross_task_macro_f1.png`` and ``cross_task_delta.png``: the headline
  datasets x conditions heatmaps (absolute macro-F1 and delta from ``ce_only``).
- ``figures/confusion/<dataset>__<condition>.png``: per-condition confusion
  matrices rendered from the stored counts.
- ``tables/*.csv``: cross-task matrices plus tidy calibration, teacher-student,
  and per-dataset factorial-effect tables.
- ``TABLES.md``: a generated index of the matrices for quick reading.

This stage is metadata-only: no checkpoints or GPU. Run
``scripts/08b_representation_analysis.py`` for the checkpoint-forward artifacts.
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tinybert_xai.analysis.cross_dataset import (  # noqa: E402
    aggregate_calibration,
    aggregate_main_effects,
    aggregate_teacher_student,
    cross_task_matrix,
    delta_matrix,
)
from tinybert_xai.analysis.loaders import load_all_runs  # noqa: E402
from tinybert_xai.analysis.plots import (  # noqa: E402
    plot_confusion_matrix,
    plot_cross_task_heatmap,
)

ANALYSIS_ROOT = pathlib.Path("results") / "analysis" / "cross_dataset"
RESULTS_ROOT = pathlib.Path("results")
PRIMARY_METRIC = "test_macro_f1"


def main() -> None:
    df = load_all_runs()
    if df.empty:
        raise SystemExit("no student runs found under results/students/")

    figures_dir, tables_dir = _prepare_output_dirs()

    matrix = cross_task_matrix(df, PRIMARY_METRIC)
    deltas = delta_matrix(df, PRIMARY_METRIC)
    calibration = aggregate_calibration(df)
    teacher_student = aggregate_teacher_student(df)
    effects = aggregate_main_effects(df, PRIMARY_METRIC)

    written = _write_heatmaps(matrix, deltas, figures_dir)
    written += _write_confusion_figures(df, figures_dir / "confusion")
    _write_tables(
        {
            "cross_task_macro_f1": matrix,
            "cross_task_delta": deltas,
            "calibration": calibration,
            "teacher_student": teacher_student,
            "main_effects": effects,
        },
        tables_dir,
    )
    _write_tables_index(matrix, deltas, effects, ANALYSIS_ROOT / "TABLES.md")

    _print_summary(matrix, deltas, written)


def _prepare_output_dirs() -> tuple[pathlib.Path, pathlib.Path]:
    figures_dir = ANALYSIS_ROOT / "figures"
    tables_dir = ANALYSIS_ROOT / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir, tables_dir


def _write_heatmaps(
    matrix: pd.DataFrame, deltas: pd.DataFrame, figures_dir: pathlib.Path
) -> list[pathlib.Path]:
    written = plot_cross_task_heatmap(
        matrix,
        figures_dir,
        "cross_task_macro_f1",
        "Test macro-F1 by dataset x condition",
        fmt=".3f",
        cmap="viridis",
    )
    written += plot_cross_task_heatmap(
        deltas,
        figures_dir,
        "cross_task_delta",
        "Test macro-F1 delta from ce_only",
        fmt="+.3f",
        center=0.0,
        cmap="RdBu_r",
    )
    return written


def _write_confusion_figures(df: pd.DataFrame, out_dir: pathlib.Path) -> list[pathlib.Path]:
    written: list[pathlib.Path] = []
    for row in df.loc[df["valid"].fillna(False).astype(bool)].itertuples(index=False):
        confusion = _load_confusion(row.path)
        if confusion is None:
            continue
        written += plot_confusion_matrix(
            confusion,
            out_dir,
            f"{row.dataset}__{row.condition}",
            f"{row.dataset} / {row.condition} confusion (test)",
        )
    return written


def _load_confusion(metadata_path: str) -> list[list[int]] | None:
    with open(metadata_path) as f:
        payload = json.load(f)
    matrix = payload.get("metrics", {}).get("test", {}).get("confusion_matrix")
    if not matrix:
        return None
    return matrix


def _write_tables(tables: dict[str, pd.DataFrame], tables_dir: pathlib.Path) -> None:
    for name, table in tables.items():
        path = tables_dir / f"{name}.csv"
        index = table.index.name is not None or name.startswith("cross_task")
        table.to_csv(path, index=index)


def _write_tables_index(
    matrix: pd.DataFrame,
    deltas: pd.DataFrame,
    effects: pd.DataFrame,
    path: pathlib.Path,
) -> None:
    lines = [
        "# Cross-dataset factorial tables",
        "",
        f"Datasets analyzed: {len(matrix.index)} "
        f"({', '.join(f'`{name}`' for name in matrix.index)}).",
        "",
        "## Test macro-F1 (dataset x condition)",
        "",
        _matrix_markdown(matrix.round(4)),
        "",
        "## Test macro-F1 delta from `ce_only`",
        "",
        _matrix_markdown(deltas.round(4)),
        "",
        "## Per-dataset factorial effects on test macro-F1",
        "",
        _effects_markdown(effects),
        "",
    ]
    path.write_text("\n".join(lines) + "\n")


def _effects_markdown(effects: pd.DataFrame) -> str:
    if effects.empty:
        return "_No dataset has a complete 8-condition sweep._"
    pivot = effects.pivot_table(index="dataset", columns="effect", values="estimate")
    return _matrix_markdown(pivot.round(5))


def _matrix_markdown(matrix: pd.DataFrame) -> str:
    """Render an indexed DataFrame as a GitHub markdown table (no tabulate dep)."""
    index_name = matrix.index.name or ""
    header = "| " + " | ".join([index_name, *map(str, matrix.columns)]) + " |"
    separator = "|" + "|".join(["---"] * (len(matrix.columns) + 1)) + "|"
    rows = []
    for label, series in matrix.iterrows():
        cells = [str(label)] + ["" if pd.isna(v) else f"{v}" for v in series]
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def _print_summary(
    matrix: pd.DataFrame, deltas: pd.DataFrame, written: list[pathlib.Path]
) -> None:
    print("Cross-dataset analysis")
    print(f"  datasets        : {len(matrix.index)} ({', '.join(matrix.index)})")
    print(f"  conditions      : {len(matrix.columns)}")
    print(f"  figures written : {len(written)}")
    best = matrix.idxmax(axis=1)
    print("\nBest condition per dataset (test macro-F1):")
    for dataset, condition in best.items():
        value = matrix.loc[dataset, condition]
        delta = deltas.loc[dataset, condition] if dataset in deltas.index else float("nan")
        print(f"  {dataset:24s} {condition:18s} {value:.4f} (delta {delta:+.4f})")


if __name__ == "__main__":
    main()
