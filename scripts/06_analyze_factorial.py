"""Analyze the 2^3 student-factorial sweep and write analysis artifacts."""

from __future__ import annotations

import argparse
import math
import pathlib
import sys
from dataclasses import dataclass

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tinybert_xai.analysis.factorial import effects_table  # noqa: E402
from tinybert_xai.analysis.loaders import load_runs, load_teacher  # noqa: E402
from tinybert_xai.analysis.plots import write_all_figures  # noqa: E402
from tinybert_xai.analysis.tables import (  # noqa: E402
    render_factorial_report,
)
from tinybert_xai.conditions import ALL_CONDITIONS  # noqa: E402

ANALYSIS_DIR = pathlib.Path("results") / "analysis"
FIGURES_DIR = ANALYSIS_DIR / "figures"
REPORT_PATH = ANALYSIS_DIR / "factorial_report.md"
METRIC_COLUMNS = [
    "test_macro_f1",
    "test_micro_f1",
    "test_accuracy",
    "test_ece",
    "test_nll",
    "test_brier",
    "dev_macro_f1",
]
LOSS_BY_FACTOR = {
    "ce": "loss_ce",
    "logit": "loss_logit",
    "hidden": "loss_hidden",
    "attention": "loss_attention",
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


def main() -> None:
    args = _parse_args()
    dataset = args.dataset

    df = load_runs(dataset)
    teacher = load_teacher(dataset)

    checks = _validate_inputs(df)
    effects = pd.DataFrame()
    if all(check.passed for check in checks):
        effects = effects_table(df, "test_macro_f1")
        ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
        _remove_stale_artifacts()
        figures = write_all_figures(df, teacher, effects, FIGURES_DIR)
        report = _write_report(df, teacher, effects, checks, figures, dataset)
        checks.append(_check_artifacts(figures, report))
        _write_report(df, teacher, effects, checks, figures, dataset)
    else:
        checks.append(Check("artifacts written", False, "not attempted because input validation failed"))

    _print_report(df, teacher, effects, checks)

    if not all(check.passed for check in checks):
        raise SystemExit(1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="tweet_eval-sentiment")
    return parser.parse_args()


def _write_report(
    df: pd.DataFrame,
    teacher: pd.Series,
    effects: pd.DataFrame,
    checks: list[Check],
    figures: list[pathlib.Path],
    dataset: str,
) -> pathlib.Path:
    REPORT_PATH.write_text(
        render_factorial_report(df, teacher, effects, checks, figures, REPORT_PATH, dataset)
    )
    return REPORT_PATH


def _remove_stale_artifacts() -> None:
    for path in [
        ANALYSIS_DIR / "student_ablation_table.md",
        ANALYSIS_DIR / "main_effects_table.md",
    ]:
        path.unlink(missing_ok=True)
    for path in FIGURES_DIR.glob("*.svg"):
        path.unlink()


def _validate_inputs(df: pd.DataFrame) -> list[Check]:
    return [
        _check_conditions_present(df),
        _check_epochs(df),
        _check_finite_recorded_values(df),
        _check_teacher_forward(df),
        _check_metric_ranges(df),
    ]


def _check_conditions_present(df: pd.DataFrame) -> Check:
    expected = {condition.name for condition in ALL_CONDITIONS}
    present = set(df["condition"])
    frame = df.loc[df["condition"].isin(expected)]
    valid = frame["valid"].fillna(False).astype(bool)
    missing = sorted(expected - present)
    invalid = sorted(frame.loc[~valid, "condition"].tolist())
    passed = not missing and not invalid and len(df) == len(expected)
    detail = "all 8 condition metadata files are present and valid"
    if not passed:
        detail = f"missing={missing or 'none'} invalid={invalid or 'none'}"
    return Check("all 8 conditions present and valid", passed, detail)


def _check_from_failures(name: str, failures: list[str], ok_detail: str) -> Check:
    detail = ", ".join(failures) if failures else ok_detail
    return Check(name, not failures, detail)


def _check_epochs(df: pd.DataFrame) -> Check:
    failures = []
    for row in df.itertuples(index=False):
        early_stopped = bool(row.early_stopped) if pd.notna(row.early_stopped) else False
        expected_epochs = row.num_epochs if pd.notna(row.num_epochs) else 3
        if not early_stopped and (
            not _finite(row.epochs_completed) or row.epochs_completed != expected_epochs
        ):
            failures.append(f"{row.condition}={row.epochs_completed}/{expected_epochs}")
    return _check_from_failures(
        "epochs completed", failures, "all runs completed configured epochs or documented early-stop"
    )


def _check_finite_recorded_values(df: pd.DataFrame) -> Check:
    failures = []
    for row in df.itertuples(index=False):
        for column in METRIC_COLUMNS:
            if not _finite(getattr(row, column)):
                failures.append(f"{row.condition}.{column}")
        for loss_name, column in LOSS_BY_FACTOR.items():
            if _loss_required(row, loss_name) and not _finite(getattr(row, column)):
                failures.append(f"{row.condition}.{column}")
    return _check_from_failures(
        "finite metrics/losses", failures, "all required metrics and active losses are finite"
    )


def _check_teacher_forward(df: pd.DataFrame) -> Check:
    failures = []
    kd_rows = df.loc[df[["logit", "hidden", "attention"]].any(axis=1)]
    for row in kd_rows.itertuples(index=False):
        if not _finite(row.num_labels):
            failures.append(f"{row.condition}.num_labels={row.num_labels}")
            continue
        baseline = 1.0 / float(row.num_labels)
        if not _finite(row.top1_agreement) or row.top1_agreement <= baseline:
            failures.append(f"{row.condition}.top1_agreement={row.top1_agreement}")
    return _check_from_failures(
        "teacher forward sane", failures, "top1_agreement is present and above random for every KD condition"
    )


def _check_metric_ranges(df: pd.DataFrame) -> Check:
    bounded_columns = [
        "test_macro_f1",
        "test_micro_f1",
        "test_accuracy",
        "test_ece",
        "top1_agreement",
    ]
    failures = []
    for row in df.itertuples(index=False):
        for column in bounded_columns:
            value = getattr(row, column)
            if pd.isna(value):
                continue
            if value < 0.0 or value > 1.0:
                failures.append(f"{row.condition}.{column}={value}")
    return _check_from_failures(
        "metric ranges", failures, "F1/accuracy/agreement/ECE values are within [0, 1]"
    )


def _check_artifacts(figures: list[pathlib.Path], report: pathlib.Path) -> Check:
    paths = figures + [report]
    missing = [str(path) for path in paths if not path.exists() or path.stat().st_size == 0]
    unexpected_svg = [str(path) for path in FIGURES_DIR.glob("*.svg")]
    stale_tables = [
        str(path)
        for path in [
            ANALYSIS_DIR / "student_ablation_table.md",
            ANALYSIS_DIR / "main_effects_table.md",
        ]
        if path.exists()
    ]
    passed = len(figures) == 4 and report.exists() and not missing and not unexpected_svg and not stale_tables
    detail = "4 PNG figures and 1 markdown report written"
    if not passed:
        detail = (
            f"figure_files={len(figures)} report={report.exists()} missing={missing or 'none'} "
            f"svg={unexpected_svg or 'none'} stale_tables={stale_tables or 'none'}"
        )
    return Check("artifacts written", passed, detail)


def _print_report(
    df: pd.DataFrame,
    teacher: pd.Series,
    effects: pd.DataFrame,
    checks: list[Check],
) -> None:
    print("\nValidity checks")
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(f"  [{status}] {check.name}: {check.detail}")

    if effects.empty:
        print("\nVerdict: NO-GO")
        return

    spread = df["test_macro_f1"].max() - df["test_macro_f1"].min()
    best = df.loc[df["test_macro_f1"].idxmax()]
    ce = df.loc[df["condition"] == "ce_only", "test_macro_f1"].iloc[0]
    attention_losses = df.loc[df["attention"], "loss_attention"].dropna()

    print("\nSummary")
    print(f"  teacher test macro-F1 : {teacher['test_macro_f1']:.4f}")
    print(f"  best student          : {best['condition']} ({best['test_macro_f1']:.4f})")
    print(f"  ce_only test macro-F1 : {ce:.4f}")
    print(f"  student F1 spread     : {spread:.4f}")
    if not attention_losses.empty:
        print(f"  attention loss mean   : {attention_losses.mean():.5f}")

    print("\nEffects on test_macro_f1")
    for row in effects.itertuples(index=False):
        print(f"  {row.effect:28s} {row.estimate:+.5f}")

    passed = all(check.passed for check in checks)
    verdict = "GO to iter-7" if passed else "NO-GO"
    print(f"\nVerdict: {verdict}")
    if passed:
        print(
            "Rationale: pipeline-validity checks pass; single-seed effect sizes are "
            "informational only. Attention KD is near-inert by final-loss magnitude "
            "and should be fixed before scaling."
        )


def _loss_required(row: tuple, loss_name: str) -> bool:
    if loss_name == "ce":
        return True
    return bool(getattr(row, loss_name))


def _finite(value: object) -> bool:
    if pd.isna(value):
        return False
    return math.isfinite(float(value))


if __name__ == "__main__":
    main()
