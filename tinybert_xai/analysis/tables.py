"""Markdown table rendering for analysis artifacts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def render_student_ablation_table(df: pd.DataFrame, teacher: pd.Series, dataset: str) -> str:
    """Render the condition table used in the project notes."""
    ce = df.loc[df["condition"] == "ce_only", "test_macro_f1"].iloc[0]
    rows = _ablation_rows(df, teacher, ce)
    best = {
        "test_macro_f1": max(row["test_macro_f1"] for row in rows),
        "delta": max(row["delta"] for row in rows),
        "test_accuracy": max(row["test_accuracy"] for row in rows),
        "test_ece": min(row["test_ece"] for row in rows),
        "top1_agreement": max(
            row["top1_agreement"] for row in rows if pd.notna(row["top1_agreement"])
        ),
    }

    lines = [
        "# Student Ablation Results",
        "",
        f"Dataset: `{dataset}`",
        "",
        "Source files:",
        f"`results/teachers/{dataset}/run_metadata.json` and",
        f"`results/students/{dataset}/*/run_metadata.json`",
        "",
        "Primary metric: test macro-F1. `Delta` is test macro-F1 relative to `ce_only`.",
        "Rows are ordered by test macro-F1 descending.",
        "Bold marks the best value in each metric column: higher is better for F1,",
        "accuracy, and agreement; lower is better for ECE.",
        "",
        "| Condition | Logit | Hidden | Attention | Test Macro-F1 | Delta | Test Acc. | Test ECE | Top-1 Agree |",
        "|---|:---:|:---:|:---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["test_macro_f1"], reverse=True):
        lines.append(_format_ablation_row(row, best))

    students = [row for row in rows if row["condition"] != "teacher"]
    best_student = max(students, key=lambda row: row["test_macro_f1"])
    lines.extend(
        [
            "",
            (
                "Best student test macro-F1 is "
                f"`{best_student['condition']}` at {best_student['test_macro_f1']:.4f}, "
                f"{best_student['delta']:+.4f} over `ce_only`."
            ),
            f"The teacher reference is higher at {teacher['test_macro_f1']:.4f}.",
            "",
        ]
    )
    return "\n".join(lines)


def render_main_effects_table(effects: pd.DataFrame, metric: str) -> str:
    """Render factorial effects and interactions as markdown."""
    lines = [
        "# Factorial Main Effects",
        "",
        f"Metric: `{metric}`",
        "",
        "Positive estimates mean the factor or interaction increases the metric under",
        "standard +/-1 factorial coding. Magnitudes are informational for this",
        "single-seed pilot.",
        "",
        "| Effect | Kind | Estimate | Absolute |",
        "|---|---:|---:|---:|",
    ]
    for row in effects.itertuples(index=False):
        lines.append(
            f"| `{row.effect}` | {row.kind} | {row.estimate:+.5f} | {row.abs_estimate:.5f} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_factorial_report(
    df: pd.DataFrame,
    teacher: pd.Series,
    effects: pd.DataFrame,
    checks: list,
    figure_paths: list[Path],
    report_path: Path,
    dataset: str,
) -> str:
    """Render the full factorial-analysis markdown report."""
    ce = df.loc[df["condition"] == "ce_only", "test_macro_f1"].iloc[0]
    best = df.loc[df["test_macro_f1"].idxmax()]
    spread = df["test_macro_f1"].max() - df["test_macro_f1"].min()
    attention_losses = df.loc[df["attention"], "loss_attention"].dropna()
    attention_mean = attention_losses.mean() if not attention_losses.empty else pd.NA

    lines = [
        "# Factorial Analysis Report",
        "",
        f"Dataset: `{dataset}`",
        "",
        "## Artifact Summary",
        "",
        f"- Teacher metadata: `results/teachers/{dataset}/run_metadata.json`",
        f"- Student metadata: `results/students/{dataset}/*/run_metadata.json`",
        f"- Report: `{report_path.as_posix()}`",
        f"- Figures: `{report_path.parent.joinpath('figures').as_posix()}/`",
        "",
        "## Validity Checklist",
        "",
        "| Check | Status | Detail |",
        "|---|:---:|---|",
    ]
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        lines.append(f"| {check.name} | {status} | {check.detail} |")

    lines.extend(
        [
            "",
            "## Key Results",
            "",
            f"- Teacher test macro-F1: `{teacher['test_macro_f1']:.4f}`.",
            f"- Best student: `{best['condition']}` with test macro-F1 `{best['test_macro_f1']:.4f}`.",
            f"- CE-only student test macro-F1: `{ce:.4f}`.",
            f"- Student macro-F1 spread across conditions: `{spread:.4f}`.",
        ]
    )
    if pd.notna(attention_mean):
        lines.append(f"- Mean final attention-loss magnitude: `{attention_mean:.5f}`.")

    lines.extend(
        [
            "",
            "The best pilot student is `kd_logit`, but the full student spread is within",
            "single-seed noise. The factorial effects below should therefore be read as",
            "pipeline diagnostics and descriptive pilot statistics, not resolved causal",
            "estimates.",
            "",
            "## Student Ablation Table",
            "",
            _without_title(render_student_ablation_table(df, teacher, dataset)),
            "## Factorial Effects",
            "",
            _without_title(render_main_effects_table(effects, "test_macro_f1")),
            "## Attention-Loss Caveat",
            "",
            "Attention KD used post-softmax attention probabilities in this pilot. Its",
            "final loss magnitude is near-inert compared with CE, logit, and hidden",
            "losses, so the attention factor was only weakly applied. Fix this signal or",
            "explicitly document the caveat before scaling the experiment.",
            "",
            "## Figures",
            "",
        ]
    )
    for figure_path in figure_paths:
        title = _figure_title(figure_path)
        relative = figure_path.relative_to(report_path.parent)
        lines.extend([f"### {title}", "", f"![{title}]({relative.as_posix()})", ""])

    return "\n".join(lines)


def _ablation_rows(df: pd.DataFrame, teacher: pd.Series, ce: float) -> list[dict]:
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            {
                "condition": row.condition,
                "logit": row.logit,
                "hidden": row.hidden,
                "attention": row.attention,
                "test_macro_f1": row.test_macro_f1,
                "delta": row.test_macro_f1 - ce,
                "test_accuracy": row.test_accuracy,
                "test_ece": row.test_ece,
                "top1_agreement": row.top1_agreement,
            }
        )
    rows.append(
        {
            "condition": "teacher",
            "logit": pd.NA,
            "hidden": pd.NA,
            "attention": pd.NA,
            "test_macro_f1": teacher["test_macro_f1"],
            "delta": teacher["test_macro_f1"] - ce,
            "test_accuracy": teacher["test_accuracy"],
            "test_ece": teacher["test_ece"],
            "top1_agreement": pd.NA,
        }
    )
    return rows


def _format_ablation_row(row: dict, best: dict) -> str:
    return (
        f"| `{row['condition']}` | {_flag(row['logit'])} | {_flag(row['hidden'])} | "
        f"{_flag(row['attention'])} | "
        f"{_metric(row['test_macro_f1'], best['test_macro_f1'])} | "
        f"{_metric(row['delta'], best['delta'], signed=True)} | "
        f"{_metric(row['test_accuracy'], best['test_accuracy'])} | "
        f"{_metric(row['test_ece'], best['test_ece'])} | "
        f"{_metric(row['top1_agreement'], best['top1_agreement'])} |"
    )


def _flag(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    return "Y" if bool(value) else ""


def _metric(value: object, best: float, *, signed: bool = False) -> str:
    if pd.isna(value):
        return "N/A"
    number = float(value)
    text = f"{number:+.4f}" if signed else f"{number:.4f}"
    return f"**{text}**" if abs(number - best) < 1e-12 else text


def _without_title(markdown: str) -> str:
    lines = markdown.splitlines()
    if lines and lines[0].startswith("# "):
        lines = lines[2:] if len(lines) > 1 and lines[1] == "" else lines[1:]
    return "\n".join(lines).strip() + "\n"


def _figure_title(path: Path) -> str:
    return path.stem.replace("_", " ").title()
