"""Static matplotlib/seaborn figures for factorial analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tinybert_xai.conditions import all_conditions

FIGURE_DPI = 180
CONDITION_ORDER = [condition.name for condition in all_conditions()]


def write_all_figures(
    df: pd.DataFrame, teacher: pd.Series, effects: pd.DataFrame, out_dir: Path, dataset: str
) -> list[Path]:
    """Write all iteration-6 figures as PNG files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    written.extend(plot_condition_bars(df, teacher, out_dir, dataset))
    written.extend(plot_main_effects(effects, out_dir))
    written.extend(plot_loss_magnitudes(df, out_dir))
    written.extend(plot_calibration(df, out_dir, dataset))
    return written


def plot_condition_bars(df: pd.DataFrame, teacher: pd.Series, out_dir: Path, dataset: str) -> list[Path]:
    frame = _ordered(df)
    ce = frame.loc[frame["condition"] == "ce_only", "test_macro_f1"].iloc[0]
    teacher_f1 = float(teacher["test_macro_f1"])
    ymin, ymax = _metric_ylim(frame["test_macro_f1"], teacher_f1)
    yspan = ymax - ymin
    label_offset = max(yspan * 0.03, 0.001)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        range(len(frame)),
        frame["test_macro_f1"] - ymin,
        bottom=ymin,
        color="#3274a1",
        width=0.8,
    )
    ax.axhline(teacher_f1, color="#4c566a", linestyle="--", linewidth=1.4)
    ax.text(
        len(CONDITION_ORDER) - 0.6,
        min(teacher_f1 + label_offset, ymax - label_offset),
        "teacher",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4c566a",
    )
    for index, row in enumerate(frame.itertuples(index=False)):
        delta = row.test_macro_f1 - ce
        ax.text(
            index,
            min(row.test_macro_f1 + label_offset, ymax - label_offset),
            f"{delta:+.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_title(f"{dataset} test macro-F1 by condition")
    ax.set_xlabel("")
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(range(len(frame)))
    ax.set_xticklabels(frame["condition"])
    ax.tick_params(axis="x", rotation=35)
    fig.subplots_adjust(bottom=0.28, top=0.88)
    return _save(fig, out_dir / "condition_bars")


def plot_main_effects(effects: pd.DataFrame, out_dir: Path) -> list[Path]:
    frame = effects.copy()
    frame["direction"] = frame["estimate"].map(lambda value: "positive" if value >= 0 else "negative")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.barplot(
        data=frame,
        x="effect",
        y="estimate",
        hue="direction",
        dodge=False,
        palette={"positive": "#4c78a8", "negative": "#d65f5f"},
        ax=ax,
    )
    ax.axhline(0, color="#2e3440", linewidth=0.9)
    ax.set_title("Factorial effects on test macro-F1")
    ax.set_xlabel("")
    ax.set_ylabel("Effect estimate")
    ax.tick_params(axis="x", rotation=35)
    ax.get_legend().remove()
    fig.tight_layout()
    return _save(fig, out_dir / "main_effects")


def plot_loss_magnitudes(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    value_vars = ["loss_ce", "loss_logit", "loss_hidden", "loss_attention"]
    frame = _ordered(df).melt(
        id_vars=["condition"],
        value_vars=value_vars,
        var_name="loss",
        value_name="magnitude",
    )
    frame = frame.dropna(subset=["magnitude"])
    frame["loss"] = frame["loss"].str.removeprefix("loss_")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=frame, x="condition", y="magnitude", hue="loss", ax=ax)
    ax.set_title("Final-epoch loss magnitudes")
    ax.set_xlabel("")
    ax.set_ylabel("Loss magnitude")
    ax.tick_params(axis="x", rotation=35)
    ax.legend(title="")
    fig.tight_layout()
    return _save(fig, out_dir / "loss_magnitudes")


def plot_calibration(df: pd.DataFrame, out_dir: Path, dataset: str) -> list[Path]:
    frame = _ordered(df)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=frame, x="condition", y="test_ece", order=CONDITION_ORDER, ax=ax)
    ax.set_title(f"{dataset} test ECE by condition")
    ax.set_xlabel("")
    ax.set_ylabel("Expected calibration error")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return _save(fig, out_dir / "calibration")


def plot_cross_task_heatmap(
    matrix: pd.DataFrame,
    out_dir: Path,
    stem: str,
    title: str,
    *,
    fmt: str = ".3f",
    center: float | None = None,
    cmap: str = "viridis",
) -> list[Path]:
    """Render a datasets x conditions matrix as an annotated heatmap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    height = max(3.0, 0.6 * len(matrix.index) + 1.5)
    width = max(8.0, 0.9 * len(matrix.columns) + 2.0)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        center=center,
        linewidths=0.5,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    return _save(fig, out_dir / stem)


def plot_confusion_matrix(
    matrix: list[list[int]],
    out_dir: Path,
    stem: str,
    title: str,
    *,
    labels: list[str] | None = None,
) -> list[Path]:
    """Render one confusion matrix (rows=true, cols=predicted) as a heatmap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(matrix)
    if labels is not None and len(labels) == frame.shape[0]:
        frame.index = labels
        frame.columns = labels
    size = max(3.5, 0.7 * frame.shape[0] + 2.0)
    fig, ax = plt.subplots(figsize=(size, size))
    sns.heatmap(frame, annot=True, fmt="d", cmap="Blues", cbar=False, square=True, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    return _save(fig, out_dir / stem)


def plot_attention_pair(
    teacher_map,
    student_map,
    tokens: list[str],
    out_dir: Path,
    stem: str,
    title: str,
) -> list[Path]:
    """Render teacher vs student head-averaged attention side by side."""
    out_dir.mkdir(parents=True, exist_ok=True)
    size = max(5.0, 0.32 * len(tokens) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(2 * size, size))
    for ax, attn, name in zip(axes, [teacher_map, student_map], ["teacher (L12)", "student (L4)"]):
        sns.heatmap(
            attn,
            cmap="magma",
            square=True,
            cbar_kws={"shrink": 0.7},
            xticklabels=tokens,
            yticklabels=tokens,
            ax=ax,
        )
        ax.set_title(name)
        ax.tick_params(axis="x", rotation=90, labelsize=6)
        ax.tick_params(axis="y", rotation=0, labelsize=6)
    fig.suptitle(title)
    fig.tight_layout()
    return _save(fig, out_dir / stem)


def plot_efficiency(efficiency: dict, out_dir: Path, stem: str = "efficiency") -> list[Path]:
    """Render teacher-vs-student parameter count and forward latency bars."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (ax_params, ax_latency) = plt.subplots(1, 2, figsize=(10, 4.5))
    names = ["teacher", "student"]
    params_millions = [
        efficiency["teacher_parameters"] / 1e6,
        efficiency["student_parameters"] / 1e6,
    ]
    latencies = [efficiency["teacher_latency_ms"], efficiency["student_latency_ms"]]

    ax_params.bar(names, params_millions, color=["#4c566a", "#3274a1"])
    ax_params.set_title("Parameters (millions)")
    ax_params.bar_label(ax_params.containers[0], fmt="%.1f")

    ax_latency.bar(names, latencies, color=["#4c566a", "#3274a1"])
    ax_latency.set_title("Forward latency (ms / batch)")
    ax_latency.bar_label(ax_latency.containers[0], fmt="%.2f")

    fig.suptitle(
        f"Student is {efficiency['parameter_ratio']:.1f}x smaller, "
        f"{efficiency['speedup']:.1f}x faster"
    )
    fig.tight_layout()
    return _save(fig, out_dir / stem)


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["condition"] = pd.Categorical(frame["condition"], CONDITION_ORDER, ordered=True)
    return frame.sort_values("condition")


def _metric_ylim(values: pd.Series, reference: float) -> tuple[float, float]:
    lower = float(min(values.min(), reference))
    upper = float(max(values.max(), reference))
    span = max(upper - lower, 0.01)
    padding = max(span * 0.15, 0.01)
    return max(0.0, lower - padding), min(1.0, upper + padding)


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    path = stem.with_suffix(".png")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return [path]
