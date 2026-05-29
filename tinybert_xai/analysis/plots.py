"""Static matplotlib/seaborn figures for factorial analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tinybert_xai.conditions import ALL_CONDITIONS

FIGURE_DPI = 180
CONDITION_ORDER = [condition.name for condition in ALL_CONDITIONS]


def write_all_figures(df: pd.DataFrame, teacher: pd.Series, effects: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write all iteration-6 figures as PNG and SVG."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    written.extend(plot_condition_bars(df, teacher, out_dir))
    written.extend(plot_main_effects(effects, out_dir))
    written.extend(plot_loss_magnitudes(df, out_dir))
    written.extend(plot_calibration(df, out_dir))
    return written


def plot_condition_bars(df: pd.DataFrame, teacher: pd.Series, out_dir: Path) -> list[Path]:
    frame = _ordered(df)
    ce = frame.loc[frame["condition"] == "ce_only", "test_macro_f1"].iloc[0]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=frame, x="condition", y="test_macro_f1", order=CONDITION_ORDER, ax=ax)
    ax.axhline(float(teacher["test_macro_f1"]), color="#4c566a", linestyle="--", linewidth=1.4)
    ax.text(
        len(CONDITION_ORDER) - 0.6,
        float(teacher["test_macro_f1"]) + 0.001,
        "teacher",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4c566a",
    )
    for index, row in enumerate(frame.itertuples(index=False)):
        delta = row.test_macro_f1 - ce
        ax.text(index, row.test_macro_f1 + 0.001, f"{delta:+.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("TweetEval-sentiment test macro-F1 by condition")
    ax.set_xlabel("")
    ax.set_ylabel("Test macro-F1")
    ax.set_ylim(0.62, max(float(teacher["test_macro_f1"]), frame["test_macro_f1"].max()) + 0.02)
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
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


def plot_calibration(df: pd.DataFrame, out_dir: Path) -> list[Path]:
    frame = _ordered(df)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    sns.barplot(data=frame, x="condition", y="test_ece", order=CONDITION_ORDER, ax=ax)
    ax.set_title("TweetEval-sentiment test ECE by condition")
    ax.set_xlabel("")
    ax.set_ylabel("Expected calibration error")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    return _save(fig, out_dir / "calibration")


def _ordered(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["condition"] = pd.Categorical(frame["condition"], CONDITION_ORDER, ordered=True)
    return frame.sort_values("condition")


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".svg")]
    for path in paths:
        fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    return paths
