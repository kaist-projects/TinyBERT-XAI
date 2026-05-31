"""Factorial-effect estimates for the 2^3 KD ablation."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from tinybert_xai.distill.conditions import all_conditions

FACTORS = ("logit", "hidden", "attention")
_EXPECTED_CONDITIONS = {condition.name for condition in all_conditions()}


def main_effect(df: pd.DataFrame, factor: str, metric: str) -> float:
    """Return mean(metric | factor on) - mean(metric | factor off)."""
    _validate_factor(factor)
    frame = _complete_metric_frame(df, metric)
    return float((_sign(frame[factor]) * frame[metric]).sum() / 4.0)


def interaction_2way(df: pd.DataFrame, fa: str, fb: str, metric: str) -> float:
    """Return the signed two-way interaction estimate for two factors."""
    _validate_factor(fa)
    _validate_factor(fb)
    if fa == fb:
        raise ValueError("two-way interaction requires two distinct factors")
    frame = _complete_metric_frame(df, metric)
    signs = _sign(frame[fa]) * _sign(frame[fb])
    return float((signs * frame[metric]).sum() / 4.0)


def interaction_3way(df: pd.DataFrame, metric: str) -> float:
    """Return the signed three-way Logit x Hidden x Attention estimate."""
    frame = _complete_metric_frame(df, metric)
    signs = _sign(frame["logit"]) * _sign(frame["hidden"]) * _sign(frame["attention"])
    return float((signs * frame[metric]).sum() / 4.0)


def effects_table(df: pd.DataFrame, metric: str = "test_macro_f1") -> pd.DataFrame:
    """Return all main effects and interactions for ``metric``."""
    rows = [
        {
            "effect": factor,
            "kind": "main",
            "metric": metric,
            "estimate": main_effect(df, factor, metric),
        }
        for factor in FACTORS
    ]
    rows.extend(
        {
            "effect": f"{fa} x {fb}",
            "kind": "2-way",
            "metric": metric,
            "estimate": interaction_2way(df, fa, fb, metric),
        }
        for fa, fb in combinations(FACTORS, 2)
    )
    rows.append(
        {
            "effect": "logit x hidden x attention",
            "kind": "3-way",
            "metric": metric,
            "estimate": interaction_3way(df, metric),
        }
    )
    effects = pd.DataFrame(rows)
    effects["abs_estimate"] = effects["estimate"].abs()
    return effects


def _sign(values: pd.Series) -> pd.Series:
    return values.astype(bool).map({True: 1.0, False: -1.0})


def _validate_factor(factor: str) -> None:
    if factor not in FACTORS:
        raise ValueError(f"unknown factor {factor!r}; expected one of {FACTORS}")


def _complete_metric_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    required = {"condition", *FACTORS, metric}
    missing_columns = required - set(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"missing required column(s): {missing}")

    frame = df.loc[df["condition"].isin(_EXPECTED_CONDITIONS), list(required)].copy()
    if len(frame) != len(_EXPECTED_CONDITIONS):
        raise ValueError(f"expected 8 condition rows, found {len(frame)}")
    if set(frame["condition"]) != _EXPECTED_CONDITIONS:
        missing = ", ".join(sorted(_EXPECTED_CONDITIONS - set(frame["condition"])))
        raise ValueError(f"missing condition row(s): {missing}")
    if frame["condition"].duplicated().any():
        duplicated = frame.loc[frame["condition"].duplicated(), "condition"].tolist()
        raise ValueError(f"duplicate condition row(s): {duplicated}")
    if frame[metric].isna().any():
        missing = frame.loc[frame[metric].isna(), "condition"].tolist()
        raise ValueError(f"metric {metric!r} is missing for condition(s): {missing}")

    return frame.sort_values("condition").reset_index(drop=True)
