import pandas as pd
import pytest

from tinybert_xai.analysis.cross_dataset import (
    cross_task_matrix,
    delta_matrix,
)
from tinybert_xai.distill.conditions import all_conditions


def _runs(datasets, base_by_dataset):
    rows = []
    for dataset in datasets:
        for offset, condition in enumerate(all_conditions()):
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition.name,
                    "logit": condition.logit,
                    "hidden": condition.hidden,
                    "attention": condition.attention,
                    "valid": True,
                    "test_macro_f1": base_by_dataset[dataset] + 0.01 * offset,
                }
            )
    return pd.DataFrame(rows)


def test_cross_task_matrix_shape_and_order():
    df = _runs(["fever", "imdb"], {"fever": 0.5, "imdb": 0.8})
    matrix = cross_task_matrix(df)

    assert list(matrix.index) == ["fever", "imdb"]
    assert list(matrix.columns) == [c.name for c in all_conditions()]
    assert matrix.loc["imdb", "ce_only"] == 0.8


def test_cross_task_matrix_drops_invalid_rows():
    df = _runs(["fever"], {"fever": 0.5})
    df.loc[df["condition"] == "kd_full", "valid"] = False
    matrix = cross_task_matrix(df)

    assert pd.isna(matrix.loc["fever", "kd_full"])


def test_delta_matrix_is_zero_at_baseline():
    df = _runs(["fever", "imdb"], {"fever": 0.5, "imdb": 0.8})
    deltas = delta_matrix(df)

    assert deltas.loc["fever", "ce_only"] == 0.0
    assert deltas.loc["imdb", "ce_only"] == 0.0
    # ce_only is index 0; kd_full is the last condition (offset 7) -> +0.07.
    assert deltas.loc["fever", "kd_full"] == pytest.approx(0.07, abs=1e-9)
