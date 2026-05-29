import pandas as pd
import pytest

from tinybert_xai.analysis.factorial import (
    effects_table,
    interaction_2way,
    interaction_3way,
    main_effect,
)
from tinybert_xai.conditions import ALL_CONDITIONS


def _frame(values):
    rows = []
    for condition in ALL_CONDITIONS:
        rows.append(
            {
                "condition": condition.name,
                "logit": condition.logit,
                "hidden": condition.hidden,
                "attention": condition.attention,
                "metric": values[condition.name],
            }
        )
    return pd.DataFrame(rows)


def test_factorial_effects_match_hand_computed_values():
    df = _frame(
        {
            "ce_only": 40.5,
            "kd_logit": 43.5,
            "kd_hidden": 44.5,
            "kd_attn": 45.5,
            "kd_logit_hidden": 49.5,
            "kd_logit_attn": 52.5,
            "kd_hidden_attn": 55.5,
            "kd_full": 68.5,
        }
    )

    assert main_effect(df, "logit", "metric") == pytest.approx(7.0)
    assert main_effect(df, "hidden", "metric") == pytest.approx(9.0)
    assert main_effect(df, "attention", "metric") == pytest.approx(11.0)
    assert interaction_2way(df, "logit", "hidden", "metric") == pytest.approx(2.0)
    assert interaction_2way(df, "logit", "attention", "metric") == pytest.approx(3.0)
    assert interaction_2way(df, "hidden", "attention", "metric") == pytest.approx(4.0)
    assert interaction_3way(df, "metric") == pytest.approx(1.0)

    table = effects_table(df, "metric")
    assert list(table["kind"]) == ["main", "main", "main", "2-way", "2-way", "2-way", "3-way"]


def test_degenerate_equal_cells_have_zero_effects():
    df = _frame({condition.name: 0.5 for condition in ALL_CONDITIONS})
    table = effects_table(df, "metric")

    assert table["estimate"].abs().max() == pytest.approx(0.0)
