import pandas as pd

from tinybert_xai.distill.conditions import all_conditions
from tinybert_xai.analysis.plots import _metric_ylim, plot_condition_bars


def test_metric_ylim_includes_low_scoring_dataset_values():
    ymin, ymax = _metric_ylim(pd.Series([0.41952, 0.43227]), 0.51215)

    assert ymin < 0.41952
    assert ymax > 0.51215
    assert ymin < ymax


def test_metric_ylim_stays_inside_metric_bounds():
    ymin, ymax = _metric_ylim(pd.Series([0.83844, 0.85189]), 0.88851)

    assert 0.0 <= ymin < 0.83844
    assert 0.88851 < ymax <= 1.0


def test_condition_bars_handles_hateval_like_narrow_low_range(tmp_path):
    values = [0.5436, 0.5515, 0.5373, 0.5555, 0.5493, 0.5316, 0.5355, 0.5479]
    df = pd.DataFrame(
        {
            "condition": [condition.name for condition in all_conditions()],
            "test_macro_f1": values,
        }
    )
    teacher = pd.Series({"test_macro_f1": 0.5788})

    paths = plot_condition_bars(df, teacher, tmp_path, "hateval")

    assert paths == [tmp_path / "condition_bars.png"]
    assert paths[0].stat().st_size > 0
