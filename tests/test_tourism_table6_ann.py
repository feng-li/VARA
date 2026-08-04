import numpy as np
import pandas as pd

from scripts.build_tourism_table6_ann import (
    ANN_BENCHMARK_MODELS,
    ANN_VARA_MODEL_NAMES,
    NON_AUTOETS_MODELS,
    build_table6_ann,
)


def _summary(models: list[str], start: float) -> pd.DataFrame:
    values = start + np.arange(len(models), dtype=float)
    return pd.DataFrame({
        "model": models,
        "rmse": values,
        "mae": values,
        "mase": values,
        "rmsse": values,
        "oos_r2": -values,
        "crps": values,
        "bias": values,
    })


def test_table6_ann_uses_exactly_eleven_methods_and_recomputes_rankscore() -> None:
    standard = _summary(list(NON_AUTOETS_MODELS), 3.0)
    ann_benchmark = _summary(list(ANN_BENCHMARK_MODELS), 9.0)
    ann_vara = _summary(list(ANN_VARA_MODEL_NAMES), 1.0)

    table = build_table6_ann(standard, ann_benchmark, ann_vara)

    assert len(table) == 11
    assert table["model"].nunique() == 11
    assert table.iloc[0]["model"] == "vara_auto_ets_lambda"
    assert table.iloc[0]["rank_score"] == 100.0
    assert table.iloc[-1]["rank_score"] == 0.0
    ann_methods = table[table["autoets_family"] == "ANN"]
    assert set(ann_methods["model"]) == {
        *ANN_BENCHMARK_MODELS,
        *ANN_VARA_MODEL_NAMES.values(),
    }
