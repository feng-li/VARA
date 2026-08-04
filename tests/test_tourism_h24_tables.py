import numpy as np
import pandas as pd

from scripts.build_tourism_h24_tables import (
    EXPECTED_MODELS,
    calculate_accuracy,
    calculate_pairwise_dm_tests,
    render_dm_latex,
    render_dm_markdown,
)


def _synthetic_origins() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", periods=24, freq="MS")
    rows = []
    for model_index, model in enumerate(sorted(EXPECTED_MODELS)):
        for time_index, date in enumerate(dates):
            rows.append({
                "model": model,
                "date": date,
                "rmse": (
                    10.0
                    + model_index
                    + 0.25 * np.sin(time_index + model_index)
                ),
                "mae": (
                    5.0
                    + model_index
                    + 0.20 * np.cos(time_index + model_index)
                ),
            })
    return pd.DataFrame(rows)


def test_h24_accuracy_and_dm_cover_all_methods_and_pairs() -> None:
    accuracy, squared_losses, audit = calculate_accuracy(
        _synthetic_origins()
    )
    method_order = tuple(accuracy["model"])
    results = calculate_pairwise_dm_tests(squared_losses, method_order)

    assert len(accuracy) == 11
    assert len(squared_losses) == 24
    assert len(audit) == 24 * 11
    assert len(results) == 55
    assert np.all(np.diff(accuracy["rmse"]) >= 0.0)
    assert (results["row_number"] > results["column_number"]).all()
    assert results["p_value_raw_two_sided"].between(0.0, 1.0).all()
    assert results["forecast_horizon"].eq(1).all()
    assert results["hac_lag"].eq(0).all()


def test_h24_dm_renderers_explain_evaluation_window_and_forecast_lead() -> None:
    accuracy, squared_losses, _ = calculate_accuracy(_synthetic_origins())
    method_order = tuple(accuracy["model"])
    results = calculate_pairwise_dm_tests(squared_losses, method_order)

    latex = render_dm_latex(results, method_order)
    markdown = render_dm_markdown(results, method_order)

    assert r"\begin{tabular}{@{}l*{10}{c}@{}}" in latex
    assert "24 months are an evaluation" in latex
    assert "one-month lead" in latex
    assert "all 55 tests" in latex
    assert markdown.splitlines()[0].count("|") == 12
