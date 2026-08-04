import numpy as np
import pandas as pd

from scripts.build_tourism_scaled_mse_tables import (
    MODELS,
    add_training_zscore_errors,
    calculate_accuracy,
    calculate_pairwise_dm_tests,
    render_accuracy_latex,
    render_dm_latex,
)


def test_training_zscore_evaluation_divides_error_by_origin_training_scale() -> None:
    dates = pd.date_range("2020-01-01", periods=6, freq="MS")
    tourism = pd.DataFrame({
        "date": dates,
        "s1": [1.0, 2.0, 4.0, 8.0, 16.0, 32.0],
        "s2": [3.0, 6.0, 9.0, 12.0, 15.0, 18.0],
    })
    rows = []
    for target_date in dates[-2:]:
        for series_id in ("s1", "s2"):
            actual = float(
                tourism.loc[tourism["date"].eq(target_date), series_id].iloc[0]
            )
            row = {
                "date": target_date,
                "series_id": series_id,
                "actual": actual,
            }
            row.update({model: actual + 2.0 for model in MODELS})
            rows.append(row)
    panel = pd.DataFrame(rows)

    cells = add_training_zscore_errors(panel, tourism)

    observed = cells[
        cells["model"].eq(MODELS[0])
        & cells["date"].eq(dates[-2])
        & cells["series_id"].eq("s1")
    ].iloc[0]
    expected_scale = np.std([1.0, 2.0, 4.0, 8.0], ddof=0)
    assert np.isclose(observed["training_scale"], expected_scale)
    assert np.isclose(observed["scaled_error"], 2.0 / expected_scale)
    assert observed["training_end"] == dates[-3]


def _synthetic_cells() -> pd.DataFrame:
    dates = pd.date_range("2016-01-01", periods=12, freq="MS")
    rows = []
    for date_index, date in enumerate(dates):
        for series_index in range(76):
            for model_index, model in enumerate(MODELS):
                error = (
                    0.2
                    + 0.05 * model_index
                    + 0.01 * np.sin(date_index + series_index)
                )
                rows.append({
                    "date": date,
                    "series_id": f"s{series_index:02d}",
                    "model": model,
                    "scaled_error": error,
                    "scaled_squared_error": error**2,
                    "scaled_absolute_error": abs(error),
                })
    return pd.DataFrame(rows)


def test_scaled_accuracy_and_dm_use_mse_order_and_all_pairs() -> None:
    accuracy, origins, audit = calculate_accuracy(_synthetic_cells())
    order = tuple(accuracy["model"])
    results, losses = calculate_pairwise_dm_tests(origins, order)

    assert len(accuracy) == 11
    assert len(origins) == 12 * 11
    assert len(audit) == 12 * 11
    assert len(results) == 55
    assert losses.shape == (12, 11)
    assert np.all(np.diff(accuracy["mse"]) >= 0.0)
    np.testing.assert_allclose(accuracy["rmse"] ** 2, accuracy["mse"])
    assert results["forecast_horizon"].eq(1).all()
    assert results["hac_lag"].eq(0).all()


def test_scaled_table_renderers_label_mse_and_scaled_dm_loss() -> None:
    accuracy, origins, _ = calculate_accuracy(_synthetic_cells())
    order = tuple(accuracy["model"])
    results, _ = calculate_pairwise_dm_tests(origins, order)

    accuracy_tex = render_accuracy_latex(
        accuracy,
        numbered=False,
        horizon_months=12,
    )
    dm_tex = render_dm_latex(results, order, horizon_months=12)

    assert "Method & MSE" in accuracy_tex
    assert "RankScore" not in accuracy_tex
    assert "training-only standard deviation" in accuracy_tex
    assert "training-z-score-scaled" in dm_tex
    assert "lower-MSE column method" in dm_tex
