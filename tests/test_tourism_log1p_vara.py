from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.evaluate_tourism_76_autoets import _transform_response
from scripts.evaluate_tourism_log1p_vara import (
    AUTOETS_MODEL,
    MODEL,
    TUNED_MODEL,
    add_rolling_tuned_lambda,
    build_log1p_predictions,
    build_paper_comparison,
)


def test_log1p_response_transform_requires_nonnegative_data() -> None:
    values = np.asarray([[0.0, 3.0], [8.0, 15.0]])
    np.testing.assert_allclose(
        _transform_response(values, "log1p"),
        np.log1p(values),
    )
    with pytest.raises(ValueError, match="nonnegative"):
        _transform_response(np.asarray([[-1.0]]), "log1p")


def test_uncalibrated_correction_is_added_in_log_space_before_inversion() -> None:
    date = pd.Timestamp("2016-01-01")
    actual_original = np.asarray([8.0, 24.0])
    baseline_original = np.asarray([3.0, 4.0])
    baseline_log = np.log1p(baseline_original)
    correction = np.log1p(actual_original) - baseline_log
    autoets = pd.DataFrame({
        "date": [date, date],
        "series_id": ["A", "B"],
        "actual": np.log1p(actual_original),
        "actual_original": actual_original,
        "direct_auto_ets": baseline_log,
        "autoets_history_end": [pd.Timestamp("2015-12-01")] * 2,
        "response_transform": ["log1p"] * 2,
    })
    vara = pd.DataFrame({
        "date": [date, date],
        "series_id": ["A", "B"],
        "strategy_id": ["chosen", "chosen"],
        "vara_correction": correction,
        "history_start": [pd.Timestamp("2014-01-01")] * 2,
        "history_end": [pd.Timestamp("2015-12-01")] * 2,
    })

    result = build_log1p_predictions(
        autoets,
        vara,
        strategy_id="chosen",
        final_start="2016-01-01",
        final_end="2016-12-01",
    )

    np.testing.assert_allclose(result[MODEL], actual_original)
    np.testing.assert_allclose(
        result["raw_forecast_log1p"],
        np.log1p(actual_original),
    )
    np.testing.assert_allclose(result[AUTOETS_MODEL], baseline_original)


def test_normalized_autoets_forecast_is_audited_after_training_inverse() -> None:
    date = pd.Timestamp("2016-01-01")
    training_mean = np.asarray([2.0, 3.0])
    training_scale = np.asarray([0.5, 0.25])
    normalized_forecast = np.asarray([1.0, -1.0])
    baseline_log = training_mean + training_scale * normalized_forecast
    autoets = pd.DataFrame({
        "date": [date, date],
        "series_id": ["A", "B"],
        "actual": baseline_log,
        "actual_original": np.expm1(baseline_log),
        "direct_auto_ets": baseline_log,
        "direct_auto_ets_normalized": normalized_forecast,
        "autoets_training_mean": training_mean,
        "autoets_training_variance": np.square(training_scale),
        "autoets_training_scale": training_scale,
        "autoets_clip_nonnegative": [True, True],
        "autoets_history_end": [pd.Timestamp("2015-12-01")] * 2,
        "response_transform": ["log1p"] * 2,
        "response_normalization": ["training_zscore"] * 2,
    })
    vara = pd.DataFrame({
        "date": [date, date],
        "series_id": ["A", "B"],
        "strategy_id": ["chosen", "chosen"],
        "vara_correction": [0.0, 0.0],
        "history_start": [pd.Timestamp("2014-01-01")] * 2,
        "history_end": [pd.Timestamp("2015-12-01")] * 2,
    })

    result = build_log1p_predictions(
        autoets,
        vara,
        strategy_id="chosen",
        final_start="2016-01-01",
        final_end="2016-12-01",
    )

    np.testing.assert_allclose(result[AUTOETS_MODEL], np.expm1(baseline_log))
    np.testing.assert_allclose(
        result["direct_auto_ets"],
        result["autoets_training_mean"]
        + result["autoets_training_scale"]
        * result["direct_auto_ets_normalized"],
    )


def test_rolling_lambda_uses_only_prior_log_autoets_errors() -> None:
    dates = pd.date_range("2015-10-01", periods=4, freq="MS")
    rows = []
    for date_index, date in enumerate(dates):
        for series_index, series_id in enumerate(("A", "B")):
            correction = float(1 + date_index + series_index)
            baseline = np.log1p(10.0 + series_index)
            actual = baseline + 0.25 * correction
            rows.append({
                "date": date,
                "series_id": series_id,
                "actual": actual,
                "direct_auto_ets": baseline,
                "vara_correction": correction,
                MODEL: np.expm1(baseline + correction),
            })
    panel = pd.DataFrame(rows)

    result = add_rolling_tuned_lambda(
        panel,
        final_start="2016-01-01",
        final_end="2016-01-01",
        calibration_window_months=3,
        lambda_cap=1.25,
    )

    np.testing.assert_allclose(result["lambda"], 0.25)
    np.testing.assert_allclose(
        result[TUNED_MODEL],
        np.expm1(
            result["direct_auto_ets"]
            + 0.25 * result["vara_correction"]
        ),
    )
    assert (result["calibration_end"] < result["date"]).all()


def test_paper_rank_score_is_recomputed_after_adding_log1p_method() -> None:
    columns = {
        "mae": [1.0, 3.0],
        "mase": [1.0, 3.0],
        "rmsse": [1.0, 3.0],
    }
    benchmark = pd.DataFrame({
        "model": ["auto_ets", "seasonal_naive"],
        "rmse": [1.0, 3.0],
        **columns,
    })
    paper_vara = pd.DataFrame({
        "model": [
            "reliability_gated_vara_auto_ets_oos",
            "raw_oos_vara",
        ],
        "rmse": [0.5, 2.0],
        "mae": [0.5, 2.0],
        "mase": [0.5, 2.0],
        "rmsse": [0.5, 2.0],
    })
    log1p = pd.DataFrame({
        "model": [MODEL],
        "rmse": [1.5],
        "mae": [1.5],
        "mase": [1.5],
        "rmsse": [1.5],
    })

    ranked = build_paper_comparison(
        log1p,
        benchmark,
        paper_vara,
    ).set_index("model")

    assert ranked.loc["vara_auto_ets_lambda", "rank_score"] == 100.0
    assert ranked.loc["seasonal_naive", "rank_score"] == 0.0
    assert ranked.loc[MODEL, "rank_score"] == 50.0


def test_paper_comparison_prefers_refreshed_application_autoets_metrics() -> None:
    benchmark = pd.DataFrame({
        "model": ["auto_ets", "seasonal_naive"],
        "rmse": [99.0, 3.0],
        "mae": [99.0, 3.0],
        "mase": [99.0, 3.0],
        "rmsse": [99.0, 3.0],
    })
    paper_vara = pd.DataFrame({
        "model": [
            "reliability_gated_vara_auto_ets_oos",
            "raw_oos_vara",
            "direct_auto_ets",
        ],
        "rmse": [0.5, 2.0, 1.0],
        "mae": [0.5, 2.0, 1.0],
        "mase": [0.5, 2.0, 1.0],
        "rmsse": [0.5, 2.0, 1.0],
    })
    log1p = pd.DataFrame({
        "model": [MODEL],
        "rmse": [1.5],
        "mae": [1.5],
        "mase": [1.5],
        "rmsse": [1.5],
    })

    comparison = build_paper_comparison(
        log1p,
        benchmark,
        paper_vara,
    )
    autoets = comparison[comparison["model"] == "auto_ets"]

    assert len(autoets) == 1
    assert autoets.iloc[0]["source"] == "paper_vara"
    assert autoets.iloc[0]["rmse"] == 1.0
