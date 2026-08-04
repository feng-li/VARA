from dataclasses import asdict

import numpy as np
import pandas as pd

import scripts.evaluate_tourism_insample_error_vara as module
from scripts.evaluate_tourism_insample_error_vara import (
    MODEL_NAME,
    _fit_origin,
    calculate_metrics,
    calculate_strict_rank_scores,
    fixed_insample_strategy,
)


def test_fit_origin_uses_only_last_24_fitted_residuals(monkeypatch) -> None:
    values = np.arange(30 * 2, dtype=float).reshape(30, 2)
    dates = [
        str(date.date())
        for date in pd.date_range("2020-01-01", periods=30, freq="MS")
    ]
    captured = {}

    def fake_autoets(history, **kwargs):
        captured["autoets_history"] = np.asarray(history).copy()
        captured["autoets_kwargs"] = kwargs
        return {
            "forecast": np.asarray([[10.0, 20.0]]),
            "residuals": np.asarray(history, dtype=float) + 100.0,
        }

    def fake_vara(
        error_history,
        series_ids,
        strategy_dict,
        target_index,
        target_date,
        history_start,
        history_end,
    ):
        captured["error_history"] = np.asarray(error_history).copy()
        captured["target_index"] = target_index
        captured["target_date"] = target_date
        captured["history_start"] = history_start
        captured["history_end"] = history_end
        return {
            "ok": True,
            "correction": np.asarray([1.0, -2.0]),
            "eta_hat": 0.25,
            "eta_on_boundary": False,
            "stationarity_proxy": 0.4,
        }

    monkeypatch.setattr(module, "auto_ets_baseline_panel", fake_autoets)
    monkeypatch.setattr(module, "_error_vara_task", fake_vara)
    result = _fit_origin(
        values,
        dates,
        ["s1", "s2"],
        29,
        asdict(fixed_insample_strategy()),
        season_length=12,
        model_specification="ZZA",
        response_normalization="training_zscore",
    )

    assert captured["autoets_history"].shape == (29, 2)
    np.testing.assert_allclose(
        captured["error_history"],
        values[5:29] + 100.0,
    )
    assert captured["autoets_kwargs"]["response_normalization"] == (
        "training_zscore"
    )
    assert captured["history_start"] == "2020-06-01"
    assert captured["history_end"] == "2022-05-01"
    np.testing.assert_allclose(result["forecast"], [11.0, 18.0])
    assert result["n_vara_train"] == 24


def test_metrics_are_evaluated_on_prediction_values() -> None:
    predictions = pd.DataFrame({
        "date": pd.to_datetime([
            "2016-01-01",
            "2016-01-01",
            "2016-02-01",
            "2016-02-01",
        ]),
        "series_id": ["a", "b", "a", "b"],
        "actual": [1.0, 2.0, 3.0, 4.0],
        "direct_auto_ets": [2.0, 3.0, 4.0, 5.0],
        MODEL_NAME: [1.0, 2.0, 3.0, 4.0],
    })
    summary, origins = calculate_metrics(predictions)

    observed = summary.set_index("model")
    assert observed.loc["auto_ets", "rmse"] == 1.0
    assert observed.loc[MODEL_NAME, "rmse"] == 0.0
    assert observed.loc[MODEL_NAME, "rmse_improvement_vs_autoets_pct"] == 100.0
    assert len(origins) == 4


def test_strict_rankscore_does_not_count_ties() -> None:
    dates = pd.date_range("2016-01-01", periods=12, freq="MS")
    frame = pd.concat([
        pd.DataFrame({"model": "best", "date": dates, "mae": 1.0}),
        pd.DataFrame({"model": "tie_a", "date": dates, "mae": 2.0}),
        pd.DataFrame({"model": "tie_b", "date": dates, "mae": 2.0}),
    ], ignore_index=True)

    summary, audit = calculate_strict_rank_scores(frame)
    scores = summary.set_index("model")["rank_score"]
    assert scores["best"] == 100.0
    assert scores["tie_a"] == 0.0
    assert scores["tie_b"] == 0.0
    tied = audit[audit["model"].isin(["tie_a", "tie_b"])]
    assert tied["absolute_error_wins"].eq(0).all()
