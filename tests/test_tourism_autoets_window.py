import json

import numpy as np
import pandas as pd

from scripts.evaluate_tourism_76_autoets import (
    _normalize_training_history,
    run_backtest,
)


def test_fixed_autoets_window_uses_exact_trailing_history(
    tmp_path,
    monkeypatch,
) -> None:
    dates = pd.date_range("2000-01-01", "2001-04-01", freq="MS")
    data = pd.DataFrame({
        "date": dates,
        "AAAAll": np.arange(len(dates), dtype=float) + 1.0,
        "AABAll": np.arange(len(dates), dtype=float) + 2.0,
        "ABAAll": np.arange(len(dates), dtype=float) + 3.0,
    })
    data_path = tmp_path / "tourism.csv"
    output_path = tmp_path / "rolling_forecasts.csv"
    data.to_csv(data_path, index=False)

    received = []

    def fake_autoets(history, **kwargs):
        values = np.asarray(history, dtype=float)
        received.append(values.copy())
        mean = values.mean(axis=0)
        variance = values.var(axis=0)
        scale = np.where(np.sqrt(variance) > 1e-12, np.sqrt(variance), 1.0)
        forecast = np.asarray(values[-1:], dtype=float)
        return {
            "forecast": forecast,
            "forecast_variance": np.ones((1, values.shape[1]), dtype=float),
            "forecast_normalized": forecast,
            "forecast_variance_normalized": np.ones(
                (1, values.shape[1]),
                dtype=float,
            ),
            "training_mean": mean,
            "training_variance": variance,
            "training_scale": scale,
        }

    monkeypatch.setattr(
        "scripts.evaluate_tourism_76_autoets.auto_ets_baseline_panel",
        fake_autoets,
    )
    forecasts = run_backtest(
        data_path=data_path,
        output_path=output_path,
        first_year=2001,
        final_year=2001,
        series_depth=3,
        expected_n_series=3,
        season_length=12,
        ray_num_cpus=1,
        max_cpus_per_task=1,
        clip_nonnegative=True,
        model_specification="ZZA",
        training_window_months=12,
    )

    assert [len(history) for history in received] == [12, 12, 12, 12]
    assert forecasts["autoets_n_train"].eq(12).all()
    starts = (
        forecasts.drop_duplicates("date")
        .sort_values("date")["autoets_history_start"]
        .tolist()
    )
    assert starts == [
        "2000-01-01",
        "2000-02-01",
        "2000-03-01",
        "2000-04-01",
    ]
    metadata = json.loads(
        output_path.with_suffix(".csv.metadata.json").read_text()
    )
    assert metadata["autoets_config"]["training_window"] == "fixed"
    assert metadata["autoets_config"]["training_window_months"] == 12


def test_training_zscore_uses_each_origins_training_mean_and_variance(
    tmp_path,
    monkeypatch,
) -> None:
    dates = pd.date_range("2000-01-01", "2001-02-01", freq="MS")
    original = np.column_stack([
        np.arange(len(dates), dtype=float) + 10.0,
        2.0 * np.arange(len(dates), dtype=float) + 40.0,
    ])
    data = pd.DataFrame({
        "date": dates,
        "AAAAll": original[:, 0],
        "AABAll": original[:, 1],
    })
    data_path = tmp_path / "tourism.csv"
    output_path = tmp_path / "rolling_forecasts.csv"
    data.to_csv(data_path, index=False)

    received = []

    def fake_autoets(history, **kwargs):
        values = np.asarray(history, dtype=float)
        normalized, mean, variance, scale = _normalize_training_history(
            values,
            kwargs["response_normalization"],
        )
        received.append(normalized.copy())
        np.testing.assert_allclose(normalized.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(normalized.var(axis=0), 1.0, atol=1e-12)
        forecast_normalized = np.full((1, normalized.shape[1]), 2.0)
        variance_normalized = np.full((1, normalized.shape[1]), 3.0)
        return {
            "forecast": mean[None, :] + scale[None, :] * forecast_normalized,
            "forecast_variance": (
                variance_normalized * np.square(scale)[None, :]
            ),
            "forecast_normalized": forecast_normalized,
            "forecast_variance_normalized": variance_normalized,
            "training_mean": mean,
            "training_variance": variance,
            "training_scale": scale,
        }

    monkeypatch.setattr(
        "scripts.evaluate_tourism_76_autoets.auto_ets_baseline_panel",
        fake_autoets,
    )
    forecasts = run_backtest(
        data_path=data_path,
        output_path=output_path,
        first_year=2001,
        final_year=2001,
        series_depth=3,
        expected_n_series=2,
        season_length=12,
        ray_num_cpus=1,
        max_cpus_per_task=1,
        clip_nonnegative=True,
        response_transform="log1p",
        response_normalization="training_zscore",
        model_specification="ZZA",
        training_window_months=12,
    )

    assert len(received) == 2
    for date, origin in forecasts.groupby("date", sort=True):
        target_index = dates.get_loc(pd.Timestamp(date))
        transformed_history = np.log1p(
            original[target_index - 12:target_index]
        )
        training_mean = transformed_history.mean(axis=0)
        training_variance = transformed_history.var(axis=0)
        training_scale = np.sqrt(training_variance)
        ordered = origin.sort_values("series_id")
        np.testing.assert_allclose(
            ordered["autoets_training_mean"],
            training_mean,
        )
        np.testing.assert_allclose(
            ordered["autoets_training_variance"],
            training_variance,
        )
        np.testing.assert_allclose(
            ordered["autoets_training_scale"],
            training_scale,
        )
        np.testing.assert_allclose(
            ordered["direct_auto_ets_normalized"],
            2.0,
        )
        np.testing.assert_allclose(
            ordered["direct_auto_ets"],
            training_mean + 2.0 * training_scale,
        )
        np.testing.assert_allclose(
            ordered["direct_auto_ets_variance"],
            3.0 * training_variance,
        )
        np.testing.assert_allclose(
            ordered["actual"],
            np.log1p(original[target_index]),
        )

    metadata = json.loads(
        output_path.with_suffix(".csv.metadata.json").read_text()
    )
    assert metadata["response_normalization"] == "training_zscore"
    assert metadata["normalization"]["variance_ddof"] == 0
    assert (
        metadata["normalization"]["statistics_source"]
        == "training observations through t-1 only"
    )


def test_constant_series_normalization_uses_safe_unit_scale() -> None:
    history = np.asarray([
        [4.0, 1.0],
        [4.0, 2.0],
        [4.0, 3.0],
    ])

    normalized, mean, variance, scale = _normalize_training_history(
        history,
        "training_zscore",
    )

    np.testing.assert_allclose(mean, [4.0, 2.0])
    np.testing.assert_allclose(variance, [0.0, 2.0 / 3.0])
    np.testing.assert_allclose(scale, [1.0, np.sqrt(2.0 / 3.0)])
    np.testing.assert_allclose(normalized[:, 0], 0.0)
