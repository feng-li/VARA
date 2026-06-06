"""Shared forecast metric tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import erf, pi, sqrt
from typing import Any

import numpy as np
import pandas as pd


def _safe_pct(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return numer / np.where(np.abs(denom) > 1e-12, np.abs(denom), np.nan)


def _seasonal_scales(train: np.ndarray, season_length: int) -> tuple[np.ndarray, np.ndarray]:
    if train.shape[0] <= season_length:
        raise ValueError("Training data must have more rows than season_length for MASE/RMSSE.")
    diff = train[season_length:] - train[:-season_length]
    mae_scale = np.nanmean(np.abs(diff), axis=0)
    rmse_scale = np.nanmean(diff ** 2, axis=0)
    mae_scale = np.where(mae_scale > 1e-12, mae_scale, np.nan)
    rmse_scale = np.where(rmse_scale > 1e-12, rmse_scale, np.nan)
    return mae_scale, rmse_scale


def _nanmean_or_nan(values: np.ndarray, axis=None):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    count = finite.sum(axis=axis)
    total = np.where(finite, arr, 0.0).sum(axis=axis)
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def _normal_crps(mean: np.ndarray, variance: np.ndarray, actual: np.ndarray) -> np.ndarray:
    mean = np.asarray(mean, dtype=float)
    actual = np.asarray(actual, dtype=float)
    variance = np.asarray(variance, dtype=float)
    out = np.full(mean.shape, np.nan, dtype=float)
    finite = np.isfinite(mean) & np.isfinite(actual) & np.isfinite(variance)
    degenerate = finite & (variance == 0.0)
    out[degenerate] = np.abs(mean[degenerate] - actual[degenerate])
    valid = finite & (variance > 0.0)
    if not valid.any():
        return out
    sigma = np.sqrt(variance[valid])
    z = (actual[valid] - mean[valid]) / sigma
    cdf = 0.5 * (1.0 + np.vectorize(erf, otypes=[float])(z / sqrt(2.0)))
    pdf = np.exp(-0.5 * z ** 2) / sqrt(2.0 * pi)
    out[valid] = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / sqrt(pi))
    return out


def forecast_metric_tables(
    forecasts: Mapping[str, np.ndarray],
    test: np.ndarray,
    train: np.ndarray,
    dates: Sequence[Any],
    series_ids: Sequence[str],
    *,
    model_names: Sequence[str] | None = None,
    season_length: int,
    forecast_mode: str,
    horizon: int,
    summary_context: Mapping[str, Any] | None = None,
    forecast_variances: Mapping[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build benchmark-style summary, by-series, and by-origin metric tables."""
    model_names = list(model_names) if model_names is not None else list(forecasts)
    series_ids = list(series_ids)
    test = np.asarray(test, dtype=float)
    train = np.asarray(train, dtype=float)
    mase_scale, rmsse_scale = _seasonal_scales(train, season_length)

    by_series_rows = []
    by_origin_rows = []
    summary_rows = []
    for model in model_names:
        forecast = np.asarray(forecasts[model], dtype=float)
        variance = None if forecast_variances is None else forecast_variances.get(model)
        if variance is not None:
            variance = np.asarray(variance, dtype=float)
            if variance.shape != forecast.shape:
                raise ValueError(
                    f"Forecast variance for model {model!r} has shape {variance.shape}; "
                    f"expected {forecast.shape}."
                )
        error = forecast - test
        abs_error = np.abs(error)
        squared_error = error ** 2
        crps = np.full(forecast.shape, np.nan, dtype=float) if variance is None else _normal_crps(forecast, variance, test)
        pct_error = _safe_pct(abs_error, test)
        smape = 2.0 * abs_error / np.where(
            np.abs(test) + np.abs(forecast) > 1e-12,
            np.abs(test) + np.abs(forecast),
            np.nan,
        )

        series_rmse = np.sqrt(np.nanmean(squared_error, axis=0))
        series_mae = np.nanmean(abs_error, axis=0)
        series_rows = pd.DataFrame({
            "model": model,
            "series_id": series_ids,
            "rmse": series_rmse,
            "mae": series_mae,
            "mape_pct": np.nanmean(pct_error, axis=0) * 100.0,
            "smape_pct": np.nanmean(smape, axis=0) * 100.0,
            "mase": series_mae / mase_scale,
            "rmsse": np.sqrt(np.nanmean(squared_error, axis=0) / rmsse_scale),
            "bias": np.nanmean(error, axis=0),
            "crps": _nanmean_or_nan(crps, axis=0),
            "forecast_variance_mean": (
                np.full(len(series_ids), np.nan)
                if variance is None
                else _nanmean_or_nan(variance, axis=0)
            ),
            "mean_actual": np.nanmean(test, axis=0),
            "mean_forecast": np.nanmean(forecast, axis=0),
        })
        by_series_rows.append(series_rows)

        by_origin_rows.append(pd.DataFrame({
            "model": model,
            "date": dates,
            "rmse": np.sqrt(np.nanmean(squared_error, axis=1)),
            "mae": np.nanmean(abs_error, axis=1),
            "mape_pct": np.nanmean(pct_error, axis=1) * 100.0,
            "smape_pct": np.nanmean(smape, axis=1) * 100.0,
            "bias": np.nanmean(error, axis=1),
            "crps": _nanmean_or_nan(crps, axis=1),
            "forecast_variance_mean": (
                np.full(test.shape[0], np.nan)
                if variance is None
                else _nanmean_or_nan(variance, axis=1)
            ),
        }))

        summary_rows.append({
            "model": model,
            "forecast_mode": forecast_mode,
            "horizon": int(horizon),
            "n_series": len(series_ids),
            "n_test": int(test.shape[0]),
            "rmse": float(np.sqrt(np.nanmean(squared_error))),
            "mae": float(np.nanmean(abs_error)),
            "mape_pct": float(np.nanmean(pct_error) * 100.0),
            "smape_pct": float(np.nanmean(smape) * 100.0),
            "mase": float(np.nanmean(series_rows["mase"])),
            "rmsse": float(np.nanmean(series_rows["rmsse"])),
            "bias": float(np.nanmean(error)),
            "crps": float(_nanmean_or_nan(crps)),
            "forecast_variance_mean": (
                np.nan if variance is None else float(_nanmean_or_nan(variance))
            ),
        })

    summary = pd.DataFrame(summary_rows)
    if summary_context:
        insert_at = 1
        for key, value in summary_context.items():
            summary.insert(insert_at, key, value)
            insert_at += 1

    return (
        pd.concat(by_series_rows, ignore_index=True),
        pd.concat(by_origin_rows, ignore_index=True),
        summary,
    )
