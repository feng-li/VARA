"""Shared forecast metric tables."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import pi, sqrt
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


def _nansum_or_nan(values: np.ndarray, axis=None):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    count = finite.sum(axis=axis)
    total = np.where(finite, arr, 0.0).sum(axis=axis)
    if axis is None:
        return float(total) if int(count) > 0 else np.nan
    return np.where(count > 0, total, np.nan)


def _one_minus_ratio(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    numer = np.asarray(numer, dtype=float)
    denom = np.asarray(denom, dtype=float)
    return 1.0 - np.divide(
        numer,
        denom,
        out=np.full_like(denom, np.nan, dtype=float),
        where=np.isfinite(numer) & np.isfinite(denom) & (np.abs(denom) > 1e-12),
    )


def out_of_sample_r2(error: np.ndarray, test: np.ndarray, axis=None) -> np.ndarray:
    """Out-of-sample R2 against each series' test-window mean."""
    error = np.asarray(error, dtype=float)
    test = np.asarray(test, dtype=float)
    if error.shape != test.shape:
        raise ValueError(f"error and test must have the same shape; got {error.shape} and {test.shape}.")

    test_mean = _nanmean_or_nan(test, axis=0)
    centered_test_sq = (test - test_mean) ** 2
    valid = np.isfinite(error) & np.isfinite(centered_test_sq)
    sse = _nansum_or_nan(np.where(valid, error ** 2, np.nan), axis=axis)
    sst = _nansum_or_nan(np.where(valid, centered_test_sq, np.nan), axis=axis)
    return _one_minus_ratio(sse, sst)


def out_of_sample_r2_zero(error: np.ndarray, test: np.ndarray, axis=None) -> np.ndarray:
    """Out-of-sample R2 against a zero forecast denominator."""
    error = np.asarray(error, dtype=float)
    test = np.asarray(test, dtype=float)
    if error.shape != test.shape:
        raise ValueError(f"error and test must have the same shape; got {error.shape} and {test.shape}.")

    valid = np.isfinite(error) & np.isfinite(test)
    sse = _nansum_or_nan(np.where(valid, error ** 2, np.nan), axis=axis)
    sst = _nansum_or_nan(np.where(valid, test ** 2, np.nan), axis=axis)
    return _one_minus_ratio(sse, sst)


def normal_crps(mean: np.ndarray, variance: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return elementwise Gaussian CRPS, broadcasting all three inputs."""
    from scipy.special import ndtr

    mean, variance, actual = np.broadcast_arrays(
        np.asarray(mean, dtype=float),
        np.asarray(variance, dtype=float),
        np.asarray(actual, dtype=float),
    )
    out = np.full(mean.shape, np.nan, dtype=float)
    finite = np.isfinite(mean) & np.isfinite(actual) & np.isfinite(variance)
    degenerate = finite & (variance == 0.0)
    out[degenerate] = np.abs(mean[degenerate] - actual[degenerate])
    valid = finite & (variance > 0.0)
    if not valid.any():
        return out
    sigma = np.sqrt(variance[valid])
    z = (actual[valid] - mean[valid]) / sigma
    cdf = ndtr(z)
    pdf = np.exp(-0.5 * z ** 2) / sqrt(2.0 * pi)
    out[valid] = sigma * (z * (2.0 * cdf - 1.0) + 2.0 * pdf - 1.0 / sqrt(pi))
    return out


# Keep existing workflow imports working while callers migrate to the public name.
_normal_crps = normal_crps


def add_composite_rank_score(
    comparison: pd.DataFrame,
    *,
    lower_is_better: Sequence[str],
    higher_is_better: Sequence[str] = (),
    closest_to_zero: Sequence[str] = (),
    method_column: str = "model",
) -> pd.DataFrame:
    """Add a normalized Borda rank score across the supplied metrics.

    Ranks use average ties. RankScore ranges from zero to 100 within the
    supplied method set, with larger values indicating better average rank.
    """
    metric_groups = {
        "lower_is_better": tuple(lower_is_better),
        "higher_is_better": tuple(higher_is_better),
        "closest_to_zero": tuple(closest_to_zero),
    }
    metrics = tuple(metric for group in metric_groups.values() for metric in group)
    if not metrics:
        raise ValueError("At least one component metric is required for RankScore.")
    if len(set(metrics)) != len(metrics):
        raise ValueError("Each RankScore component metric must appear exactly once.")
    required = {method_column, *metrics}
    missing = sorted(required.difference(comparison.columns))
    if missing:
        raise ValueError(f"RankScore input is missing columns: {missing}")
    if comparison.empty:
        raise ValueError("At least one method is required to compute RankScore.")
    if comparison[method_column].duplicated().any():
        duplicates = sorted(
            comparison.loc[
                comparison[method_column].duplicated(False), method_column
            ].astype(str).unique()
        )
        raise ValueError(f"RankScore input has duplicate methods: {duplicates}")

    ranked = comparison.copy()
    numeric = ranked.loc[:, metrics].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("RankScore component metrics must all be finite.")
    ranked.loc[:, metrics] = numeric

    rank_columns: list[str] = []
    for metric in metric_groups["lower_is_better"]:
        column = f"rank_{metric}"
        ranked[column] = ranked[metric].rank(method="average", ascending=True)
        rank_columns.append(column)
    for metric in metric_groups["higher_is_better"]:
        column = f"rank_{metric}"
        ranked[column] = ranked[metric].rank(method="average", ascending=False)
        rank_columns.append(column)
    for metric in metric_groups["closest_to_zero"]:
        absolute_column = f"abs_{metric}"
        rank_column = f"rank_{absolute_column}"
        ranked[absolute_column] = ranked[metric].abs()
        ranked[rank_column] = ranked[absolute_column].rank(
            method="average",
            ascending=True,
        )
        rank_columns.append(rank_column)

    ranked["mean_accuracy_rank"] = ranked[rank_columns].mean(axis=1)
    method_count = len(ranked)
    if method_count == 1:
        ranked["rank_score"] = 100.0
    else:
        ranked["rank_score"] = 100.0 * (
            method_count - ranked["mean_accuracy_rank"]
        ) / (method_count - 1)
    ranked["overall_rank"] = (
        ranked["rank_score"].rank(method="min", ascending=False).astype(int)
    )
    return ranked


def forecast_metric_tables(
    forecasts: Mapping[str, np.ndarray],
    test: np.ndarray,
    train: np.ndarray | None,
    dates: Sequence[Any],
    series_ids: Sequence[str],
    *,
    model_names: Sequence[str] | None = None,
    season_length: int,
    forecast_mode: str,
    horizon: int,
    summary_context: Mapping[str, Any] | None = None,
    forecast_variances: Mapping[str, np.ndarray] | None = None,
    seasonal_mae_scale: np.ndarray | None = None,
    seasonal_mse_scale: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build benchmark-style summary, by-series, and by-origin metric tables."""
    model_names = list(model_names) if model_names is not None else list(forecasts)
    series_ids = list(series_ids)
    test = np.asarray(test, dtype=float)
    if (seasonal_mae_scale is None) != (seasonal_mse_scale is None):
        raise ValueError(
            "seasonal_mae_scale and seasonal_mse_scale must be provided together."
        )
    if seasonal_mae_scale is None:
        if train is None:
            raise ValueError(
                "train is required when precomputed seasonal scales are not provided."
            )
        train = np.asarray(train, dtype=float)
        mase_scale, rmsse_scale = _seasonal_scales(train, season_length)
    else:
        mase_scale = np.asarray(seasonal_mae_scale, dtype=float)
        rmsse_scale = np.asarray(seasonal_mse_scale, dtype=float)
        expected_shape = (len(series_ids),)
        if mase_scale.shape != expected_shape or rmsse_scale.shape != expected_shape:
            raise ValueError(
                "Precomputed seasonal scales must contain one value per series; "
                f"expected {expected_shape}, got {mase_scale.shape} and {rmsse_scale.shape}."
            )
        mase_scale = np.where(mase_scale > 1e-12, mase_scale, np.nan)
        rmsse_scale = np.where(rmsse_scale > 1e-12, rmsse_scale, np.nan)

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
        crps = np.full(forecast.shape, np.nan, dtype=float) if variance is None else normal_crps(forecast, variance, test)
        pct_error = _safe_pct(abs_error, test)
        smape = 2.0 * abs_error / np.where(
            np.abs(test) + np.abs(forecast) > 1e-12,
            np.abs(test) + np.abs(forecast),
            np.nan,
        )

        series_rmse = np.sqrt(_nanmean_or_nan(squared_error, axis=0))
        series_mae = _nanmean_or_nan(abs_error, axis=0)
        series_oos_r2 = out_of_sample_r2(error, test, axis=0)
        series_oos_r2_zero = out_of_sample_r2_zero(error, test, axis=0)
        series_rows = pd.DataFrame({
            "model": model,
            "series_id": series_ids,
            "rmse": series_rmse,
            "mae": series_mae,
            "mape_pct": _nanmean_or_nan(pct_error, axis=0) * 100.0,
            "smape_pct": _nanmean_or_nan(smape, axis=0) * 100.0,
            "mase": series_mae / mase_scale,
            "rmsse": np.sqrt(_nanmean_or_nan(squared_error, axis=0) / rmsse_scale),
            "seasonal_mae_scale": mase_scale,
            "seasonal_mse_scale": rmsse_scale,
            "bias": _nanmean_or_nan(error, axis=0),
            "oos_r2": series_oos_r2,
            "oos_r2_zero": series_oos_r2_zero,
            "crps": _nanmean_or_nan(crps, axis=0),
            "forecast_variance_mean": (
                np.full(len(series_ids), np.nan)
                if variance is None
                else _nanmean_or_nan(variance, axis=0)
            ),
            "mean_actual": _nanmean_or_nan(test, axis=0),
            "mean_forecast": _nanmean_or_nan(forecast, axis=0),
        })
        by_series_rows.append(series_rows)

        by_origin_rows.append(pd.DataFrame({
            "model": model,
            "date": dates,
            "rmse": np.sqrt(_nanmean_or_nan(squared_error, axis=1)),
            "mae": _nanmean_or_nan(abs_error, axis=1),
            "mape_pct": _nanmean_or_nan(pct_error, axis=1) * 100.0,
            "smape_pct": _nanmean_or_nan(smape, axis=1) * 100.0,
            "bias": _nanmean_or_nan(error, axis=1),
            "oos_r2": out_of_sample_r2(error, test, axis=1),
            "oos_r2_zero": out_of_sample_r2_zero(error, test, axis=1),
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
            "rmse": float(np.sqrt(_nanmean_or_nan(squared_error))),
            "mae": float(_nanmean_or_nan(abs_error)),
            "mape_pct": float(_nanmean_or_nan(pct_error) * 100.0),
            "smape_pct": float(_nanmean_or_nan(smape) * 100.0),
            "mase": float(_nanmean_or_nan(series_rows["mase"])),
            "rmsse": float(_nanmean_or_nan(series_rows["rmsse"])),
            "bias": float(_nanmean_or_nan(error)),
            "oos_r2": float(out_of_sample_r2(error, test)),
            "oos_r2_zero": float(out_of_sample_r2_zero(error, test)),
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
