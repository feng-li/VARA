"""Statistical tests for comparing matched out-of-sample forecasts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import sqrt
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as student_t


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Return Holm family-wise-error adjusted p-values."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("p_values must be a non-empty one-dimensional sequence.")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p_values must be finite values between zero and one.")

    order = np.argsort(values)
    ordered = values[order]
    multipliers = values.size - np.arange(values.size)
    adjusted_ordered = np.maximum.accumulate(ordered * multipliers)
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted


def diebold_mariano_test(
    focal_loss: Sequence[float],
    comparator_loss: Sequence[float],
    *,
    forecast_horizon: int = 1,
) -> dict[str, float | int]:
    """Run the Harvey-Leybourne-Newbold corrected Diebold-Mariano test.

    Loss differentials are defined as focal minus comparator loss, so a
    negative statistic favors the focal forecast. Autocovariances through
    ``forecast_horizon - 1`` are included in the long-run variance.
    """
    focal = np.asarray(focal_loss, dtype=float)
    comparator = np.asarray(comparator_loss, dtype=float)
    if focal.ndim != 1 or comparator.ndim != 1 or focal.shape != comparator.shape:
        raise ValueError("Matched focal and comparator losses must be equal-length vectors.")
    if not np.isfinite(focal).all() or not np.isfinite(comparator).all():
        raise ValueError("Diebold-Mariano losses must all be finite.")
    n_observations = int(focal.size)
    horizon = int(forecast_horizon)
    if n_observations < 3:
        raise ValueError("At least three forecast origins are required for a DM test.")
    if horizon < 1 or horizon >= n_observations:
        raise ValueError("forecast_horizon must be positive and smaller than the sample.")

    differential = focal - comparator
    mean_differential = float(np.mean(differential))
    centered = differential - mean_differential
    autocovariances = [float(np.dot(centered, centered) / n_observations)]
    for lag in range(1, horizon):
        autocovariances.append(
            float(np.dot(centered[lag:], centered[:-lag]) / n_observations)
        )
    long_run_variance = autocovariances[0] + 2.0 * sum(autocovariances[1:])
    variance_scale = max(float(np.mean(differential**2)), 1.0)
    tolerance = np.finfo(float).eps * variance_scale * 100.0

    correction_term = (
        n_observations
        + 1
        - 2 * horizon
        + horizon * (horizon - 1) / n_observations
    ) / n_observations
    if correction_term <= 0.0:
        raise ValueError("The Harvey correction is undefined for this sample and horizon.")
    harvey_correction = sqrt(correction_term)

    if long_run_variance < -tolerance:
        raise ValueError("Estimated DM long-run variance is negative.")
    if long_run_variance <= tolerance:
        if not np.allclose(centered, 0.0, rtol=0.0, atol=sqrt(tolerance)):
            raise ValueError("Estimated DM long-run variance is numerically zero.")
        standard_error = 0.0
        if abs(mean_differential) <= sqrt(tolerance):
            raw_statistic = 0.0
            statistic = 0.0
            p_value_raw_two_sided = 1.0
            p_value_raw_focal_better = 0.5
            p_value_two_sided = 1.0
            p_value_focal_better = 0.5
        else:
            raw_statistic = float(np.copysign(np.inf, mean_differential))
            statistic = raw_statistic
            p_value_raw_two_sided = 0.0
            p_value_raw_focal_better = 0.0 if raw_statistic < 0.0 else 1.0
            p_value_two_sided = 0.0
            p_value_focal_better = 0.0 if statistic < 0.0 else 1.0
    else:
        standard_error = sqrt(long_run_variance / n_observations)
        raw_statistic = mean_differential / standard_error
        statistic = raw_statistic * harvey_correction
        degrees_of_freedom = n_observations - 1
        p_value_raw_two_sided = float(
            2.0 * norm.sf(abs(raw_statistic))
        )
        p_value_raw_focal_better = float(norm.cdf(raw_statistic))
        p_value_two_sided = float(
            2.0 * student_t.sf(abs(statistic), df=degrees_of_freedom)
        )
        p_value_focal_better = float(
            student_t.cdf(statistic, df=degrees_of_freedom)
        )

    return {
        "n_origins": n_observations,
        "forecast_horizon": horizon,
        "hac_lag": horizon - 1,
        "mean_loss_difference": mean_differential,
        "long_run_variance": float(max(long_run_variance, 0.0)),
        "standard_error": float(standard_error),
        "harvey_correction": float(harvey_correction),
        "dm_statistic_raw": float(raw_statistic),
        "p_value_raw_two_sided": float(p_value_raw_two_sided),
        "p_value_raw_focal_better": float(p_value_raw_focal_better),
        "dm_statistic": float(statistic),
        "p_value_two_sided": float(p_value_two_sided),
        "p_value_focal_better": float(p_value_focal_better),
    }


def panel_diebold_mariano_tests(
    forecasts: Mapping[str, np.ndarray],
    actual: np.ndarray,
    dates: Sequence[Any],
    *,
    focal_model: str,
    forecast_horizon: int = 1,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare a focal method with all others using panel-average origin losses."""
    if focal_model not in forecasts:
        raise ValueError(f"Focal model {focal_model!r} is not available.")
    if len(forecasts) < 2:
        raise ValueError("At least two forecast methods are required.")
    if not 0.0 < float(alpha) < 1.0:
        raise ValueError("alpha must be between zero and one.")

    actual_array = np.asarray(actual, dtype=float)
    if actual_array.ndim != 2 or not np.isfinite(actual_array).all():
        raise ValueError("actual must be a finite origin-by-series matrix.")
    if len(dates) != actual_array.shape[0]:
        raise ValueError("dates must contain one value per forecast origin.")
    parsed_dates = pd.to_datetime(list(dates))
    if parsed_dates.duplicated().any():
        raise ValueError("Forecast-origin dates must be unique.")

    arrays: dict[str, np.ndarray] = {}
    for model, values in forecasts.items():
        array = np.asarray(values, dtype=float)
        if array.shape != actual_array.shape:
            raise ValueError(
                f"Forecast {model!r} has shape {array.shape}; expected {actual_array.shape}."
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Forecast {model!r} contains non-finite values.")
        arrays[str(model)] = array

    loss_specs = {
        "squared_error": (lambda error: error**2, "rmse", np.sqrt),
        "absolute_error": (np.abs, "mae", lambda value: value),
    }
    rows = []
    focal_error = arrays[focal_model] - actual_array
    for loss_name, (loss_function, metric_name, metric_transform) in loss_specs.items():
        focal_origin_loss = np.mean(loss_function(focal_error), axis=1)
        focal_mean_loss = float(np.mean(focal_origin_loss))
        focal_metric = float(metric_transform(focal_mean_loss))
        for comparator_model, comparator_forecast in arrays.items():
            if comparator_model == focal_model:
                continue
            comparator_error = comparator_forecast - actual_array
            comparator_origin_loss = np.mean(
                loss_function(comparator_error),
                axis=1,
            )
            comparator_mean_loss = float(np.mean(comparator_origin_loss))
            comparator_metric = float(metric_transform(comparator_mean_loss))
            test = diebold_mariano_test(
                focal_origin_loss,
                comparator_origin_loss,
                forecast_horizon=forecast_horizon,
            )
            improvement = (
                np.nan
                if abs(comparator_metric) <= 1e-12
                else 100.0 * (comparator_metric - focal_metric) / comparator_metric
            )
            rows.append({
                "focal_model": focal_model,
                "comparator_model": comparator_model,
                "loss": loss_name,
                "metric": metric_name,
                "test_start": parsed_dates.min(),
                "test_end": parsed_dates.max(),
                "n_series": int(actual_array.shape[1]),
                "focal_metric": focal_metric,
                "comparator_metric": comparator_metric,
                "metric_improvement_pct": float(improvement),
                "focal_mean_loss": focal_mean_loss,
                "comparator_mean_loss": comparator_mean_loss,
                "focal_better": bool(focal_mean_loss < comparator_mean_loss),
                **test,
            })

    results = pd.DataFrame(rows)
    for _, indexes in results.groupby("loss", sort=False).groups.items():
        index = list(indexes)
        results.loc[index, "p_value_raw_two_sided_holm"] = holm_adjust(
            results.loc[index, "p_value_raw_two_sided"].to_numpy(dtype=float)
        )
        results.loc[index, "p_value_raw_focal_better_holm"] = holm_adjust(
            results.loc[index, "p_value_raw_focal_better"].to_numpy(dtype=float)
        )
        results.loc[index, "p_value_two_sided_holm"] = holm_adjust(
            results.loc[index, "p_value_two_sided"].to_numpy(dtype=float)
        )
        results.loc[index, "p_value_focal_better_holm"] = holm_adjust(
            results.loc[index, "p_value_focal_better"].to_numpy(dtype=float)
        )
    results["significant_difference_holm"] = (
        results["p_value_two_sided_holm"] < float(alpha)
    )
    results["significant_difference_raw_holm"] = (
        results["p_value_raw_two_sided_holm"] < float(alpha)
    )
    results["significant_improvement_raw_holm"] = (
        results["focal_better"]
        & (results["p_value_raw_focal_better_holm"] < float(alpha))
    )
    results["significant_improvement_holm"] = (
        results["focal_better"]
        & (results["p_value_focal_better_holm"] < float(alpha))
    )
    results["alpha"] = float(alpha)
    loss_order = pd.Categorical(
        results["loss"],
        categories=list(loss_specs),
        ordered=True,
    )
    return (
        results.assign(_loss_order=loss_order)
        .sort_values(["_loss_order", "p_value_focal_better", "comparator_model"])
        .drop(columns="_loss_order")
        .reset_index(drop=True)
    )
