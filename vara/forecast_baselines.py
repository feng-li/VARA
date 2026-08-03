"""Shared baseline forecast helpers for residual VARA workflows."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


AUTO_ETS_ALIAS = "auto_ets"
AUTO_ETS_MODEL_SPECIFICATION = "ZZZ"
AUTO_ETS_DAMPED = None
AUTO_ETS_PHI = None
STATSMODELS_ETS_ALIAS = "statsmodels_ets"
SEASONAL_ETS_ALIAS = "seasonal_ets"
FORECAST_VARIANCE_LEVEL = 95
_FORECAST_VARIANCE_Z = 1.959963984540054
RESPONSE_NORMALIZATIONS = ("none", "training_zscore")


def auto_ets_config(
    seasonal_lag: int,
    *,
    model_specification: str = AUTO_ETS_MODEL_SPECIFICATION,
    damped: bool | None = AUTO_ETS_DAMPED,
    phi: float | None = AUTO_ETS_PHI,
) -> dict[str, Any]:
    """Return a StatsForecast AutoETS specification.

    The defaults preserve the frozen standard configuration: ``ZZZ`` delegates
    error, trend, and seasonal-form selection to AutoETS; ``damped=None`` lets
    it compare damped and undamped additive trends; and ``phi=None`` estimates
    the damping coefficient when needed. A fixed three-letter ETS family such
    as ``ANN`` can be supplied for a prespecified sensitivity analysis without
    changing those defaults.
    """
    specification = str(model_specification).strip().upper()
    allowed = (set("AMZ"), set("NAMZ"), set("NAMZ"))
    if len(specification) != 3 or any(
        symbol not in choices
        for symbol, choices in zip(specification, allowed, strict=True)
    ):
        raise ValueError(
            "AutoETS model_specification must be a three-letter ETS code with "
            "error in {A,M,Z} and trend/seasonality in {N,A,M,Z}."
        )
    if specification[1] == "N" and damped is True:
        raise ValueError("An ETS model without trend cannot be damped.")
    selects_family = (
        "Z" in specification
        or (specification[1] != "N" and damped is None)
    )
    return {
        "model": "AutoETS",
        "implementation": "statsforecast.models.AutoETS",
        "alias": AUTO_ETS_ALIAS,
        "model_specification": specification,
        "season_length": max(1, int(seasonal_lag)),
        "damped": damped,
        "phi": phi,
        "selection_criterion": "AICc" if selects_family else None,
        "optimization_criterion": "likelihood",
        "allow_multiplicative_trend": False,
        "freq": 1,
    }


def make_auto_ets_model(
    seasonal_lag: int,
    *,
    alias: str = AUTO_ETS_ALIAS,
    model_specification: str = AUTO_ETS_MODEL_SPECIFICATION,
    damped: bool | None = AUTO_ETS_DAMPED,
    phi: float | None = AUTO_ETS_PHI,
):
    """Construct AutoETS from the shared specification."""
    from statsforecast.models import AutoETS

    config = auto_ets_config(
        seasonal_lag,
        model_specification=model_specification,
        damped=damped,
        phi=phi,
    )
    return AutoETS(
        season_length=config["season_length"],
        model=config["model_specification"],
        damped=config["damped"],
        phi=config["phi"],
        alias=str(alias),
    )


def statsmodels_ets_config() -> dict[str, Any]:
    return {
        "model": "statsmodels.ETSModel",
        "alias": STATSMODELS_ETS_ALIAS,
        "error": "add",
        "trend": None,
        "damped_trend": False,
        "seasonal": None,
        "initialization_method": "estimated",
    }


def seasonal_ets_config(seasonal_lag: int) -> dict[str, Any]:
    return {
        "model": "statsmodels.ETSModel",
        "alias": SEASONAL_ETS_ALIAS,
        "error": "add",
        "trend": None,
        "damped_trend": False,
        "seasonal": "add",
        "seasonal_periods": max(1, int(seasonal_lag)),
        "initialization_method": "estimated",
    }


def _statsforecast_long_from_panel(values: np.ndarray, series_ids: Sequence[str]) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    frame = pd.DataFrame(values, columns=list(series_ids))
    frame.insert(0, "ds", np.arange(values.shape[0]))
    return frame.melt(id_vars="ds", var_name="unique_id", value_name="y")


def _statsforecast_wide(frame: pd.DataFrame, series_ids: Sequence[str], value_col: str) -> np.ndarray:
    wide = frame.pivot(index="ds", columns="unique_id", values=value_col).sort_index()
    return wide.loc[:, list(series_ids)].to_numpy(dtype=float)


def _interval_variance(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    var = ((hi - lo) / (2.0 * _FORECAST_VARIANCE_Z)) ** 2
    return np.where(np.isfinite(var) & (var >= 0.0), var, np.nan)


def normalize_training_panel(
    values: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each panel column using statistics from the supplied history."""
    history = np.asarray(values, dtype=float)
    if history.ndim != 2:
        raise ValueError("Training normalization expects a two-dimensional panel.")
    training_mean = np.mean(history, axis=0)
    training_variance = np.var(history, axis=0)
    training_scale = np.sqrt(training_variance)
    training_scale = np.where(training_scale > 1e-12, training_scale, 1.0)
    if method == "none":
        return (
            history.copy(),
            training_mean,
            training_variance,
            training_scale,
        )
    if method == "training_zscore":
        normalized = (
            history - training_mean[None, :]
        ) / training_scale[None, :]
        return normalized, training_mean, training_variance, training_scale
    raise ValueError(
        f"response_normalization must be one of {RESPONSE_NORMALIZATIONS}."
    )


def _statsmodels_prediction_variance(result, *, start: int, horizon: int) -> np.ndarray:
    try:
        frame = result.get_prediction(start=start, end=start + int(horizon) - 1).summary_frame(alpha=0.05)
        return _interval_variance(frame["pi_lower"].to_numpy(dtype=float), frame["pi_upper"].to_numpy(dtype=float))
    except Exception:
        return np.full(int(horizon), np.nan, dtype=float)


def auto_ets_baseline_panel(
    history: np.ndarray,
    horizon: int,
    seasonal_lag: int,
    series_ids: Sequence[str],
    *,
    model_specification: str = AUTO_ETS_MODEL_SPECIFICATION,
    damped: bool | None = AUTO_ETS_DAMPED,
    phi: float | None = AUTO_ETS_PHI,
    response_normalization: str = "none",
) -> dict[str, Any]:
    """Fit AutoETS on ``history`` and return fitted, residual, and forecast panels.

    ``history`` has shape ``(T, p)`` and all returned panels use the same series
    order as ``series_ids``. This function is intentionally the single source of
    AutoETS configuration for residual feature extraction and forecast recovery.
    """
    from statsforecast import StatsForecast
    values = np.asarray(history, dtype=float)
    if values.ndim != 2:
        raise ValueError("history must have shape (T, p).")
    if values.shape[0] < 2:
        raise ValueError("AutoETS baseline requires at least two training observations.")
    if values.shape[1] != len(series_ids):
        raise ValueError("history column count must match series_ids.")

    fit_values, training_mean, training_variance, training_scale = (
        normalize_training_panel(values, response_normalization)
    )
    config = auto_ets_config(
        seasonal_lag,
        model_specification=model_specification,
        damped=damped,
        phi=phi,
    )
    long = _statsforecast_long_from_panel(fit_values, series_ids)
    sf = StatsForecast(
        models=[
            make_auto_ets_model(
                seasonal_lag,
                model_specification=model_specification,
                damped=damped,
                phi=phi,
            )
        ],
        freq=config["freq"],
        n_jobs=1,
    )
    forecast_long = sf.forecast(h=int(horizon), df=long, fitted=True, level=[FORECAST_VARIANCE_LEVEL])
    fitted_long = sf.forecast_fitted_values()
    fitted_normalized = _statsforecast_wide(
        fitted_long,
        series_ids,
        AUTO_ETS_ALIAS,
    )
    forecast_normalized = _statsforecast_wide(
        forecast_long,
        series_ids,
        AUTO_ETS_ALIAS,
    )
    forecast_variance_normalized = _interval_variance(
        _statsforecast_wide(forecast_long, series_ids, f"{AUTO_ETS_ALIAS}-lo-{FORECAST_VARIANCE_LEVEL}"),
        _statsforecast_wide(forecast_long, series_ids, f"{AUTO_ETS_ALIAS}-hi-{FORECAST_VARIANCE_LEVEL}"),
    )
    if response_normalization == "training_zscore":
        fitted = (
            fitted_normalized * training_scale[None, :]
            + training_mean[None, :]
        )
        forecast = (
            forecast_normalized * training_scale[None, :]
            + training_mean[None, :]
        )
        forecast_variance = (
            forecast_variance_normalized
            * np.square(training_scale)[None, :]
        )
    else:
        fitted = fitted_normalized.copy()
        forecast = forecast_normalized.copy()
        forecast_variance = forecast_variance_normalized.copy()
    return {
        "fitted": fitted,
        "residuals": values - fitted,
        "forecast": forecast,
        "forecast_variance": forecast_variance,
        "fitted_normalized": fitted_normalized,
        "residuals_normalized": fit_values - fitted_normalized,
        "forecast_normalized": forecast_normalized,
        "forecast_variance_normalized": forecast_variance_normalized,
        "training_mean": training_mean,
        "training_variance": training_variance,
        "training_scale": training_scale,
        "response_normalization": response_normalization,
        "config": config,
    }


def statsmodels_ets_baseline_panel(
    history: np.ndarray,
    horizon: int,
    seasonal_lag: int,
    series_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit statsmodels ETSModel defaults and return fitted, residual, and forecast panels."""
    del seasonal_lag

    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    values = np.asarray(history, dtype=float)
    if values.ndim != 2:
        raise ValueError("history must have shape (T, p).")
    if values.shape[0] < 2:
        raise ValueError("statsmodels ETS baseline requires at least two training observations.")
    if values.shape[1] != len(series_ids):
        raise ValueError("history column count must match series_ids.")

    config = statsmodels_ets_config()
    fitted_cols = []
    forecast_cols = []
    variance_cols = []
    for idx in range(values.shape[1]):
        series = pd.Series(values[:, idx], index=pd.RangeIndex(values.shape[0]))
        result = ETSModel(series).fit(disp=False)
        fitted_cols.append(np.asarray(result.fittedvalues, dtype=float))
        forecast_cols.append(np.asarray(result.forecast(int(horizon)), dtype=float))
        variance_cols.append(_statsmodels_prediction_variance(result, start=values.shape[0], horizon=horizon))

    fitted = np.column_stack(fitted_cols)
    forecast = np.column_stack(forecast_cols)
    forecast_variance = np.column_stack(variance_cols)
    return {
        "fitted": fitted,
        "residuals": values - fitted,
        "forecast": forecast,
        "forecast_variance": forecast_variance,
        "config": config,
    }


def seasonal_ets_baseline_panel(
    history: np.ndarray,
    horizon: int,
    seasonal_lag: int,
    series_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit a fixed additive seasonal ETSModel baseline."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    values = np.asarray(history, dtype=float)
    if values.ndim != 2:
        raise ValueError("history must have shape (T, p).")
    if values.shape[0] <= 2 * int(seasonal_lag):
        raise ValueError("seasonal ETS baseline requires more than two seasonal cycles.")
    if values.shape[1] != len(series_ids):
        raise ValueError("history column count must match series_ids.")

    config = seasonal_ets_config(seasonal_lag)
    fitted_cols = []
    forecast_cols = []
    variance_cols = []
    for idx in range(values.shape[1]):
        series = pd.Series(values[:, idx], index=pd.RangeIndex(values.shape[0]))
        result = ETSModel(
            series,
            error=config["error"],
            trend=config["trend"],
            damped_trend=config["damped_trend"],
            seasonal=config["seasonal"],
            seasonal_periods=config["seasonal_periods"],
            initialization_method=config["initialization_method"],
        ).fit(disp=False)
        fitted_cols.append(np.asarray(result.fittedvalues, dtype=float))
        forecast_cols.append(np.asarray(result.forecast(int(horizon)), dtype=float))
        variance_cols.append(_statsmodels_prediction_variance(result, start=values.shape[0], horizon=horizon))

    fitted = np.column_stack(fitted_cols)
    forecast = np.column_stack(forecast_cols)
    forecast_variance = np.column_stack(variance_cols)
    return {
        "fitted": fitted,
        "residuals": values - fitted,
        "forecast": forecast,
        "forecast_variance": forecast_variance,
        "config": config,
    }
