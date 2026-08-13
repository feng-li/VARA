"""Shared baseline forecast helpers for residual VARA workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


AUTO_ETS_ALIAS = "auto_ets"
AUTO_ETS_MODEL_SPECIFICATION = "ZZZ"
AUTO_ETS_DAMPED = None
AUTO_ETS_PHI = None
AUTO_ARIMA_ALIAS = "auto_arima"
NBEATS_ALIAS = "nbeats"
STATSMODELS_ETS_ALIAS = "statsmodels_ets"
SEASONAL_ETS_ALIAS = "seasonal_ets"
FORECAST_VARIANCE_LEVEL = 95
_FORECAST_VARIANCE_Z = 1.959963984540054
RESPONSE_NORMALIZATIONS = ("none", "training_zscore")


@dataclass(frozen=True, slots=True)
class NeuralModelConfig:
    """Prespecified neural benchmark settings shared with the legacy runner."""

    input_size: int = 0
    max_steps: int = 200
    hidden_size: int = 64
    num_layers: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    scaler_type: str = "robust"
    random_seed: int = 2026

    def __post_init__(self) -> None:
        if isinstance(self.input_size, (bool, np.bool_)) or int(self.input_size) != self.input_size:
            raise TypeError("input_size must be an integer.")
        if int(self.input_size) < 0:
            raise ValueError("input_size must be nonnegative.")
        for name in ("max_steps", "hidden_size", "num_layers", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or int(value) != value:
                raise TypeError(f"{name} must be an integer.")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
        if not np.isfinite(self.learning_rate) or float(self.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be finite and positive.")
        if not str(self.scaler_type).strip():
            raise ValueError("scaler_type must be nonempty.")

    def legacy_config(self) -> dict[str, Any]:
        """Return names used by existing benchmark configuration files."""
        return {
            "deepar_input_size": int(self.input_size),
            "deepar_max_steps": int(self.max_steps),
            "deepar_hidden_size": int(self.hidden_size),
            "deepar_lstm_layers": int(self.num_layers),
            "deepar_batch_size": int(self.batch_size),
            "deepar_learning_rate": float(self.learning_rate),
            "deepar_scaler_type": str(self.scaler_type),
            "deepar_random_seed": int(self.random_seed),
            "nbeats_input_size": int(self.input_size),
            "nbeats_max_steps": int(self.max_steps),
            "nbeats_hidden_size": int(self.hidden_size),
            "nbeats_num_layers": int(self.num_layers),
            "nbeats_batch_size": int(self.batch_size),
            "nbeats_learning_rate": float(self.learning_rate),
            "nbeats_scaler_type": str(self.scaler_type),
            "nbeats_random_seed": int(self.random_seed),
        }


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


def auto_arima_config(seasonal_lag: int) -> dict[str, Any]:
    """Return the existing StatsForecast AutoARIMA specification."""
    return {
        "model": "AutoARIMA",
        "implementation": "statsforecast.models.AutoARIMA",
        "alias": AUTO_ARIMA_ALIAS,
        "season_length": max(1, int(seasonal_lag)),
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
    return frame.melt(
        id_vars="ds",
        var_name="unique_id",
        value_name="y",
    ).dropna(subset=["y"])


def _statsforecast_wide(
    frame: pd.DataFrame,
    series_ids: Sequence[str],
    value_col: str,
    *,
    expected_index: Sequence[int] | np.ndarray | None = None,
) -> np.ndarray:
    wide = frame.pivot(index="ds", columns="unique_id", values=value_col).sort_index()
    if expected_index is not None:
        wide = wide.reindex(np.asarray(expected_index))
    return wide.loc[:, list(series_ids)].to_numpy(dtype=float)


def _interval_variance(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    var = ((hi - lo) / (2.0 * _FORECAST_VARIANCE_Z)) ** 2
    return np.where(np.isfinite(var) & (var >= 0.0), var, np.nan)


def _panel_forecast_horizon(values: np.ndarray, horizon: int) -> int:
    """Extend a common horizon enough to span trailing missing observations."""
    observed = np.isfinite(values)
    if np.any(observed.sum(axis=0) == 0):
        raise ValueError("Every training panel series needs one observed value.")
    reverse_last = np.argmax(observed[::-1], axis=0)
    return int(horizon) + int(reverse_last.max())


def normalize_training_panel(
    values: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each panel column using statistics from the supplied history."""
    history = np.asarray(values, dtype=float)
    if history.ndim != 2:
        raise ValueError("Training normalization expects a two-dimensional panel.")
    observed = np.isfinite(history)
    if np.any(observed.sum(axis=0) == 0):
        raise ValueError("Every training panel series needs one observed value.")
    training_mean = np.nanmean(history, axis=0)
    training_variance = np.nanvar(history, axis=0)
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
    forecast_horizon = _panel_forecast_horizon(values, int(horizon))
    forecast_long = sf.forecast(
        h=forecast_horizon,
        df=long,
        fitted=True,
        level=[FORECAST_VARIANCE_LEVEL],
    )
    fitted_long = sf.forecast_fitted_values()
    fitted_normalized = _statsforecast_wide(
        fitted_long,
        series_ids,
        AUTO_ETS_ALIAS,
        expected_index=np.arange(values.shape[0]),
    )
    forecast_normalized = _statsforecast_wide(
        forecast_long,
        series_ids,
        AUTO_ETS_ALIAS,
        expected_index=np.arange(values.shape[0], values.shape[0] + int(horizon)),
    )
    forecast_variance_normalized = _interval_variance(
        _statsforecast_wide(
            forecast_long,
            series_ids,
            f"{AUTO_ETS_ALIAS}-lo-{FORECAST_VARIANCE_LEVEL}",
            expected_index=np.arange(values.shape[0], values.shape[0] + int(horizon)),
        ),
        _statsforecast_wide(
            forecast_long,
            series_ids,
            f"{AUTO_ETS_ALIAS}-hi-{FORECAST_VARIANCE_LEVEL}",
            expected_index=np.arange(values.shape[0], values.shape[0] + int(horizon)),
        ),
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


def auto_arima_baseline_panel(
    history: np.ndarray,
    horizon: int,
    seasonal_lag: int,
    series_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit the existing StatsForecast AutoARIMA benchmark to a panel."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoARIMA

    values = np.asarray(history, dtype=float)
    if values.ndim != 2:
        raise ValueError("history must have shape (T, p).")
    if values.shape[0] < 3:
        raise ValueError("AutoARIMA requires at least three training observations.")
    if values.shape[1] != len(series_ids):
        raise ValueError("history column count must match series_ids.")
    if int(horizon) < 1:
        raise ValueError("horizon must be positive.")

    config = auto_arima_config(seasonal_lag)
    model = AutoARIMA(
        season_length=config["season_length"],
        alias=AUTO_ARIMA_ALIAS,
    )
    forecast_horizon = _panel_forecast_horizon(values, int(horizon))
    forecast_long = StatsForecast(
        models=[model],
        freq=config["freq"],
        n_jobs=1,
    ).forecast(
        h=forecast_horizon,
        df=_statsforecast_long_from_panel(values, series_ids),
        level=[FORECAST_VARIANCE_LEVEL],
    )
    forecast = _statsforecast_wide(
        forecast_long,
        series_ids,
        AUTO_ARIMA_ALIAS,
        expected_index=np.arange(values.shape[0], values.shape[0] + int(horizon)),
    )
    lower_name = f"{AUTO_ARIMA_ALIAS}-lo-{FORECAST_VARIANCE_LEVEL}"
    upper_name = f"{AUTO_ARIMA_ALIAS}-hi-{FORECAST_VARIANCE_LEVEL}"
    if lower_name in forecast_long.columns and upper_name in forecast_long.columns:
        variance = _interval_variance(
            _statsforecast_wide(
                forecast_long,
                series_ids,
                lower_name,
                expected_index=np.arange(values.shape[0], values.shape[0] + int(horizon)),
            ),
            _statsforecast_wide(
                forecast_long,
                series_ids,
                upper_name,
                expected_index=np.arange(values.shape[0], values.shape[0] + int(horizon)),
            ),
        )
    else:
        variance = np.full(forecast.shape, np.nan, dtype=float)
    return {
        "forecast": forecast,
        "forecast_variance": variance,
        "config": config,
    }


def _nbeats_stack_config(
    horizon: int,
    hidden_size: int,
    num_layers: int,
) -> tuple[list[str], list[int], list[list[int]]]:
    layers = [int(hidden_size)] * max(int(num_layers), 2)
    if int(horizon) <= 1:
        return ["identity"], [1], [layers.copy()]
    return (
        ["identity", "trend", "seasonality"],
        [1, 1, 1],
        [layers.copy(), layers.copy(), layers.copy()],
    )


def nbeats_baseline_panel(
    history: np.ndarray,
    horizon: int,
    seasonal_lag: int,
    series_ids: Sequence[str],
    *,
    neural_config: NeuralModelConfig = NeuralModelConfig(),
) -> dict[str, Any]:
    """Fit the existing pooled NeuralForecast N-BEATS benchmark."""
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS
    except Exception as exc:
        raise RuntimeError(
            "N-BEATS requires neuralforecast and its PyTorch dependencies."
        ) from exc

    values = np.asarray(history, dtype=float)
    if values.ndim != 2:
        raise ValueError("history must have shape (T, p).")
    if values.shape[1] != len(series_ids):
        raise ValueError("history column count must match series_ids.")
    requested_horizon = int(horizon)
    if requested_horizon < 1:
        raise ValueError("horizon must be positive.")
    h = _panel_forecast_horizon(values, requested_horizon)
    input_size = (
        int(neural_config.input_size)
        if int(neural_config.input_size) > 0
        else int(max(3 * h, 2 * int(seasonal_lag), 1))
    )
    if values.shape[0] <= input_size:
        raise ValueError("N-BEATS history must exceed its input_size.")

    try:
        import torch

        torch.set_num_threads(1)
    except Exception:
        pass
    stack_types, n_blocks, mlp_units = _nbeats_stack_config(
        h,
        neural_config.hidden_size,
        neural_config.num_layers,
    )
    model = NBEATS(
        h=h,
        input_size=input_size,
        max_steps=int(neural_config.max_steps),
        learning_rate=float(neural_config.learning_rate),
        batch_size=int(neural_config.batch_size),
        scaler_type=str(neural_config.scaler_type),
        random_seed=int(neural_config.random_seed),
        alias=NBEATS_ALIAS,
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
        stack_types=stack_types,
        n_blocks=n_blocks,
        mlp_units=mlp_units,
    )
    long = _statsforecast_long_from_panel(values, series_ids)
    neural_forecast = NeuralForecast(models=[model], freq=1)
    neural_forecast.fit(
        df=long,
        verbose=False,
    )
    forecast_long = neural_forecast.predict(verbose=False)
    forecast = _statsforecast_wide(
        forecast_long.reset_index(),
        series_ids,
        NBEATS_ALIAS,
        expected_index=np.arange(
            values.shape[0],
            values.shape[0] + requested_horizon,
        ),
    )
    return {
        "forecast": forecast,
        "forecast_variance": np.full(forecast.shape, np.nan, dtype=float),
        "config": {
            "model": "N-BEATS",
            "implementation": "neuralforecast.models.NBEATS",
            "alias": NBEATS_ALIAS,
            "season_length": int(seasonal_lag),
            "input_size": input_size,
            **asdict(neural_config),
        },
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
