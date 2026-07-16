#!/usr/bin/env python3
"""Run VARA forecasting on panel time-series data."""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd

from vara.forecast_baselines import (
    auto_ets_baseline_panel,
    auto_ets_config,
    seasonal_ets_baseline_panel,
    seasonal_ets_config,
    statsmodels_ets_baseline_panel,
    statsmodels_ets_config,
)
from vara.forecast_data import load_panel_data
from vara.forecast_metrics import forecast_metric_tables

BASE_SUFFIX = "All"
DETAILED_CODE_DEPTH = 3
TOURISM_TSFEATURES = (
    "acf_features",
    "heterogeneity",
    "series_length",
    "arch_stat",
    "holt_parameters",
    "sparsity",
    "count_entropy",
    "hurst",
    "stability",
    "crossing_points",
    "hw_parameters",
    "stl_features",
    "entropy",
    "intervals",
    "unitroot_kpss",
    "flat_spots",
    "lumpiness",
    "unitroot_pp",
    "frequency",
    "nonlinearity",
    "guerrero",
    "pacf_features",
)
DEFAULT_TOURISM_TSFEATURES = ",".join(TOURISM_TSFEATURES)
TOURISM_FEATURE_FUNCTIONS = "hierarchy,seasonal_profile,recent_dynamics"
SEASONAL_CORR_FEATURE_FUNCTIONS = (
    "seasonal_naive_residual_correlation,seasonal_profile_pca,recent_dynamics,hierarchy"
)
SEASONAL_PREDICTIVE_FEATURE_FUNCTIONS = "seasonal_diff_predictive_corr_lag1_lag12"
FORECAST_SCENARIOS = ("direct", "seasonal-naive-residual", "auto_ets", "statsmodels_ets", "seasonal_ets")
FORECAST_MODES = ("recursive", "rolling-origin", "closed-loop")
FORECAST_MODE_ALIASES = {
    "rolling_origin": "rolling-origin",
    "closed_loop": "closed-loop",
}
FEATURE_SOURCES = (
    "file",
    "tsfeatures",
    "tourism",
    "hybrid",
    "seasonal-corr",
    "seasonal-predictive",
)
SCORE_FEATURE_SOURCES = {"seasonal-predictive"}
TOPK_ATTENTION_MODES = {"topk", "score-topk"}
SCORE_ATTENTION_MODES = {"score", "score-topk"}
ATTENTION_MODES = ("dense", "topk", "score", "score-topk")
FEATURE_UPDATE_POLICIES = ("exact", "periodic")
Y_TRANSFORMS = (
    "none",
    "center",
    "standardize",
    "diff",
    "log1p-center",
    "log1p-standardize",
    "log1p-diff",
    "log1p-seasonal-diff",
)
FEATURE_SOURCE_TRANSFORM_LABELS = {
    "file": "prepared_feature_file",
    "tourism": "raw_tourism_panel",
    "seasonal-corr": "log1p_seasonal_naive_residuals",
    "seasonal-predictive": "log1p_seasonal_diff_predictive_scores",
}
FEATURE_SOURCE_FUNCTION_LABELS = {
    "file": "prepared_feature_file",
    "tourism": TOURISM_FEATURE_FUNCTIONS,
    "seasonal-corr": SEASONAL_CORR_FEATURE_FUNCTIONS,
    "seasonal-predictive": SEASONAL_PREDICTIVE_FEATURE_FUNCTIONS,
}
META_COLUMNS = ("series_id", "transform", "n_obs", "train_start", "train_end")


@dataclass(frozen=True)
class VARAForecastConfig:
    data_dir: Path = Path("data/tourism")
    output_dir: Path = Path("outputs/vara_forecast")
    data_path: Path | None = None
    data_format: str = "auto"
    date_col: str = "date"
    series_col: str = "series_id"
    value_col: str = "value"
    max_series: int = 0
    series_depth: int | None = DETAILED_CODE_DEPTH
    series_suffix: str | None = BASE_SUFFIX
    horizon: int = 12
    grid_size: int = 7
    ridge: float = 1e-2
    forecast_scenario: str = "direct"
    residual_transform: str = "center"
    attention_mode: str = "dense"
    attention_top_k: int | None = None
    seasonal_lag: int = 12
    feature_freq: int = 12
    feature_functions: str | Sequence[str] = DEFAULT_TOURISM_TSFEATURES
    feature_source: str = "tsfeatures"
    feature_path: Path | None = None
    feature_update_policy: str = "exact"
    feature_update_interval: int = 1
    forecast_mode: str = "recursive"
    y_transform: str = "log1p-seasonal-diff"
    ray_num_cpus: int = 1
    max_cpus_per_task: int = 1
    ray_debug: bool = False
    clip_nonnegative: bool = False
    residual_shrinkage: float = 1.0
    residual_shrinkage_file: Path | None = None

    @property
    def input_path(self) -> Path:
        return self.data_path if self.data_path is not None else self.data_dir / "tourism.csv"

    def run_kwargs(self) -> dict:
        return {
            "data_dir": self.data_dir,
            "max_series": self.max_series,
            "horizon": self.horizon,
            "grid_size": self.grid_size,
            "y_transform": self.y_transform,
            "seasonal_lag": self.seasonal_lag,
            "ridge": self.ridge,
            "data_path": self.data_path,
            "data_format": self.data_format,
            "date_col": self.date_col,
            "series_col": self.series_col,
            "value_col": self.value_col,
            "series_depth": self.series_depth,
            "series_suffix": self.series_suffix,
            "feature_freq": self.feature_freq,
            "feature_functions": self.feature_functions,
            "feature_source": self.feature_source,
            "feature_path": self.feature_path,
            "feature_update_policy": self.feature_update_policy,
            "feature_update_interval": self.feature_update_interval,
            "attention_mode": self.attention_mode,
            "attention_top_k": self.attention_top_k,
            "forecast_scenario": self.forecast_scenario,
            "residual_transform": self.residual_transform,
            "ray_num_cpus": self.ray_num_cpus,
            "max_cpus_per_task": self.max_cpus_per_task,
            "ray_debug": self.ray_debug,
            "clip_nonnegative": self.clip_nonnegative,
            "residual_shrinkage": self.residual_shrinkage,
            "residual_shrinkage_file": self.residual_shrinkage_file,
        }


DEFAULT_VARA_FORECAST_CONFIG = VARAForecastConfig()

_JNP = None
_VARAForecaster = None


def _normalize_forecast_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("_", "-")
    normalized = FORECAST_MODE_ALIASES.get(normalized, normalized)
    if normalized not in set(FORECAST_MODES):
        valid = ", ".join(FORECAST_MODES)
        raise ValueError(f"forecast_mode must be one of {valid}. Got {mode!r}.")
    return normalized


def _parse_forecast_modes(value: str) -> list[str]:
    modes = [_normalize_forecast_mode(part) for part in str(value).split(",") if part.strip()]
    if not modes:
        raise ValueError("forecast_mode must contain at least one mode.")
    out = []
    seen = set()
    for mode in modes:
        if mode not in seen:
            out.append(mode)
            seen.add(mode)
    return out


def _runtime_imports():
    global _JNP, _VARAForecaster
    if _JNP is None or _VARAForecaster is None:
        import jax.numpy as jnp
        from vara.forecast import VARAForecaster

        _JNP = jnp
        _VARAForecaster = VARAForecaster
    return _JNP, _VARAForecaster


def _print_progress(message: str) -> None:
    print(f"[{pd.Timestamp.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def _fit_y_transform(train: np.ndarray, method: str, seasonal_lag: int) -> tuple[np.ndarray, dict]:
    if method == "none":
        base = np.asarray(train, dtype=float)
        mean = np.zeros((1, base.shape[1]), dtype=float)
        scale = np.ones_like(mean)
        return base, {"method": method, "mean": mean, "scale": scale, "model_scale": "original"}

    if method in {"diff", "log1p-diff"}:
        if train.shape[0] <= 1:
            raise ValueError("Training data must have at least two rows for first differences.")
        history = np.log1p(np.maximum(train, 0.0)) if method == "log1p-diff" else np.asarray(train, dtype=float)
        base = history[1:] - history[:-1]
        mean = base.mean(axis=0, keepdims=True)
        scale = base.std(axis=0, keepdims=True)
        scale = np.where(scale > 0.0, scale, 1.0)
        return (base - mean) / scale, {
            "method": method,
            "mean": mean,
            "scale": scale,
            "history": history,
            "model_scale": "standardized",
        }

    if method == "log1p-seasonal-diff":
        if train.shape[0] <= seasonal_lag:
            raise ValueError("Training data must have more rows than seasonal_lag.")
        log_train = np.log1p(np.maximum(train, 0.0))
        base = log_train[seasonal_lag:] - log_train[:-seasonal_lag]
        mean = base.mean(axis=0, keepdims=True)
        scale = base.std(axis=0, keepdims=True)
        scale = np.where(scale > 0.0, scale, 1.0)
        return (base - mean) / scale, {
            "method": method,
            "mean": mean,
            "scale": scale,
            "seasonal_lag": int(seasonal_lag),
            "log_history": log_train,
            "model_scale": "standardized",
        }

    if method == "center":
        base = train
        mean = base.mean(axis=0, keepdims=True)
        scale = np.ones_like(mean)
        return (base - mean), {"method": method, "mean": mean, "scale": scale, "model_scale": "centered"}
    if method == "standardize":
        base = train
    elif method == "log1p-center":
        base = np.log1p(np.maximum(train, 0.0))
    elif method == "log1p-standardize":
        base = np.log1p(np.maximum(train, 0.0))
    else:
        raise ValueError("Unknown y_transform.")

    mean = base.mean(axis=0, keepdims=True)
    if method.endswith("standardize"):
        scale = base.std(axis=0, keepdims=True)
        scale = np.where(scale > 0.0, scale, 1.0)
    else:
        scale = np.ones_like(mean)
    return (base - mean) / scale, {"method": method, "mean": mean, "scale": scale, "model_scale": "standardized"}


def _unscale_y_transform(Y: np.ndarray, transform: dict) -> np.ndarray:
    return Y * transform["scale"] + transform["mean"]


def _inverse_y_transform(Y: np.ndarray, transform: dict) -> np.ndarray:
    method = str(transform["method"])
    out = _unscale_y_transform(Y, transform)
    if method in {"diff", "log1p-diff"}:
        history = [row.copy() for row in np.asarray(transform["history"], dtype=float)]
        forecasts = []
        for first_diff in out:
            forecast = history[-1] + first_diff
            history.append(forecast)
            forecasts.append(forecast)
        out = np.vstack(forecasts)
        if method == "log1p-diff":
            return np.maximum(np.expm1(out), 0.0)
        return out
    if method == "log1p-seasonal-diff":
        seasonal_lag = int(transform["seasonal_lag"])
        history = [row.copy() for row in np.asarray(transform["log_history"], dtype=float)]
        forecasts = []
        for seasonal_diff in out:
            log_forecast = history[-seasonal_lag] + seasonal_diff
            history.append(log_forecast)
            forecasts.append(log_forecast)
        return np.maximum(np.expm1(np.vstack(forecasts)), 0.0)
    if str(transform["method"]).startswith("log1p"):
        out = np.maximum(np.expm1(out), 0.0)
    return out


def _seasonal_naive_forecast(history: np.ndarray, horizon: int, seasonal_lag: int) -> np.ndarray:
    if history.shape[0] < seasonal_lag:
        raise ValueError("History must have at least seasonal_lag rows for seasonal naive forecasting.")
    values = [row.copy() for row in np.asarray(history, dtype=float)]
    forecasts = []
    for _ in range(int(horizon)):
        y_hat = values[-seasonal_lag].copy()
        values.append(y_hat)
        forecasts.append(y_hat)
    return np.vstack(forecasts)


def _seasonal_naive_residual_panel(history: np.ndarray, seasonal_lag: int) -> np.ndarray:
    if history.shape[0] <= seasonal_lag:
        raise ValueError("History must have more rows than seasonal_lag for residual correction.")
    values = np.asarray(history, dtype=float)
    return values[seasonal_lag:] - values[:-seasonal_lag]


def _baseline_config_for_scenario(forecast_scenario: str, seasonal_lag: int) -> dict:
    if forecast_scenario == "auto_ets":
        return auto_ets_config(seasonal_lag)
    if forecast_scenario == "statsmodels_ets":
        return statsmodels_ets_config()
    if forecast_scenario == "seasonal_ets":
        return seasonal_ets_config(seasonal_lag)
    if forecast_scenario == "seasonal-naive-residual":
        return {"model": "SeasonalNaive", "season_length": int(seasonal_lag)}
    return {}


def _feature_metadata_path(feature_path: Path) -> Path:
    return Path(feature_path).with_suffix(Path(feature_path).suffix + ".metadata.json")


def _validate_baseline_feature_metadata(
    *,
    feature_path: Path | None,
    forecast_scenario: str,
    seasonal_lag: int,
) -> dict:
    if forecast_scenario != "auto_ets" or feature_path is None:
        return {"checked": False, "metadata_path": ""}

    metadata_path = _feature_metadata_path(feature_path)
    out = {"checked": False, "metadata_path": str(metadata_path)}
    if not metadata_path.exists():
        warnings.warn(
            "AutoETS forecast recovery cannot verify the feature file baseline because "
            f"metadata is missing: {metadata_path}",
            RuntimeWarning,
        )
        return out

    metadata = json.loads(metadata_path.read_text())
    out.update({
        "checked": True,
        "filter_method": str(metadata.get("filter_method", "")),
        "seasonal_lag": metadata.get("seasonal_lag", ""),
    })
    if str(metadata.get("filter_method", "")) != "auto_ets":
        raise ValueError(
            "--forecast-scenario auto_ets requires an auto_ets-filtered feature file. "
            f"Got filter_method={metadata.get('filter_method')!r} in {metadata_path}."
        )
    if int(metadata.get("seasonal_lag", -1)) != int(seasonal_lag):
        raise ValueError(
            "AutoETS feature metadata seasonal_lag does not match forecast seasonal_lag: "
            f"{metadata.get('seasonal_lag')} != {seasonal_lag}."
        )

    expected_config = auto_ets_config(seasonal_lag)
    actual_config = metadata.get("baseline_config", {}) or {}
    for key in ("model", "alias", "season_length", "freq"):
        if key in actual_config and actual_config[key] != expected_config[key]:
            raise ValueError(
                "AutoETS feature metadata baseline_config does not match forecast config: "
                f"{key}={actual_config[key]!r} != {expected_config[key]!r}."
            )
    fallback = actual_config.get("fallback", "")
    if fallback not in ("", None):
        raise ValueError(
            "AutoETS feature metadata should not specify a fallback model; "
            "use filter_method=seasonal_naive as a separate filter instead."
        )
    return out


def _residual_baseline_name(forecast_scenario: str) -> str:
    if forecast_scenario == "seasonal-naive-residual":
        return "seasonal_naive"
    if forecast_scenario == "auto_ets":
        return "auto_ets"
    if forecast_scenario == "statsmodels_ets":
        return "statsmodels_ets"
    if forecast_scenario == "seasonal_ets":
        return "seasonal_ets"
    return ""


def _baseline_forecast_residual_and_variance(
    history: np.ndarray,
    horizon: int,
    seasonal_lag: int,
    series_ids: list[str],
    forecast_scenario: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if forecast_scenario == "seasonal-naive-residual":
        forecast = _seasonal_naive_forecast(history, horizon, seasonal_lag)
        residuals = _seasonal_naive_residual_panel(history, seasonal_lag)
        return forecast, residuals, np.zeros_like(forecast, dtype=float)
    if forecast_scenario == "auto_ets":
        baseline = auto_ets_baseline_panel(history, horizon, seasonal_lag, series_ids)
        return baseline["forecast"], baseline["residuals"], baseline["forecast_variance"]
    if forecast_scenario == "statsmodels_ets":
        baseline = statsmodels_ets_baseline_panel(history, horizon, seasonal_lag, series_ids)
        return baseline["forecast"], baseline["residuals"], baseline["forecast_variance"]
    if forecast_scenario == "seasonal_ets":
        baseline = seasonal_ets_baseline_panel(history, horizon, seasonal_lag, series_ids)
        return baseline["forecast"], baseline["residuals"], baseline["forecast_variance"]
    raise ValueError(f"Unknown residual forecast_scenario: {forecast_scenario!r}.")


def _baseline_forecast_and_residual_panel(
    history: np.ndarray,
    horizon: int,
    seasonal_lag: int,
    series_ids: list[str],
    forecast_scenario: str,
) -> tuple[np.ndarray, np.ndarray]:
    forecast, residuals, _ = _baseline_forecast_residual_and_variance(
        history,
        horizon,
        seasonal_lag,
        series_ids,
        forecast_scenario,
    )
    return forecast, residuals


def _inverse_y_variance(variance: np.ndarray, forecast_original_scale: np.ndarray, transform: dict) -> np.ndarray:
    variance = np.asarray(variance, dtype=float)
    scale = np.asarray(transform["scale"], dtype=float)
    out = variance * (scale ** 2)
    if str(transform["method"]).startswith("log1p"):
        out = out * np.maximum(np.asarray(forecast_original_scale, dtype=float) + 1.0, 0.0) ** 2
    return np.where(np.isfinite(out) & (out >= 0.0), out, np.nan)


def _one_step_residual_forecast_variance(forecaster, forecast_original_scale: np.ndarray, transform: dict) -> np.ndarray:
    sigma2_hat = getattr(forecaster, "sigma2_hat", None)
    if sigma2_hat is None:
        return np.full_like(forecast_original_scale, np.nan, dtype=float)
    variance = np.asarray(sigma2_hat, dtype=float)[None, :]
    return _inverse_y_variance(variance, forecast_original_scale, transform)


def _combine_forecast_variance(baseline_variance: np.ndarray, residual_variance: np.ndarray) -> np.ndarray:
    baseline_variance = np.asarray(baseline_variance, dtype=float)
    residual_variance = np.asarray(residual_variance, dtype=float)
    if baseline_variance.shape != residual_variance.shape:
        raise ValueError(
            f"Baseline variance shape {baseline_variance.shape} does not match "
            f"residual variance shape {residual_variance.shape}."
        )
    return baseline_variance + residual_variance


def _series_code(series_id: str, suffix: str = BASE_SUFFIX) -> str:
    return series_id[: -len(suffix)] if suffix and series_id.endswith(suffix) else series_id


def _series_depth(series_id: str, suffix: str = BASE_SUFFIX) -> int:
    code = _series_code(series_id, suffix)
    return 0 if code == "Total" else len(code)


def _base_series_columns(
    wide: pd.DataFrame,
    series_depth: int | None = DEFAULT_VARA_FORECAST_CONFIG.series_depth,
    series_suffix: str | None = DEFAULT_VARA_FORECAST_CONFIG.series_suffix,
) -> list[str]:
    cols = [c for c in wide.columns if c != "date"]
    if series_suffix:
        cols = [c for c in cols if c.endswith(series_suffix)]
    if series_depth is not None and series_suffix:
        cols = [c for c in cols if _series_depth(c, series_suffix) == series_depth]
    if not cols:
        suffix_msg = f" ending with {series_suffix!r}" if series_suffix else ""
        raise ValueError(f"No series columns{suffix_msg} found for series_depth={series_depth}.")
    return cols


def _feature_transform_from_y_transform(y_transform: str) -> str:
    if y_transform in {"none", "center", "standardize"}:
        return "level"
    if y_transform in {"log1p-center", "log1p-standardize"}:
        return "log1p"
    return y_transform


def _finalize_feature_matrix(
    feature_df: pd.DataFrame,
    *,
    label: str,
    meta_columns: Sequence[str] = META_COLUMNS,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    feature_cols = [col for col in feature_df.columns if col not in set(meta_columns)]
    if not feature_cols:
        raise ValueError(f"No {label} feature columns were produced.")
    Z = feature_df.loc[:, feature_cols].to_numpy(dtype=float)
    if not np.isfinite(Z).all():
        raise ValueError(f"{label} feature matrix contains non-finite values.")
    return Z, feature_df, feature_cols


def _tourism_tsfeature_matrix(
    wide: pd.DataFrame,
    series_ids: list[str],
    *,
    horizon: int,
    y_transform: str,
    seasonal_lag: int,
    feature_freq: int = 12,
    feature_functions: str | Sequence[str] = TOURISM_TSFEATURES,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    from vara.series_features import extract_series_tsfeatures

    feature_df = extract_series_tsfeatures(
        wide.loc[:, ["date", *series_ids]],
        date_col="date",
        series_cols=series_ids,
        holdout=horizon,
        transform=_feature_transform_from_y_transform(y_transform),
        seasonal_lag=seasonal_lag,
        freq=feature_freq,
        features=feature_functions,
        scale_series=True,
        standardize_output=True,
        clean=True,
        threads=1,
    )
    return _finalize_feature_matrix(feature_df, label="tourism tsfeature")


def _cyclic_month(month: int, prefix: str) -> dict[str, float]:
    angle = 2.0 * np.pi * (int(month) - 1) / 12.0
    return {f"{prefix}_sin": float(np.sin(angle)), f"{prefix}_cos": float(np.cos(angle))}


def _log_growth(log_values: np.ndarray, lag: int) -> float:
    if log_values.size <= lag:
        return 0.0
    return float(log_values[-1] - log_values[-1 - lag])


def _fill_nonfinite_by_column(values: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    if out.ndim == 1:
        out = out[:, None]
    for j in range(out.shape[1]):
        col = out[:, j]
        finite = np.isfinite(col)
        fill = float(np.mean(col[finite])) if finite.any() else 0.0
        col[~finite] = fill
        out[:, j] = col
    return out


def _standardize_array(values: np.ndarray) -> np.ndarray:
    out = _fill_nonfinite_by_column(values)
    scale = out.std(axis=0, keepdims=True)
    scale = np.where(scale > 0.0, scale, 1.0)
    return (out - out.mean(axis=0, keepdims=True)) / scale


def _orient_columns(scores: np.ndarray) -> np.ndarray:
    out = np.asarray(scores, dtype=float).copy()
    for j in range(out.shape[1]):
        pivot = int(np.argmax(np.abs(out[:, j])))
        if out[pivot, j] < 0.0:
            out[:, j] *= -1.0
    return out


def _pca_scores(values: np.ndarray, n_components: int) -> np.ndarray:
    X = _standardize_array(values)
    k = min(int(n_components), X.shape[0], X.shape[1])
    if k <= 0:
        return np.zeros((X.shape[0], 0), dtype=float)
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    return _orient_columns(X @ vt[:k].T)


def _code_ordinal_features(series_id: str) -> dict[str, float]:
    code = _series_code(series_id)
    if code == "Total":
        code = ""
    out = {"code_depth": float(len(code))}
    for pos in range(3):
        if pos < len(code):
            char = code[pos].upper()
            out[f"code_{pos + 1}_ord"] = float(ord(char) - ord("A") + 1) if "A" <= char <= "Z" else 0.0
        else:
            out[f"code_{pos + 1}_ord"] = 0.0
    return out


def _lagged_cross_correlation_matrix(values: np.ndarray, lag: int) -> np.ndarray:
    if values.shape[0] <= lag:
        return np.zeros((values.shape[1], values.shape[1]), dtype=float)
    target = _standardize_array(values[lag:])
    source = _standardize_array(values[:-lag])
    return (target.T @ source) / max(target.shape[0], 1)


def _tourism_specific_feature_matrix(
    wide: pd.DataFrame,
    series_ids: list[str],
    *,
    horizon: int,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    train = wide.iloc[: len(wide) - horizon].loc[:, ["date", *series_ids]].copy()
    dates = pd.to_datetime(train["date"])
    values = train.loc[:, series_ids].to_numpy(dtype=float)
    months = dates.dt.month.to_numpy()

    rows = []
    for j, series_id in enumerate(series_ids):
        code = _series_code(series_id)
        is_total = code == "Total"
        y = np.asarray(values[:, j], dtype=float)
        y = np.where(np.isfinite(y), y, np.nan)
        log_y = np.log1p(np.maximum(y, 0.0))
        mean_y = float(np.nanmean(y))
        denom = mean_y if abs(mean_y) > 1e-12 else np.nan

        month_means = pd.DataFrame({"month": months, "y": y}).groupby("month")["y"].mean()
        seasonal = np.asarray([month_means.get(m, np.nan) for m in range(1, 13)], dtype=float)
        if np.all(np.isnan(seasonal)):
            seasonal = np.zeros(12, dtype=float)
        fill_value = float(np.nanmean(seasonal)) if np.isfinite(np.nanmean(seasonal)) else 0.0
        seasonal = np.where(np.isfinite(seasonal), seasonal, fill_value)
        normalized_seasonal = seasonal / denom - 1.0 if np.isfinite(denom) else np.zeros(12, dtype=float)
        peak_month = int(np.argmax(seasonal) + 1)
        trough_month = int(np.argmin(seasonal) + 1)

        recent_diff = np.diff(log_y[-13:]) if log_y.size >= 2 else np.asarray([0.0])
        row = {
            "series_id": series_id,
            "transform": "tourism",
            "n_obs": int(len(train)),
            "train_start": str(dates.iloc[0].date()),
            "train_end": str(dates.iloc[-1].date()),
            "code_depth": 0 if is_total else len(code),
            "is_total": int(is_total),
            "code_1": "Total" if is_total else code[0],
            "code_2": "None" if is_total or len(code) < 2 else code[1],
            "code_3": "None" if is_total or len(code) < 3 else code[2],
            "seasonal_amplitude": float((np.max(seasonal) - np.min(seasonal)) / denom) if np.isfinite(denom) else 0.0,
            "growth_1": _log_growth(log_y, 1),
            "growth_3": _log_growth(log_y, 3),
            "growth_12": _log_growth(log_y, 12),
            "recent_vol_12": float(np.nanstd(recent_diff)),
            "recent_level_log": float(log_y[-1]) if log_y.size else 0.0,
            "recent_mean_log_12": float(np.nanmean(log_y[-12:])) if log_y.size else 0.0,
        }
        row.update(_cyclic_month(peak_month, "season_peak"))
        row.update(_cyclic_month(trough_month, "season_trough"))
        for month, value in enumerate(normalized_seasonal, start=1):
            row[f"season_m{month:02d}"] = float(value)
        rows.append(row)

    feature_df = pd.DataFrame(rows)
    feature_df = pd.get_dummies(feature_df, columns=["code_1", "code_2", "code_3"], dtype=float)
    from vara.series_features import clean_feature_frame

    feature_df = clean_feature_frame(feature_df, standardize_output=True, meta_columns=META_COLUMNS)
    return _finalize_feature_matrix(feature_df, label="tourism-specific")


def _tourism_seasonal_predictive_score_matrix(
    wide: pd.DataFrame,
    series_ids: list[str],
    *,
    horizon: int,
    seasonal_lag: int,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    train = wide.iloc[: len(wide) - horizon].loc[:, ["date", *series_ids]].copy()
    if len(train) <= 2 * seasonal_lag:
        raise ValueError("Training data must have more than 2 * seasonal_lag rows for seasonal-predictive attention.")

    dates = pd.to_datetime(train["date"])
    values = _fill_nonfinite_by_column(train.loc[:, series_ids].to_numpy(dtype=float))
    log_values = np.log1p(np.maximum(values, 0.0))
    residuals = log_values[seasonal_lag:] - log_values[:-seasonal_lag]

    score_1 = _lagged_cross_correlation_matrix(residuals, lag=1)
    score_12 = _lagged_cross_correlation_matrix(residuals, lag=seasonal_lag)
    scores = 0.5 * (score_1 + score_12)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(scores, 0.0)

    rows = []
    for j, series_id in enumerate(series_ids):
        row = {
            "series_id": series_id,
            "transform": "seasonal_predictive",
            "n_obs": int(len(train)),
            "train_start": str(dates.iloc[0].date()),
            "train_end": str(dates.iloc[-1].date()),
        }
        for source_id, value in zip(series_ids, scores[j], strict=True):
            row[f"score_from_{source_id}"] = float(value)
        rows.append(row)

    feature_df = pd.DataFrame(rows)
    return _finalize_feature_matrix(feature_df, label="seasonal-predictive score")


def _tourism_seasonal_corr_feature_matrix(
    wide: pd.DataFrame,
    series_ids: list[str],
    *,
    horizon: int,
    seasonal_lag: int,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    train = wide.iloc[: len(wide) - horizon].loc[:, ["date", *series_ids]].copy()
    if len(train) <= seasonal_lag:
        raise ValueError("Training data must have more rows than seasonal_lag for seasonal-corr features.")

    dates = pd.to_datetime(train["date"])
    months = dates.dt.month.to_numpy()
    values = _fill_nonfinite_by_column(train.loc[:, series_ids].to_numpy(dtype=float))
    log_values = np.log1p(np.maximum(values, 0.0))
    residuals = log_values[seasonal_lag:] - log_values[:-seasonal_lag]
    residual_z = _standardize_array(residuals)

    corr = (residual_z.T @ residual_z) / max(residual_z.shape[0], 1)
    corr = np.nan_to_num((corr + corr.T) / 2.0)
    eigvals, eigvecs = np.linalg.eigh(corr)
    order = np.argsort(eigvals)[::-1]
    n_corr = min(8, len(series_ids))
    corr_scores = eigvecs[:, order[:n_corr]] * np.sqrt(np.maximum(eigvals[order[:n_corr]], 0.0))
    corr_scores = _orient_columns(corr_scores)

    profile = np.zeros((len(series_ids), 12), dtype=float)
    center = np.mean(log_values, axis=0)
    for month in range(1, 13):
        mask = months == month
        month_mean = np.mean(log_values[mask, :], axis=0) if mask.any() else center
        profile[:, month - 1] = month_mean - center
    seasonal_scores = _pca_scores(profile, n_components=4)

    growth_1 = log_values[-1] - log_values[-2] if log_values.shape[0] > 1 else np.zeros(len(series_ids))
    growth_3 = log_values[-1] - log_values[-4] if log_values.shape[0] > 3 else np.zeros(len(series_ids))
    growth_12 = log_values[-1] - log_values[-13] if log_values.shape[0] > 12 else np.zeros(len(series_ids))
    recent_resid = residuals[-12:] if residuals.shape[0] else np.zeros((1, len(series_ids)))
    resid_vol_12 = np.std(recent_resid, axis=0)

    rows = []
    for j, series_id in enumerate(series_ids):
        row = {
            "series_id": series_id,
            "transform": "seasonal_corr",
            "n_obs": int(len(train)),
            "train_start": str(dates.iloc[0].date()),
            "train_end": str(dates.iloc[-1].date()),
            "growth_1": float(growth_1[j]),
            "growth_3": float(growth_3[j]),
            "growth_12": float(growth_12[j]),
            "resid_vol_12": float(resid_vol_12[j]),
            "recent_level_log": float(log_values[-1, j]),
        }
        row.update(_code_ordinal_features(series_id))
        for k in range(corr_scores.shape[1]):
            row[f"corr_embed_{k + 1:02d}"] = float(corr_scores[j, k])
        for k in range(seasonal_scores.shape[1]):
            row[f"seasonal_pc_{k + 1:02d}"] = float(seasonal_scores[j, k])
        rows.append(row)

    feature_df = pd.DataFrame(rows)
    from vara.series_features import clean_feature_frame

    feature_df = clean_feature_frame(feature_df, standardize_output=True, meta_columns=META_COLUMNS)
    return _finalize_feature_matrix(feature_df, label="seasonal-corr tourism")


def _tourism_feature_matrix(
    wide: pd.DataFrame,
    series_ids: list[str],
    *,
    horizon: int,
    feature_source: str,
    y_transform: str,
    seasonal_lag: int,
    feature_freq: int,
    feature_functions: str | Sequence[str],
    feature_path: Path | None = None,
    forecast_scenario: str | None = None,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    del forecast_scenario
    if feature_source == "file":
        if feature_path is None:
            raise ValueError("feature_path must be provided when feature_source='file'.")
        from vara.series_features import load_series_feature_matrix

        return load_series_feature_matrix(feature_path, series_ids, standardize_output=True)
    if feature_source == "tsfeatures":
        return _tourism_tsfeature_matrix(
            wide,
            series_ids,
            horizon=horizon,
            y_transform=y_transform,
            seasonal_lag=seasonal_lag,
            feature_freq=feature_freq,
            feature_functions=feature_functions,
        )
    if feature_source == "tourism":
        return _tourism_specific_feature_matrix(wide, series_ids, horizon=horizon)
    if feature_source == "seasonal-corr":
        return _tourism_seasonal_corr_feature_matrix(
            wide,
            series_ids,
            horizon=horizon,
            seasonal_lag=seasonal_lag,
        )
    if feature_source == "seasonal-predictive":
        return _tourism_seasonal_predictive_score_matrix(
            wide,
            series_ids,
            horizon=horizon,
            seasonal_lag=seasonal_lag,
        )
    if feature_source == "hybrid":
        _, tourism_df, tourism_cols = _tourism_specific_feature_matrix(wide, series_ids, horizon=horizon)
        _, tsfeature_df, tsfeature_cols = _tourism_tsfeature_matrix(
            wide,
            series_ids,
            horizon=horizon,
            y_transform=y_transform,
            seasonal_lag=seasonal_lag,
            feature_freq=feature_freq,
            feature_functions=feature_functions,
        )
        if list(tourism_df["series_id"]) != list(tsfeature_df["series_id"]):
            raise ValueError("Tourism and tsfeatures series order does not match.")

        feature_df = tourism_df.loc[:, [col for col in META_COLUMNS if col in tourism_df.columns]].copy()
        feature_df["transform"] = "hybrid"
        tourism_block = tourism_df.loc[:, tourism_cols].add_prefix("tourism__")
        tsfeature_block = tsfeature_df.loc[:, tsfeature_cols].add_prefix("tsfeatures__")
        feature_df = pd.concat([feature_df, tourism_block, tsfeature_block], axis=1)
        return _finalize_feature_matrix(feature_df, label="hybrid tourism")
    valid = ", ".join(repr(item) for item in FEATURE_SOURCES)
    raise ValueError(f"feature_source must be one of {valid}.")


def _new_forecaster(
    grid_size: int,
    ridge: float,
    attention_mode: str = "dense",
    attention_top_k: int | None = None,
    eta_bounds: tuple[float, float] = (-2.0, 2.0),
):
    _, VARAForecaster = _runtime_imports()
    return VARAForecaster(
        grid_size=grid_size,
        eta_bounds=eta_bounds,
        ridge=ridge,
        dense_threshold=2500,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
    )



def _feature_train_end(feature_df: pd.DataFrame, fallback) -> str:
    if "train_end" in feature_df.columns and len(feature_df) > 0:
        value = feature_df["train_end"].iloc[0]
        if pd.notna(value):
            return str(value)
    return str(fallback)


def _feature_source_step(step: int, feature_update_policy: str, feature_update_interval: int) -> tuple[int, bool]:
    if feature_update_policy == "exact":
        return int(step), True
    source_step = (int(step) // int(feature_update_interval)) * int(feature_update_interval)
    return source_step, source_step == int(step)


def _fit_vara_origin_step(
    wide: pd.DataFrame,
    Y_raw: np.ndarray,
    series_ids: list[str],
    *,
    step: int,
    horizon: int,
    Z: np.ndarray,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
    y_transform: str,
    seasonal_lag: int,
    grid_size: int,
    ridge: float,
    attention_mode: str,
    attention_top_k: int | None,
    forecast_scenario: str,
    residual_transform: str,
    feature_update_policy: str,
    feature_exact: bool,
    feature_source_origin: int,
    feature_source_train_end: str,
    origin_train: np.ndarray | None = None,
    train_end_override=None,
    eta_bounds: tuple[float, float] = (-2.0, 2.0),
) -> dict:
    jnp, _ = _runtime_imports()
    origin = int(step) + 1
    origin_end = len(wide) - int(horizon) + int(step)
    if origin_train is None:
        origin_train = Y_raw[:origin_end]
    else:
        origin_train = np.asarray(origin_train, dtype=float)
    baseline_one = np.zeros((1, len(series_ids)), dtype=float)
    baseline_variance_one = np.zeros_like(baseline_one, dtype=float)
    if forecast_scenario == "direct":
        model_train = origin_train
        model_transform = y_transform
    else:
        baseline_one, model_train, baseline_variance_one = _baseline_forecast_residual_and_variance(
            origin_train,
            1,
            seasonal_lag,
            series_ids,
            forecast_scenario,
        )
        model_transform = residual_transform

    Y_train, fitted_y_transform = _fit_y_transform(model_train, model_transform, seasonal_lag)
    forecaster = _new_forecaster(
        grid_size,
        ridge,
        attention_mode,
        attention_top_k,
        eta_bounds=eta_bounds,
    )
    forecaster.fit(jnp.asarray(Y_train), jnp.asarray(Z))
    forecast_model_scale = np.asarray(forecaster.predict(h=1))
    transformed_one = _unscale_y_transform(forecast_model_scale, fitted_y_transform)
    residual_one = _inverse_y_transform(forecast_model_scale, fitted_y_transform)
    residual_variance_one = _one_step_residual_forecast_variance(forecaster, residual_one, fitted_y_transform)
    forecast_one = residual_one if forecast_scenario == "direct" else baseline_one + residual_one
    forecast_variance_one = (
        residual_variance_one
        if forecast_scenario == "direct"
        else _combine_forecast_variance(baseline_variance_one, residual_variance_one)
    )

    forecast_date = str(wide["date"].iloc[origin_end])
    train_end = str(train_end_override if train_end_override is not None else wide["date"].iloc[origin_end - 1])
    origin_features = feature_df.copy()
    origin_features.insert(0, "origin", origin)
    origin_features.insert(1, "forecast_date", forecast_date)
    origin_features.insert(2, "feature_exact", bool(feature_exact))
    origin_features.insert(3, "feature_source_origin", int(feature_source_origin))
    origin_features.insert(4, "feature_source_train_end", str(feature_source_train_end))

    fit_diagnostics = forecaster.diagnostics()
    diag = dict(fit_diagnostics)
    diag.update({
        "origin": origin,
        "forecast_date": forecast_date,
        "train_end": train_end,
        "n_train": int(origin_train.shape[0]),
        "n_train_model": int(Y_train.shape[0]),
        "n_attention_feature_columns": len(feature_cols),
        "feature_update_policy": feature_update_policy,
        "feature_exact": bool(feature_exact),
        "feature_source_origin": int(feature_source_origin),
        "feature_source_train_end": str(feature_source_train_end),
    })
    return {
        "origin": origin,
        "forecast": forecast_one[0],
        "baseline": baseline_one[0],
        "residual": residual_one[0],
        "forecast_variance": forecast_variance_one[0],
        "baseline_variance": baseline_variance_one[0],
        "residual_variance": residual_variance_one[0],
        "forecast_transformed": transformed_one[0],
        "feature_df": origin_features,
        "diag": diag,
        "fit_diagnostics": fit_diagnostics,
        "n_train_model": int(Y_train.shape[0]),
        "feature_cols": list(feature_cols),
        "feature_exact": bool(feature_exact),
        "model_scale": fitted_y_transform["model_scale"],
    }


def _fit_vara_origin_remote(
    wide: pd.DataFrame,
    Y_raw: np.ndarray,
    series_ids: list[str],
    *,
    step: int,
    horizon: int,
    y_transform: str,
    seasonal_lag: int,
    feature_freq: int,
    feature_functions: str | Sequence[str],
    feature_source: str,
    feature_path: Path | None,
    feature_update_policy: str,
    feature_update_interval: int,
    grid_size: int,
    ridge: float,
    attention_mode: str,
    attention_top_k: int | None,
    forecast_scenario: str,
    residual_transform: str,
    ray_debug: bool,
    eta_bounds: tuple[float, float] = (-2.0, 2.0),
) -> dict:
    start = time.time()
    origin = int(step) + 1
    if ray_debug:
        _print_progress(f"[Ray VARA origin start] origin={origin}, horizon={horizon}")
    source_step, feature_exact = _feature_source_step(step, feature_update_policy, feature_update_interval)
    feature_horizon = int(horizon) - int(source_step)
    Z, feature_df, feature_cols = _tourism_feature_matrix(
        wide,
        series_ids,
        horizon=feature_horizon,
        feature_source=feature_source,
        y_transform=y_transform,
        seasonal_lag=seasonal_lag,
        feature_freq=feature_freq,
        feature_functions=feature_functions,
        feature_path=feature_path,
    )
    source_origin = int(source_step) + 1
    fallback_train_end = wide["date"].iloc[len(wide) - feature_horizon - 1]
    result = _fit_vara_origin_step(
        wide,
        Y_raw,
        series_ids,
        step=step,
        horizon=horizon,
        Z=Z,
        feature_df=feature_df,
        feature_cols=feature_cols,
        y_transform=y_transform,
        seasonal_lag=seasonal_lag,
        grid_size=grid_size,
        ridge=ridge,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        forecast_scenario=forecast_scenario,
        residual_transform=residual_transform,
        feature_update_policy=feature_update_policy,
        feature_exact=feature_exact,
        feature_source_origin=source_origin,
        feature_source_train_end=_feature_train_end(feature_df, fallback_train_end),
        eta_bounds=eta_bounds,
    )
    if ray_debug:
        _print_progress(f"[Ray VARA origin done] origin={origin}, elapsed_sec={time.time() - start:.2f}")
    return result


def _start_ray(ray_num_cpus: int, max_cpus_per_task: int):
    import ray

    ray_started_here = False
    if not ray.is_initialized():
        _print_progress(f"Starting Ray: ray_num_cpus={ray_num_cpus}, max_cpus_per_task={max_cpus_per_task}")
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=ray_num_cpus,
            _node_ip_address="127.0.0.1",
        )
        ray_started_here = True
        _print_progress("Ray started")
    available = ray.available_resources()
    available_cpus = float(available.get("CPU", 0.0))
    if available_cpus < float(max_cpus_per_task):
        raise RuntimeError(
            f"Ray has no available CPUs after initialization: {available}. "
            "Stop stale Ray workers with `ray stop --force` or reduce other Ray jobs."
        )
    return ray, ray_started_here


def _fit_vara_origins_with_ray(
    ray,
    wide: pd.DataFrame,
    Y_raw: np.ndarray,
    series_ids: list[str],
    *,
    horizon: int,
    max_cpus_per_task: int,
    y_transform: str,
    seasonal_lag: int,
    feature_freq: int,
    feature_functions: str | Sequence[str],
    feature_source: str,
    feature_path: Path | None,
    feature_update_policy: str,
    feature_update_interval: int,
    grid_size: int,
    ridge: float,
    attention_mode: str,
    attention_top_k: int | None,
    forecast_scenario: str,
    residual_transform: str,
    ray_debug: bool,
) -> list[dict]:
    remote_origin = ray.remote(num_cpus=max_cpus_per_task)(_fit_vara_origin_remote)
    wide_ref = ray.put(wide)
    Y_ref = ray.put(Y_raw)
    series_ref = ray.put(series_ids)
    futures = []
    submit_times = {}
    for step in range(int(horizon)):
        ref = remote_origin.remote(
            wide_ref,
            Y_ref,
            series_ref,
            step=step,
            horizon=horizon,
            y_transform=y_transform,
            seasonal_lag=seasonal_lag,
            feature_freq=feature_freq,
            feature_functions=feature_functions,
            feature_source=feature_source,
            feature_path=feature_path,
            feature_update_policy=feature_update_policy,
            feature_update_interval=feature_update_interval,
            grid_size=grid_size,
            ridge=ridge,
            attention_mode=attention_mode,
            attention_top_k=attention_top_k,
            forecast_scenario=forecast_scenario,
            residual_transform=residual_transform,
            ray_debug=ray_debug,
        )
        futures.append(ref)
        submit_times[ref] = time.time()

    _print_progress(f"Submitted Ray VARA rolling-origin tasks: origins={len(futures)}")
    pending = list(futures)
    completed = 0
    results = []
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=60.0)
        if not ready:
            now = time.time()
            slowest_wait = now - min(submit_times.values()) if submit_times else 0.0
            _print_progress(
                "Ray VARA wait: "
                f"done={completed}/{len(futures)}, pending={len(pending)}, "
                f"slowest_pending_sec={slowest_wait:.1f}, available={ray.available_resources()}"
            )
            continue
        for ref in ready:
            result = ray.get(ref)
            results.append(result)
            submit_times.pop(ref, None)
            completed += 1
            if ray_debug:
                _print_progress(f"Ray VARA collected: origin={result['origin']}, done={completed}/{len(futures)}")
    return sorted(results, key=lambda item: int(item["origin"]))


def _fit_vara_origins_closed_loop(
    wide: pd.DataFrame,
    Y_raw: np.ndarray,
    series_ids: list[str],
    *,
    horizon: int,
    y_transform: str,
    seasonal_lag: int,
    feature_freq: int,
    feature_functions: str | Sequence[str],
    feature_source: str,
    feature_path: Path | None,
    feature_update_policy: str,
    feature_update_interval: int,
    grid_size: int,
    ridge: float,
    attention_mode: str,
    attention_top_k: int | None,
    forecast_scenario: str,
    residual_transform: str,
) -> list[dict]:
    history_wide = wide.iloc[: len(wide) - horizon].loc[:, ["date", *series_ids]].copy().reset_index(drop=True)
    history_values = history_wide.loc[:, series_ids].to_numpy(dtype=float)
    cached_Z = None
    cached_feature_df = None
    cached_feature_cols = None
    cached_feature_source_origin = None
    cached_feature_source_train_end = None
    results = []

    for step in range(int(horizon)):
        source_step, feature_exact = _feature_source_step(step, feature_update_policy, feature_update_interval)
        if feature_exact or cached_Z is None:
            Z, feature_df, feature_cols = _tourism_feature_matrix(
                history_wide,
                series_ids,
                horizon=0,
                feature_source=feature_source,
                y_transform=y_transform,
                seasonal_lag=seasonal_lag,
                feature_freq=feature_freq,
                feature_functions=feature_functions,
                feature_path=feature_path,
            )
            cached_Z = Z
            cached_feature_df = feature_df.copy()
            cached_feature_cols = list(feature_cols)
            cached_feature_source_origin = source_step + 1
            cached_feature_source_train_end = _feature_train_end(feature_df, history_wide["date"].iloc[-1])
        else:
            Z = cached_Z
            feature_df = cached_feature_df.copy()
            feature_cols = list(cached_feature_cols)

        result = _fit_vara_origin_step(
            wide,
            Y_raw,
            series_ids,
            step=step,
            horizon=horizon,
            Z=Z,
            feature_df=feature_df,
            feature_cols=feature_cols,
            y_transform=y_transform,
            seasonal_lag=seasonal_lag,
            grid_size=grid_size,
            ridge=ridge,
            attention_mode=attention_mode,
            attention_top_k=attention_top_k,
            forecast_scenario=forecast_scenario,
            residual_transform=residual_transform,
            feature_update_policy=feature_update_policy,
            feature_exact=feature_exact,
            feature_source_origin=cached_feature_source_origin,
            feature_source_train_end=cached_feature_source_train_end,
            origin_train=history_values,
            train_end_override=history_wide["date"].iloc[-1],
        )
        results.append(result)

        forecast_date = wide["date"].iloc[len(wide) - horizon + step]
        next_row = {"date": forecast_date}
        next_row.update({series_id: float(value) for series_id, value in zip(series_ids, result["forecast"], strict=True)})
        history_wide = pd.concat([history_wide, pd.DataFrame([next_row])], ignore_index=True)
        history_values = np.vstack([history_values, result["forecast"]])

    return results


def _collect_origin_results(origin_results: list[dict]) -> dict:
    if not origin_results:
        raise ValueError("No VARA origin results were produced.")
    return {
        "forecast": np.vstack([result["forecast"] for result in origin_results]),
        "baseline": np.vstack([result["baseline"] for result in origin_results]),
        "residual": np.vstack([result["residual"] for result in origin_results]),
        "forecast_variance": np.vstack([result["forecast_variance"] for result in origin_results]),
        "baseline_variance": np.vstack([result["baseline_variance"] for result in origin_results]),
        "residual_variance": np.vstack([result["residual_variance"] for result in origin_results]),
        "forecast_transformed": np.vstack([result["forecast_transformed"] for result in origin_results]),
        "feature_frames": [result["feature_df"] for result in origin_results],
        "rolling_diagnostics": [result["diag"] for result in origin_results],
        "exact_feature_origins": [int(result["origin"]) for result in origin_results if result["feature_exact"]],
        "fit_diagnostics": origin_results[-1]["fit_diagnostics"],
        "fitted_y_transform": {"model_scale": origin_results[-1]["model_scale"]},
        "n_train_model_initial": int(origin_results[0]["n_train_model"]),
        "n_train_model_final": int(origin_results[-1]["n_train_model"]),
        "n_feature_columns": len(origin_results[-1]["feature_cols"]),
    }


def _write_wide_output(
    output_dir: Path,
    filename: str,
    values: np.ndarray,
    dates,
    series_ids: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=series_ids)
    frame.insert(0, "date", np.asarray(dates))
    frame.to_csv(output_dir / filename, index=False)
    return frame


def _load_residual_shrinkage_by_series(
    path: Path,
    series_ids: Sequence[str],
) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Residual shrinkage file does not exist: {path}")
    frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    required = {"series_id", "residual_shrinkage"}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Residual shrinkage file {path} is missing required columns: {missing_columns}"
        )

    frame = frame.copy()
    if frame["series_id"].isna().any():
        raise ValueError(f"Residual shrinkage file {path} contains missing series_id values.")
    frame["series_id"] = frame["series_id"].astype(str)
    duplicate_ids = sorted(
        frame.loc[frame["series_id"].duplicated(keep=False), "series_id"].unique()
    )
    if duplicate_ids:
        raise ValueError(
            f"Residual shrinkage file {path} contains duplicate series_id values: {duplicate_ids[:10]}"
        )

    try:
        frame["residual_shrinkage"] = pd.to_numeric(
            frame["residual_shrinkage"], errors="raise"
        ).astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Residual shrinkage file {path} contains nonnumeric residual_shrinkage values."
        ) from exc
    weights = frame["residual_shrinkage"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or np.any((weights < 0.0) | (weights > 1.0)):
        raise ValueError(
            f"Residual shrinkage file {path} must contain finite weights between 0 and 1."
        )

    selected_ids = [str(series_id) for series_id in series_ids]
    indexed = frame.set_index("series_id", drop=False)
    missing_ids = [series_id for series_id in selected_ids if series_id not in indexed.index]
    if missing_ids:
        raise ValueError(
            f"Residual shrinkage file {path} is missing selected series: {missing_ids[:10]}"
        )
    return indexed.loc[selected_ids].reset_index(drop=True)


def _apply_residual_shrinkage(
    baseline_forecast: np.ndarray,
    residual_forecast: np.ndarray,
    baseline_variance: np.ndarray,
    residual_variance: np.ndarray,
    residual_shrinkage: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = np.asarray(residual_shrinkage, dtype=float)
    if weights.ndim != 1 or weights.shape[0] != residual_forecast.shape[1]:
        raise ValueError(
            "residual_shrinkage must contain one weight for each forecast series; "
            f"got shape {weights.shape} for {residual_forecast.shape[1]} series."
        )
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("residual_shrinkage weights must be finite and nonnegative.")

    scale = weights[None, :]
    scaled_residual = residual_forecast * scale
    scaled_residual_variance = residual_variance * (scale ** 2)
    forecast = baseline_forecast + scaled_residual
    forecast_variance = baseline_variance + scaled_residual_variance
    return forecast, scaled_residual, forecast_variance, scaled_residual_variance


def _feature_function_label(feature_functions: str | Sequence[str]) -> str:
    if isinstance(feature_functions, str):
        return feature_functions
    return ",".join(str(item) for item in feature_functions)


def _feature_transform_label(feature_source: str, y_transform: str) -> str:
    if feature_source == "hybrid":
        return f"raw_tourism_panel+{_feature_transform_from_y_transform(y_transform)}"
    return FEATURE_SOURCE_TRANSFORM_LABELS.get(feature_source, _feature_transform_from_y_transform(y_transform))


def _feature_function_metric(feature_source: str, feature_functions: str | Sequence[str]) -> str:
    if feature_source == "hybrid":
        return f"{TOURISM_FEATURE_FUNCTIONS},{_feature_function_label(feature_functions)}"
    return FEATURE_SOURCE_FUNCTION_LABELS.get(feature_source, _feature_function_label(feature_functions))


def run_vara_forecast(
    data_dir: Path = DEFAULT_VARA_FORECAST_CONFIG.data_dir,
    output_dir: Path = DEFAULT_VARA_FORECAST_CONFIG.output_dir,
    max_series: int = DEFAULT_VARA_FORECAST_CONFIG.max_series,
    horizon: int = DEFAULT_VARA_FORECAST_CONFIG.horizon,
    grid_size: int = DEFAULT_VARA_FORECAST_CONFIG.grid_size,
    y_transform: str = DEFAULT_VARA_FORECAST_CONFIG.y_transform,
    seasonal_lag: int = DEFAULT_VARA_FORECAST_CONFIG.seasonal_lag,
    ridge: float = DEFAULT_VARA_FORECAST_CONFIG.ridge,
    data_path: Path | None = DEFAULT_VARA_FORECAST_CONFIG.data_path,
    data_format: str = DEFAULT_VARA_FORECAST_CONFIG.data_format,
    date_col: str = DEFAULT_VARA_FORECAST_CONFIG.date_col,
    series_col: str = DEFAULT_VARA_FORECAST_CONFIG.series_col,
    value_col: str = DEFAULT_VARA_FORECAST_CONFIG.value_col,
    series_depth: int | None = DEFAULT_VARA_FORECAST_CONFIG.series_depth,
    series_suffix: str | None = DEFAULT_VARA_FORECAST_CONFIG.series_suffix,
    feature_freq: int = DEFAULT_VARA_FORECAST_CONFIG.feature_freq,
    feature_functions: str | Sequence[str] = DEFAULT_VARA_FORECAST_CONFIG.feature_functions,
    feature_source: str = DEFAULT_VARA_FORECAST_CONFIG.feature_source,
    feature_path: Path | None = DEFAULT_VARA_FORECAST_CONFIG.feature_path,
    forecast_mode: str = DEFAULT_VARA_FORECAST_CONFIG.forecast_mode,
    feature_update_policy: str = DEFAULT_VARA_FORECAST_CONFIG.feature_update_policy,
    feature_update_interval: int = DEFAULT_VARA_FORECAST_CONFIG.feature_update_interval,
    attention_mode: str = DEFAULT_VARA_FORECAST_CONFIG.attention_mode,
    attention_top_k: int | None = DEFAULT_VARA_FORECAST_CONFIG.attention_top_k,
    forecast_scenario: str = DEFAULT_VARA_FORECAST_CONFIG.forecast_scenario,
    residual_transform: str = DEFAULT_VARA_FORECAST_CONFIG.residual_transform,
    ray_num_cpus: int = DEFAULT_VARA_FORECAST_CONFIG.ray_num_cpus,
    max_cpus_per_task: int = DEFAULT_VARA_FORECAST_CONFIG.max_cpus_per_task,
    ray_debug: bool = DEFAULT_VARA_FORECAST_CONFIG.ray_debug,
    clip_nonnegative: bool = DEFAULT_VARA_FORECAST_CONFIG.clip_nonnegative,
    residual_shrinkage: float = DEFAULT_VARA_FORECAST_CONFIG.residual_shrinkage,
    residual_shrinkage_file: Path | None = (
        DEFAULT_VARA_FORECAST_CONFIG.residual_shrinkage_file
    ),
) -> dict:
    if forecast_scenario not in set(FORECAST_SCENARIOS):
        raise ValueError(f"forecast_scenario must be one of {FORECAST_SCENARIOS}.")
    forecast_mode = _normalize_forecast_mode(forecast_mode)
    if residual_transform not in {"center", "standardize"}:
        raise ValueError("residual_transform must be 'center' or 'standardize'.")
    residual_shrinkage = float(residual_shrinkage)
    if not np.isfinite(residual_shrinkage) or residual_shrinkage < 0.0:
        raise ValueError("residual_shrinkage must be a finite nonnegative number.")
    residual_shrinkage_file = (
        None if residual_shrinkage_file is None else Path(residual_shrinkage_file)
    )
    if residual_shrinkage_file is not None and residual_shrinkage != 1.0:
        raise ValueError(
            "Use either residual_shrinkage or residual_shrinkage_file, not both."
        )
    if forecast_scenario == "direct" and (
        residual_shrinkage != 1.0 or residual_shrinkage_file is not None
    ):
        raise ValueError("residual_shrinkage is only defined for residual-correction scenarios.")
    input_feature_path = feature_path
    if input_feature_path is not None:
        if feature_source == "tsfeatures":
            feature_source = "file"
        elif feature_source != "file":
            raise ValueError("feature_path can only be used with feature_source='file' or the default feature_source.")
    if feature_source not in FEATURE_SOURCES:
        valid = ", ".join(repr(item) for item in FEATURE_SOURCES)
        raise ValueError(f"feature_source must be one of {valid}.")
    if attention_mode not in ATTENTION_MODES:
        valid = ", ".join(repr(item) for item in ATTENTION_MODES)
        raise ValueError(f"attention_mode must be one of {valid}.")
    if attention_mode in TOPK_ATTENTION_MODES and attention_top_k is None:
        raise ValueError("attention_top_k must be provided when attention_mode uses top-k.")
    if feature_source == "file" and input_feature_path is None:
        raise ValueError("feature_path must be provided when feature_source='file'.")
    if feature_source in SCORE_FEATURE_SOURCES and attention_mode not in SCORE_ATTENTION_MODES:
        raise ValueError("feature_source='seasonal-predictive' requires attention_mode='score' or 'score-topk'.")
    if feature_source not in SCORE_FEATURE_SOURCES and attention_mode in SCORE_ATTENTION_MODES:
        raise ValueError("score attention modes require feature_source='seasonal-predictive'.")
    if feature_update_policy not in FEATURE_UPDATE_POLICIES:
        valid = ", ".join(repr(item) for item in FEATURE_UPDATE_POLICIES)
        raise ValueError(f"feature_update_policy must be one of {valid}.")
    if int(feature_update_interval) < 1:
        raise ValueError("feature_update_interval must be at least 1.")
    feature_update_interval = int(feature_update_interval)
    if int(ray_num_cpus) < 1:
        raise ValueError("ray_num_cpus must be >= 1.")
    if int(max_cpus_per_task) < 1:
        raise ValueError("max_cpus_per_task must be >= 1.")
    ray_num_cpus = int(ray_num_cpus)
    max_cpus_per_task = int(max_cpus_per_task)
    if max_cpus_per_task > ray_num_cpus:
        raise ValueError("max_cpus_per_task cannot exceed ray_num_cpus.")
    use_ray = forecast_mode == "rolling-origin" and ray_num_cpus > 1

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    data_path = None if data_path is None else Path(data_path)
    input_feature_path = None if input_feature_path is None else Path(input_feature_path)
    baseline_feature_metadata = _validate_baseline_feature_metadata(
        feature_path=input_feature_path,
        forecast_scenario=forecast_scenario,
        seasonal_lag=seasonal_lag,
    )
    baseline_config = _baseline_config_for_scenario(forecast_scenario, seasonal_lag)
    series_suffix = None if series_suffix == "" else series_suffix
    config = VARAForecastConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        data_path=data_path,
        data_format=data_format,
        date_col=date_col,
        series_col=series_col,
        value_col=value_col,
        max_series=max_series,
        series_depth=series_depth,
        series_suffix=series_suffix,
        horizon=horizon,
        grid_size=grid_size,
        ridge=ridge,
        forecast_scenario=forecast_scenario,
        residual_transform=residual_transform,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        seasonal_lag=seasonal_lag,
        feature_freq=feature_freq,
        feature_functions=feature_functions,
        feature_source=feature_source,
        feature_path=input_feature_path,
        feature_update_policy=feature_update_policy,
        feature_update_interval=feature_update_interval,
        forecast_mode=forecast_mode,
        y_transform=y_transform,
        ray_num_cpus=ray_num_cpus,
        max_cpus_per_task=max_cpus_per_task,
        ray_debug=ray_debug,
        clip_nonnegative=clip_nonnegative,
        residual_shrinkage=residual_shrinkage,
        residual_shrinkage_file=residual_shrinkage_file,
    )
    input_path = config.input_path
    wide = load_panel_data(
        input_path,
        data_format=data_format,
        date_col=date_col,
        series_col=series_col,
        value_col=value_col,
    )

    all_series = _base_series_columns(wide, series_depth=series_depth, series_suffix=series_suffix)
    series_ids = all_series[: min(max_series, len(all_series))] if max_series > 0 else all_series
    if len(series_ids) < 2:
        raise ValueError("Need at least two series for VARA forecasting.")
    if horizon < 1 or horizon >= len(wide):
        raise ValueError("horizon must be positive and smaller than the number of observations.")

    if residual_shrinkage_file is None:
        shrinkage_by_series = pd.DataFrame({
            "series_id": series_ids,
            "residual_shrinkage": np.full(len(series_ids), residual_shrinkage, dtype=float),
        })
        residual_shrinkage_mode = "scalar"
    else:
        shrinkage_by_series = _load_residual_shrinkage_by_series(
            residual_shrinkage_file,
            series_ids,
        )
        residual_shrinkage_mode = "per_series"
    residual_shrinkage_weights = shrinkage_by_series["residual_shrinkage"].to_numpy(
        dtype=float
    )

    Y_raw = wide[series_ids].to_numpy(dtype=float)
    train = Y_raw[:-horizon]
    test = Y_raw[-horizon:]

    feature_frames = []
    rolling_diagnostics = []
    exact_feature_origins = []
    baseline_forecasts = []
    residual_forecasts = []
    forecast_variances = []
    baseline_variances = []
    residual_variances = []

    if forecast_mode == "recursive":
        baseline_forecast = np.zeros((horizon, len(series_ids)), dtype=float)
        baseline_variance = np.zeros_like(baseline_forecast, dtype=float)
        if forecast_scenario == "direct":
            model_train = train
            model_transform = y_transform
        else:
            baseline_forecast, model_train, baseline_variance = _baseline_forecast_residual_and_variance(
                train,
                horizon,
                seasonal_lag,
                series_ids,
                forecast_scenario,
            )
            model_transform = residual_transform
        Y_train, fitted_y_transform = _fit_y_transform(model_train, model_transform, seasonal_lag)
        Z, feature_df, feature_cols = _tourism_feature_matrix(
            wide,
            series_ids,
            horizon=horizon,
            feature_source=feature_source,
            y_transform=y_transform,
            seasonal_lag=seasonal_lag,
            feature_freq=feature_freq,
            feature_functions=feature_functions,
            forecast_scenario=forecast_scenario,
            feature_path=input_feature_path,
        )

        jnp, _ = _runtime_imports()
        forecaster = _new_forecaster(grid_size, ridge, attention_mode, attention_top_k)
        forecaster.fit(jnp.asarray(Y_train), jnp.asarray(Z))
        forecast_model_scale = np.asarray(forecaster.predict(h=horizon))
        forecast_transformed = _unscale_y_transform(forecast_model_scale, fitted_y_transform)
        residual_forecast = _inverse_y_transform(forecast_model_scale, fitted_y_transform)
        if horizon == 1:
            residual_variance = _one_step_residual_forecast_variance(forecaster, residual_forecast, fitted_y_transform)
            forecast_variance = (
                residual_variance
                if forecast_scenario == "direct"
                else _combine_forecast_variance(baseline_variance, residual_variance)
            )
        else:
            residual_variance = np.full_like(residual_forecast, np.nan, dtype=float)
            forecast_variance = np.full_like(residual_forecast, np.nan, dtype=float)
        forecast = residual_forecast if forecast_scenario == "direct" else baseline_forecast + residual_forecast
        baseline_forecasts.append(baseline_forecast)
        residual_forecasts.append(residual_forecast)
        forecast_variances.append(forecast_variance)
        baseline_variances.append(baseline_variance)
        residual_variances.append(residual_variance)
        feature_frames.append(feature_df)
        fit_diagnostics = forecaster.diagnostics()
        n_train_model_initial = int(Y_train.shape[0])
        n_train_model_final = int(Y_train.shape[0])
        n_feature_columns = len(feature_cols)
        exact_feature_origins = [1]
    elif forecast_mode == "closed-loop":
        origin_results = _fit_vara_origins_closed_loop(
            wide,
            Y_raw,
            series_ids,
            horizon=horizon,
            y_transform=y_transform,
            seasonal_lag=seasonal_lag,
            feature_freq=feature_freq,
            feature_functions=feature_functions,
            feature_source=feature_source,
            feature_path=input_feature_path,
            feature_update_policy=feature_update_policy,
            feature_update_interval=feature_update_interval,
            grid_size=grid_size,
            ridge=ridge,
            attention_mode=attention_mode,
            attention_top_k=attention_top_k,
            forecast_scenario=forecast_scenario,
            residual_transform=residual_transform,
        )
        collected = _collect_origin_results(origin_results)
        forecast = collected["forecast"]
        baseline_forecasts.append(collected["baseline"])
        residual_forecasts.append(collected["residual"])
        forecast_variances.append(collected["forecast_variance"])
        baseline_variances.append(collected["baseline_variance"])
        residual_variances.append(collected["residual_variance"])
        forecast_transformed = collected["forecast_transformed"]
        feature_frames.extend(collected["feature_frames"])
        rolling_diagnostics.extend(collected["rolling_diagnostics"])
        exact_feature_origins = collected["exact_feature_origins"]
        fit_diagnostics = collected["fit_diagnostics"]
        fitted_y_transform = collected["fitted_y_transform"]
        n_train_model_initial = collected["n_train_model_initial"]
        n_train_model_final = collected["n_train_model_final"]
        n_feature_columns = collected["n_feature_columns"]
    else:
        if use_ray:
            ray_module, ray_started_here = _start_ray(ray_num_cpus, max_cpus_per_task)
            try:
                origin_results = _fit_vara_origins_with_ray(
                    ray_module,
                    wide,
                    Y_raw,
                    series_ids,
                    horizon=horizon,
                    max_cpus_per_task=max_cpus_per_task,
                    y_transform=y_transform,
                    seasonal_lag=seasonal_lag,
                    feature_freq=feature_freq,
                    feature_functions=feature_functions,
                    feature_source=feature_source,
                    feature_path=input_feature_path,
                    feature_update_policy=feature_update_policy,
                    feature_update_interval=feature_update_interval,
                    grid_size=grid_size,
                    ridge=ridge,
                    attention_mode=attention_mode,
                    attention_top_k=attention_top_k,
                    forecast_scenario=forecast_scenario,
                    residual_transform=residual_transform,
                    ray_debug=ray_debug,
                )
            finally:
                if ray_started_here:
                    _print_progress("Shutting down Ray")
                    ray_module.shutdown()
                    _print_progress("Ray shut down")

            collected = _collect_origin_results(origin_results)
            forecast = collected["forecast"]
            baseline_forecasts.append(collected["baseline"])
            residual_forecasts.append(collected["residual"])
            forecast_variances.append(collected["forecast_variance"])
            baseline_variances.append(collected["baseline_variance"])
            residual_variances.append(collected["residual_variance"])
            forecast_transformed = collected["forecast_transformed"]
            feature_frames.extend(collected["feature_frames"])
            rolling_diagnostics.extend(collected["rolling_diagnostics"])
            exact_feature_origins = collected["exact_feature_origins"]
            fit_diagnostics = collected["fit_diagnostics"]
            fitted_y_transform = collected["fitted_y_transform"]
            n_train_model_initial = collected["n_train_model_initial"]
            n_train_model_final = collected["n_train_model_final"]
            n_feature_columns = collected["n_feature_columns"]
        else:
            origin_results = []
            cached_Z = None
            cached_feature_df = None
            cached_feature_cols = None
            cached_feature_source_origin = None
            cached_feature_source_train_end = None
            for step in range(horizon):
                source_step, feature_exact = _feature_source_step(step, feature_update_policy, feature_update_interval)
                if feature_exact or cached_Z is None:
                    feature_horizon = horizon - source_step
                    Z, feature_df, feature_cols = _tourism_feature_matrix(
                        wide,
                        series_ids,
                        horizon=feature_horizon,
                        feature_source=feature_source,
                        y_transform=y_transform,
                        seasonal_lag=seasonal_lag,
                        feature_freq=feature_freq,
                        feature_functions=feature_functions,
                        feature_path=input_feature_path,
                    )
                    cached_Z = Z
                    cached_feature_df = feature_df.copy()
                    cached_feature_cols = list(feature_cols)
                    cached_feature_source_origin = source_step + 1
                    fallback_train_end = wide["date"].iloc[len(wide) - feature_horizon - 1]
                    cached_feature_source_train_end = _feature_train_end(feature_df, fallback_train_end)
                else:
                    Z = cached_Z
                    feature_df = cached_feature_df.copy()
                    feature_cols = list(cached_feature_cols)

                origin_results.append(_fit_vara_origin_step(
                    wide,
                    Y_raw,
                    series_ids,
                    step=step,
                    horizon=horizon,
                    Z=Z,
                    feature_df=feature_df,
                    feature_cols=feature_cols,
                    y_transform=y_transform,
                    seasonal_lag=seasonal_lag,
                    grid_size=grid_size,
                    ridge=ridge,
                    attention_mode=attention_mode,
                    attention_top_k=attention_top_k,
                    forecast_scenario=forecast_scenario,
                    residual_transform=residual_transform,
                    feature_update_policy=feature_update_policy,
                    feature_exact=feature_exact,
                    feature_source_origin=cached_feature_source_origin,
                    feature_source_train_end=cached_feature_source_train_end,
                ))

            collected = _collect_origin_results(origin_results)
            forecast = collected["forecast"]
            baseline_forecasts.append(collected["baseline"])
            residual_forecasts.append(collected["residual"])
            forecast_variances.append(collected["forecast_variance"])
            baseline_variances.append(collected["baseline_variance"])
            residual_variances.append(collected["residual_variance"])
            forecast_transformed = collected["forecast_transformed"]
            feature_frames.extend(collected["feature_frames"])
            rolling_diagnostics.extend(collected["rolling_diagnostics"])
            exact_feature_origins = collected["exact_feature_origins"]
            fit_diagnostics = collected["fit_diagnostics"]
            fitted_y_transform = collected["fitted_y_transform"]
            n_train_model_initial = collected["n_train_model_initial"]
            n_train_model_final = collected["n_train_model_final"]
            n_feature_columns = collected["n_feature_columns"]

    baseline_forecast = np.vstack(baseline_forecasts) if baseline_forecasts else np.zeros_like(forecast)
    residual_forecast = np.vstack(residual_forecasts) if residual_forecasts else forecast.copy()
    forecast_variance = np.vstack(forecast_variances) if forecast_variances else np.full_like(forecast, np.nan, dtype=float)
    baseline_variance = np.vstack(baseline_variances) if baseline_variances else np.full_like(forecast, np.nan, dtype=float)
    residual_variance = np.vstack(residual_variances) if residual_variances else np.full_like(forecast, np.nan, dtype=float)
    residual_forecast_unshrunk = residual_forecast.copy()
    residual_variance_unshrunk = residual_variance.copy()

    if forecast_scenario != "direct":
        forecast, residual_forecast, forecast_variance, residual_variance = (
            _apply_residual_shrinkage(
                baseline_forecast,
                residual_forecast,
                baseline_variance,
                residual_variance,
                residual_shrinkage_weights,
            )
        )

    clip_nonnegative = bool(clip_nonnegative)
    if clip_nonnegative:
        forecast = np.maximum(forecast, 0.0)
        residual_forecast = forecast.copy() if forecast_scenario == "direct" else forecast - baseline_forecast

    errors = forecast - test
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    output_dir.mkdir(parents=True, exist_ok=True)
    shrinkage_by_series.to_csv(output_dir / "residual_shrinkage_by_series.csv", index=False)
    attention_feature_output_path = output_dir / "attention_features.csv"
    pd.concat(feature_frames, ignore_index=True, sort=False).to_csv(attention_feature_output_path, index=False)
    if rolling_diagnostics:
        pd.DataFrame(rolling_diagnostics).to_csv(output_dir / "rolling_origin_diagnostics.csv", index=False)

    forecast_dates = wide["date"].iloc[-horizon:].to_numpy()
    _write_wide_output(output_dir, "forecast.csv", forecast, forecast_dates, series_ids)
    _write_wide_output(output_dir, "baseline_forecast.csv", baseline_forecast, forecast_dates, series_ids)
    _write_wide_output(output_dir, "residual_forecast.csv", residual_forecast, forecast_dates, series_ids)
    _write_wide_output(
        output_dir,
        "residual_forecast_unshrunk.csv",
        residual_forecast_unshrunk,
        forecast_dates,
        series_ids,
    )
    _write_wide_output(output_dir, "forecast_transformed.csv", forecast_transformed, forecast_dates, series_ids)
    _write_wide_output(output_dir, "forecast_variance.csv", forecast_variance, forecast_dates, series_ids)
    _write_wide_output(output_dir, "baseline_forecast_variance.csv", baseline_variance, forecast_dates, series_ids)
    _write_wide_output(output_dir, "residual_forecast_variance.csv", residual_variance, forecast_dates, series_ids)
    _write_wide_output(
        output_dir,
        "residual_forecast_variance_unshrunk.csv",
        residual_variance_unshrunk,
        forecast_dates,
        series_ids,
    )
    _write_wide_output(output_dir, "test.csv", test, forecast_dates, series_ids)
    _write_wide_output(output_dir, "errors.csv", errors, forecast_dates, series_ids)

    by_series, by_origin, summary = forecast_metric_tables(
        {"vara": forecast},
        test,
        train,
        wide["date"].iloc[-horizon:].to_numpy(),
        series_ids,
        model_names=["vara"],
        season_length=seasonal_lag,
        forecast_mode=forecast_mode,
        horizon=horizon,
        forecast_variances={"vara": forecast_variance},
        summary_context={
            "data": str(input_path),
            "series_depth": -1 if series_depth is None else int(series_depth),
            "season_length": int(seasonal_lag),
            "train_start": str(wide["date"].iloc[0]),
            "train_end": str(wide["date"].iloc[-horizon - 1]),
            "test_start": str(wide["date"].iloc[-horizon]),
            "test_end": str(wide["date"].iloc[-1]),
        },
    )
    by_series.to_csv(output_dir / "metrics_by_series.csv", index=False)
    by_origin.to_csv(output_dir / "metrics_by_origin.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)

    metrics = {
        "source": str(input_path),
        "series_suffix": "" if series_suffix is None else str(series_suffix),
        "data_format": str(data_format),
        "date_col": str(date_col),
        "series_col": str(series_col),
        "value_col": str(value_col),
        "series_depth": -1 if series_depth is None else int(series_depth),
        "n_series": len(series_ids),
        "n_train": int(train.shape[0]),
        "n_train_final": int(Y_raw.shape[0] - 1 if forecast_mode in {"rolling-origin", "closed-loop"} else train.shape[0]),
        "n_train_model": n_train_model_initial,
        "n_train_model_final": n_train_model_final,
        "n_test": int(test.shape[0]),
        "horizon": int(horizon),
        "forecast_type": forecast_mode,
        "ray_used": bool(use_ray),
        "ray_num_cpus": int(ray_num_cpus),
        "max_cpus_per_task": int(max_cpus_per_task),
        "forecast_scenario": forecast_scenario,
        "clip_nonnegative": bool(clip_nonnegative),
        "residual_shrinkage": float(np.mean(residual_shrinkage_weights)),
        "residual_shrinkage_mode": residual_shrinkage_mode,
        "residual_shrinkage_file": (
            "" if residual_shrinkage_file is None else str(residual_shrinkage_file)
        ),
        "residual_shrinkage_min": float(np.min(residual_shrinkage_weights)),
        "residual_shrinkage_median": float(np.median(residual_shrinkage_weights)),
        "residual_shrinkage_max": float(np.max(residual_shrinkage_weights)),
        "residual_baseline": _residual_baseline_name(forecast_scenario),
        "residual_transform": residual_transform if forecast_scenario != "direct" else "",
        "baseline_config_model": baseline_config.get("model", ""),
        "baseline_config_alias": baseline_config.get("alias", ""),
        "baseline_config_season_length": baseline_config.get("season_length", ""),
        "baseline_config_freq": baseline_config.get("freq", ""),
        "baseline_feature_metadata_checked": bool(baseline_feature_metadata.get("checked", False)),
        "baseline_feature_metadata_path": baseline_feature_metadata.get("metadata_path", ""),
        "baseline_feature_filter_method": baseline_feature_metadata.get("filter_method", ""),
        "baseline_feature_seasonal_lag": baseline_feature_metadata.get("seasonal_lag", ""),
        "train_start": str(wide["date"].iloc[0]),
        "train_end": str(wide["date"].iloc[-horizon - 1]),
        "test_start": str(wide["date"].iloc[-horizon]),
        "test_end": str(wide["date"].iloc[-1]),
        "grid_size": int(grid_size),
        "ridge": float(ridge),
        "y_transform": y_transform,
        "seasonal_lag": int(seasonal_lag),
        "model_scale": fitted_y_transform["model_scale"],
        "attention_feature_source": feature_source,
        "attention_feature_file": str(attention_feature_output_path),
        "attention_feature_input_file": "" if input_feature_path is None else str(input_feature_path),
        "attention_feature_transform": _feature_transform_label(feature_source, y_transform),
        "attention_feature_freq": int(feature_freq),
        "attention_feature_functions": _feature_function_metric(feature_source, feature_functions),
        "attention_feature_update_policy": feature_update_policy,
        "attention_feature_update_interval": feature_update_interval,
        "attention_feature_exact_origins": ",".join(str(origin) for origin in exact_feature_origins),
        "attention_feature_exact_origin_count": len(exact_feature_origins),
        "n_attention_feature_columns": n_feature_columns,
        "rmse": rmse,
        "mae": mae,
        "mape_pct": float(summary["mape_pct"].iloc[0]),
        "smape_pct": float(summary["smape_pct"].iloc[0]),
        "mase": float(summary["mase"].iloc[0]),
        "rmsse": float(summary["rmsse"].iloc[0]),
        "bias": float(summary["bias"].iloc[0]),
        "oos_r2": float(summary["oos_r2"].iloc[0]),
        "oos_r2_zero": float(summary["oos_r2_zero"].iloc[0]),
        "crps": float(summary["crps"].iloc[0]),
        "forecast_variance_mean": float(summary["forecast_variance_mean"].iloc[0]),
        **fit_diagnostics,
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    return metrics


run_tourism_forecast = run_vara_forecast


def _vara_config_from_args(args: argparse.Namespace) -> VARAForecastConfig:
    series_depth = None if args.series_depth is None or int(args.series_depth) < 0 else int(args.series_depth)
    series_suffix = None if args.series_suffix in (None, "") else args.series_suffix
    return VARAForecastConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        data_path=args.data_path,
        data_format=args.data_format,
        date_col=args.date_col,
        series_col=args.series_col,
        value_col=args.value_col,
        max_series=args.max_series,
        series_depth=series_depth,
        series_suffix=series_suffix,
        horizon=args.horizon,
        grid_size=args.grid_size,
        ridge=args.ridge,
        forecast_scenario=args.forecast_scenario,
        residual_transform=args.residual_transform,
        attention_mode=args.attention_mode,
        attention_top_k=args.attention_top_k,
        seasonal_lag=args.seasonal_lag,
        feature_freq=args.feature_freq,
        feature_functions=args.feature_functions,
        feature_source=args.feature_source,
        feature_path=args.feature_path,
        feature_update_policy=args.feature_update_policy,
        feature_update_interval=args.feature_update_interval,
        forecast_mode=args.forecast_mode,
        y_transform=args.y_transform,
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
        ray_debug=args.ray_debug,
        clip_nonnegative=args.clip_nonnegative,
        residual_shrinkage=args.residual_shrinkage,
        residual_shrinkage_file=args.residual_shrinkage_file,
    )


def _parse_args() -> argparse.Namespace:
    default_config = DEFAULT_VARA_FORECAST_CONFIG
    parser = argparse.ArgumentParser(description="Run a VARA forecast on panel time-series data.")
    parser.add_argument("--data-dir", type=Path, default=default_config.data_dir)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_config.data_path,
        help="CSV/parquet path in wide or long panel format. Overrides --data-dir/tourism.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_config.output_dir)
    parser.add_argument("--data-format", choices=["auto", "wide", "long"], default=default_config.data_format)
    parser.add_argument("--date-col", type=str, default=default_config.date_col)
    parser.add_argument("--series-col", type=str, default=default_config.series_col)
    parser.add_argument("--value-col", type=str, default=default_config.value_col)
    parser.add_argument("--max-series", type=int, default=default_config.max_series, help="0 means use all selected series.")
    parser.add_argument(
        "--series-depth",
        type=int,
        default=-1 if default_config.series_depth is None else default_config.series_depth,
        help="Hierarchy code depth to forecast; use -1 for all All-suffix series.",
    )
    parser.add_argument(
        "--series-suffix",
        type=str,
        default="" if default_config.series_suffix is None else default_config.series_suffix,
        help="Series-column suffix used for selection; pass an empty string to use all non-date columns.",
    )
    parser.add_argument("--horizon", type=int, default=default_config.horizon)
    parser.add_argument("--grid-size", type=int, default=default_config.grid_size)
    parser.add_argument("--ridge", type=float, default=default_config.ridge)
    parser.add_argument(
        "--forecast-scenario",
        type=str,
        default=default_config.forecast_scenario,
        choices=list(FORECAST_SCENARIOS),
        help="Forecasting scenario: direct VARA or residual correction around a baseline.",
    )
    parser.add_argument(
        "--residual-transform",
        type=str,
        default=default_config.residual_transform,
        choices=["center", "standardize"],
        help="Transform used for VARA when fitting residual-correction scenarios.",
    )
    parser.add_argument(
        "--attention-mode",
        type=str,
        default=default_config.attention_mode,
        choices=list(ATTENTION_MODES),
        help="Attention computation mode for VARA.",
    )
    parser.add_argument(
        "--attention-top-k",
        type=int,
        default=default_config.attention_top_k,
        help="Top-k neighbor count when --attention-mode topk.",
    )
    parser.add_argument("--seasonal-lag", type=int, default=default_config.seasonal_lag)
    parser.add_argument("--feature-freq", type=int, default=default_config.feature_freq)
    parser.add_argument("--feature-functions", type=str, default=default_config.feature_functions)
    parser.add_argument(
        "--feature-source",
        type=str,
        default=default_config.feature_source,
        choices=list(FEATURE_SOURCES),
        help="Feature source for attention construction.",
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=default_config.feature_path,
        help="Prepared series-level feature CSV/parquet. Providing this path implies --feature-source file.",
    )
    parser.add_argument(
        "--feature-update-policy",
        type=str,
        default=default_config.feature_update_policy,
        choices=list(FEATURE_UPDATE_POLICIES),
        help="Feature update policy for rolling-origin mode.",
    )
    parser.add_argument(
        "--feature-update-interval",
        type=int,
        default=default_config.feature_update_interval,
        help="For periodic feature updates, recompute exact features every K rolling origins.",
    )
    parser.add_argument(
        "--forecast-mode",
        type=str,
        default=default_config.forecast_mode,
        help="Forecast mode, or comma-separated modes: recursive, rolling-origin, closed-loop.",
    )
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=default_config.ray_num_cpus,
        help="Use Ray for rolling-origin VARA when > 1; closed-loop is sequential.",
    )
    parser.add_argument(
        "--max-cpus-per-task",
        type=int,
        default=default_config.max_cpus_per_task,
        help="Ray CPU resources per rolling-origin task.",
    )
    parser.add_argument("--ray-debug", action="store_true", help="Print Ray rolling-origin task start/done logs.")
    parser.add_argument(
        "--clip-nonnegative",
        action="store_true",
        default=default_config.clip_nonnegative,
        help="Clip final VARA forecasts at zero before writing outputs and computing metrics.",
    )
    parser.add_argument(
        "--residual-shrinkage",
        type=float,
        default=default_config.residual_shrinkage,
        help="Multiply VARA residual forecasts by this scalar before adding them back to the baseline.",
    )
    parser.add_argument(
        "--residual-shrinkage-file",
        type=Path,
        default=default_config.residual_shrinkage_file,
        help=(
            "CSV/parquet file with series_id and residual_shrinkage columns. "
            "Weights must be between 0 and 1 and cover every selected series."
        ),
    )
    parser.add_argument(
        "--y-transform",
        type=str,
        default=default_config.y_transform,
        choices=list(Y_TRANSFORMS),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        forecast_modes = _parse_forecast_modes(args.forecast_mode)
    except ValueError as exc:
        raise SystemExit(f"run_vara_forecast.py: error: {exc}") from exc

    config = _vara_config_from_args(args)
    run_kwargs = config.run_kwargs()

    if len(forecast_modes) == 1:
        run_vara_forecast(
            output_dir=config.output_dir,
            forecast_mode=forecast_modes[0],
            **run_kwargs,
        )
        summary = pd.read_csv(config.output_dir / "metrics_summary.csv")
    else:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        summary_frames = []
        metrics_frames = []
        by_series_frames = []
        by_origin_frames = []
        for forecast_mode in forecast_modes:
            mode_output_dir = config.output_dir / forecast_mode
            _print_progress(f"Running VARA forecast mode={forecast_mode}, output_dir={mode_output_dir}")
            run_vara_forecast(
                output_dir=mode_output_dir,
                forecast_mode=forecast_mode,
                **run_kwargs,
            )
            summary_frames.append(pd.read_csv(mode_output_dir / "metrics_summary.csv"))
            metrics_frames.append(pd.read_csv(mode_output_dir / "metrics.csv"))
            by_series_frames.append(pd.read_csv(mode_output_dir / "metrics_by_series.csv"))
            by_origin_frames.append(pd.read_csv(mode_output_dir / "metrics_by_origin.csv"))

        summary = pd.concat(summary_frames, ignore_index=True, sort=False)
        summary.to_csv(config.output_dir / "metrics_summary.csv", index=False)
        pd.concat(metrics_frames, ignore_index=True, sort=False).to_csv(
            config.output_dir / "metrics.csv", index=False
        )
        pd.concat(by_series_frames, ignore_index=True, sort=False).to_csv(
            config.output_dir / "metrics_by_series.csv", index=False
        )
        pd.concat(by_origin_frames, ignore_index=True, sort=False).to_csv(
            config.output_dir / "metrics_by_origin.csv", index=False
        )

    print("VARA forecast summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
