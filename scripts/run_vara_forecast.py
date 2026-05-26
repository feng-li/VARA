#!/usr/bin/env python3
"""Run VARA forecasting on panel time-series data."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd

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
FORECAST_SCENARIOS = ("direct", "seasonal-naive-residual")
FORECAST_MODES = ("recursive", "rolling-origin", "closed-loop")
META_COLUMNS = ("series_id", "transform", "n_obs", "train_start", "train_end")

_JNP = None
_VARAForecaster = None


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


def _series_code(series_id: str, suffix: str = BASE_SUFFIX) -> str:
    return series_id[: -len(suffix)] if suffix and series_id.endswith(suffix) else series_id


def _series_depth(series_id: str, suffix: str = BASE_SUFFIX) -> int:
    code = _series_code(series_id, suffix)
    return 0 if code == "Total" else len(code)


def _base_series_columns(
    wide: pd.DataFrame,
    series_depth: int | None = DETAILED_CODE_DEPTH,
    series_suffix: str | None = BASE_SUFFIX,
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
    if y_transform in {"center", "standardize"}:
        return "level"
    if y_transform in {"log1p-center", "log1p-standardize"}:
        return "log1p"
    return y_transform


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
    meta_columns = set(META_COLUMNS)
    feature_cols = [col for col in feature_df.columns if col not in meta_columns]
    if not feature_cols:
        raise ValueError("No tourism tsfeature columns were produced.")
    Z = feature_df.loc[:, feature_cols].to_numpy(dtype=float)
    if not np.isfinite(Z).all():
        raise ValueError("Tourism tsfeature matrix contains non-finite values.")
    return Z, feature_df, feature_cols


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
    feature_cols = [col for col in feature_df.columns if col not in set(META_COLUMNS)]
    if not feature_cols:
        raise ValueError("No tourism-specific feature columns were produced.")
    Z = feature_df.loc[:, feature_cols].to_numpy(dtype=float)
    if not np.isfinite(Z).all():
        raise ValueError("Tourism-specific feature matrix contains non-finite values.")
    return Z, feature_df, feature_cols


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
    feature_cols = [col for col in feature_df.columns if col not in set(META_COLUMNS)]
    Z = feature_df.loc[:, feature_cols].to_numpy(dtype=float)
    if not np.isfinite(Z).all():
        raise ValueError("Seasonal-predictive score matrix contains non-finite values.")
    return Z, feature_df, feature_cols


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
    feature_cols = [col for col in feature_df.columns if col not in set(META_COLUMNS)]
    if not feature_cols:
        raise ValueError("No seasonal-corr tourism feature columns were produced.")
    Z = feature_df.loc[:, feature_cols].to_numpy(dtype=float)
    if not np.isfinite(Z).all():
        raise ValueError("Seasonal-corr tourism feature matrix contains non-finite values.")
    return Z, feature_df, feature_cols


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
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
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
        feature_cols = [*tourism_block.columns.tolist(), *tsfeature_block.columns.tolist()]
        Z = feature_df.loc[:, feature_cols].to_numpy(dtype=float)
        if not np.isfinite(Z).all():
            raise ValueError("Hybrid tourism feature matrix contains non-finite values.")
        return Z, feature_df, feature_cols
    raise ValueError("feature_source must be 'file', 'tsfeatures', 'tourism', 'hybrid', 'seasonal-corr', or 'seasonal-predictive'.")


def _new_forecaster(
    grid_size: int,
    ridge: float,
    attention_mode: str = "dense",
    attention_top_k: int | None = None,
):
    _, VARAForecaster = _runtime_imports()
    return VARAForecaster(
        grid_size=grid_size,
        eta_bounds=(-2.0, 2.0),
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
) -> dict:
    jnp, _ = _runtime_imports()
    origin = int(step) + 1
    origin_end = len(wide) - int(horizon) + int(step)
    if origin_train is None:
        origin_train = Y_raw[:origin_end]
    else:
        origin_train = np.asarray(origin_train, dtype=float)
    baseline_one = np.zeros((1, len(series_ids)), dtype=float)
    if forecast_scenario == "direct":
        model_train = origin_train
        model_transform = y_transform
    else:
        baseline_one = _seasonal_naive_forecast(origin_train, 1, seasonal_lag)
        model_train = _seasonal_naive_residual_panel(origin_train, seasonal_lag)
        model_transform = residual_transform

    Y_train, fitted_y_transform = _fit_y_transform(model_train, model_transform, seasonal_lag)
    forecaster = _new_forecaster(grid_size, ridge, attention_mode, attention_top_k)
    forecaster.fit(jnp.asarray(Y_train), jnp.asarray(Z))
    forecast_model_scale = np.asarray(forecaster.predict(h=1))
    transformed_one = _unscale_y_transform(forecast_model_scale, fitted_y_transform)
    residual_one = _inverse_y_transform(forecast_model_scale, fitted_y_transform)
    forecast_one = residual_one if forecast_scenario == "direct" else baseline_one + residual_one

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


def _feature_function_label(feature_functions: str | Sequence[str]) -> str:
    if isinstance(feature_functions, str):
        return feature_functions
    return ",".join(str(item) for item in feature_functions)


def _feature_transform_label(feature_source: str, y_transform: str) -> str:
    if feature_source == "file":
        return "prepared_feature_file"
    if feature_source == "tourism":
        return "raw_tourism_panel"
    if feature_source == "seasonal-corr":
        return "log1p_seasonal_naive_residuals"
    if feature_source == "seasonal-predictive":
        return "log1p_seasonal_diff_predictive_scores"
    if feature_source == "hybrid":
        return f"raw_tourism_panel+{_feature_transform_from_y_transform(y_transform)}"
    return _feature_transform_from_y_transform(y_transform)


def _feature_function_metric(feature_source: str, feature_functions: str | Sequence[str]) -> str:
    if feature_source == "file":
        return "prepared_feature_file"
    if feature_source == "tourism":
        return TOURISM_FEATURE_FUNCTIONS
    if feature_source == "seasonal-corr":
        return SEASONAL_CORR_FEATURE_FUNCTIONS
    if feature_source == "seasonal-predictive":
        return SEASONAL_PREDICTIVE_FEATURE_FUNCTIONS
    if feature_source == "hybrid":
        return f"{TOURISM_FEATURE_FUNCTIONS},{_feature_function_label(feature_functions)}"
    return _feature_function_label(feature_functions)


def run_vara_forecast(
    data_dir: Path,
    output_dir: Path,
    max_series: int,
    horizon: int,
    grid_size: int,
    y_transform: str,
    seasonal_lag: int,
    ridge: float,
    data_path: Path | None = None,
    data_format: str = "auto",
    date_col: str = "date",
    series_col: str = "series_id",
    value_col: str = "value",
    series_depth: int | None = DETAILED_CODE_DEPTH,
    series_suffix: str | None = BASE_SUFFIX,
    feature_freq: int = 12,
    feature_functions: str | Sequence[str] = TOURISM_TSFEATURES,
    feature_source: str = "tsfeatures",
    feature_path: Path | None = None,
    forecast_mode: str = "recursive",
    feature_update_policy: str = "exact",
    feature_update_interval: int = 1,
    attention_mode: str = "dense",
    attention_top_k: int | None = None,
    forecast_scenario: str = "direct",
    residual_transform: str = "center",
    ray_num_cpus: int = 1,
    max_cpus_per_task: int = 1,
    ray_debug: bool = False,
) -> dict:
    if forecast_scenario not in set(FORECAST_SCENARIOS):
        raise ValueError(f"forecast_scenario must be one of {FORECAST_SCENARIOS}.")
    if forecast_mode not in set(FORECAST_MODES):
        raise ValueError(f"forecast_mode must be one of {FORECAST_MODES}.")
    if residual_transform not in {"center", "standardize"}:
        raise ValueError("residual_transform must be 'center' or 'standardize'.")
    input_feature_path = feature_path
    if input_feature_path is not None:
        if feature_source == "tsfeatures":
            feature_source = "file"
        elif feature_source != "file":
            raise ValueError("feature_path can only be used with feature_source='file' or the default feature_source.")
    if feature_source not in {"file", "tsfeatures", "tourism", "hybrid", "seasonal-corr", "seasonal-predictive"}:
        raise ValueError("feature_source must be 'file', 'tsfeatures', 'tourism', 'hybrid', 'seasonal-corr', or 'seasonal-predictive'.")
    if attention_mode not in {"dense", "topk", "score", "score-topk"}:
        raise ValueError("attention_mode must be 'dense', 'topk', 'score', or 'score-topk'.")
    if attention_mode in {"topk", "score-topk"} and attention_top_k is None:
        raise ValueError("attention_top_k must be provided when attention_mode uses top-k.")
    if feature_source == "file" and input_feature_path is None:
        raise ValueError("feature_path must be provided when feature_source='file'.")
    if feature_source == "seasonal-predictive" and attention_mode not in {"score", "score-topk"}:
        raise ValueError("feature_source='seasonal-predictive' requires attention_mode='score' or 'score-topk'.")
    if feature_source != "seasonal-predictive" and attention_mode in {"score", "score-topk"}:
        raise ValueError("score attention modes require feature_source='seasonal-predictive'.")
    if feature_update_policy not in {"exact", "periodic"}:
        raise ValueError("feature_update_policy must be 'exact' or 'periodic'.")
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

    input_path = data_path if data_path is not None else data_dir / "tourism.csv"
    series_suffix = None if series_suffix == "" else series_suffix
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

    Y_raw = wide[series_ids].to_numpy(dtype=float)
    train = Y_raw[:-horizon]
    test = Y_raw[-horizon:]

    feature_frames = []
    rolling_diagnostics = []
    exact_feature_origins = []
    baseline_forecasts = []
    residual_forecasts = []

    if forecast_mode == "recursive":
        baseline_forecast = np.zeros((horizon, len(series_ids)), dtype=float)
        if forecast_scenario == "direct":
            model_train = train
            model_transform = y_transform
        else:
            baseline_forecast = _seasonal_naive_forecast(train, horizon, seasonal_lag)
            model_train = _seasonal_naive_residual_panel(train, seasonal_lag)
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
            feature_path=input_feature_path,
        )

        jnp, _ = _runtime_imports()
        forecaster = _new_forecaster(grid_size, ridge, attention_mode, attention_top_k)
        forecaster.fit(jnp.asarray(Y_train), jnp.asarray(Z))
        forecast_model_scale = np.asarray(forecaster.predict(h=horizon))
        forecast_transformed = _unscale_y_transform(forecast_model_scale, fitted_y_transform)
        residual_forecast = _inverse_y_transform(forecast_model_scale, fitted_y_transform)
        forecast = residual_forecast if forecast_scenario == "direct" else baseline_forecast + residual_forecast
        baseline_forecasts.append(baseline_forecast)
        residual_forecasts.append(residual_forecast)
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
        forecasts = [result["forecast"] for result in origin_results]
        baseline_steps = [result["baseline"] for result in origin_results]
        residual_steps = [result["residual"] for result in origin_results]
        transformed_forecasts = [result["forecast_transformed"] for result in origin_results]
        feature_frames.extend(result["feature_df"] for result in origin_results)
        rolling_diagnostics.extend(result["diag"] for result in origin_results)
        exact_feature_origins = [int(result["origin"]) for result in origin_results if result["feature_exact"]]
        forecast = np.vstack(forecasts)
        baseline_forecasts.append(np.vstack(baseline_steps))
        residual_forecasts.append(np.vstack(residual_steps))
        forecast_transformed = np.vstack(transformed_forecasts)
        fit_diagnostics = origin_results[-1]["fit_diagnostics"]
        fitted_y_transform = {"model_scale": origin_results[-1]["model_scale"]}
        n_train_model_initial = int(origin_results[0]["n_train_model"])
        n_train_model_final = int(origin_results[-1]["n_train_model"])
        n_feature_columns = len(origin_results[-1]["feature_cols"])
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

            forecasts = [result["forecast"] for result in origin_results]
            baseline_steps = [result["baseline"] for result in origin_results]
            residual_steps = [result["residual"] for result in origin_results]
            transformed_forecasts = [result["forecast_transformed"] for result in origin_results]
            feature_frames.extend(result["feature_df"] for result in origin_results)
            rolling_diagnostics.extend(result["diag"] for result in origin_results)
            exact_feature_origins = [int(result["origin"]) for result in origin_results if result["feature_exact"]]
            forecast = np.vstack(forecasts)
            baseline_forecasts.append(np.vstack(baseline_steps))
            residual_forecasts.append(np.vstack(residual_steps))
            forecast_transformed = np.vstack(transformed_forecasts)
            fit_diagnostics = origin_results[-1]["fit_diagnostics"]
            fitted_y_transform = {"model_scale": origin_results[-1]["model_scale"]}
            n_train_model_initial = int(origin_results[0]["n_train_model"])
            n_train_model_final = int(origin_results[-1]["n_train_model"])
            n_feature_columns = len(origin_results[-1]["feature_cols"])
        else:
            forecasts = []
            baseline_steps = []
            residual_steps = []
            transformed_forecasts = []
            cached_Z = None
            cached_feature_df = None
            cached_feature_cols = None
            cached_feature_source_origin = None
            cached_feature_source_train_end = None
            n_train_model_initial = 0
            n_train_model_final = 0
            fit_diagnostics = {}
            fitted_y_transform = {"model_scale": ""}
            feature_cols = []
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
                    exact_feature_origins.append(source_step + 1)
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
                )
                forecasts.append(result["forecast"])
                baseline_steps.append(result["baseline"])
                residual_steps.append(result["residual"])
                transformed_forecasts.append(result["forecast_transformed"])
                feature_frames.append(result["feature_df"])
                rolling_diagnostics.append(result["diag"])
                fit_diagnostics = result["fit_diagnostics"]
                fitted_y_transform = {"model_scale": result["model_scale"]}
                if step == 0:
                    n_train_model_initial = int(result["n_train_model"])
                n_train_model_final = int(result["n_train_model"])

            forecast = np.vstack(forecasts)
            baseline_forecasts.append(np.vstack(baseline_steps))
            residual_forecasts.append(np.vstack(residual_steps))
            forecast_transformed = np.vstack(transformed_forecasts)
            n_feature_columns = len(feature_cols)

    baseline_forecast = np.vstack(baseline_forecasts) if baseline_forecasts else np.zeros_like(forecast)
    residual_forecast = np.vstack(residual_forecasts) if residual_forecasts else forecast.copy()

    errors = forecast - test
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    output_dir.mkdir(parents=True, exist_ok=True)
    attention_feature_output_path = output_dir / "tourism_attention_features.csv"
    pd.concat(feature_frames, ignore_index=True, sort=False).to_csv(attention_feature_output_path, index=False)
    if rolling_diagnostics:
        pd.DataFrame(rolling_diagnostics).to_csv(output_dir / "tourism_rolling_origin_diagnostics.csv", index=False)

    forecast_df = pd.DataFrame(forecast, columns=series_ids)
    forecast_df.insert(0, "date", wide["date"].iloc[-horizon:].to_numpy())
    forecast_df.to_csv(output_dir / "tourism_forecast.csv", index=False)

    baseline_df = pd.DataFrame(baseline_forecast, columns=series_ids)
    baseline_df.insert(0, "date", wide["date"].iloc[-horizon:].to_numpy())
    baseline_df.to_csv(output_dir / "tourism_baseline_forecast.csv", index=False)

    residual_forecast_df = pd.DataFrame(residual_forecast, columns=series_ids)
    residual_forecast_df.insert(0, "date", wide["date"].iloc[-horizon:].to_numpy())
    residual_forecast_df.to_csv(output_dir / "tourism_residual_forecast.csv", index=False)

    forecast_transformed_df = pd.DataFrame(forecast_transformed, columns=series_ids)
    forecast_transformed_df.insert(0, "date", wide["date"].iloc[-horizon:].to_numpy())
    forecast_transformed_df.to_csv(output_dir / "tourism_forecast_transformed.csv", index=False)

    test_df = pd.DataFrame(test, columns=series_ids)
    test_df.insert(0, "date", wide["date"].iloc[-horizon:].to_numpy())
    test_df.to_csv(output_dir / "tourism_test.csv", index=False)

    error_df = pd.DataFrame(errors, columns=series_ids)
    error_df.insert(0, "date", wide["date"].iloc[-horizon:].to_numpy())
    error_df.to_csv(output_dir / "tourism_errors.csv", index=False)

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
        "residual_baseline": "seasonal_naive" if forecast_scenario == "seasonal-naive-residual" else "",
        "residual_transform": residual_transform if forecast_scenario == "seasonal-naive-residual" else "",
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
        **fit_diagnostics,
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "tourism_metrics.csv", index=False)
    return metrics


run_tourism_forecast = run_vara_forecast


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a VARA forecast on panel time-series data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tourism"))
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="CSV/parquet path in wide or long panel format. Overrides --data-dir/tourism.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vara_forecast"))
    parser.add_argument("--data-format", choices=["auto", "wide", "long"], default="auto")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--series-col", type=str, default="series_id")
    parser.add_argument("--value-col", type=str, default="value")
    parser.add_argument("--max-series", type=int, default=0, help="0 means use all selected series.")
    parser.add_argument(
        "--series-depth",
        type=int,
        default=DETAILED_CODE_DEPTH,
        help="Hierarchy code depth to forecast; use -1 for all All-suffix series.",
    )
    parser.add_argument(
        "--series-suffix",
        type=str,
        default=BASE_SUFFIX,
        help="Series-column suffix used for selection; pass '' to use all non-date columns.",
    )
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument(
        "--forecast-scenario",
        type=str,
        default="direct",
        choices=list(FORECAST_SCENARIOS),
        help="Forecasting scenario: direct VARA or residual correction around a baseline.",
    )
    parser.add_argument(
        "--residual-transform",
        type=str,
        default="center",
        choices=["center", "standardize"],
        help="Transform used for VARA when fitting residual-correction scenarios.",
    )
    parser.add_argument(
        "--attention-mode",
        type=str,
        default="dense",
        choices=["dense", "topk", "score", "score-topk"],
        help="Attention computation mode for VARA.",
    )
    parser.add_argument(
        "--attention-top-k",
        type=int,
        default=None,
        help="Top-k neighbor count when --attention-mode topk.",
    )
    parser.add_argument("--seasonal-lag", type=int, default=12)
    parser.add_argument("--feature-freq", type=int, default=12)
    parser.add_argument("--feature-functions", type=str, default=DEFAULT_TOURISM_TSFEATURES)
    parser.add_argument(
        "--feature-source",
        type=str,
        default="tsfeatures",
        choices=["file", "tsfeatures", "tourism", "hybrid", "seasonal-corr", "seasonal-predictive"],
        help="Feature source for attention construction.",
    )
    parser.add_argument(
        "--feature-path",
        type=Path,
        default=None,
        help="Prepared series-level feature CSV/parquet. Providing this path implies --feature-source file.",
    )
    parser.add_argument(
        "--feature-update-policy",
        type=str,
        default="exact",
        choices=["exact", "periodic"],
        help="Feature update policy for rolling-origin mode.",
    )
    parser.add_argument(
        "--feature-update-interval",
        type=int,
        default=1,
        help="For periodic feature updates, recompute exact features every K rolling origins.",
    )
    parser.add_argument(
        "--forecast-mode",
        type=str,
        default="recursive",
        choices=list(FORECAST_MODES),
        help="recursive fits once; rolling-origin uses actual test updates; closed-loop feeds forecasts back as pseudo-observations.",
    )
    parser.add_argument("--ray-num-cpus", type=int, default=1, help="Use Ray for rolling-origin VARA when > 1; closed-loop is sequential.")
    parser.add_argument("--max-cpus-per-task", type=int, default=1, help="Ray CPU resources per rolling-origin task.")
    parser.add_argument("--ray-debug", action="store_true", help="Print Ray rolling-origin task start/done logs.")
    parser.add_argument(
        "--y-transform",
        type=str,
        default="log1p-seasonal-diff",
        choices=["none", "center", "standardize", "diff", "log1p-center", "log1p-standardize", "log1p-diff", "log1p-seasonal-diff"],
    )
    args = parser.parse_args()

    run_vara_forecast(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_series=args.max_series,
        horizon=args.horizon,
        grid_size=args.grid_size,
        y_transform=args.y_transform,
        seasonal_lag=args.seasonal_lag,
        ridge=args.ridge,
        data_path=args.data_path,
        data_format=args.data_format,
        date_col=args.date_col,
        series_col=args.series_col,
        value_col=args.value_col,
        series_depth=None if args.series_depth < 0 else args.series_depth,
        series_suffix=None if args.series_suffix == "" else args.series_suffix,
        feature_freq=args.feature_freq,
        feature_functions=args.feature_functions,
        feature_source=args.feature_source,
        feature_path=args.feature_path,
        forecast_mode=args.forecast_mode,
        feature_update_policy=args.feature_update_policy,
        feature_update_interval=args.feature_update_interval,
        attention_mode=args.attention_mode,
        attention_top_k=args.attention_top_k,
        forecast_scenario=args.forecast_scenario,
        residual_transform=args.residual_transform,
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
        ray_debug=args.ray_debug,
    )
    summary = pd.read_csv(args.output_dir / "metrics_summary.csv")
    print("VARA forecast summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
