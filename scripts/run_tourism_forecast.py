#!/usr/bin/env python3
"""Run VARA forecasting on the tourism dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
import pandas as pd

from vara.forecast import VARAForecaster
from vara.series_features import META_COLUMNS, extract_series_tsfeatures

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


def _fit_y_transform(train: np.ndarray, method: str, seasonal_lag: int) -> tuple[np.ndarray, dict]:
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


def _series_code(series_id: str, suffix: str = BASE_SUFFIX) -> str:
    return series_id[: -len(suffix)] if suffix and series_id.endswith(suffix) else series_id


def _series_depth(series_id: str, suffix: str = BASE_SUFFIX) -> int:
    code = _series_code(series_id, suffix)
    return 0 if code == "Total" else len(code)


def _base_series_columns(wide: pd.DataFrame, series_depth: int | None = DETAILED_CODE_DEPTH) -> list[str]:
    cols = [c for c in wide.columns if c != "date" and c.endswith(BASE_SUFFIX)]
    if series_depth is not None:
        cols = [c for c in cols if _series_depth(c) == series_depth]
    if not cols:
        raise ValueError(f"No tourism series ending with {BASE_SUFFIX!r} found for series_depth={series_depth}.")
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


def _new_forecaster(grid_size: int, ridge: float) -> VARAForecaster:
    return VARAForecaster(
        grid_size=grid_size,
        eta_bounds=(-2.0, 2.0),
        ridge=ridge,
        dense_threshold=2500,
    )


def _feature_function_label(feature_functions: str | Sequence[str]) -> str:
    if isinstance(feature_functions, str):
        return feature_functions
    return ",".join(str(item) for item in feature_functions)


def run_tourism_forecast(
    data_dir: Path,
    output_dir: Path,
    max_series: int,
    horizon: int,
    grid_size: int,
    y_transform: str,
    seasonal_lag: int,
    ridge: float,
    series_depth: int | None = DETAILED_CODE_DEPTH,
    feature_freq: int = 12,
    feature_functions: str | Sequence[str] = TOURISM_TSFEATURES,
    forecast_mode: str = "recursive",
) -> dict:
    if forecast_mode not in {"recursive", "rolling-origin"}:
        raise ValueError("forecast_mode must be 'recursive' or 'rolling-origin'.")

    wide = pd.read_csv(data_dir / "tourism.csv")

    all_series = _base_series_columns(wide, series_depth=series_depth)
    series_ids = all_series[: min(max_series, len(all_series))] if max_series > 0 else all_series
    if len(series_ids) < 2:
        raise ValueError("Need at least two tourism series for VARA forecasting.")
    if horizon < 1 or horizon >= len(wide):
        raise ValueError("horizon must be positive and smaller than the number of months.")

    Y_raw = wide[series_ids].to_numpy(dtype=float)
    train = Y_raw[:-horizon]
    test = Y_raw[-horizon:]

    feature_frames = []
    rolling_diagnostics = []

    if forecast_mode == "recursive":
        Y_train, fitted_y_transform = _fit_y_transform(train, y_transform, seasonal_lag)
        Z, feature_df, feature_cols = _tourism_tsfeature_matrix(
            wide,
            series_ids,
            horizon=horizon,
            y_transform=y_transform,
            seasonal_lag=seasonal_lag,
            feature_freq=feature_freq,
            feature_functions=feature_functions,
        )

        forecaster = _new_forecaster(grid_size, ridge)
        forecaster.fit(jnp.asarray(Y_train), jnp.asarray(Z))
        forecast_model_scale = np.asarray(forecaster.predict(h=horizon))
        forecast_transformed = _unscale_y_transform(forecast_model_scale, fitted_y_transform)
        forecast = _inverse_y_transform(forecast_model_scale, fitted_y_transform)
        feature_frames.append(feature_df)
        fit_diagnostics = forecaster.diagnostics()
        n_train_model_initial = int(Y_train.shape[0])
        n_train_model_final = int(Y_train.shape[0])
        n_feature_columns = len(feature_cols)
    else:
        forecasts = []
        transformed_forecasts = []
        forecaster = None
        fitted_y_transform = None
        feature_cols = []
        n_train_model_initial = 0
        n_train_model_final = 0
        for step in range(horizon):
            origin_end = len(wide) - horizon + step
            origin_train = Y_raw[:origin_end]
            Y_train, fitted_y_transform = _fit_y_transform(origin_train, y_transform, seasonal_lag)
            Z, feature_df, feature_cols = _tourism_tsfeature_matrix(
                wide,
                series_ids,
                horizon=horizon - step,
                y_transform=y_transform,
                seasonal_lag=seasonal_lag,
                feature_freq=feature_freq,
                feature_functions=feature_functions,
            )

            forecaster = _new_forecaster(grid_size, ridge)
            forecaster.fit(jnp.asarray(Y_train), jnp.asarray(Z))
            if step == 0:
                n_train_model_initial = int(Y_train.shape[0])
            forecast_model_scale = np.asarray(forecaster.predict(h=1))
            transformed_one = _unscale_y_transform(forecast_model_scale, fitted_y_transform)
            forecast_one = _inverse_y_transform(forecast_model_scale, fitted_y_transform)
            transformed_forecasts.append(transformed_one[0])
            forecasts.append(forecast_one[0])

            forecast_date = str(wide["date"].iloc[origin_end])
            train_end = str(wide["date"].iloc[origin_end - 1])
            origin_features = feature_df.copy()
            origin_features.insert(0, "origin", step + 1)
            origin_features.insert(1, "forecast_date", forecast_date)
            feature_frames.append(origin_features)

            diag = forecaster.diagnostics()
            diag.update({
                "origin": step + 1,
                "forecast_date": forecast_date,
                "train_end": train_end,
                "n_train": int(origin_train.shape[0]),
                "n_train_model": int(Y_train.shape[0]),
                "n_attention_feature_columns": len(feature_cols),
            })
            rolling_diagnostics.append(diag)
            n_train_model_final = int(Y_train.shape[0])

        forecast = np.vstack(forecasts)
        forecast_transformed = np.vstack(transformed_forecasts)
        fit_diagnostics = forecaster.diagnostics()
        n_feature_columns = len(feature_cols)

    errors = forecast - test
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    output_dir.mkdir(parents=True, exist_ok=True)
    feature_path = output_dir / "tourism_attention_features.csv"
    pd.concat(feature_frames, ignore_index=True, sort=False).to_csv(feature_path, index=False)
    if rolling_diagnostics:
        pd.DataFrame(rolling_diagnostics).to_csv(output_dir / "tourism_rolling_origin_diagnostics.csv", index=False)

    forecast_df = pd.DataFrame(forecast, columns=series_ids)
    forecast_df.insert(0, "date", wide["date"].iloc[-horizon:].to_numpy())
    forecast_df.to_csv(output_dir / "tourism_forecast.csv", index=False)

    forecast_transformed_df = pd.DataFrame(forecast_transformed, columns=series_ids)
    forecast_transformed_df.insert(0, "date", wide["date"].iloc[-horizon:].to_numpy())
    forecast_transformed_df.to_csv(output_dir / "tourism_forecast_transformed.csv", index=False)

    test_df = pd.DataFrame(test, columns=series_ids)
    test_df.insert(0, "date", wide["date"].iloc[-horizon:].to_numpy())
    test_df.to_csv(output_dir / "tourism_test.csv", index=False)

    error_df = pd.DataFrame(errors, columns=series_ids)
    error_df.insert(0, "date", wide["date"].iloc[-horizon:].to_numpy())
    error_df.to_csv(output_dir / "tourism_errors.csv", index=False)

    metrics = {
        "source": str(data_dir / "tourism.csv"),
        "series_suffix": BASE_SUFFIX,
        "series_depth": -1 if series_depth is None else int(series_depth),
        "n_series": len(series_ids),
        "n_train": int(train.shape[0]),
        "n_train_final": int(Y_raw.shape[0] - 1 if forecast_mode == "rolling-origin" else train.shape[0]),
        "n_train_model": n_train_model_initial,
        "n_train_model_final": n_train_model_final,
        "n_test": int(test.shape[0]),
        "horizon": int(horizon),
        "forecast_type": forecast_mode,
        "train_start": str(wide["date"].iloc[0]),
        "train_end": str(wide["date"].iloc[-horizon - 1]),
        "test_start": str(wide["date"].iloc[-horizon]),
        "test_end": str(wide["date"].iloc[-1]),
        "grid_size": int(grid_size),
        "ridge": float(ridge),
        "y_transform": y_transform,
        "seasonal_lag": int(seasonal_lag),
        "model_scale": fitted_y_transform["model_scale"],
        "attention_feature_source": "tsfeatures",
        "attention_feature_file": str(feature_path),
        "attention_feature_transform": _feature_transform_from_y_transform(y_transform),
        "attention_feature_freq": int(feature_freq),
        "attention_feature_functions": _feature_function_label(feature_functions),
        "n_attention_feature_columns": n_feature_columns,
        "rmse": rmse,
        "mae": mae,
        **fit_diagnostics,
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "tourism_metrics.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a VARA forecast on tourism data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tourism"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tourism_forecast"))
    parser.add_argument("--max-series", type=int, default=0, help="0 means use all selected series.")
    parser.add_argument(
        "--series-depth",
        type=int,
        default=DETAILED_CODE_DEPTH,
        help="Hierarchy code depth to forecast; use -1 for all All-suffix series.",
    )
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seasonal-lag", type=int, default=12)
    parser.add_argument("--feature-freq", type=int, default=12)
    parser.add_argument("--feature-functions", type=str, default=DEFAULT_TOURISM_TSFEATURES)
    parser.add_argument(
        "--forecast-mode",
        type=str,
        default="recursive",
        choices=["recursive", "rolling-origin"],
        help="recursive fits once; rolling-origin refits one-step models through the test window.",
    )
    parser.add_argument(
        "--y-transform",
        type=str,
        default="log1p-seasonal-diff",
        choices=["center", "standardize", "log1p-center", "log1p-standardize", "log1p-seasonal-diff"],
    )
    args = parser.parse_args()

    metrics = run_tourism_forecast(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_series=args.max_series,
        horizon=args.horizon,
        grid_size=args.grid_size,
        y_transform=args.y_transform,
        seasonal_lag=args.seasonal_lag,
        ridge=args.ridge,
        series_depth=None if args.series_depth < 0 else args.series_depth,
        feature_freq=args.feature_freq,
        feature_functions=args.feature_functions,
        forecast_mode=args.forecast_mode,
    )
    print("Tourism VARA forecast metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
