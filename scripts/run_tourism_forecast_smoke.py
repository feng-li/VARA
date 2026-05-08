#!/usr/bin/env python3
"""Small VARA forecasting smoke run on the tourism dataset."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np
import pandas as pd

from vara.forecast import VARAForecaster

BASE_SUFFIX = "All"


def _standardize_features(Z: np.ndarray) -> np.ndarray:
    mean = Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, keepdims=True)
    std = np.where(std > 0.0, std, 1.0)
    return (Z - mean) / std


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


def _base_series_columns(wide: pd.DataFrame) -> list[str]:
    cols = [c for c in wide.columns if c != "date" and c.endswith(BASE_SUFFIX)]
    if not cols:
        raise ValueError(f"No base tourism series ending with {BASE_SUFFIX!r} found.")
    return cols


def _tourism_series_features(series_ids: list[str]) -> np.ndarray:
    rows = []
    for series_id in series_ids:
        code = series_id[: -len(BASE_SUFFIX)]
        is_total = code == "Total"
        rows.append(
            {
                "series_id": series_id,
                "depth": 0 if is_total else len(code),
                "is_total": int(is_total),
                "level_1": "Total" if is_total else code[0],
                "level_2": "None" if is_total or len(code) < 2 else code[1],
                "level_3": "None" if is_total or len(code) < 3 else code[2],
                "is_aggregate": int(is_total or len(code) < 3),
            }
        )
    features = pd.DataFrame(rows).set_index("series_id")
    features = pd.get_dummies(features, columns=["level_1", "level_2", "level_3"], dtype=float)
    return features.to_numpy(dtype=float)


def run_tourism_forecast_smoke(
    data_dir: Path,
    output_dir: Path,
    max_series: int,
    horizon: int,
    grid_size: int,
    y_transform: str,
    seasonal_lag: int,
    ridge: float,
) -> dict:
    wide = pd.read_csv(data_dir / "tourism.csv")

    all_series = _base_series_columns(wide)
    series_ids = all_series[: min(max_series, len(all_series))]
    if len(series_ids) < 2:
        raise ValueError("Need at least two tourism series for VARA forecasting.")
    if horizon < 1 or horizon >= len(wide):
        raise ValueError("horizon must be positive and smaller than the number of months.")

    Y_raw = wide[series_ids].to_numpy(dtype=float)
    train = Y_raw[:-horizon]
    test = Y_raw[-horizon:]

    Y_train, fitted_y_transform = _fit_y_transform(train, y_transform, seasonal_lag)

    Z = _standardize_features(_tourism_series_features(series_ids))

    forecaster = VARAForecaster(
        grid_size=grid_size,
        eta_bounds=(-2.0, 2.0),
        ridge=ridge,
        dense_threshold=2500,
    )
    forecaster.fit(jnp.asarray(Y_train), jnp.asarray(Z))
    forecast_model_scale = np.asarray(forecaster.predict(h=horizon))
    forecast_transformed = _unscale_y_transform(forecast_model_scale, fitted_y_transform)
    forecast = _inverse_y_transform(forecast_model_scale, fitted_y_transform)

    errors = forecast - test
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    mae = float(np.mean(np.abs(errors)))

    output_dir.mkdir(parents=True, exist_ok=True)
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
        "n_series": len(series_ids),
        "n_train": int(train.shape[0]),
        "n_train_model": int(Y_train.shape[0]),
        "n_test": int(test.shape[0]),
        "horizon": int(horizon),
        "forecast_type": "recursive",
        "train_start": str(wide["date"].iloc[0]),
        "train_end": str(wide["date"].iloc[-horizon - 1]),
        "test_start": str(wide["date"].iloc[-horizon]),
        "test_end": str(wide["date"].iloc[-1]),
        "grid_size": int(grid_size),
        "ridge": float(ridge),
        "y_transform": y_transform,
        "seasonal_lag": int(seasonal_lag),
        "model_scale": fitted_y_transform["model_scale"],
        "rmse": rmse,
        "mae": mae,
        **forecaster.diagnostics(),
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "tourism_metrics.csv", index=False)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small VARA forecast smoke test on tourism data.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/tourism"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tourism_forecast_smoke"))
    parser.add_argument("--max-series", type=int, default=111)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--seasonal-lag", type=int, default=12)
    parser.add_argument(
        "--y-transform",
        type=str,
        default="log1p-seasonal-diff",
        choices=["center", "standardize", "log1p-center", "log1p-standardize", "log1p-seasonal-diff"],
    )
    args = parser.parse_args()

    metrics = run_tourism_forecast_smoke(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_series=args.max_series,
        horizon=args.horizon,
        grid_size=args.grid_size,
        y_transform=args.y_transform,
        seasonal_lag=args.seasonal_lag,
        ridge=args.ridge,
    )
    print("Tourism VARA forecast smoke metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
