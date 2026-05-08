#!/usr/bin/env python3
"""Stationarity diagnostics for the tourism forecasting panel."""

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
from vara.stationarity import (
    PANEL_TRANSFORMS,
    check_panel_stationarity,
    standardize_panel,
    vara_stability_from_params,
)


def _base_series_columns(df: pd.DataFrame, suffix: str) -> list[str]:
    cols = [c for c in df.columns if c != "date" and c.endswith(suffix)]
    if not cols:
        raise ValueError(f"No series ending with {suffix!r} found.")
    return cols


def _tourism_series_features(series_ids: list[str], suffix: str) -> np.ndarray:
    rows = []
    for series_id in series_ids:
        code = series_id[: -len(suffix)] if suffix else series_id
        is_total = code == "Total"
        rows.append({
            "series_id": series_id,
            "depth": 0 if is_total else len(code),
            "is_total": int(is_total),
            "level_1": "Total" if is_total else code[0],
            "level_2": "None" if is_total or len(code) < 2 else code[1],
            "level_3": "None" if is_total or len(code) < 3 else code[2],
            "is_aggregate": int(is_total or len(code) < 3),
        })
    features = pd.DataFrame(rows).set_index("series_id")
    features = pd.get_dummies(features, columns=["level_1", "level_2", "level_3"], dtype=float)
    Z = features.to_numpy(dtype=float)
    std = Z.std(axis=0, keepdims=True)
    std = np.where(std > 0.0, std, 1.0)
    return (Z - Z.mean(axis=0, keepdims=True)) / std


def _fit_vara_stability(
    Y: np.ndarray,
    Z: np.ndarray,
    grid_size: int,
    eta_bounds: tuple[float, float],
    standardize_y: bool,
    ridge: float,
) -> dict:
    Y_fit = standardize_panel(Y) if standardize_y else Y
    forecaster = VARAForecaster(
        grid_size=grid_size,
        eta_bounds=eta_bounds,
        ridge=ridge,
        dense_threshold=2500,
    )
    forecaster.fit(jnp.asarray(Y_fit), jnp.asarray(Z))
    stability = vara_stability_from_params(
        forecaster.alpha_hat,
        forecaster.beta_hat,
        forecaster.Z_,
        float(forecaster.eta_hat),
        params_scale=forecaster.params_scale,
        attention_mode=forecaster.attention_mode,
        attention_top_k=forecaster.attention_top_k,
        attention_cache=forecaster.attention_cache,
    )
    return {
        "vara_y_standardized": bool(standardize_y),
        "vara_ridge": float(ridge),
        **stability,
        **forecaster.diagnostics(),
    }


def check_tourism_stationarity(
    data_path: Path,
    output_dir: Path,
    series_suffix: str,
    max_series: int,
    holdout: int,
    transform: str,
    seasonal_lag: int,
    regression: str,
    fit_vara: bool,
    standardize_vara_y: bool,
    vara_ridge: float,
    grid_size: int,
    eta_lower: float,
    eta_upper: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(data_path)
    series_ids = _base_series_columns(df, series_suffix)
    if max_series > 0:
        series_ids = series_ids[:max_series]
    if holdout < 0 or holdout >= len(df):
        raise ValueError("holdout must be nonnegative and smaller than the number of rows.")

    train_df = df.iloc[: len(df) - holdout] if holdout else df
    Y_raw = train_df[series_ids].to_numpy(dtype=float)
    Y, rows, stationarity_summary = check_panel_stationarity(
        Y_raw,
        series_ids=series_ids,
        transform=transform,
        seasonal_lag=seasonal_lag,
        regression=regression,
    )
    detail = pd.DataFrame(rows)

    summary = {
        "source": str(data_path),
        "series_suffix": series_suffix,
        "n_series": len(series_ids),
        "n_train_raw": int(Y_raw.shape[0]),
        "n_train_transformed": int(Y.shape[0]),
        "holdout": int(holdout),
        "train_start": str(train_df["date"].iloc[0]),
        "train_end": str(train_df["date"].iloc[-1]),
        "transform": transform,
        "seasonal_lag": int(seasonal_lag),
        "regression": regression,
        **stationarity_summary,
    }
    if fit_vara:
        Z = _tourism_series_features(series_ids, series_suffix)
        summary.update(_fit_vara_stability(
            Y,
            Z,
            grid_size,
            (eta_lower, eta_upper),
            standardize_vara_y,
            vara_ridge,
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    detail_path = output_dir / "tourism_stationarity_by_series.csv"
    summary_path = output_dir / "tourism_stationarity_summary.csv"
    detail.to_csv(detail_path, index=False)
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_path, index=False)
    return detail, summary_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Check stationarity for tourism base series.")
    parser.add_argument("--data-path", type=Path, default=Path("data/tourism/tourism.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tourism_stationarity"))
    parser.add_argument("--series-suffix", type=str, default="All")
    parser.add_argument("--max-series", type=int, default=0, help="0 means use all matching series.")
    parser.add_argument("--holdout", type=int, default=12)
    parser.add_argument(
        "--transform",
        type=str,
        default="log1p-seasonal-diff",
        choices=PANEL_TRANSFORMS,
    )
    parser.add_argument("--seasonal-lag", type=int, default=12)
    parser.add_argument("--regression", type=str, default="c", choices=["c", "ct"])
    parser.add_argument("--fit-vara", action="store_true")
    parser.add_argument("--no-standardize-vara-y", action="store_true")
    parser.add_argument("--vara-ridge", type=float, default=1e-2)
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--eta-lower", type=float, default=-2.0)
    parser.add_argument("--eta-upper", type=float, default=2.0)
    args = parser.parse_args()

    detail, summary = check_tourism_stationarity(
        data_path=args.data_path,
        output_dir=args.output_dir,
        series_suffix=args.series_suffix,
        max_series=args.max_series,
        holdout=args.holdout,
        transform=args.transform,
        seasonal_lag=args.seasonal_lag,
        regression=args.regression,
        fit_vara=args.fit_vara,
        standardize_vara_y=not args.no_standardize_vara_y,
        vara_ridge=args.vara_ridge,
        grid_size=args.grid_size,
        eta_lower=args.eta_lower,
        eta_upper=args.eta_upper,
    )

    print("Tourism stationarity summary:")
    for key, value in summary.iloc[0].items():
        print(f"  {key}: {value}")
    print(f"Saved per-series diagnostics for {len(detail)} series to {args.output_dir}")


if __name__ == "__main__":
    main()
