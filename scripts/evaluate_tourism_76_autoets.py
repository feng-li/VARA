#!/usr/bin/env python3
"""Generate long-history rolling AutoETS forecasts for 76 Tourism totals."""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version as package_version
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from _bootstrap import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT  # noqa: F401
from vara.forecast_baselines import (
    AUTO_ETS_DAMPED,
    AUTO_ETS_MODEL_SPECIFICATION,
    AUTO_ETS_PHI,
    auto_ets_baseline_panel,
    auto_ets_config,
    normalize_training_panel,
)


RESPONSE_TRANSFORMS = ("none", "log1p")
RESPONSE_NORMALIZATIONS = ("none", "training_zscore")


def _all_series_ids(columns: list[str], series_depth: int) -> list[str]:
    ids = sorted(
        str(column)
        for column in columns
        if str(column).endswith("All")
        and len(str(column)[:-3]) == int(series_depth)
    )
    if not ids:
        raise ValueError(
            f"No all-purpose Tourism series found at depth {series_depth}."
        )
    return ids


def _transform_response(values: np.ndarray, method: str) -> np.ndarray:
    """Transform Tourism observations before fitting the rolling AutoETS models."""
    data = np.asarray(values, dtype=float)
    if method == "none":
        return data.copy()
    if method == "log1p":
        if (data < 0.0).any():
            raise ValueError("log1p response transformation requires nonnegative data.")
        return np.log1p(data)
    raise ValueError(f"Unknown response transformation: {method}")


def _normalize_training_history(
    values: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Normalize each series using statistics from one origin's training data."""
    return normalize_training_panel(values, method)


def _forecast_origin(
    history: np.ndarray,
    series_ids: list[str],
    origin_index: int,
    target_date: str,
    history_start: str,
    history_end: str,
    season_length: int,
    clip_nonnegative: bool,
    model_specification: str,
    damped: bool | None,
    phi: float | None,
    response_normalization: str,
) -> dict:
    values = np.asarray(history, dtype=float)
    target = pd.Timestamp(target_date)
    information_start = pd.Timestamp(history_start)
    information_end = pd.Timestamp(history_end)
    if values.ndim != 2 or values.shape[1] != len(series_ids):
        raise ValueError("AutoETS history must have shape (T, 76).")
    if len(values) < 2 or len(values) > int(origin_index):
        raise ValueError("AutoETS history length is invalid for the origin.")
    if information_start > information_end:
        raise ValueError("AutoETS history start must not follow its end.")
    if information_end >= target:
        raise ValueError("AutoETS history must end before the target date.")
    result = auto_ets_baseline_panel(
        values,
        horizon=1,
        seasonal_lag=int(season_length),
        series_ids=series_ids,
        model_specification=model_specification,
        damped=damped,
        phi=phi,
        response_normalization=response_normalization,
    )
    forecast = np.asarray(result["forecast"][0], dtype=float)
    forecast_variance = np.asarray(
        result["forecast_variance"][0],
        dtype=float,
    )
    normalized_forecast = np.asarray(
        result["forecast_normalized"][0],
        dtype=float,
    )
    normalized_forecast_variance = np.asarray(
        result["forecast_variance_normalized"][0],
        dtype=float,
    )
    if forecast.shape != (len(series_ids),):
        raise ValueError("AutoETS returned an unexpected forecast shape.")
    if forecast_variance.shape != (len(series_ids),):
        raise ValueError("AutoETS returned an unexpected variance shape.")
    training_mean = np.asarray(result["training_mean"], dtype=float)
    training_variance = np.asarray(result["training_variance"], dtype=float)
    training_scale = np.asarray(result["training_scale"], dtype=float)
    if clip_nonnegative:
        forecast = np.maximum(forecast, 0.0)
    return {
        "origin_index": int(origin_index),
        "date": str(target.date()),
        "history_start": str(information_start.date()),
        "history_end": str(information_end.date()),
        "n_train": int(len(values)),
        "forecast": forecast,
        "forecast_variance": forecast_variance,
        "forecast_normalized": normalized_forecast,
        "forecast_variance_normalized": normalized_forecast_variance,
        "training_mean": training_mean,
        "training_variance": training_variance,
        "training_scale": training_scale,
    }


def run_backtest(
    *,
    data_path: Path,
    output_path: Path,
    first_year: int,
    final_year: int,
    series_depth: int,
    expected_n_series: int,
    season_length: int,
    ray_num_cpus: int,
    max_cpus_per_task: int,
    clip_nonnegative: bool,
    response_transform: str = "none",
    response_normalization: str = "none",
    model_specification: str = AUTO_ETS_MODEL_SPECIFICATION,
    damped: bool | None = AUTO_ETS_DAMPED,
    phi: float | None = AUTO_ETS_PHI,
    training_window_months: int | None = None,
) -> pd.DataFrame:
    if int(first_year) > int(final_year):
        raise ValueError("first_year must not exceed final_year.")
    if training_window_months is not None and int(training_window_months) < 2:
        raise ValueError("training_window_months must be at least two.")
    if response_normalization not in RESPONSE_NORMALIZATIONS:
        raise ValueError(
            f"response_normalization must be one of {RESPONSE_NORMALIZATIONS}."
        )
    configured_autoets = auto_ets_config(
        season_length,
        model_specification=model_specification,
        damped=damped,
        phi=phi,
    )
    data = pd.read_csv(data_path)
    if "date" not in data.columns:
        raise ValueError(f"Expected a date column in {data_path}.")
    data["date"] = pd.to_datetime(data["date"])
    if data["date"].duplicated().any() or not data["date"].is_monotonic_increasing:
        raise ValueError("Tourism dates must be unique and increasing.")
    series_ids = _all_series_ids(list(data.columns), series_depth)
    if len(series_ids) != int(expected_n_series):
        raise ValueError(
            f"Expected {expected_n_series} all-purpose series, found {len(series_ids)}."
        )
    original_values = data.loc[:, series_ids].to_numpy(dtype=float)
    if not np.isfinite(original_values).all():
        raise ValueError("Tourism observations contain non-finite values.")
    values = _transform_response(original_values, response_transform)
    origin_indices = [
        index
        for index, date in enumerate(data["date"])
        if int(first_year) <= int(date.year) <= int(final_year)
    ]
    if not origin_indices or min(origin_indices) < 2:
        raise ValueError("The requested origins need at least two prior observations.")
    if (
        training_window_months is not None
        and min(origin_indices) < int(training_window_months)
    ):
        raise ValueError(
            "Every requested origin must have the full fixed training window."
        )

    dates = data["date"].to_numpy()
    history_slices = {
        int(origin_index): (
            0
            if training_window_months is None
            else int(origin_index) - int(training_window_months),
            int(origin_index),
        )
        for origin_index in origin_indices
    }
    started = time.time()
    if int(ray_num_cpus) > 1:
        import ray

        started_here = False
        if not ray.is_initialized():
            ray.init(
                num_cpus=int(ray_num_cpus),
                include_dashboard=False,
                ignore_reinit_error=True,
                _node_ip_address="127.0.0.1",
            )
            started_here = True
        try:
            remote = ray.remote(num_cpus=int(max_cpus_per_task))(_forecast_origin)
            series_ref = ray.put(series_ids)
            futures = []
            for origin_index in origin_indices:
                history_start_index, history_stop_index = history_slices[
                    int(origin_index)
                ]
                history_ref = ray.put(
                    values[history_start_index:history_stop_index].copy()
                )
                futures.append(remote.remote(
                    history_ref,
                    series_ref,
                    origin_index,
                    str(pd.Timestamp(dates[origin_index]).date()),
                    str(pd.Timestamp(dates[history_start_index]).date()),
                    str(pd.Timestamp(dates[origin_index - 1]).date()),
                    int(season_length),
                    bool(clip_nonnegative),
                    configured_autoets["model_specification"],
                    configured_autoets["damped"],
                    configured_autoets["phi"],
                    response_normalization,
                ))
            print(
                f"Submitted {len(futures)} rolling 76-series AutoETS origins.",
                flush=True,
            )
            results = []
            pending = list(futures)
            while pending:
                ready, pending = ray.wait(
                    pending,
                    num_returns=min(12, len(pending)),
                    timeout=60.0,
                )
                results.extend(ray.get(ready))
                print(
                    f"Collected {len(results)}/{len(futures)} AutoETS origins.",
                    flush=True,
                )
        finally:
            if started_here:
                ray.shutdown()
    else:
        results = [
            _forecast_origin(
                values[
                    history_slices[int(origin_index)][0]:
                    history_slices[int(origin_index)][1]
                ].copy(),
                series_ids,
                origin_index,
                str(pd.Timestamp(dates[origin_index]).date()),
                str(pd.Timestamp(
                    dates[history_slices[int(origin_index)][0]]
                ).date()),
                str(pd.Timestamp(dates[origin_index - 1]).date()),
                int(season_length),
                bool(clip_nonnegative),
                configured_autoets["model_specification"],
                configured_autoets["damped"],
                configured_autoets["phi"],
                response_normalization,
            )
            for origin_index in origin_indices
        ]

    rows = []
    for result in sorted(results, key=lambda item: item["origin_index"]):
        date = pd.Timestamp(result["date"])
        origin_index = int(result["origin_index"])
        for column_index, series_id in enumerate(series_ids):
            rows.append({
                "date": date,
                "year": int(date.year),
                "month": int(date.month),
                "series_id": series_id,
                "group": series_id[0],
                "actual": float(values[origin_index, column_index]),
                "direct_auto_ets": float(result["forecast"][column_index]),
                "direct_auto_ets_variance": float(
                    result["forecast_variance"][column_index]
                ),
                "direct_auto_ets_normalized": float(
                    result["forecast_normalized"][column_index]
                ),
                "direct_auto_ets_variance_normalized": float(
                    result["forecast_variance_normalized"][column_index]
                ),
                "autoets_training_mean": float(
                    result["training_mean"][column_index]
                ),
                "autoets_training_variance": float(
                    result["training_variance"][column_index]
                ),
                "autoets_training_scale": float(
                    result["training_scale"][column_index]
                ),
                "autoets_history_start": result["history_start"],
                "autoets_history_end": result["history_end"],
                "autoets_n_train": int(result["n_train"]),
                "response_transform": response_transform,
                "response_normalization": response_normalization,
                "autoets_clip_nonnegative": bool(clip_nonnegative),
                "actual_original": float(
                    original_values[origin_index, column_index]
                ),
            })
    forecasts = pd.DataFrame(rows).sort_values(
        ["date", "series_id"]
    ).reset_index(drop=True)
    if forecasts["series_id"].nunique() != int(expected_n_series):
        raise RuntimeError("The output does not contain exactly 76 series.")
    history_end = pd.to_datetime(forecasts["autoets_history_end"])
    if not (history_end < forecasts["date"]).all():
        raise RuntimeError("Saved AutoETS histories do not precede their targets.")
    if (
        training_window_months is not None
        and not forecasts["autoets_n_train"].eq(int(training_window_months)).all()
    ):
        raise RuntimeError("A saved AutoETS forecast lacks the fixed training window.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output_path, index=False)
    variance_output_path = output_path.parent / "forecast_variance_auto_ets.csv"
    variance_wide = forecasts.pivot(
        index="date",
        columns="series_id",
        values="direct_auto_ets_variance",
    ).sort_index()
    variance_wide.columns.name = None
    variance_wide.reset_index().to_csv(variance_output_path, index=False)
    metadata = {
        "data_path": str(data_path),
        "output_path": str(output_path),
        "first_year": int(first_year),
        "final_year": int(final_year),
        "series_depth": int(series_depth),
        "n_series": int(expected_n_series),
        "n_origins": int(forecasts["date"].nunique()),
        "n_rows": int(len(forecasts)),
        "purpose_component_series_used": False,
        "response_transform": response_transform,
        "response_normalization": response_normalization,
        "model_scale": (
            "log1p visitor nights"
            if response_transform == "log1p"
            else "original visitor nights"
        ),
        "normalization": {
            "method": response_normalization,
            "scope": "per origin and per series",
            "statistics_source": "training observations through t-1 only",
            "variance_ddof": 0,
            "fit_formula": "(response - training_mean) / sqrt(training_variance)",
            "forecast_inverse_formula": (
                "training_mean + sqrt(training_variance) * "
                "normalized_forecast"
            ),
            "constant_series_scale_fallback": 1.0,
            "saved_statistic_columns": [
                "autoets_training_mean",
                "autoets_training_variance",
                "autoets_training_scale",
            ],
        },
        "autoets_config": {
            **configured_autoets,
            "package": "statsforecast",
            "package_version": package_version("statsforecast"),
            "series_fit": "separate",
            "training_window": (
                "expanding"
                if training_window_months is None
                else "fixed"
            ),
            "training_window_months": (
                None
                if training_window_months is None
                else int(training_window_months)
            ),
            "forecast_horizon": 1,
            "refit_every_origin": True,
            "validation_hyperparameter_search": False,
            "clip_nonnegative_after_forecast": bool(clip_nonnegative),
        },
        "forecast_variance_path": str(variance_output_path),
        "leakage_control": (
            "each worker receives observations through t-1 only; response "
            "normalization statistics are computed inside that training slice"
        ),
        "elapsed_seconds": float(time.time() - started),
    }
    output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/tourism/tourism.csv"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tourism_76_autoets_long/rolling_forecasts.csv"),
    )
    parser.add_argument("--first-year", type=int, default=2001)
    parser.add_argument("--final-year", type=int, default=2016)
    parser.add_argument("--series-depth", type=int, default=3)
    parser.add_argument("--expected-n-series", type=int, default=76)
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--ray-num-cpus", type=int, default=104)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    parser.add_argument(
        "--training-window-months",
        type=int,
        default=None,
        help=(
            "Use a fixed trailing training window; omit for the existing "
            "expanding-window behavior."
        ),
    )
    parser.add_argument(
        "--clip-nonnegative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--response-transform",
        choices=RESPONSE_TRANSFORMS,
        default="none",
        help="Transformation applied before fitting every rolling AutoETS model.",
    )
    parser.add_argument(
        "--response-normalization",
        choices=RESPONSE_NORMALIZATIONS,
        default="none",
        help=(
            "Optional per-origin, per-series normalization fitted only on the "
            "AutoETS training observations."
        ),
    )
    parser.add_argument(
        "--model-specification",
        default=AUTO_ETS_MODEL_SPECIFICATION,
        help=(
            "Three-letter ETS family. ZZZ retains standard AutoETS selection; "
            "ANN fixes additive error, no trend, and no seasonality."
        ),
    )
    parser.add_argument(
        "--damped",
        action=argparse.BooleanOptionalAction,
        default=AUTO_ETS_DAMPED,
        help="Force or forbid damping; omit to let standard AutoETS decide.",
    )
    parser.add_argument(
        "--phi",
        type=float,
        default=AUTO_ETS_PHI,
        help="Optional fixed damping coefficient.",
    )
    args = parser.parse_args()
    run_backtest(
        data_path=args.data_path,
        output_path=args.output_path,
        first_year=args.first_year,
        final_year=args.final_year,
        series_depth=args.series_depth,
        expected_n_series=args.expected_n_series,
        season_length=args.season_length,
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
        clip_nonnegative=args.clip_nonnegative,
        response_transform=args.response_transform,
        response_normalization=args.response_normalization,
        model_specification=args.model_specification,
        damped=args.damped,
        phi=args.phi,
        training_window_months=args.training_window_months,
    )


if __name__ == "__main__":
    main()
