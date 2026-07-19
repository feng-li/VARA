#!/usr/bin/env python3
"""Fit unchanged VARA models in leakage-safe local M5 panels.

This driver deliberately lives outside :mod:`vara`.  It partitions the M5
item-store panel before calling the existing ``VARAForecaster`` unchanged, so
it can test the economically homogeneous-panel recommendation without
altering VARA's estimator, likelihood, or attention parameterization.

Each local panel is evaluated as a sequence of one-step rolling origins.  A
single feature file is valid for the entire evaluation block only when it was
created with a holdout at least as long as that block.  This makes the command
useful both for historical validation blocks and for the final held-out month.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

if __package__:
    from scripts.run_vara_forecast import (
        _baseline_forecast_residual_and_variance,
        _fit_y_transform,
        _inverse_y_transform,
        _new_forecaster,
        _unscale_y_transform,
    )
else:
    from run_vara_forecast import (
        _baseline_forecast_residual_and_variance,
        _fit_y_transform,
        _inverse_y_transform,
        _new_forecaster,
        _unscale_y_transform,
    )

from vara.forecast_metrics import forecast_metric_tables
from vara.series_features import load_series_feature_matrix


def _print_progress(message: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}", flush=True)


def _evaluation_origin_offsets(horizon: int, origin_stride: int) -> list[int]:
    """Return ordered one-step origins, always retaining the final date."""
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    if origin_stride < 1:
        raise ValueError("origin_stride must be at least one.")
    offsets = list(range(horizon - 1, -1, -origin_stride))
    offsets.reverse()
    return offsets


def _transform_sales(values: np.ndarray, method: str) -> np.ndarray:
    """Transform original nonnegative sales before fitting the baseline."""
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Sales values must be finite before transformation.")
    if method == "none":
        return values
    if method == "log1p":
        if np.any(values < 0.0):
            raise ValueError("log1p sales transformation requires nonnegative values.")
        return np.log1p(values)
    raise ValueError("sales_transform must be 'none' or 'log1p'.")


def _recover_sales_forecast(
    baseline_model_scale: np.ndarray,
    residual_model_scale: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover an original-scale baseline and additive correction.

    The residual is added before reversing ``log1p``.  Returning the difference
    between the recovered forecasts preserves the exact additive component
    interface expected by the validation/recombination utilities.
    """
    baseline_model_scale = np.asarray(baseline_model_scale, dtype=float)
    residual_model_scale = np.asarray(residual_model_scale, dtype=float)
    if baseline_model_scale.shape != residual_model_scale.shape:
        raise ValueError("Baseline and residual forecasts must have identical shapes.")
    if method == "none":
        baseline = baseline_model_scale
        forecast = baseline_model_scale + residual_model_scale
    elif method == "log1p":
        baseline = np.expm1(baseline_model_scale)
        forecast = np.expm1(baseline_model_scale + residual_model_scale)
        # Original M5 demand is nonnegative.  AutoETS can extrapolate a tiny
        # negative log level, so impose the response support after inversion.
        baseline = np.maximum(baseline, 0.0)
        forecast = np.maximum(forecast, 0.0)
    else:
        raise ValueError("sales_transform must be 'none' or 'log1p'.")
    return baseline, forecast - baseline


def _read_panel(
    path: Path,
    *,
    date_col: str,
    series_col: str,
    value_col: str,
    group_cols: list[str],
    evaluation_end: str | None,
) -> pd.DataFrame:
    columns = [date_col, series_col, value_col, *group_cols]
    if path.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, columns=columns)
    else:
        frame = pd.read_csv(path, usecols=columns)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"M5 panel is missing required columns: {missing}")
    frame = frame.copy()
    frame[date_col] = pd.to_datetime(frame[date_col])
    if evaluation_end is not None:
        end = pd.Timestamp(evaluation_end)
        frame = frame.loc[frame[date_col] <= end].copy()
    if frame.empty:
        raise ValueError("No observations remain after applying evaluation_end.")
    if frame[columns].isna().any().any():
        raise ValueError("M5 local-panel input contains missing required values.")
    return frame


def _group_mapping(
    frame: pd.DataFrame,
    *,
    series_col: str,
    group_cols: list[str],
) -> pd.DataFrame:
    mapping = frame.loc[:, [series_col, *group_cols]].drop_duplicates().copy()
    duplicate = mapping[series_col].duplicated(keep=False)
    if duplicate.any():
        examples = mapping.loc[duplicate, series_col].astype(str).unique().tolist()[:10]
        raise ValueError(
            "Local-panel group columns must be constant within each series; "
            f"conflicting examples: {examples}"
        )
    mapping = mapping.rename(columns={series_col: "series_id"})
    mapping["series_id"] = mapping["series_id"].astype(str)
    mapping["group_id"] = mapping.loc[:, group_cols].astype(str).apply(
        lambda row: "|".join(f"{column}={row[column]}" for column in group_cols),
        axis=1,
    )
    return mapping


def _fit_group_rolling(
    group_id: str,
    series_ids: list[str],
    values: np.ndarray,
    features: np.ndarray,
    *,
    horizon: int,
    origin_stride: int,
    origin_offsets: list[int] | None = None,
    seasonal_lag: int,
    sales_transform: str,
    residual_transform: str,
    grid_size: int,
    ridge: float,
    attention_mode: str,
    attention_top_k: int,
    train_window: int,
    fit_vara: bool = True,
) -> dict[str, Any]:
    """Fit one local panel through all rolling origins in an evaluation block."""
    if values.ndim != 2 or values.shape[1] != len(series_ids):
        raise ValueError("Local panel values must have shape (time, n_series).")
    if values.shape[0] <= horizon + 1:
        raise ValueError("Local panel is too short for the requested forecast block.")
    if features.shape[0] != len(series_ids):
        raise ValueError("Local feature rows do not align to local series ids.")

    import jax.numpy as jnp

    p = len(series_ids)
    # With the default stride of one, this is every rolling one-step origin in
    # the evaluation block.  A wider stride retains repeated, pre-test origins
    # for validation at a cost that is practical for the full M5 hierarchy.
    if origin_offsets is None:
        origin_steps = _evaluation_origin_offsets(horizon, origin_stride)
    else:
        origin_steps = [int(step) for step in origin_offsets]
        if not origin_steps:
            raise ValueError("origin_offsets must contain at least one origin.")
        if len(set(origin_steps)) != len(origin_steps):
            raise ValueError("origin_offsets must not contain duplicates.")
        if min(origin_steps) < 0 or max(origin_steps) >= horizon:
            raise ValueError("origin_offsets must be within the evaluation horizon.")
    n_origins = len(origin_steps)
    baseline = np.empty((n_origins, p), dtype=float)
    residual = np.empty((n_origins, p), dtype=float)
    transformed = np.empty((n_origins, p), dtype=float)
    diagnostics: list[dict[str, Any]] = []
    for output_index, step in enumerate(origin_steps):
        origin_end = values.shape[0] - horizon + step
        history_start = 0 if train_window <= 0 else max(0, origin_end - train_window)
        history = values[history_start:origin_end]
        model_history = _transform_sales(history, sales_transform)
        baseline_model_one, residual_train, _ = _baseline_forecast_residual_and_variance(
            model_history,
            1,
            seasonal_lag,
            series_ids,
            "auto_ets",
        )
        if fit_vara:
            model_train, fitted_transform = _fit_y_transform(
                residual_train,
                residual_transform,
                seasonal_lag,
            )
            forecaster = _new_forecaster(
                grid_size,
                ridge,
                attention_mode,
                attention_top_k,
            )
            forecaster.fit(jnp.asarray(model_train), jnp.asarray(features))
            forecast_standardized = np.asarray(forecaster.predict(h=1))
            residual_model_one = _inverse_y_transform(
                forecast_standardized,
                fitted_transform,
            )
            transformed[output_index] = _unscale_y_transform(
                forecast_standardized,
                fitted_transform,
            )[0]
            diag = forecaster.diagnostics()
        else:
            residual_model_one = np.zeros_like(baseline_model_one)
            transformed[output_index] = 0.0
            diag = {}
        baseline_one, residual_one = _recover_sales_forecast(
            baseline_model_one,
            residual_model_one,
            sales_transform,
        )
        baseline[output_index] = baseline_one[0]
        residual[output_index] = residual_one[0]
        diag.update({
            "group_id": group_id,
            "origin": int(output_index + 1),
            "evaluation_offset": int(step),
            "n_series": p,
            "history_start": int(history_start),
            "n_train": int(history.shape[0]),
            "sales_transform": sales_transform,
            "fit_vara": bool(fit_vara),
        })
        diagnostics.append(diag)
    return {
        "group_id": group_id,
        "series_ids": series_ids,
        "origin_offsets": origin_steps,
        "baseline": baseline,
        "residual": residual,
        "transformed": transformed,
        "diagnostics": diagnostics,
    }


def _write_wide(
    path: Path,
    dates: np.ndarray,
    series_ids: list[str],
    values: np.ndarray,
) -> None:
    out = pd.DataFrame(values, columns=series_ids)
    out.insert(0, "date", dates)
    out.to_csv(path, index=False)


def _read_wide_forecast(
    path: Path,
    dates: np.ndarray,
    series_ids: list[str],
) -> np.ndarray:
    """Read an externally computed baseline and align it exactly to this run."""
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"Baseline forecast {path} has no 'date' column.")
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].duplicated().any():
        raise ValueError(f"Baseline forecast {path} has duplicate dates.")
    missing_columns = sorted(set(series_ids) - set(frame.columns))
    if missing_columns:
        raise ValueError(
            f"Baseline forecast {path} is missing series columns: "
            f"{missing_columns[:10]}"
        )
    aligned = frame.set_index("date").reindex(pd.to_datetime(dates))
    if aligned.loc[:, series_ids].isna().any().any():
        missing_dates = aligned.index[aligned.loc[:, series_ids].isna().any(axis=1)]
        raise ValueError(
            f"Baseline forecast {path} does not cover all requested dates; "
            f"examples: {[str(date) for date in missing_dates[:5]]}"
        )
    return aligned.loc[:, series_ids].to_numpy(dtype=float)


def _run_groups_with_ray(
    tasks: list[dict[str, Any]],
    *,
    ray_num_cpus: int,
) -> list[dict[str, Any]]:
    import ray

    started = False
    if not ray.is_initialized():
        _print_progress(f"Starting Ray: ray_num_cpus={ray_num_cpus}")
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=ray_num_cpus,
            _node_ip_address="127.0.0.1",
        )
        started = True
    remote_fit = ray.remote(num_cpus=1)(_fit_group_rolling)
    panel_refs: dict[tuple[int, int], tuple[Any, Any]] = {}
    futures = []
    for task in tasks:
        panel_key = (id(task["values"]), id(task["features"]))
        if panel_key not in panel_refs:
            panel_refs[panel_key] = (ray.put(task["values"]), ray.put(task["features"]))
        values_ref, features_ref = panel_refs[panel_key]
        futures.append(
            remote_fit.remote(
                task["group_id"],
                task["series_ids"],
                values_ref,
                features_ref,
                **task["kwargs"],
            )
        )
    _print_progress(f"Submitted local VARA panel tasks: groups={len(futures)}")
    pending = list(futures)
    results = []
    progress = tqdm(total=len(futures), desc="M5 local panels", ncols=100)
    try:
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=60.0)
            if not ready:
                _print_progress(
                    f"Local VARA wait: done={len(results)}/{len(futures)}, pending={len(pending)}"
                )
                continue
            for ref in ready:
                results.append(ray.get(ref))
                progress.update(1)
    finally:
        progress.close()
        if started:
            _print_progress("Shutting down Ray")
            ray.shutdown()
            _print_progress("Ray shut down")
    return results


def run_m5_local_vara(
    *,
    data_path: Path,
    feature_path: Path,
    output_dir: Path,
    horizon: int = 28,
    origin_stride: int = 1,
    evaluation_end: str | None = None,
    group_cols: tuple[str, ...] = ("dept_id", "store_id"),
    seasonal_lag: int = 7,
    sales_transform: str = "none",
    residual_transform: str = "standardize",
    grid_size: int = 7,
    ridge: float = 0.01,
    attention_mode: str = "dense",
    attention_top_k: int = 50,
    train_window: int = 0,
    ray_num_cpus: int = 70,
    max_groups: int = 0,
    clip_nonnegative: bool = True,
    origins_per_task: int = 0,
    group_shrinkage_path: Path | None = None,
    baseline_forecast_path: Path | None = None,
) -> dict[str, Any]:
    """Run local-panel VARA and persist unshrunk correction components.

    ``train_window=0`` uses every historical observation preceding each origin.
    Positive values select a recent fixed-length window without touching the
    VARA implementation.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive.")
    if origin_stride < 1:
        raise ValueError("origin_stride must be at least one.")
    if origins_per_task < 0:
        raise ValueError("origins_per_task must be zero or a positive integer.")
    if sales_transform not in {"none", "log1p"}:
        raise ValueError("sales_transform must be 'none' or 'log1p'.")
    if residual_transform not in {"center", "standardize", "robust-standardize"}:
        raise ValueError(
            "residual_transform must be 'center', 'standardize', or "
            "'robust-standardize'."
        )
    if ray_num_cpus < 1:
        raise ValueError("ray_num_cpus must be at least one.")
    group_cols_list = [str(column) for column in group_cols]
    if not group_cols_list:
        raise ValueError("At least one group column is required.")

    frame = _read_panel(
        Path(data_path),
        date_col="date",
        series_col="series_id",
        value_col="value",
        group_cols=group_cols_list,
        evaluation_end=evaluation_end,
    )
    mapping = _group_mapping(
        frame,
        series_col="series_id",
        group_cols=group_cols_list,
    )
    series_ids = mapping["series_id"].tolist()
    dates = pd.Series(frame["date"].unique()).sort_values().to_numpy()
    if len(dates) <= horizon + 1:
        raise ValueError("M5 panel is too short for the requested horizon.")
    origin_offsets = np.asarray(
        _evaluation_origin_offsets(horizon, origin_stride), dtype=int
    )
    test_dates = dates[-horizon + origin_offsets]

    Z, feature_frame, feature_cols = load_series_feature_matrix(
        Path(feature_path),
        series_ids,
        standardize_output=True,
    )
    mapping["column_index"] = np.arange(len(mapping), dtype=int)
    available_group_ids = sorted(mapping["group_id"].unique().tolist())
    group_ids = list(available_group_ids)
    n_available_groups = len(available_group_ids)
    if group_shrinkage_path is not None:
        shrinkage = pd.read_csv(group_shrinkage_path)
        required = {"group_id", "residual_shrinkage"}
        missing = sorted(required - set(shrinkage.columns))
        if missing:
            raise ValueError(
                f"Group shrinkage file {group_shrinkage_path} is missing columns: {missing}"
            )
        if shrinkage["group_id"].duplicated().any():
            raise ValueError(f"Group shrinkage file {group_shrinkage_path} has duplicate groups.")
        known_groups = set(shrinkage["group_id"].astype(str))
        absent = sorted(set(group_ids) - known_groups)
        if absent:
            raise ValueError(
                f"Group shrinkage file {group_shrinkage_path} is missing M5 groups: {absent[:10]}"
            )
        active_groups = set(
            shrinkage.loc[
                pd.to_numeric(shrinkage["residual_shrinkage"], errors="raise") > 0.0,
                "group_id",
            ].astype(str)
        )
        group_ids = [group_id for group_id in group_ids if group_id in active_groups]
    if max_groups > 0:
        group_ids = group_ids[:max_groups]
    if not group_ids and baseline_forecast_path is not None:
        _print_progress("No local VARA groups selected; using the external baseline only.")

    tasks = []
    model_group_ids = set(group_ids)
    # Without an external baseline, fit AutoETS for every panel but skip VARA
    # for validation-suppressed panels.  This guarantees that the benchmark
    # uses the same sales transform and rolling origins as the focal model.
    task_group_ids = (
        group_ids if baseline_forecast_path is not None else available_group_ids
    )
    for group_id in task_group_ids:
        group_mapping = mapping.loc[mapping["group_id"] == group_id]
        local_ids = group_mapping["series_id"].tolist()
        values = (
            frame.loc[frame["series_id"].astype(str).isin(local_ids), ["date", "series_id", "value"]]
            .assign(series_id=lambda x: x["series_id"].astype(str))
            .pivot(index="date", columns="series_id", values="value")
            .reindex(index=dates, columns=local_ids)
            .to_numpy(dtype=float)
        )
        if not np.isfinite(values).all():
            raise ValueError(f"Local M5 panel {group_id} has missing/non-finite values.")
        positions = group_mapping["column_index"].to_numpy(dtype=int)
        local_features = Z[positions]
        selected_offsets = origin_offsets.tolist()
        # Rejected panels still need an exactly matched logged-scale AutoETS
        # baseline, but no VARA refits. Keep those origins in one baseline-only
        # task; retained panels remain split to expose VARA parallelism.
        effective_origins_per_task = (
            origins_per_task if group_id in model_group_ids else 0
        )
        offset_chunks = (
            [selected_offsets]
            if effective_origins_per_task == 0
            else [
                selected_offsets[start:start + effective_origins_per_task]
                for start in range(0, len(selected_offsets), effective_origins_per_task)
            ]
        )
        for chunk in offset_chunks:
            tasks.append({
                "group_id": group_id,
                "series_ids": local_ids,
                "values": values,
                "features": local_features,
                "kwargs": {
                    "horizon": int(horizon),
                    "origin_stride": int(origin_stride),
                    "origin_offsets": chunk,
                    "seasonal_lag": int(seasonal_lag),
                    "sales_transform": sales_transform,
                    "residual_transform": residual_transform,
                    "grid_size": int(grid_size),
                    "ridge": float(ridge),
                    "attention_mode": attention_mode,
                    "attention_top_k": int(attention_top_k),
                    "train_window": int(train_window),
                    "fit_vara": group_id in model_group_ids,
                },
            })

    started = time.time()
    if not tasks:
        results = []
    elif len(tasks) == 1 or ray_num_cpus == 1:
        results = [
            _fit_group_rolling(
                task["group_id"],
                task["series_ids"],
                task["values"],
                task["features"],
                **task["kwargs"],
            )
            for task in tasks
        ]
    else:
        results = _run_groups_with_ray(tasks, ray_num_cpus=min(ray_num_cpus, len(tasks)))

    test = (
        frame.loc[frame["date"].isin(test_dates), ["date", "series_id", "value"]]
        .assign(series_id=lambda x: x["series_id"].astype(str))
        .pivot(index="date", columns="series_id", values="value")
        .reindex(index=test_dates, columns=series_ids)
        .to_numpy(dtype=float)
    )
    if not np.isfinite(test).all():
        raise ValueError("Final M5 test block contains missing/non-finite values.")

    baseline = (
        _read_wide_forecast(Path(baseline_forecast_path), test_dates, series_ids)
        if baseline_forecast_path is not None
        else np.full((len(test_dates), len(series_ids)), np.nan, dtype=float)
    )
    residual = np.zeros_like(baseline)
    transformed = np.zeros_like(baseline)
    diagnostics = []
    output_row_by_offset = {
        int(offset): row for row, offset in enumerate(origin_offsets.tolist())
    }
    for result in results:
        indexed = mapping.set_index("series_id")
        positions = indexed.loc[result["series_ids"], "column_index"].to_numpy(dtype=int)
        result_offsets = [int(offset) for offset in result["origin_offsets"]]
        output_rows = np.asarray(
            [output_row_by_offset[offset] for offset in result_offsets], dtype=int
        )
        expected_shape = (len(output_rows), len(positions))
        for name in ("baseline", "residual", "transformed"):
            if np.asarray(result[name]).shape != expected_shape:
                raise RuntimeError(
                    f"Local result {name} for group {result['group_id']} has shape "
                    f"{np.asarray(result[name]).shape}; expected {expected_shape}."
                )
        if baseline_forecast_path is None:
            baseline[np.ix_(output_rows, positions)] = result["baseline"]
        residual[np.ix_(output_rows, positions)] = result["residual"]
        transformed[np.ix_(output_rows, positions)] = result["transformed"]
        diagnostics.extend(result["diagnostics"])
    missing_series = np.isnan(baseline).all(axis=0)
    if missing_series.any():
        missing = np.asarray(series_ids, dtype=object)[missing_series].tolist()[:10]
        raise RuntimeError(
            "Local VARA did not produce a baseline for every series. "
            f"Increase --max-groups or provide all groups; missing examples: {missing}"
        )
    forecast = baseline + residual
    if clip_nonnegative:
        forecast = np.maximum(forecast, 0.0)
    errors = forecast - test

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_wide(output_dir / "forecast.csv", test_dates, series_ids, forecast)
    _write_wide(output_dir / "baseline_forecast.csv", test_dates, series_ids, baseline)
    _write_wide(output_dir / "residual_forecast.csv", test_dates, series_ids, residual)
    _write_wide(output_dir / "residual_forecast_unshrunk.csv", test_dates, series_ids, residual)
    _write_wide(output_dir / "forecast_transformed.csv", test_dates, series_ids, transformed)
    _write_wide(output_dir / "test.csv", test_dates, series_ids, test)
    _write_wide(output_dir / "errors.csv", test_dates, series_ids, errors)
    mapping.drop(columns="column_index").to_csv(output_dir / "series_groups.csv", index=False)
    feature_frame.to_csv(output_dir / "attention_features.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(output_dir / "rolling_origin_diagnostics.csv", index=False)

    train_values = (
        frame.loc[frame["date"] < test_dates[0], ["date", "series_id", "value"]]
        .assign(series_id=lambda x: x["series_id"].astype(str))
        .pivot(index="date", columns="series_id", values="value")
        .reindex(columns=series_ids)
        .to_numpy(dtype=float)
    )
    by_series, by_origin, summary = forecast_metric_tables(
        {"vara_local": forecast},
        test,
        train_values,
        test_dates,
        series_ids,
        season_length=seasonal_lag,
        forecast_mode="rolling-origin",
        horizon=1,
    )
    by_series.to_csv(output_dir / "metrics_by_series.csv", index=False)
    by_origin.to_csv(output_dir / "metrics_by_origin.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    metrics = {
        "source": str(data_path),
        "feature_path": str(feature_path),
        "n_series": int(len(series_ids)),
        "n_groups": int(len(group_ids)),
        "n_tasks": int(len(tasks)),
        "n_available_groups": int(n_available_groups),
        "group_cols": ",".join(group_cols_list),
        "horizon": int(horizon),
        "n_evaluation_origins": int(len(test_dates)),
        "origin_stride": int(origin_stride),
        "origins_per_task": int(origins_per_task),
        "forecast_type": "rolling-origin",
        "forecast_scenario": "auto_ets",
        "sales_transform": sales_transform,
        "residual_transform": residual_transform,
        "residual_shrinkage": 1.0,
        "clip_nonnegative": bool(clip_nonnegative),
        "seasonal_lag": int(seasonal_lag),
        "train_window": int(train_window),
        "group_shrinkage_path": (
            None if group_shrinkage_path is None else str(group_shrinkage_path)
        ),
        "baseline_forecast_path": (
            None if baseline_forecast_path is None else str(baseline_forecast_path)
        ),
        "n_attention_feature_columns": int(len(feature_cols)),
        "test_start": str(pd.Timestamp(test_dates[0])),
        "test_end": str(pd.Timestamp(test_dates[-1])),
        "elapsed_seconds": float(time.time() - started),
        "rmse": float(summary["rmse"].iloc[0]),
        "mae": float(summary["mae"].iloc[0]),
    }
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "run_config.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_progress(
        "M5 local VARA complete: "
        f"groups={metrics['n_groups']}, rmse={metrics['rmse']:.6g}, mae={metrics['mae']:.6g}"
    )
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit unchanged VARA separately in local M5 economic panels."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/m5/long/sales_train_evaluation_long.parquet"),
    )
    parser.add_argument("--feature-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=28)
    parser.add_argument(
        "--origin-stride",
        type=int,
        default=1,
        help=(
            "Evaluate every Nth one-step origin in the trailing horizon; the "
            "last origin is always retained. Default 1 evaluates every date."
        ),
    )
    parser.add_argument(
        "--origins-per-task",
        type=int,
        default=0,
        help=(
            "Number of selected origins fitted by each Ray task. Zero keeps one "
            "task per local panel; a positive value splits origins for parallelism."
        ),
    )
    parser.add_argument(
        "--evaluation-end",
        type=str,
        default=None,
        help="Last observed date of the evaluation block; default uses the data end.",
    )
    parser.add_argument("--group-cols", type=str, default="dept_id,store_id")
    parser.add_argument("--seasonal-lag", type=int, default=7)
    parser.add_argument(
        "--sales-transform",
        choices=["none", "log1p"],
        default="none",
        help="Transform every original sales series before fitting AutoETS.",
    )
    parser.add_argument(
        "--residual-transform",
        choices=["center", "standardize", "robust-standardize"],
        default="standardize",
    )
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--attention-mode", choices=["dense", "topk"], default="dense")
    parser.add_argument("--attention-top-k", type=int, default=50)
    parser.add_argument(
        "--train-window",
        type=int,
        default=0,
        help="Recent observations retained at each origin; zero uses the full history.",
    )
    parser.add_argument("--ray-num-cpus", type=int, default=70)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument(
        "--group-shrinkage-path",
        type=Path,
        default=None,
        help=(
            "Optional group_residual_shrinkage.csv. Only groups with a positive "
            "selected weight are fitted; suppressed groups use --baseline-forecast-path."
        ),
    )
    parser.add_argument(
        "--baseline-forecast-path",
        type=Path,
        default=None,
        help="Externally computed AutoETS baseline used to fill suppressed local groups.",
    )
    parser.add_argument("--clip-nonnegative", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    group_cols = tuple(part.strip() for part in args.group_cols.split(",") if part.strip())
    run_m5_local_vara(
        data_path=args.data_path,
        feature_path=args.feature_path,
        output_dir=args.output_dir,
        horizon=args.horizon,
        origin_stride=args.origin_stride,
        origins_per_task=args.origins_per_task,
        evaluation_end=args.evaluation_end,
        group_cols=group_cols,
        seasonal_lag=args.seasonal_lag,
        sales_transform=args.sales_transform,
        residual_transform=args.residual_transform,
        grid_size=args.grid_size,
        ridge=args.ridge,
        attention_mode=args.attention_mode,
        attention_top_k=args.attention_top_k,
        train_window=args.train_window,
        ray_num_cpus=args.ray_num_cpus,
        max_groups=args.max_groups,
        clip_nonnegative=args.clip_nonnegative,
        group_shrinkage_path=args.group_shrinkage_path,
        baseline_forecast_path=args.baseline_forecast_path,
    )


if __name__ == "__main__":
    main()
