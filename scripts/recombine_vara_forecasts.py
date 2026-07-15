#!/usr/bin/env python3
"""Recombine existing VARA components with per-series shrinkage weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

if __package__:
    from scripts.run_vara_forecast import (
        _apply_residual_shrinkage,
        _load_residual_shrinkage_by_series,
    )
else:
    from run_vara_forecast import (
        _apply_residual_shrinkage,
        _load_residual_shrinkage_by_series,
    )
from vara.forecast_metrics import forecast_metric_tables


def _read_wide_values(
    path: Path,
    *,
    expected_series: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing forecast component: {path}")
    frame = pd.read_csv(path)
    if frame.shape[1] < 2:
        raise ValueError(f"Forecast component must contain dates and series columns: {path}")
    dates = frame.iloc[:, 0].to_numpy()
    series_ids = [str(column) for column in frame.columns[1:]]
    if expected_series is not None and series_ids != list(expected_series):
        raise ValueError(f"Series columns in {path} do not match the source forecast columns.")
    return dates, series_ids, frame.iloc[:, 1:].to_numpy(dtype=float)


def _write_wide_values(
    output_dir: Path,
    filename: str,
    dates: np.ndarray,
    series_ids: Sequence[str],
    values: np.ndarray,
) -> None:
    frame = pd.DataFrame(values, columns=list(series_ids))
    frame.insert(0, "date", dates)
    frame.to_csv(output_dir / filename, index=False)


def _read_source_metadata(source_dir: Path) -> tuple[pd.Series, pd.Series]:
    metrics_path = source_dir / "metrics.csv"
    summary_path = source_dir / "metrics_summary.csv"
    if not metrics_path.exists() or not summary_path.exists():
        raise FileNotFoundError(
            f"Source directory must contain metrics.csv and metrics_summary.csv: {source_dir}"
        )
    metrics = pd.read_csv(metrics_path)
    summary = pd.read_csv(summary_path)
    if metrics.empty or summary.empty:
        raise ValueError(f"Source metric files cannot be empty: {source_dir}")
    return metrics.iloc[0], summary.iloc[0]


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no", ""}:
            return False
    return bool(value)


def _source_shrinkage_weights(
    source_dir: Path,
    series_ids: Sequence[str],
    source_metrics: pd.Series,
) -> np.ndarray:
    path = source_dir / "residual_shrinkage_by_series.csv"
    if path.exists():
        frame = _load_residual_shrinkage_by_series(path, series_ids)
        return frame["residual_shrinkage"].to_numpy(dtype=float)
    if "residual_shrinkage" not in source_metrics or pd.isna(source_metrics["residual_shrinkage"]):
        raise ValueError("Source metrics do not identify the applied residual shrinkage.")
    weight = float(source_metrics["residual_shrinkage"])
    if not np.isfinite(weight) or weight < 0.0:
        raise ValueError("Source residual shrinkage must be finite and nonnegative.")
    return np.full(len(series_ids), weight, dtype=float)


def _resolve_unshrunk_residual(
    source_dir: Path,
    series_ids: Sequence[str],
    baseline: np.ndarray,
    source_forecast: np.ndarray,
    source_weights: np.ndarray,
    source_metrics: pd.Series,
    *,
    clip_nonnegative: bool,
) -> tuple[np.ndarray, str]:
    explicit_path = source_dir / "residual_forecast_unshrunk.csv"
    if explicit_path.exists():
        _, _, raw_residual = _read_wide_values(
            explicit_path,
            expected_series=series_ids,
        )
        source_name = explicit_path.name
    else:
        transformed_path = source_dir / "forecast_transformed.csv"
        residual_transform = str(source_metrics.get("residual_transform", ""))
        if transformed_path.exists() and residual_transform in {"center", "standardize"}:
            _, _, raw_residual = _read_wide_values(
                transformed_path,
                expected_series=series_ids,
            )
            source_name = transformed_path.name
        else:
            residual_path = source_dir / "residual_forecast.csv"
            _, _, applied_residual = _read_wide_values(
                residual_path,
                expected_series=series_ids,
            )
            if clip_nonnegative or np.any(source_weights <= 1e-12):
                raise ValueError(
                    "Cannot recover an exact unshrunk residual from clipped or zero-weight "
                    "source forecasts. Re-run VARA with the updated output files."
                )
            raw_residual = applied_residual / source_weights[None, :]
            source_name = f"{residual_path.name}/source_weight"

    reconstructed = baseline + source_weights[None, :] * raw_residual
    if clip_nonnegative:
        reconstructed = np.maximum(reconstructed, 0.0)
    finite = np.isfinite(reconstructed) & np.isfinite(source_forecast)
    max_difference = (
        float(np.max(np.abs(reconstructed[finite] - source_forecast[finite])))
        if finite.any()
        else float("inf")
    )
    tolerance = 1e-8 * (1.0 + float(np.nanmax(np.abs(source_forecast))))
    if max_difference > tolerance:
        raise ValueError(
            f"Unshrunk residual source {source_name} does not reconstruct the source "
            f"forecast; max difference={max_difference:.6g}, tolerance={tolerance:.6g}."
        )
    return raw_residual, source_name


def _resolve_unshrunk_residual_variance(
    source_dir: Path,
    series_ids: Sequence[str],
    source_weights: np.ndarray,
    target_weights: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, str]:
    explicit_path = source_dir / "residual_forecast_variance_unshrunk.csv"
    if explicit_path.exists():
        _, _, raw_variance = _read_wide_values(
            explicit_path,
            expected_series=series_ids,
        )
        return raw_variance, explicit_path.name

    applied_path = source_dir / "residual_forecast_variance.csv"
    if not applied_path.exists():
        return np.full(shape, np.nan, dtype=float), "unavailable"
    _, _, applied_variance = _read_wide_values(
        applied_path,
        expected_series=series_ids,
    )
    unknown = (source_weights <= 1e-12) & (target_weights > 1e-12)
    if unknown.any():
        missing_ids = np.asarray(series_ids, dtype=object)[unknown]
        raise ValueError(
            "Cannot recover unshrunk residual variance for zero-weight source series: "
            f"{missing_ids[:10].tolist()}"
        )
    raw_variance = np.zeros_like(applied_variance, dtype=float)
    recoverable = source_weights > 1e-12
    raw_variance[:, recoverable] = (
        applied_variance[:, recoverable] / source_weights[None, recoverable] ** 2
    )
    return raw_variance, f"{applied_path.name}/source_weight^2"


def _candidate_scale_paths(source_dir: Path, explicit_paths: Sequence[Path]) -> list[Path]:
    paths = [source_dir / "metrics_by_series.csv", *[Path(path) for path in explicit_paths]]
    workflow_config_path = source_dir.parent / "config_forecast_workflow.json"
    if workflow_config_path.exists():
        config = json.loads(workflow_config_path.read_text(encoding="utf-8"))
        benchmark_dir = config.get("benchmark_dir") or config.get("effective_benchmark_dir")
        if benchmark_dir:
            paths.append(Path(benchmark_dir) / "metrics_by_series.csv")
    unique = []
    seen = set()
    for path in paths:
        normalized = str(path)
        if normalized not in seen and path.exists():
            unique.append(path)
            seen.add(normalized)
    return unique


def _seasonal_scales_from_metrics(
    paths: Sequence[Path],
    series_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    candidates = []
    for path in paths:
        frame = pd.read_csv(path)
        if "series_id" not in frame.columns:
            continue
        series = frame["series_id"].astype(str)
        if {"seasonal_mae_scale", "seasonal_mse_scale"}.issubset(frame.columns):
            mae_scale = pd.to_numeric(frame["seasonal_mae_scale"], errors="coerce")
            mse_scale = pd.to_numeric(frame["seasonal_mse_scale"], errors="coerce")
        elif {"mae", "mase", "rmse", "rmsse"}.issubset(frame.columns):
            mae = pd.to_numeric(frame["mae"], errors="coerce")
            mase = pd.to_numeric(frame["mase"], errors="coerce")
            rmse = pd.to_numeric(frame["rmse"], errors="coerce")
            rmsse = pd.to_numeric(frame["rmsse"], errors="coerce")
            mae_scale = mae / mase.where(np.abs(mase) > 1e-12)
            mse_scale = rmse ** 2 / (rmsse ** 2).where(np.abs(rmsse) > 1e-12)
        else:
            continue
        candidates.append(pd.DataFrame({
            "series_id": series,
            "seasonal_mae_scale": mae_scale,
            "seasonal_mse_scale": mse_scale,
        }))
    if not candidates:
        raise ValueError("No metric table contained enough information to recover seasonal scales.")

    combined = pd.concat(candidates, ignore_index=True, sort=False)
    for column in ("seasonal_mae_scale", "seasonal_mse_scale"):
        combined.loc[
            ~np.isfinite(combined[column]) | (combined[column] <= 1e-12),
            column,
        ] = np.nan
    scales = combined.groupby("series_id", sort=False)[
        ["seasonal_mae_scale", "seasonal_mse_scale"]
    ].median()
    scales = scales.reindex([str(series_id) for series_id in series_ids])
    missing = scales.isna().any(axis=1)
    if missing.any():
        missing_ids = scales.index[missing].tolist()
        raise ValueError(
            "Could not recover seasonal scales for selected series; provide another "
            f"--scale-metrics-path. Missing examples: {missing_ids[:10]}"
        )
    return (
        scales["seasonal_mae_scale"].to_numpy(dtype=float),
        scales["seasonal_mse_scale"].to_numpy(dtype=float),
    )


def _summary_context(source_summary: pd.Series) -> dict[str, Any]:
    keys = (
        "data",
        "series_depth",
        "season_length",
        "train_start",
        "train_end",
        "test_start",
        "test_end",
    )
    return {
        key: source_summary[key]
        for key in keys
        if key in source_summary and pd.notna(source_summary[key])
    }


def _group_metric_table(
    forecast: np.ndarray,
    forecast_variance: np.ndarray,
    test: np.ndarray,
    dates: np.ndarray,
    series_ids: Sequence[str],
    shrinkage_table: pd.DataFrame,
    seasonal_mae_scale: np.ndarray,
    seasonal_mse_scale: np.ndarray,
    *,
    season_length: int,
    forecast_mode: str,
    model_name: str,
) -> pd.DataFrame:
    if "group_id" not in shrinkage_table.columns:
        return pd.DataFrame()
    rows = []
    labels = shrinkage_table["group_id"].astype(str).to_numpy()
    for group_id in sorted(pd.unique(labels)):
        mask = labels == group_id
        _, _, summary = forecast_metric_tables(
            {model_name: forecast[:, mask]},
            test[:, mask],
            None,
            dates,
            np.asarray(series_ids, dtype=object)[mask].tolist(),
            season_length=season_length,
            forecast_mode=forecast_mode,
            horizon=test.shape[0],
            forecast_variances={model_name: forecast_variance[:, mask]},
            seasonal_mae_scale=seasonal_mae_scale[mask],
            seasonal_mse_scale=seasonal_mse_scale[mask],
        )
        row = summary.iloc[0].to_dict()
        row["group_id"] = group_id
        row["residual_shrinkage"] = float(
            shrinkage_table.loc[mask, "residual_shrinkage"].iloc[0]
        )
        rows.append(row)
    columns = ["group_id", "residual_shrinkage"]
    out = pd.DataFrame(rows)
    return out.loc[:, [*columns, *[column for column in out.columns if column not in columns]]]


def recombine_vara_forecasts(
    source_dir: Path,
    shrinkage_file: Path,
    output_dir: Path,
    *,
    scale_metrics_paths: Sequence[Path] = (),
    clip_nonnegative: bool | None = None,
    model_name: str = "vara_group_shrinkage",
    overwrite: bool = False,
) -> dict[str, Any]:
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    source_metrics, source_summary = _read_source_metadata(source_dir)
    dates, series_ids, baseline = _read_wide_values(source_dir / "baseline_forecast.csv")
    test_dates, _, test = _read_wide_values(
        source_dir / "test.csv",
        expected_series=series_ids,
    )
    _, _, source_forecast = _read_wide_values(
        source_dir / "forecast.csv",
        expected_series=series_ids,
    )
    if not np.array_equal(dates.astype(str), test_dates.astype(str)):
        raise ValueError("Source baseline and test dates do not match.")

    shrinkage_table = _load_residual_shrinkage_by_series(shrinkage_file, series_ids)
    target_weights = shrinkage_table["residual_shrinkage"].to_numpy(dtype=float)
    source_weights = _source_shrinkage_weights(source_dir, series_ids, source_metrics)
    effective_clip = (
        _as_bool(source_metrics.get("clip_nonnegative", False))
        if clip_nonnegative is None
        else bool(clip_nonnegative)
    )
    raw_residual, raw_residual_source = _resolve_unshrunk_residual(
        source_dir,
        series_ids,
        baseline,
        source_forecast,
        source_weights,
        source_metrics,
        clip_nonnegative=effective_clip,
    )

    baseline_variance_path = source_dir / "baseline_forecast_variance.csv"
    if baseline_variance_path.exists():
        _, _, baseline_variance = _read_wide_values(
            baseline_variance_path,
            expected_series=series_ids,
        )
    else:
        baseline_variance = np.full_like(baseline, np.nan, dtype=float)
    raw_residual_variance, raw_variance_source = _resolve_unshrunk_residual_variance(
        source_dir,
        series_ids,
        source_weights,
        target_weights,
        baseline.shape,
    )
    forecast, residual, forecast_variance, residual_variance = _apply_residual_shrinkage(
        baseline,
        raw_residual,
        baseline_variance,
        raw_residual_variance,
        target_weights,
    )
    if effective_clip:
        forecast = np.maximum(forecast, 0.0)
        residual = forecast - baseline
    errors = forecast - test

    scale_paths = _candidate_scale_paths(source_dir, scale_metrics_paths)
    seasonal_mae_scale, seasonal_mse_scale = _seasonal_scales_from_metrics(
        scale_paths,
        series_ids,
    )
    season_length = int(
        source_summary.get("season_length", source_metrics.get("seasonal_lag", 1))
    )
    forecast_mode = str(
        source_summary.get("forecast_mode", source_metrics.get("forecast_type", "unknown"))
    )
    by_series, by_origin, summary = forecast_metric_tables(
        {model_name: forecast},
        test,
        None,
        dates,
        series_ids,
        season_length=season_length,
        forecast_mode=forecast_mode,
        horizon=test.shape[0],
        summary_context=_summary_context(source_summary),
        forecast_variances={model_name: forecast_variance},
        seasonal_mae_scale=seasonal_mae_scale,
        seasonal_mse_scale=seasonal_mse_scale,
    )

    source_variance_path = source_dir / "forecast_variance.csv"
    if source_variance_path.exists():
        _, _, source_variance = _read_wide_values(
            source_variance_path,
            expected_series=series_ids,
        )
    else:
        source_variance = np.full_like(source_forecast, np.nan, dtype=float)
    comparison_models = ["baseline", "vara_source", model_name]
    _, _, comparison = forecast_metric_tables(
        {
            "baseline": baseline,
            "vara_source": source_forecast,
            model_name: forecast,
        },
        test,
        None,
        dates,
        series_ids,
        model_names=comparison_models,
        season_length=season_length,
        forecast_mode=forecast_mode,
        horizon=test.shape[0],
        forecast_variances={
            "baseline": baseline_variance,
            "vara_source": source_variance,
            model_name: forecast_variance,
        },
        seasonal_mae_scale=seasonal_mae_scale,
        seasonal_mse_scale=seasonal_mse_scale,
    )
    group_metrics = _group_metric_table(
        forecast,
        forecast_variance,
        test,
        dates,
        series_ids,
        shrinkage_table,
        seasonal_mae_scale,
        seasonal_mse_scale,
        season_length=season_length,
        forecast_mode=forecast_mode,
        model_name=model_name,
    )

    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; pass --overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_wide_values(output_dir, "forecast.csv", dates, series_ids, forecast)
    _write_wide_values(output_dir, "baseline_forecast.csv", dates, series_ids, baseline)
    _write_wide_values(output_dir, "residual_forecast.csv", dates, series_ids, residual)
    _write_wide_values(
        output_dir,
        "residual_forecast_unshrunk.csv",
        dates,
        series_ids,
        raw_residual,
    )
    _write_wide_values(output_dir, "forecast_variance.csv", dates, series_ids, forecast_variance)
    _write_wide_values(
        output_dir,
        "baseline_forecast_variance.csv",
        dates,
        series_ids,
        baseline_variance,
    )
    _write_wide_values(
        output_dir,
        "residual_forecast_variance.csv",
        dates,
        series_ids,
        residual_variance,
    )
    _write_wide_values(
        output_dir,
        "residual_forecast_variance_unshrunk.csv",
        dates,
        series_ids,
        raw_residual_variance,
    )
    _write_wide_values(output_dir, "test.csv", dates, series_ids, test)
    _write_wide_values(output_dir, "errors.csv", dates, series_ids, errors)
    shrinkage_table.to_csv(output_dir / "residual_shrinkage_by_series.csv", index=False)
    by_series.to_csv(output_dir / "metrics_by_series.csv", index=False)
    by_origin.to_csv(output_dir / "metrics_by_origin.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    comparison.to_csv(output_dir / "metrics_summary_comparison.csv", index=False)
    if not group_metrics.empty:
        group_metrics.to_csv(output_dir / "metrics_by_group.csv", index=False)

    metrics = source_metrics.to_dict()
    metrics.update({
        "source_vara_dir": str(source_dir),
        "residual_shrinkage_mode": "per_series_group",
        "residual_shrinkage_file": str(shrinkage_file),
        "residual_shrinkage": float(np.mean(target_weights)),
        "residual_shrinkage_min": float(np.min(target_weights)),
        "residual_shrinkage_median": float(np.median(target_weights)),
        "residual_shrinkage_max": float(np.max(target_weights)),
        "raw_residual_source": raw_residual_source,
        "raw_residual_variance_source": raw_variance_source,
    })
    for column, value in summary.iloc[0].items():
        if column not in {"model", "data"}:
            metrics[column] = value
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)

    config = {
        "source_dir": str(source_dir),
        "shrinkage_file": str(shrinkage_file),
        "output_dir": str(output_dir),
        "model_name": model_name,
        "clip_nonnegative": effective_clip,
        "formula": "forecast = baseline_forecast + residual_shrinkage * unshrunk_residual_forecast",
        "variance_formula": "forecast_variance = baseline_variance + residual_shrinkage^2 * unshrunk_residual_variance",
        "raw_residual_source": raw_residual_source,
        "raw_residual_variance_source": raw_variance_source,
        "scale_metrics_paths": [str(path) for path in scale_paths],
    }
    (output_dir / "recombination_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )

    show = [
        "model",
        "rmse",
        "mae",
        "oos_r2",
        "oos_r2_zero",
        "bias",
        "crps",
    ]
    print("Recombined forecast metric comparison:")
    print(comparison.loc[:, show].to_string(index=False))
    return {
        "metrics_by_series": by_series,
        "metrics_by_origin": by_origin,
        "metrics_summary": summary,
        "metrics_summary_comparison": comparison,
        "metrics_by_group": group_metrics,
        "config": config,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recombine existing VARA components with per-series shrinkage weights."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--shrinkage-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scale-metrics-path",
        type=Path,
        action="append",
        default=[],
        help="Additional metrics_by_series.csv used to recover MASE/RMSSE scales.",
    )
    parser.add_argument(
        "--clip-nonnegative",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override source nonnegative clipping behavior.",
    )
    parser.add_argument("--model-name", default="vara_group_shrinkage")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    recombine_vara_forecasts(
        args.source_dir,
        args.shrinkage_file,
        args.output_dir,
        scale_metrics_paths=args.scale_metrics_path,
        clip_nonnegative=args.clip_nonnegative,
        model_name=args.model_name,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
