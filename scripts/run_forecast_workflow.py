#!/usr/bin/env python3
"""Run benchmark, VARA, comparison, and horizon-plot workflows."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vara.forecast_metrics import _normal_crps


DEFAULT_MODELS = "naive,seasonal_naive,drift,auto_ets,auto_arima,ar1,var1,deepar,nbeats"
DEFAULT_FORECAST_MODES = "recursive,rolling-origin,closed-loop"
DEFAULT_FEATURE_FUNCTIONS = (
    "acf_features,heterogeneity,series_length,arch_stat,holt_parameters,sparsity,"
    "count_entropy,hurst,stability,crossing_points,hw_parameters,stl_features,"
    "entropy,intervals,unitroot_kpss,flat_spots,lumpiness,unitroot_pp,frequency,"
    "nonlinearity,guerrero,pacf_features"
)
TOURISM_TABLE7_MODELS = (
    "seasonal_naive,auto_ets,auto_arima,nbeats,ar1_auto_ets,var1_auto_ets,"
    "deepar,ar1,var1"
)
TOURISM_TABLE7_FEATURE_FUNCTIONS = (
    "stl_features,acf_features,pacf_features,entropy,holt_parameters,"
    "hw_parameters,frequency,guerrero"
)
TOURISM_TABLE7_MODEL_LABELS = {
    "vara": "VARA on AutoETS residuals, reduced features",
    "ar1_auto_ets": "AR(1) on AutoETS residuals",
    "var1_auto_ets": "VAR(1) on AutoETS residuals",
    "auto_ets": "AutoETS",
    "auto_arima": "AutoARIMA",
    "nbeats": "N-BEATS",
    "seasonal_naive": "Seasonal naive",
    "deepar": "DeepAR",
    "ar1": "AR(1)",
    "var1": "VAR(1)",
}
M5_ROLLING_MODELS = "naive,seasonal_naive,drift,auto_ets,auto_arima,ar1,ar1_snaive,deepar,nbeats"
M5_MODEL_LABELS = {
    "vara": "VARA on AutoETS residuals",
    "naive": "Naive",
    "seasonal_naive": "Seasonal naive",
    "drift": "Drift",
    "auto_ets": "AutoETS",
    "auto_arima": "AutoARIMA",
    "ar1": "AR(1)",
    "ar1_snaive": "AR(1) on seasonal-naive residuals",
    "deepar": "DeepAR",
    "nbeats": "N-BEATS",
}

DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "m5": {
        "data_path": Path("data/m5/long/sales_train_evaluation_long.parquet"),
        "output_dir": Path("outputs/m5_vara_auto_ets_dense_workflow"),
        "benchmark_dir": Path("outputs/m5_benchmark_all_modes"),
        "benchmark_mode_subdirs": True,
        "data_format": "long",
        "date_col": "date",
        "series_col": "series_id",
        "value_col": "value",
        "series_suffix": "",
        "series_depth": -1,
        "max_series": 500,
        "horizon": 28,
        "seasonal_lag": 7,
        "feature_freq": 7,
        "freq": "D",
        "feature_source": "file",
        "feature_functions": "",
        "feature_path_name": "m5_attention_features_auto_ets.parquet",
        "clip_nonnegative": True,
        "models": M5_ROLLING_MODELS,
        "forecast_mode": "rolling-origin",
        "forecast_scenario": "auto_ets",
        "residual_transform": "center",
        "y_transform": "none",
        "grid_size": 7,
        "ridge": 0.01,
        "attention_mode": "dense",
        "attention_top_k": 50,
        "feature_update_policy": "exact",
        "feature_update_interval": 1,
        "include_full_tsfeatures_vara": False,
    },
    "tourism": {
        "data_path": Path("data/tourism/tourism.csv"),
        "output_dir": Path("outputs/tourism_vara_auto_ets_core8_features"),
        "benchmark_dir": None,
        "benchmark_mode_subdirs": False,
        "data_format": "wide",
        "date_col": "date",
        "series_col": "series_id",
        "value_col": "value",
        "series_suffix": "All",
        "series_depth": 3,
        "max_series": 0,
        "horizon": 12,
        "seasonal_lag": 12,
        "feature_freq": 12,
        "freq": "MS",
        "feature_source": "tsfeatures",
        "feature_functions": TOURISM_TABLE7_FEATURE_FUNCTIONS,
        "feature_path_name": "",
        "clip_nonnegative": True,
        "models": TOURISM_TABLE7_MODELS,
        "forecast_mode": "rolling-origin",
        "forecast_scenario": "auto_ets",
        "residual_transform": "center",
        "y_transform": "none",
        "grid_size": 7,
        "ridge": 0.01,
        "attention_mode": "dense",
        "attention_top_k": 50,
        "feature_update_policy": "exact",
        "feature_update_interval": 1,
        "include_full_tsfeatures_vara": True,
    },
}


def _as_config_value(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _arg_value(args: argparse.Namespace, defaults: dict[str, Any], name: str) -> Any:
    value = getattr(args, name)
    return defaults[name] if value is None else value


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("\n" + shlex.join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _comparison_table(
    benchmark_dir: Path,
    vara_dirs: list[tuple[Path, str | None]],
    output_dir: Path,
) -> pd.DataFrame:
    bench = pd.read_csv(benchmark_dir / "metrics_summary.csv")
    vara_frames = []
    for vara_dir, model_label in vara_dirs:
        frame = pd.read_csv(vara_dir / "metrics_summary.csv")
        if model_label is not None:
            frame = frame.copy()
            frame["model"] = model_label
        vara_frames.append(frame)
    vara = pd.concat(vara_frames, ignore_index=True, sort=False)

    keep = [
        "model",
        "data",
        "forecast_mode",
        "horizon",
        "n_series",
        "n_test",
        "rmse",
        "mae",
        "mape_pct",
        "smape_pct",
        "mase",
        "rmsse",
        "bias",
    ]
    optional_keep = ["crps", "forecast_variance_mean"]
    keep.extend([col for col in optional_keep if col in bench.columns and col in vara.columns])
    missing = [col for col in keep if col not in bench.columns or col not in vara.columns]
    if missing:
        raise ValueError(f"Missing comparison columns: {missing}")

    out = pd.concat([bench[keep], vara[keep]], ignore_index=True, sort=False)
    out = out.sort_values(["forecast_mode", "rmse", "mae"]).copy()
    out["rank_rmse"] = out.groupby("forecast_mode")["rmse"].rank(method="first").astype(int)

    output_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_dir / "metrics_summary_comparison.csv", index=False)
    return out


def _feature_metadata_path(feature_path: Path) -> Path:
    return feature_path.with_suffix(feature_path.suffix + ".metadata.json")


def _feature_outputs_exist(args: argparse.Namespace, config: dict[str, Any]) -> bool:
    if config["feature_source"] != "file":
        return False
    feature_path = config["feature_path"]
    if feature_path is None or not feature_path.exists():
        return False
    if config.get("forecast_scenario") == "auto_ets" and not _feature_metadata_path(feature_path).exists():
        return False
    return True


def _benchmark_outputs_exist(benchmark_dir: Path) -> bool:
    return (
        benchmark_dir.exists()
        and (benchmark_dir / "metrics_summary.csv").exists()
        and (benchmark_dir / "metrics_by_origin.csv").exists()
        and (benchmark_dir / "benchmark_forecast_variance_long.csv").exists()
    )


def _forecast_modes(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _benchmark_dir_for_mode(base_dir: Path, forecast_mode: str, *, use_mode_subdirs: bool) -> Path:
    modes = _forecast_modes(forecast_mode)
    if len(modes) != 1:
        return base_dir
    mode = modes[0]
    if base_dir.name == mode:
        return base_dir
    mode_dir = base_dir / mode
    if use_mode_subdirs or _benchmark_outputs_exist(mode_dir):
        return mode_dir
    return base_dir


def _safe_pct(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return numer / np.where(np.abs(denom) > 1e-12, np.abs(denom), np.nan)


def _nanmean_or_nan(values: np.ndarray, axis=None):
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    count = finite.sum(axis=axis)
    total = np.where(finite, arr, 0.0).sum(axis=axis)
    if axis is None:
        return float(total / count) if int(count) > 0 else np.nan
    return np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)


def _benchmark_needs_series_subset(benchmark_dir: Path, max_series: int) -> bool:
    if max_series <= 0 or not (benchmark_dir / "metrics_summary.csv").exists():
        return False
    summary = pd.read_csv(benchmark_dir / "metrics_summary.csv")
    if "n_series" not in summary.columns or summary.empty:
        return False
    return int(pd.to_numeric(summary["n_series"], errors="coerce").max()) > max_series


def _subset_existing_benchmark_outputs(benchmark_dir: Path, output_dir: Path, max_series: int) -> Path:
    """Create benchmark metric files for the first max_series columns of an existing wide forecast output."""
    if max_series <= 0 or not _benchmark_needs_series_subset(benchmark_dir, max_series):
        return benchmark_dir

    test_path = benchmark_dir / "test.csv"
    summary_path = benchmark_dir / "metrics_summary.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"Cannot subset benchmark outputs without {test_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(summary_path)
    test_wide = pd.read_csv(test_path)
    date_col = test_wide.columns[0]
    selected_series = list(test_wide.columns[1: 1 + max_series])
    if len(selected_series) < max_series:
        raise ValueError(
            f"Requested max_series={max_series}, but {test_path} only contains {len(selected_series)} series."
        )

    test = test_wide[selected_series].to_numpy(dtype=float)
    dates = test_wide[date_col]
    by_origin_rows = []
    by_series_rows = []
    summary_rows = []
    point_long_frames = []
    variance_long_frames = []

    original_by_series = None
    by_series_path = benchmark_dir / "metrics_by_series.csv"
    if by_series_path.exists():
        original_by_series = pd.read_csv(by_series_path)

    for _, summary_row in summary.iterrows():
        model = str(summary_row["model"])
        forecast_path = benchmark_dir / f"forecast_{model}.csv"
        if not forecast_path.exists():
            raise FileNotFoundError(f"Missing forecast file for model {model!r}: {forecast_path}")
        forecast_wide = pd.read_csv(forecast_path)
        forecast = forecast_wide[selected_series].to_numpy(dtype=float)
        forecast_date_col = forecast_wide.columns[0]
        forecast_subset = forecast_wide.loc[:, [forecast_date_col, *selected_series]].copy()
        forecast_subset.to_csv(output_dir / f"forecast_{model}.csv", index=False)
        point_long_frames.append(
            forecast_subset.rename(columns={forecast_date_col: "date"}).melt(
                id_vars="date",
                var_name="series_id",
                value_name=model,
            )
        )

        variance = None
        variance_path = benchmark_dir / f"forecast_variance_{model}.csv"
        if variance_path.exists():
            variance_wide = pd.read_csv(variance_path)
            variance_date_col = variance_wide.columns[0]
            variance_subset = variance_wide.loc[:, [variance_date_col, *selected_series]].copy()
            variance_subset.to_csv(output_dir / f"forecast_variance_{model}.csv", index=False)
            variance = variance_subset[selected_series].to_numpy(dtype=float)
            variance = np.where(np.isfinite(variance) & (variance >= 0.0), variance, np.nan)
            variance_long_frames.append(
                variance_subset.rename(columns={variance_date_col: "date"}).melt(
                    id_vars="date",
                    var_name="series_id",
                    value_name=model,
                )
            )

        error = forecast - test
        abs_error = np.abs(error)
        squared_error = error ** 2
        pct_error = _safe_pct(abs_error, test)
        smape = 2.0 * abs_error / np.where(
            np.abs(test) + np.abs(forecast) > 1e-12,
            np.abs(test) + np.abs(forecast),
            np.nan,
        )
        crps = (
            np.full_like(forecast, np.nan, dtype=float)
            if variance is None
            else _normal_crps(forecast, variance, test)
        )
        include_variance_metrics = (
            variance is not None
            or "crps" in summary.columns
            or "forecast_variance_mean" in summary.columns
        )

        error_subset = forecast_subset.copy()
        error_subset.loc[:, selected_series] = error
        error_subset.to_csv(output_dir / f"errors_{model}.csv", index=False)

        model_by_series = pd.DataFrame({
            "model": model,
            "series_id": selected_series,
            "rmse": np.sqrt(_nanmean_or_nan(squared_error, axis=0)),
            "mae": _nanmean_or_nan(abs_error, axis=0),
            "mape_pct": _nanmean_or_nan(pct_error, axis=0) * 100.0,
            "smape_pct": _nanmean_or_nan(smape, axis=0) * 100.0,
            "bias": _nanmean_or_nan(error, axis=0),
            "mean_actual": _nanmean_or_nan(test, axis=0),
            "mean_forecast": _nanmean_or_nan(forecast, axis=0),
        })
        if include_variance_metrics:
            model_by_series["crps"] = _nanmean_or_nan(crps, axis=0)
            model_by_series["forecast_variance_mean"] = (
                np.full(len(selected_series), np.nan)
                if variance is None
                else _nanmean_or_nan(variance, axis=0)
            )
        if original_by_series is not None:
            scale_cols = [col for col in ["mase", "rmsse"] if col in original_by_series.columns]
            if scale_cols:
                scales = original_by_series.loc[
                    original_by_series["model"].astype(str).eq(model)
                    & original_by_series["series_id"].astype(str).isin(selected_series),
                    ["series_id", *scale_cols],
                ]
                model_by_series = model_by_series.merge(scales, on="series_id", how="left")
        for col in ["mase", "rmsse"]:
            if col not in model_by_series.columns:
                model_by_series[col] = np.nan
        by_series_rows.append(model_by_series)

        model_by_origin = pd.DataFrame({
            "model": model,
            "date": dates,
            "rmse": np.sqrt(_nanmean_or_nan(squared_error, axis=1)),
            "mae": _nanmean_or_nan(abs_error, axis=1),
            "mape_pct": _nanmean_or_nan(pct_error, axis=1) * 100.0,
            "smape_pct": _nanmean_or_nan(smape, axis=1) * 100.0,
            "bias": _nanmean_or_nan(error, axis=1),
        })
        if include_variance_metrics:
            model_by_origin["crps"] = _nanmean_or_nan(crps, axis=1)
            model_by_origin["forecast_variance_mean"] = (
                np.full(len(dates), np.nan)
                if variance is None
                else _nanmean_or_nan(variance, axis=1)
            )
        by_origin_rows.append(model_by_origin)

        new_summary = summary_row.to_dict()
        new_summary.update({
            "n_series": len(selected_series),
            "rmse": float(np.sqrt(_nanmean_or_nan(squared_error))),
            "mae": float(_nanmean_or_nan(abs_error)),
            "mape_pct": float(_nanmean_or_nan(pct_error) * 100.0),
            "smape_pct": float(_nanmean_or_nan(smape) * 100.0),
            "mase": float(_nanmean_or_nan(model_by_series["mase"].to_numpy(dtype=float))),
            "rmsse": float(_nanmean_or_nan(model_by_series["rmsse"].to_numpy(dtype=float))),
            "bias": float(_nanmean_or_nan(error)),
        })
        if include_variance_metrics:
            new_summary["crps"] = float(_nanmean_or_nan(crps))
            new_summary["forecast_variance_mean"] = (
                np.nan if variance is None else float(_nanmean_or_nan(variance))
            )
        else:
            new_summary.pop("crps", None)
            new_summary.pop("forecast_variance_mean", None)
        summary_rows.append(new_summary)

    subset_summary = pd.DataFrame(summary_rows)
    subset_by_series = pd.concat(by_series_rows, ignore_index=True, sort=False)
    subset_by_origin = pd.concat(by_origin_rows, ignore_index=True, sort=False)

    subset_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    subset_by_series.to_csv(output_dir / "metrics_by_series.csv", index=False)
    subset_by_origin.to_csv(output_dir / "metrics_by_origin.csv", index=False)
    test_wide.loc[:, [date_col, *selected_series]].to_csv(output_dir / "test.csv", index=False)
    if point_long_frames:
        point_long = point_long_frames[0]
        for frame in point_long_frames[1:]:
            point_long = point_long.merge(frame, on=["date", "series_id"], how="outer")
        point_long.to_csv(output_dir / "benchmark_forecasts_long.csv", index=False)
    if variance_long_frames:
        variance_long = variance_long_frames[0]
        for frame in variance_long_frames[1:]:
            variance_long = variance_long.merge(frame, on=["date", "series_id"], how="outer")
        variance_long.to_csv(output_dir / "benchmark_forecast_variance_long.csv", index=False)
    return output_dir


def _comparison_benchmark_dir(config: dict[str, Any]) -> Path:
    benchmark_dir = config["benchmark_dir"]
    if config.get("dataset") != "m5":
        return benchmark_dir
    modes = _forecast_modes(config["forecast_mode"])
    if len(modes) != 1:
        return benchmark_dir
    max_series = int(config.get("max_series") or 0)
    if max_series <= 0:
        return benchmark_dir
    subset_dir = config["comparison_dir"] / f"benchmark_first_{max_series}_{modes[0]}"
    if _benchmark_needs_series_subset(benchmark_dir, max_series):
        print(
            f"Subsetting existing M5 benchmark outputs to first {max_series} series: {subset_dir}",
            flush=True,
        )
    return _subset_existing_benchmark_outputs(benchmark_dir, subset_dir, max_series)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a forecasting benchmark plus VARA workflow.")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), default="m5")
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--benchmark-dir", type=Path, default=None)
    parser.add_argument("--data-format", choices=["auto", "wide", "long"], default=None)
    parser.add_argument("--date-col", type=str, default=None)
    parser.add_argument("--series-col", type=str, default=None)
    parser.add_argument("--value-col", type=str, default=None)
    parser.add_argument("--series-suffix", type=str, default=None)
    parser.add_argument("--series-depth", type=int, default=None)
    parser.add_argument("--max-series", type=int, default=None)

    parser.add_argument("--feature-path", type=Path, default=None)
    parser.add_argument("--feature-source", choices=["file", "tsfeatures", "tourism", "hybrid", "seasonal-corr", "seasonal-predictive"], default=None)
    parser.add_argument("--feature-functions", type=str, default=None)
    parser.add_argument("--feature-filter-method", choices=["direct", "seasonal_naive", "auto_ets"], default="auto_ets")
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--skip-features", action="store_true")
    parser.add_argument("--standardize-features", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--skip-benchmarks", action="store_true")
    parser.add_argument("--skip-vara", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--include-full-tsfeatures-vara", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--forecast-mode", type=str, default=None)
    parser.add_argument("--freq", type=str, default=None)
    parser.add_argument("--clip-nonnegative", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seasonal-lag", type=int, default=None)
    parser.add_argument("--feature-freq", type=int, default=None)
    parser.add_argument("--forecast-scenario", choices=["direct", "seasonal-naive-residual", "auto_ets", "statsmodels_ets", "seasonal_ets"], default=None)
    parser.add_argument("--residual-transform", choices=["center", "standardize"], default=None)
    parser.add_argument("--y-transform", default=None)

    parser.add_argument("--grid-size", type=int, default=None)
    parser.add_argument("--ridge", type=float, default=None)
    parser.add_argument("--attention-mode", choices=["dense", "topk", "score", "score-topk"], default=None)
    parser.add_argument("--attention-top-k", type=int, default=None)
    parser.add_argument("--feature-update-policy", choices=["exact", "periodic"], default=None)
    parser.add_argument("--feature-update-interval", type=int, default=None)
    parser.add_argument("--ray-num-cpus", type=int, default=52)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    parser.add_argument("--ray-debug", action="store_true")
    return parser.parse_args()


def _resolved_config(args: argparse.Namespace) -> dict[str, Any]:
    defaults = DATASET_DEFAULTS[args.dataset]
    output_dir = args.output_dir or defaults["output_dir"]
    feature_source = _arg_value(args, defaults, "feature_source")
    feature_path = args.feature_path
    if feature_source == "file" and feature_path is None and defaults.get("feature_path_name"):
        feature_path = output_dir / "features" / str(defaults["feature_path_name"])
    forecast_mode = _arg_value(args, defaults, "forecast_mode")
    benchmark_root_dir = args.benchmark_dir or defaults.get("benchmark_dir") or output_dir / "benchmarks"
    benchmark_dir = _benchmark_dir_for_mode(
        benchmark_root_dir,
        forecast_mode,
        use_mode_subdirs=bool(defaults.get("benchmark_mode_subdirs")),
    )

    config = {
        "dataset": args.dataset,
        "data_path": _arg_value(args, defaults, "data_path"),
        "output_dir": output_dir,
        "benchmark_root_dir": benchmark_root_dir,
        "benchmark_dir": benchmark_dir,
        "vara_dir": output_dir / "vara",
        "comparison_dir": output_dir / "comparison",
        "data_format": _arg_value(args, defaults, "data_format"),
        "date_col": _arg_value(args, defaults, "date_col"),
        "series_col": _arg_value(args, defaults, "series_col"),
        "value_col": _arg_value(args, defaults, "value_col"),
        "series_suffix": _arg_value(args, defaults, "series_suffix"),
        "series_depth": _arg_value(args, defaults, "series_depth"),
        "max_series": _arg_value(args, defaults, "max_series"),
        "horizon": _arg_value(args, defaults, "horizon"),
        "seasonal_lag": _arg_value(args, defaults, "seasonal_lag"),
        "feature_freq": _arg_value(args, defaults, "feature_freq"),
        "freq": _arg_value(args, defaults, "freq"),
        "feature_source": feature_source,
        "feature_path": feature_path,
        "feature_functions": _arg_value(args, defaults, "feature_functions"),
        "clip_nonnegative": defaults["clip_nonnegative"] if args.clip_nonnegative is None else args.clip_nonnegative,
        "models": _arg_value(args, defaults, "models"),
        "forecast_mode": forecast_mode,
        "forecast_scenario": _arg_value(args, defaults, "forecast_scenario"),
        "residual_transform": _arg_value(args, defaults, "residual_transform"),
        "y_transform": _arg_value(args, defaults, "y_transform"),
        "grid_size": _arg_value(args, defaults, "grid_size"),
        "ridge": _arg_value(args, defaults, "ridge"),
        "attention_mode": _arg_value(args, defaults, "attention_mode"),
        "attention_top_k": _arg_value(args, defaults, "attention_top_k"),
        "feature_update_policy": _arg_value(args, defaults, "feature_update_policy"),
        "feature_update_interval": _arg_value(args, defaults, "feature_update_interval"),
        "include_full_tsfeatures_vara": (
            defaults["include_full_tsfeatures_vara"]
            if args.include_full_tsfeatures_vara is None
            else args.include_full_tsfeatures_vara
        ),
    }
    return config


def _ray_args(args: argparse.Namespace) -> list[str]:
    out = [
        "--ray-num-cpus",
        str(args.ray_num_cpus),
        "--max-cpus-per-task",
        str(args.max_cpus_per_task),
    ]
    if args.ray_debug:
        out.append("--ray-debug")
    return out


def _prepare_feature_cmd(args: argparse.Namespace, config: dict[str, Any], ray_args: list[str]) -> list[str] | None:
    if args.dataset != "m5" or config["feature_source"] != "file":
        return None
    feature_path = config["feature_path"]
    if feature_path is None:
        raise ValueError("M5 file-based features require --feature-path or the dataset default feature path.")
    if args.skip_features or (_feature_outputs_exist(args, config) and not args.recompute_features):
        if not args.skip_features:
            print(f"Using existing feature file: {feature_path}", flush=True)
        return None

    feature_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "scripts/prepare_m5_attention_features.py",
        "--out",
        feature_path.as_posix(),
        "--holdout",
        str(config["horizon"]),
        "--filter-method",
        args.feature_filter_method,
        "--seasonal-lag",
        str(config["seasonal_lag"]),
        "--max-series",
        str(config["max_series"]),
        *ray_args,
    ]
    if args.standardize_features:
        cmd.append("--standardize-output")
    if args.recompute_features:
        cmd.append("--overwrite")
    return cmd


def _common_panel_args(config: dict[str, Any]) -> list[str]:
    return [
        "--data-path",
        config["data_path"].as_posix(),
        "--data-format",
        str(config["data_format"]),
        "--date-col",
        str(config["date_col"]),
        "--series-col",
        str(config["series_col"]),
        "--value-col",
        str(config["value_col"]),
        "--series-suffix",
        str(config["series_suffix"]),
        "--series-depth",
        str(config["series_depth"]),
        "--max-series",
        str(config["max_series"]),
        "--horizon",
        str(config["horizon"]),
    ]


def _benchmark_cmd(args: argparse.Namespace, config: dict[str, Any], panel_args: list[str], ray_args: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_benchmarks.py",
        *panel_args,
        "--output-dir",
        config["benchmark_dir"].as_posix(),
        "--season-length",
        str(config["seasonal_lag"]),
        "--freq",
        str(config["freq"]),
        "--forecast-mode",
        str(config["forecast_mode"]),
        "--models",
        str(config["models"]),
        *ray_args,
    ]
    if config["clip_nonnegative"]:
        cmd.append("--clip-nonnegative")
    return cmd


def _vara_cmd(
    args: argparse.Namespace,
    config: dict[str, Any],
    panel_args: list[str],
    ray_args: list[str],
    *,
    output_dir: Path | None = None,
    feature_functions: str | None = None,
) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_vara_forecast.py",
        *panel_args,
        "--output-dir",
        (output_dir or config["vara_dir"]).as_posix(),
        "--seasonal-lag",
        str(config["seasonal_lag"]),
        "--feature-freq",
        str(config["feature_freq"]),
        "--feature-source",
        str(config["feature_source"]),
        "--forecast-scenario",
        str(config["forecast_scenario"]),
        "--residual-transform",
        str(config["residual_transform"]),
        "--y-transform",
        str(config["y_transform"]),
        "--forecast-mode",
        str(config["forecast_mode"]),
        "--feature-update-policy",
        str(config["feature_update_policy"]),
        "--feature-update-interval",
        str(config["feature_update_interval"]),
        "--grid-size",
        str(config["grid_size"]),
        "--ridge",
        str(config["ridge"]),
        "--attention-mode",
        str(config["attention_mode"]),
        "--attention-top-k",
        str(config["attention_top_k"]),
        *ray_args,
    ]
    if config["feature_path"] is not None:
        cmd.extend(["--feature-path", config["feature_path"].as_posix()])
    effective_feature_functions = config["feature_functions"] if feature_functions is None else feature_functions
    if effective_feature_functions:
        cmd.extend(["--feature-functions", str(effective_feature_functions)])
    return cmd


def _plot_cmd(config: dict[str, Any]) -> list[str]:
    comparison_dir = config["comparison_dir"]
    if config.get("dataset") == "tourism":
        vara_label = config.get("plot_vara_label") or (
            "VARA on AutoETS residuals (full tsfeatures)"
            if config.get("include_full_tsfeatures_vara")
            else "VARA on AutoETS residuals (reduced features)"
        )
        model_labels = (
            f"vara={vara_label},"
            "ar1_auto_ets=AR(1) on AutoETS residuals,"
            "var1_auto_ets=VAR(1) on AutoETS residuals,"
            "auto_ets=AutoETS,auto_arima=AutoARIMA,nbeats=N-BEATS,"
            "seasonal_naive=Seasonal naive,deepar=DeepAR,ar1=AR(1),var1=VAR(1)"
        )
        method_order = (
            f"{vara_label},"
            "AR(1) on AutoETS residuals,AutoETS,AutoARIMA,N-BEATS,"
            "VAR(1) on AutoETS residuals,Seasonal naive,DeepAR,AR(1),VAR(1)"
        )
    else:
        model_labels = (
            "vara=VARA on AutoETS residuals,auto_ets=AutoETS,auto_arima=AutoARIMA,"
            "nbeats=N-BEATS,deepar=DeepAR,ar1=AR(1),ar1_snaive=AR(1) on seasonal-naive residuals,"
            "seasonal_naive=Seasonal naive,naive=Naive,drift=Drift"
        )
        method_order = (
            "VARA on AutoETS residuals,AutoETS,AutoARIMA,N-BEATS,DeepAR,AR(1),"
            "AR(1) on seasonal-naive residuals,Seasonal naive,Naive,Drift"
        )
    results_dirs = [
        config.get("plot_benchmark_dir", config["benchmark_dir"]),
        *config.get("plot_vara_dirs", [config["vara_dir"]]),
    ]
    cmd = [
        sys.executable,
        "scripts/plot_forecast_horizon_metrics.py",
        "--results-dirs",
        ",".join(Path(path).as_posix() for path in results_dirs),
        "--model-labels",
        model_labels,
        "--method-order",
        method_order,
        "--plot-metrics",
        "rmse=RMSE,crps=CRPS,bias=Bias",
        "--output-csv",
        (comparison_dir / "horizon_metrics.csv").as_posix(),
        "--output-pdf",
        (comparison_dir / "horizon_metrics.pdf").as_posix(),
        "--output-png",
        (comparison_dir / "horizon_metrics.png").as_posix(),
        "--log-positive-metrics",
    ]
    return cmd


def main() -> None:
    args = _parse_args()
    if (
        args.dataset == "m5"
        and (args.forecast_scenario or DATASET_DEFAULTS[args.dataset]["forecast_scenario"]) == "auto_ets"
        and args.feature_filter_method != "auto_ets"
        and not args.skip_features
    ):
        raise ValueError("M5 --forecast-scenario auto_ets requires --feature-filter-method auto_ets.")
    config = _resolved_config(args)
    output_dir = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_config = {key: _as_config_value(value) for key, value in vars(args).items()}
    saved_config.update({f"effective_{key}": _as_config_value(value) for key, value in config.items()})
    (output_dir / "workflow_config.json").write_text(json.dumps(saved_config, indent=2, sort_keys=True) + "\n")

    ray_args = _ray_args(args)
    feature_cmd = _prepare_feature_cmd(args, config, ray_args)
    if feature_cmd is not None:
        _run(feature_cmd, dry_run=args.dry_run)

    panel_args = _common_panel_args(config)
    if not args.skip_benchmarks and _benchmark_outputs_exist(config["benchmark_dir"]):
        print(f"Using existing benchmark outputs: {config['benchmark_dir']}", flush=True)
    elif not args.skip_benchmarks:
        _run(_benchmark_cmd(args, config, panel_args, ray_args), dry_run=args.dry_run)
    comparison_benchmark_dir = config["benchmark_dir"] if args.dry_run else _comparison_benchmark_dir(config)
    vara_outputs = [(config["vara_dir"], "VARA on AutoETS residuals, reduced features" if args.dataset == "tourism" else None)]
    full_vara_dir = config["output_dir"] / "vara_full_tsfeatures"
    if config.get("include_full_tsfeatures_vara"):
        vara_outputs.append((full_vara_dir, "VARA on AutoETS residuals, full tsfeatures"))

    if not args.skip_vara:
        _run(_vara_cmd(args, config, panel_args, ray_args), dry_run=args.dry_run)
        if config.get("include_full_tsfeatures_vara"):
            _run(
                _vara_cmd(
                    args,
                    config,
                    panel_args,
                    ray_args,
                    output_dir=full_vara_dir,
                    feature_functions=DEFAULT_FEATURE_FUNCTIONS,
                ),
                dry_run=args.dry_run,
            )

    if not args.dry_run:
        comparison = _comparison_table(comparison_benchmark_dir, vara_outputs, config["comparison_dir"])
        if args.dataset == "tourism":
            comparison_display = comparison.copy()
            comparison_display["model"] = comparison_display["model"].map(TOURISM_TABLE7_MODEL_LABELS).fillna(
                comparison_display["model"]
            )
            show = ["forecast_mode", "rank_rmse", "model", "n_series", "rmse", "bias"]
        else:
            comparison_display = comparison.copy()
            comparison_display["model"] = comparison_display["model"].map(M5_MODEL_LABELS).fillna(
                comparison_display["model"]
            )
            show = ["forecast_mode", "rank_rmse", "model", "n_series", "rmse", "mae", "mase", "rmsse", "bias"]
        show.extend([col for col in ["crps", "forecast_variance_mean"] if col in comparison_display.columns])
        print(f"\n{args.dataset} benchmark and VARA comparison:")
        print(comparison_display[show].to_string(index=False))

    if not args.skip_plot:
        plot_config = dict(config)
        plot_config["plot_benchmark_dir"] = comparison_benchmark_dir
        if args.dataset == "tourism" and config.get("include_full_tsfeatures_vara"):
            plot_config["plot_vara_dirs"] = [full_vara_dir]
            plot_config["plot_vara_label"] = "VARA on AutoETS residuals (full tsfeatures)"
        elif args.dataset == "tourism":
            plot_config["plot_vara_dirs"] = [config["vara_dir"]]
            plot_config["plot_vara_label"] = "VARA on AutoETS residuals (reduced features)"
        else:
            plot_config["plot_vara_dirs"] = [path for path, _ in vara_outputs]
        _run(_plot_cmd(plot_config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

