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

import pandas as pd


DEFAULT_MODELS = "naive,seasonal_naive,drift,auto_ets,auto_arima,ar1,var1,deepar,nbeats"
DEFAULT_FORECAST_MODES = "recursive,rolling-origin,closed-loop"
DEFAULT_FEATURE_FUNCTIONS = (
    "acf_features,heterogeneity,series_length,arch_stat,holt_parameters,sparsity,"
    "count_entropy,hurst,stability,crossing_points,hw_parameters,stl_features,"
    "entropy,intervals,unitroot_kpss,flat_spots,lumpiness,unitroot_pp,frequency,"
    "nonlinearity,guerrero,pacf_features"
)

DATASET_DEFAULTS: dict[str, dict[str, Any]] = {
    "m5": {
        "data_path": Path("data/m5/long/sales_train_evaluation_long.parquet"),
        "output_dir": Path("outputs/m5_auto_ets_dense_workflow"),
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
    },
    "tourism": {
        "data_path": Path("data/tourism/tourism.csv"),
        "output_dir": Path("outputs/tourism_auto_ets_dense_workflow"),
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
        "feature_functions": DEFAULT_FEATURE_FUNCTIONS,
        "feature_path_name": "",
        "clip_nonnegative": True,
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


def _comparison_table(benchmark_dir: Path, vara_dir: Path, output_dir: Path) -> pd.DataFrame:
    bench = pd.read_csv(benchmark_dir / "metrics_summary.csv")
    vara = pd.read_csv(vara_dir / "metrics_summary.csv")

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
    if args.forecast_scenario == "auto_ets" and not _feature_metadata_path(feature_path).exists():
        return False
    return True


def _benchmark_outputs_exist(benchmark_dir: Path) -> bool:
    return (
        benchmark_dir.exists()
        and (benchmark_dir / "metrics_summary.csv").exists()
        and (benchmark_dir / "metrics_by_origin.csv").exists()
    )


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

    parser.add_argument("--models", type=str, default=DEFAULT_MODELS)
    parser.add_argument("--forecast-mode", type=str, default=DEFAULT_FORECAST_MODES)
    parser.add_argument("--freq", type=str, default=None)
    parser.add_argument("--clip-nonnegative", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seasonal-lag", type=int, default=None)
    parser.add_argument("--feature-freq", type=int, default=None)
    parser.add_argument("--forecast-scenario", choices=["direct", "seasonal-naive-residual", "auto_ets"], default="auto_ets")
    parser.add_argument("--residual-transform", choices=["center", "standardize"], default="center")
    parser.add_argument("--y-transform", default="none")

    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--attention-mode", choices=["dense", "topk", "score", "score-topk"], default="dense")
    parser.add_argument("--attention-top-k", type=int, default=50)
    parser.add_argument("--feature-update-policy", choices=["exact", "periodic"], default="exact")
    parser.add_argument("--feature-update-interval", type=int, default=1)
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

    config = {
        "dataset": args.dataset,
        "data_path": _arg_value(args, defaults, "data_path"),
        "output_dir": output_dir,
        "benchmark_dir": args.benchmark_dir or output_dir / "benchmarks",
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
        args.forecast_mode,
        "--models",
        args.models,
        *ray_args,
    ]
    if config["clip_nonnegative"]:
        cmd.append("--clip-nonnegative")
    return cmd


def _vara_cmd(args: argparse.Namespace, config: dict[str, Any], panel_args: list[str], ray_args: list[str]) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/run_vara_forecast.py",
        *panel_args,
        "--output-dir",
        config["vara_dir"].as_posix(),
        "--seasonal-lag",
        str(config["seasonal_lag"]),
        "--feature-freq",
        str(config["feature_freq"]),
        "--feature-source",
        str(config["feature_source"]),
        "--forecast-scenario",
        args.forecast_scenario,
        "--residual-transform",
        args.residual_transform,
        "--y-transform",
        args.y_transform,
        "--forecast-mode",
        args.forecast_mode,
        "--feature-update-policy",
        args.feature_update_policy,
        "--feature-update-interval",
        str(args.feature_update_interval),
        "--grid-size",
        str(args.grid_size),
        "--ridge",
        str(args.ridge),
        "--attention-mode",
        args.attention_mode,
        "--attention-top-k",
        str(args.attention_top_k),
        *ray_args,
    ]
    if config["feature_path"] is not None:
        cmd.extend(["--feature-path", config["feature_path"].as_posix()])
    if config["feature_functions"]:
        cmd.extend(["--feature-functions", str(config["feature_functions"])])
    return cmd


def _plot_cmd(config: dict[str, Any]) -> list[str]:
    comparison_dir = config["comparison_dir"]
    return [
        sys.executable,
        "scripts/plot_forecast_horizon_metrics.py",
        "--results-dirs",
        f"{config['benchmark_dir'].as_posix()},{config['vara_dir'].as_posix()}",
        "--model-labels",
        (
            "vara=VARA(1) on AutoETS residuals,auto_ets=AutoETS,auto_arima=AutoARIMA,"
            "seasonal_naive=SNaive,ar1=AR(1),var1=VAR(1),deepar=DeepAR,nbeats=NBEATS,"
            "naive=Naive,drift=Drift"
        ),
        "--method-order",
        (
            "VARA(1) on AutoETS residuals,AutoETS,AutoARIMA,AR(1),VAR(1),SNaive,"
            "DeepAR,NBEATS,Naive,Drift"
        ),
        "--output-csv",
        (comparison_dir / "horizon_metrics.csv").as_posix(),
        "--output-pdf",
        (comparison_dir / "horizon_metrics.pdf").as_posix(),
        "--output-png",
        (comparison_dir / "horizon_metrics.png").as_posix(),
        "--log-positive-metrics",
    ]


def main() -> None:
    args = _parse_args()
    if (
        args.dataset == "m5"
        and args.forecast_scenario == "auto_ets"
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
    if not args.skip_vara:
        _run(_vara_cmd(args, config, panel_args, ray_args), dry_run=args.dry_run)

    if not args.dry_run:
        comparison = _comparison_table(config["benchmark_dir"], config["vara_dir"], config["comparison_dir"])
        show = ["forecast_mode", "rank_rmse", "model", "n_series", "rmse", "mae", "mase", "rmsse", "bias"]
        print(f"\n{args.dataset} benchmark and VARA comparison:")
        print(comparison[show].to_string(index=False))

    if not args.skip_plot:
        _run(_plot_cmd(config), dry_run=args.dry_run)


if __name__ == "__main__":
    main()

