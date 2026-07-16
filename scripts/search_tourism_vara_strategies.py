#!/usr/bin/env python3
"""Run a Ray-backed, validation-only sweep of Tourism VARA strategies."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

try:
    from _bootstrap import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT  # noqa: F401
from scripts.run_vara_forecast import (
    DEFAULT_TOURISM_TSFEATURES,
    _base_series_columns,
    _fit_vara_origin_remote,
)
from vara.forecast_data import load_panel_data


CURRENT_FEATURES = "stl_features,acf_features,pacf_features,entropy,holt_parameters,hw_parameters,frequency,guerrero"
COMPACT_FEATURES = "stl_features,acf_features,pacf_features,entropy"
SEASONAL_FEATURES = "stl_features,acf_features,pacf_features,holt_parameters,hw_parameters,frequency"


@dataclass(frozen=True)
class Strategy:
    strategy_id: str
    feature_source: str = "tsfeatures"
    feature_functions: str = CURRENT_FEATURES
    y_transform: str = "standardize"
    residual_transform: str = "standardize"
    attention_mode: str = "dense"
    attention_top_k: int | None = None
    grid_size: int = 7
    ridge: float = 0.01
    feature_update_policy: str = "exact"
    feature_update_interval: int = 1
    history_window: int | None = None
    eta_lower: float = -2.0
    eta_upper: float = 2.0


def default_strategies() -> list[Strategy]:
    strategies = [
        Strategy("current_dense"),
        Strategy("compact_dense", feature_functions=COMPACT_FEATURES),
        Strategy("seasonal_dense", feature_functions=SEASONAL_FEATURES),
        Strategy("full_dense", feature_functions=DEFAULT_TOURISM_TSFEATURES),
        Strategy("tourism_dense", feature_source="tourism"),
        Strategy("hybrid_compact_dense", feature_source="hybrid", feature_functions=COMPACT_FEATURES),
        Strategy("hybrid_current_dense", feature_source="hybrid"),
        Strategy("seasonal_corr_dense", feature_source="seasonal-corr"),
    ]
    for top_k in (5, 10, 20, 50):
        strategies.append(
            Strategy(
                f"current_topk_{top_k}",
                attention_mode="topk",
                attention_top_k=top_k,
            )
        )
    strategies.extend([
        Strategy(
            "seasonal_predictive_topk_5_eta4",
            feature_source="seasonal-predictive",
            attention_mode="score-topk",
            attention_top_k=5,
            eta_lower=-4.0,
            eta_upper=4.0,
        ),
        Strategy(
            "seasonal_predictive_topk_20_eta4",
            feature_source="seasonal-predictive",
            attention_mode="score-topk",
            attention_top_k=20,
            eta_lower=-4.0,
            eta_upper=4.0,
        ),
        Strategy(
            "seasonal_predictive_topk_50_eta4",
            feature_source="seasonal-predictive",
            attention_mode="score-topk",
            attention_top_k=50,
            eta_lower=-4.0,
            eta_upper=4.0,
        ),
        Strategy(
            "seasonal_predictive_topk_50_eta8",
            feature_source="seasonal-predictive",
            attention_mode="score-topk",
            attention_top_k=50,
            eta_lower=-8.0,
            eta_upper=8.0,
        ),
    ])
    strategies.extend([
        Strategy(
            "compact_topk_10",
            feature_functions=COMPACT_FEATURES,
            attention_mode="topk",
            attention_top_k=10,
        ),
        Strategy(
            "tourism_topk_10",
            feature_source="tourism",
            attention_mode="topk",
            attention_top_k=10,
        ),
        Strategy(
            "hybrid_topk_10",
            feature_source="hybrid",
            attention_mode="topk",
            attention_top_k=10,
        ),
        Strategy(
            "seasonal_corr_topk_10",
            feature_source="seasonal-corr",
            attention_mode="topk",
            attention_top_k=10,
        ),
        Strategy(
            "seasonal_predictive_score",
            feature_source="seasonal-predictive",
            attention_mode="score",
        ),
    ])
    for top_k in (5, 10, 20, 50):
        strategies.append(
            Strategy(
                f"seasonal_predictive_topk_{top_k}",
                feature_source="seasonal-predictive",
                attention_mode="score-topk",
                attention_top_k=top_k,
            )
        )
    strategies.extend([
        Strategy("current_center", residual_transform="center"),
        Strategy("current_log_features", y_transform="log1p-standardize"),
        Strategy(
            "current_log_features_center",
            y_transform="log1p-standardize",
            residual_transform="center",
        ),
        Strategy("current_ridge_1e4", ridge=1e-4),
        Strategy("current_ridge_1e3", ridge=1e-3),
        Strategy("current_ridge_1e1", ridge=1e-1),
        Strategy("current_grid_11", grid_size=11),
        Strategy("current_grid_21", grid_size=21),
        Strategy(
            "current_topk_10_grid_11",
            attention_mode="topk",
            attention_top_k=10,
            grid_size=11,
        ),
        Strategy(
            "current_fixed_features",
            feature_update_policy="periodic",
            feature_update_interval=12,
        ),
        Strategy(
            "current_quarterly_features",
            feature_update_policy="periodic",
            feature_update_interval=3,
        ),
    ])
    ids = [strategy.strategy_id for strategy in strategies]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Default strategy identifiers must be unique.")
    return strategies


def _parse_years(value: str) -> list[int]:
    years = sorted({int(part.strip()) for part in str(value).split(",") if part.strip()})
    if len(years) < 2:
        raise ValueError("At least two validation years are required for nested scoring.")
    return years


def _select_strategies(value: str | None) -> list[Strategy]:
    strategies = default_strategies()
    if value is None or not str(value).strip():
        return strategies
    selected_ids = {part.strip() for part in str(value).split(",") if part.strip()}
    by_id = {strategy.strategy_id: strategy for strategy in strategies}
    missing = sorted(selected_ids.difference(by_id))
    if missing:
        raise ValueError(f"Unknown strategy ids: {missing}")
    return [strategy for strategy in strategies if strategy.strategy_id in selected_ids]


def _slice_history_window(
    wide: pd.DataFrame,
    *,
    step: int,
    horizon: int,
    history_window: int | None,
) -> pd.DataFrame:
    if history_window is None:
        return wide
    origin_end = len(wide) - int(horizon) + int(step)
    start = max(0, origin_end - int(history_window))
    return wide.iloc[start:].reset_index(drop=True)


def _strategy_origin_task(
    wide: pd.DataFrame,
    series_ids: list[str],
    strategy_dict: dict,
    evaluation_year: int,
    step: int,
    horizon: int,
    seasonal_lag: int,
    feature_freq: int,
) -> dict:
    strategy = Strategy(**strategy_dict)
    try:
        local_wide = _slice_history_window(
            wide,
            step=step,
            horizon=horizon,
            history_window=strategy.history_window,
        )
        values = local_wide.loc[:, series_ids].to_numpy(dtype=float)
        result = _fit_vara_origin_remote(
            local_wide,
            values,
            series_ids,
            step=int(step),
            horizon=int(horizon),
            y_transform=strategy.y_transform,
            seasonal_lag=int(seasonal_lag),
            feature_freq=int(feature_freq),
            feature_functions=strategy.feature_functions,
            feature_source=strategy.feature_source,
            feature_path=None,
            feature_update_policy=strategy.feature_update_policy,
            feature_update_interval=int(strategy.feature_update_interval),
            grid_size=int(strategy.grid_size),
            ridge=float(strategy.ridge),
            attention_mode=strategy.attention_mode,
            attention_top_k=strategy.attention_top_k,
            forecast_scenario="auto_ets",
            residual_transform=strategy.residual_transform,
            ray_debug=False,
            eta_bounds=(float(strategy.eta_lower), float(strategy.eta_upper)),
        )
        actual_index = len(local_wide) - int(horizon) + int(step)
        return {
            "ok": True,
            "strategy_id": strategy.strategy_id,
            "evaluation_year": int(evaluation_year),
            "origin": int(step) + 1,
            "date": str(pd.Timestamp(local_wide["date"].iloc[actual_index]).date()),
            "series_ids": series_ids,
            "actual": values[actual_index],
            "baseline": np.asarray(result["baseline"], dtype=float),
            "residual": np.asarray(result["residual"], dtype=float),
            "eta_hat": float(result["diag"]["eta_hat"]),
            "eta_on_boundary": bool(result["diag"]["eta_on_boundary"]),
            "max_abs_alpha": float(result["diag"]["max_abs_alpha"]),
            "max_abs_beta": float(result["diag"]["max_abs_beta"]),
            "stationarity_proxy": float(result["diag"]["stationarity_proxy"]),
        }
    except Exception as exc:
        return {
            "ok": False,
            "strategy_id": strategy.strategy_id,
            "evaluation_year": int(evaluation_year),
            "origin": int(step) + 1,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _completed_task_keys(checkpoint: pd.DataFrame, n_series: int) -> set[tuple[str, int, int]]:
    if checkpoint.empty:
        return set()
    counts = checkpoint.groupby(
        ["strategy_id", "evaluation_year", "origin"], observed=True
    ).size()
    return {
        (str(strategy_id), int(year), int(origin))
        for (strategy_id, year, origin), count in counts.items()
        if int(count) == int(n_series)
    }


def _result_frame(result: dict) -> pd.DataFrame:
    n_series = len(result["series_ids"])
    return pd.DataFrame({
        "strategy_id": result["strategy_id"],
        "evaluation_year": int(result["evaluation_year"]),
        "origin": int(result["origin"]),
        "date": result["date"],
        "series_id": result["series_ids"],
        "actual": np.asarray(result["actual"], dtype=float),
        "baseline": np.asarray(result["baseline"], dtype=float),
        "vara_residual": np.asarray(result["residual"], dtype=float),
        "eta_hat": np.full(n_series, float(result["eta_hat"])),
        "eta_on_boundary": np.full(n_series, bool(result["eta_on_boundary"])),
        "max_abs_alpha": np.full(n_series, float(result["max_abs_alpha"])),
        "max_abs_beta": np.full(n_series, float(result["max_abs_beta"])),
        "stationarity_proxy": np.full(n_series, float(result["stationarity_proxy"])),
    })


def run_ray_sweep(
    *,
    data_path: Path,
    output_dir: Path,
    years: list[int],
    strategies: list[Strategy],
    horizon: int,
    seasonal_lag: int,
    feature_freq: int,
    ray_num_cpus: int,
    max_cpus_per_task: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "origin_results.csv"
    failures_path = output_dir / "failures.csv"
    checkpoint = pd.read_csv(checkpoint_path) if checkpoint_path.exists() else pd.DataFrame()

    wide = load_panel_data(data_path, data_format="wide", date_col="date")
    wide["date"] = pd.to_datetime(wide["date"])
    series_ids = _base_series_columns(wide, series_depth=3, series_suffix="All")
    completed = _completed_task_keys(checkpoint, len(series_ids))

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
    remote_task = ray.remote(num_cpus=int(max_cpus_per_task))(_strategy_origin_task)
    try:
        wide_refs = {}
        for year in years:
            evaluation_panel = wide[wide["date"].dt.year <= int(year)].reset_index(drop=True)
            if len(evaluation_panel[evaluation_panel["date"].dt.year == int(year)]) != int(horizon):
                raise ValueError(f"Evaluation year {year} does not contain {horizon} rows.")
            wide_refs[year] = ray.put(evaluation_panel)
        series_ref = ray.put(series_ids)

        futures = []
        submitted_at = {}
        for strategy in strategies:
            strategy_dict = asdict(strategy)
            for year in years:
                for step in range(int(horizon)):
                    key = (strategy.strategy_id, int(year), int(step) + 1)
                    if key in completed:
                        continue
                    ref = remote_task.remote(
                        wide_refs[year],
                        series_ref,
                        strategy_dict,
                        int(year),
                        int(step),
                        int(horizon),
                        int(seasonal_lag),
                        int(feature_freq),
                    )
                    futures.append(ref)
                    submitted_at[ref] = time.time()
        print(
            f"Submitted {len(futures)} VARA tasks for {len(strategies)} strategies, "
            f"{len(years)} years, and {horizon} origins per year; "
            f"resumed={len(completed)}.",
            flush=True,
        )

        pending = list(futures)
        completed_now = 0
        failure_rows = []
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=60.0)
            if not ready:
                oldest = time.time() - min(submitted_at.values()) if submitted_at else 0.0
                print(
                    f"Sweep progress: done={completed_now}/{len(futures)}, "
                    f"pending={len(pending)}, oldest_sec={oldest:.1f}, "
                    f"available={ray.available_resources()}",
                    flush=True,
                )
                continue
            result = ray.get(ready[0])
            submitted_at.pop(ready[0], None)
            completed_now += 1
            if result["ok"]:
                frame = _result_frame(result)
                frame.to_csv(
                    checkpoint_path,
                    mode="a",
                    header=not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0,
                    index=False,
                )
            else:
                failure_rows.append(result)
                pd.DataFrame([result]).to_csv(
                    failures_path,
                    mode="a",
                    header=not failures_path.exists() or failures_path.stat().st_size == 0,
                    index=False,
                )
            if completed_now % 25 == 0 or not pending:
                print(
                    f"Collected {completed_now}/{len(futures)} tasks; "
                    f"failures={len(failure_rows)}.",
                    flush=True,
                )
    finally:
        if started_here:
            ray.shutdown()

    results = pd.read_csv(checkpoint_path) if checkpoint_path.exists() else pd.DataFrame()
    failures = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
    return results, failures


def _fit_stack(train: pd.DataFrame) -> np.ndarray:
    design = train.loc[:, ["intercept", "purpose_correction", "vara_residual"]].to_numpy(dtype=float)
    target = train["actual"].to_numpy(dtype=float) - train["direct_auto_ets"].to_numpy(dtype=float)
    return np.asarray(
        lsq_linear(
            design,
            target,
            bounds=(np.asarray([-np.inf, 0.0, 0.0]), np.asarray([np.inf, 1.0, 1.0])),
        ).x,
        dtype=float,
    )


def _fit_calibrated_purpose(train: pd.DataFrame) -> np.ndarray:
    design = train.loc[:, ["intercept", "purpose_correction"]].to_numpy(dtype=float)
    target = train["actual"].to_numpy(dtype=float) - train["direct_auto_ets"].to_numpy(dtype=float)
    return np.asarray(
        lsq_linear(
            design,
            target,
            bounds=(np.asarray([-np.inf, 0.0]), np.asarray([np.inf, 1.0])),
        ).x,
        dtype=float,
    )


def score_sweep(
    results: pd.DataFrame,
    purpose_forecasts_path: Path,
    strategies: list[Strategy],
    years: list[int],
    output_dir: Path,
) -> pd.DataFrame:
    if results.empty:
        raise ValueError("No successful strategy results are available to score.")
    purpose = pd.read_csv(purpose_forecasts_path)
    purpose["date"] = pd.to_datetime(purpose["date"])
    results = results.copy()
    results["date"] = pd.to_datetime(results["date"])
    results["evaluation_year"] = results["evaluation_year"].astype(int)
    merged = results.merge(
        purpose.loc[:, [
            "date",
            "series_id",
            "year",
            "actual",
            "direct_auto_ets",
            "purpose_sum_auto_ets",
        ]].rename(columns={"actual": "purpose_actual"}),
        on=["date", "series_id"],
        how="inner",
        validate="many_to_one",
    )
    if not np.allclose(merged["actual"], merged["purpose_actual"]):
        raise ValueError("Sweep actuals do not match the purpose forecast panel.")
    merged["intercept"] = 1.0
    merged["purpose_correction"] = (
        merged["purpose_sum_auto_ets"] - merged["direct_auto_ets"]
    )

    fold_rows = []
    coefficient_rows = []
    prediction_rows = []
    ranking_rows = []
    strategy_ids = [strategy.strategy_id for strategy in strategies]
    for strategy_id in strategy_ids:
        strategy_panel = merged[merged["strategy_id"] == strategy_id]
        complete_years = sorted(strategy_panel["evaluation_year"].unique())
        if complete_years != years:
            print(
                f"Skipping incomplete strategy {strategy_id}: years={complete_years}, expected={years}",
                flush=True,
            )
            continue
        stack_errors = []
        purpose_errors = []
        baseline_errors = []
        full_vara_errors = []
        for test_year in years[1:]:
            train = strategy_panel[strategy_panel["evaluation_year"] < int(test_year)]
            test = strategy_panel[strategy_panel["evaluation_year"] == int(test_year)].copy()
            stack_coef = _fit_stack(train)
            purpose_coef = _fit_calibrated_purpose(train)
            stack_forecast = np.maximum(
                test["direct_auto_ets"].to_numpy(dtype=float)
                + test.loc[:, ["intercept", "purpose_correction", "vara_residual"]].to_numpy(dtype=float)
                @ stack_coef,
                0.0,
            )
            purpose_forecast = np.maximum(
                test["direct_auto_ets"].to_numpy(dtype=float)
                + test.loc[:, ["intercept", "purpose_correction"]].to_numpy(dtype=float)
                @ purpose_coef,
                0.0,
            )
            full_vara_forecast = np.maximum(
                test["baseline"].to_numpy(dtype=float)
                + test["vara_residual"].to_numpy(dtype=float),
                0.0,
            )
            actual = test["actual"].to_numpy(dtype=float)
            errors = {
                "auto_ets": test["direct_auto_ets"].to_numpy(dtype=float) - actual,
                "calibrated_purpose": purpose_forecast - actual,
                "calibrated_purpose_vara": stack_forecast - actual,
                "full_vara": full_vara_forecast - actual,
            }
            baseline_errors.extend(errors["auto_ets"])
            purpose_errors.extend(errors["calibrated_purpose"])
            stack_errors.extend(errors["calibrated_purpose_vara"])
            full_vara_errors.extend(errors["full_vara"])
            for model, error in errors.items():
                fold_rows.append({
                    "strategy_id": strategy_id,
                    "test_year": int(test_year),
                    "model": model,
                    "n_observations": int(error.size),
                    "rmse": float(np.sqrt(np.mean(error**2))),
                    "mae": float(np.mean(np.abs(error))),
                    "bias": float(np.mean(error)),
                })
            coefficient_rows.append({
                "strategy_id": strategy_id,
                "test_year": int(test_year),
                "intercept": float(stack_coef[0]),
                "purpose_weight": float(stack_coef[1]),
                "vara_weight": float(stack_coef[2]),
                "n_training_observations": int(len(train)),
            })
            prediction = test.loc[:, [
                "strategy_id",
                "evaluation_year",
                "origin",
                "date",
                "series_id",
                "actual",
                "direct_auto_ets",
                "purpose_sum_auto_ets",
                "baseline",
                "vara_residual",
            ]].copy()
            prediction["calibrated_purpose_forecast"] = purpose_forecast
            prediction["calibrated_purpose_vara_forecast"] = stack_forecast
            prediction["full_vara_forecast"] = full_vara_forecast
            prediction_rows.append(prediction)

        baseline_rmse = float(np.sqrt(np.mean(np.asarray(baseline_errors) ** 2)))
        purpose_rmse = float(np.sqrt(np.mean(np.asarray(purpose_errors) ** 2)))
        stack_rmse = float(np.sqrt(np.mean(np.asarray(stack_errors) ** 2)))
        full_vara_rmse = float(np.sqrt(np.mean(np.asarray(full_vara_errors) ** 2)))
        origins = strategy_panel.drop_duplicates(["evaluation_year", "origin"])
        ranking_rows.append({
            "strategy_id": strategy_id,
            "n_nested_observations": len(stack_errors),
            "auto_ets_rmse": baseline_rmse,
            "calibrated_purpose_rmse": purpose_rmse,
            "calibrated_purpose_vara_rmse": stack_rmse,
            "full_vara_rmse": full_vara_rmse,
            "rmse_improvement_vs_auto_ets_pct": 100.0 * (baseline_rmse - stack_rmse) / baseline_rmse,
            "incremental_rmse_improvement_vs_calibrated_purpose_pct": (
                100.0 * (purpose_rmse - stack_rmse) / purpose_rmse
            ),
            "mean_nested_vara_weight": float(
                np.mean([
                    row["vara_weight"]
                    for row in coefficient_rows
                    if row["strategy_id"] == strategy_id
                ])
            ),
            "eta_hat_mean": float(origins["eta_hat"].mean()),
            "eta_boundary_rate": float(origins["eta_on_boundary"].astype(float).mean()),
            "stationarity_proxy_max": float(origins["stationarity_proxy"].max()),
        })

    ranking = pd.DataFrame(ranking_rows).sort_values(
        "calibrated_purpose_vara_rmse"
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    pd.DataFrame(fold_rows).to_csv(output_dir / "nested_fold_metrics.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output_dir / "nested_coefficients.csv", index=False)
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            output_dir / "nested_predictions.csv", index=False
        )
    ranking.to_csv(output_dir / "ranking.csv", index=False)
    print("Validation-only strategy ranking:", flush=True)
    print(ranking.to_string(index=False), flush=True)
    return ranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/tourism/tourism.csv"))
    parser.add_argument(
        "--purpose-forecasts-path",
        type=Path,
        default=Path("outputs/tourism_purpose_autoets_rolling_2011_2016/rolling_forecasts.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_vara_strategy_search"),
    )
    parser.add_argument("--years", type=str, default="2013,2014,2015")
    parser.add_argument("--strategy-ids", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seasonal-lag", type=int, default=12)
    parser.add_argument("--feature-freq", type=int, default=12)
    parser.add_argument("--ray-num-cpus", type=int, default=104)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    args = parser.parse_args()

    years = _parse_years(args.years)
    strategies = _select_strategies(args.strategy_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "data_path": str(args.data_path),
        "purpose_forecasts_path": str(args.purpose_forecasts_path),
        "output_dir": str(args.output_dir),
        "years": years,
        "horizon": int(args.horizon),
        "seasonal_lag": int(args.seasonal_lag),
        "feature_freq": int(args.feature_freq),
        "ray_num_cpus": int(args.ray_num_cpus),
        "max_cpus_per_task": int(args.max_cpus_per_task),
        "strategies": [asdict(strategy) for strategy in strategies],
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    results, failures = run_ray_sweep(
        data_path=args.data_path,
        output_dir=args.output_dir,
        years=years,
        strategies=strategies,
        horizon=args.horizon,
        seasonal_lag=args.seasonal_lag,
        feature_freq=args.feature_freq,
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
    )
    if not failures.empty:
        print(f"Sweep failures: {len(failures)}; see {args.output_dir / 'failures.csv'}", flush=True)
    score_sweep(
        results,
        args.purpose_forecasts_path,
        strategies,
        years,
        args.output_dir,
    )


if __name__ == "__main__":
    main()
