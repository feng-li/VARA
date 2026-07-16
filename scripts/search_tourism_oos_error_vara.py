#!/usr/bin/env python3
"""Search VARA models trained on historical rolling-origin AutoETS errors."""

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

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear


@dataclass(frozen=True)
class ErrorStrategy:
    strategy_id: str
    feature_source: str = "error"
    attention_mode: str = "dense"
    attention_top_k: int | None = None
    transform: str = "standardize"
    history_window: int | None = None
    ridge: float = 0.01
    grid_size: int = 7
    eta_lower: float = -2.0
    eta_upper: float = 2.0
    stability_cap: float | None = None


def default_strategies() -> list[ErrorStrategy]:
    strategies = []
    for window in (12, 24, 36, 48, None):
        label = "all" if window is None else str(window)
        strategies.extend([
            ErrorStrategy(f"error_dense_w{label}", history_window=window),
            ErrorStrategy(
                f"hybrid_dense_w{label}",
                feature_source="hybrid",
                history_window=window,
            ),
            ErrorStrategy(
                f"hierarchy_dense_w{label}",
                feature_source="hierarchy",
                history_window=window,
            ),
            ErrorStrategy(
                f"predictive_lag1_topk10_w{label}",
                feature_source="predictive-lag1",
                attention_mode="score-topk",
                attention_top_k=10,
                history_window=window,
            ),
            ErrorStrategy(
                f"predictive_avg_topk10_w{label}",
                feature_source="predictive-average",
                attention_mode="score-topk",
                attention_top_k=10,
                history_window=window,
            ),
        ])
    strategies.extend([
        ErrorStrategy(
            "predictive_lag1_topk5_w24",
            feature_source="predictive-lag1",
            attention_mode="score-topk",
            attention_top_k=5,
            history_window=24,
        ),
        ErrorStrategy(
            "predictive_lag1_topk20_w24",
            feature_source="predictive-lag1",
            attention_mode="score-topk",
            attention_top_k=20,
            history_window=24,
        ),
        ErrorStrategy(
            "predictive_lag1_topk50_w24",
            feature_source="predictive-lag1",
            attention_mode="score-topk",
            attention_top_k=50,
            history_window=24,
        ),
        ErrorStrategy(
            "predictive_lag1_score_w24",
            feature_source="predictive-lag1",
            attention_mode="score",
            history_window=24,
        ),
        ErrorStrategy("error_topk5_w24", attention_mode="topk", attention_top_k=5, history_window=24),
        ErrorStrategy("error_topk10_w24", attention_mode="topk", attention_top_k=10, history_window=24),
        ErrorStrategy("error_topk20_w24", attention_mode="topk", attention_top_k=20, history_window=24),
        ErrorStrategy("error_center_w24", transform="center", history_window=24),
        ErrorStrategy("error_dense_w24_ridge1e3", history_window=24, ridge=1e-3),
        ErrorStrategy("error_dense_w24_ridge1e1", history_window=24, ridge=1e-1),
    ])
    for cap_label, cap in (("05", 0.5), ("08", 0.8), ("095", 0.95)):
        strategies.extend([
            ErrorStrategy(
                f"error_dense_w48_stable{cap_label}",
                history_window=48,
                stability_cap=cap,
            ),
            ErrorStrategy(
                f"predictive_avg_topk10_w24_stable{cap_label}",
                feature_source="predictive-average",
                attention_mode="score-topk",
                attention_top_k=10,
                history_window=24,
                stability_cap=cap,
            ),
            ErrorStrategy(
                f"hierarchy_dense_w48_stable{cap_label}",
                feature_source="hierarchy",
                history_window=48,
                stability_cap=cap,
            ),
        ])
    ids = [strategy.strategy_id for strategy in strategies]
    if len(ids) != len(set(ids)):
        raise RuntimeError("OOS-error strategy identifiers must be unique.")
    return strategies


def _select_strategies(value: str | None) -> list[ErrorStrategy]:
    strategies = default_strategies()
    if value is None or not str(value).strip():
        return strategies
    requested = {part.strip() for part in str(value).split(",") if part.strip()}
    by_id = {strategy.strategy_id: strategy for strategy in strategies}
    missing = sorted(requested.difference(by_id))
    if missing:
        raise ValueError(f"Unknown OOS-error strategy ids: {missing}")
    return [strategy for strategy in strategies if strategy.strategy_id in requested]


def _standardize_columns(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    mean = np.mean(values, axis=0, keepdims=True)
    scale = np.std(values, axis=0, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (values - mean) / scale


def _clean_standardize_features(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = np.where(np.isfinite(values), values, np.nan)
    for column in range(values.shape[1]):
        finite = np.isfinite(values[:, column])
        fill = float(np.mean(values[finite, column])) if finite.any() else 0.0
        values[~finite, column] = fill
    return _standardize_columns(values)


def _acf(values: np.ndarray, lag: int) -> np.ndarray:
    if values.shape[0] <= int(lag):
        return np.zeros(values.shape[1], dtype=float)
    left = _standardize_columns(values[int(lag):])
    right = _standardize_columns(values[:-int(lag)])
    return np.mean(left * right, axis=0)


def _error_feature_matrix(history: np.ndarray) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    z = _standardize_columns(values)
    recent_3 = np.mean(z[-min(3, len(z)):], axis=0)
    recent_6 = np.mean(z[-min(6, len(z)):], axis=0)
    recent_12 = np.mean(z[-min(12, len(z)):], axis=0)
    trend_3 = z[-1] - z[-min(4, len(z))]
    trend_12 = z[-1] - z[-min(13, len(z))]
    features = np.column_stack([
        np.mean(values, axis=0),
        np.std(values, axis=0),
        z[-1],
        recent_3,
        recent_6,
        recent_12,
        trend_3,
        trend_12,
        _acf(values, 1),
        _acf(values, 12),
    ])
    return _clean_standardize_features(features)


def _hierarchy_feature_matrix(series_ids: list[str]) -> np.ndarray:
    rows = []
    for series_id in series_ids:
        code = series_id[:-3] if series_id.endswith("All") else series_id
        row = []
        for position, width in ((0, 7), (1, 6), (2, 8)):
            values = np.zeros(width, dtype=float)
            if position < len(code):
                index = max(0, min(width - 1, ord(code[position].upper()) - ord("A")))
                values[index] = 1.0
            row.extend(values)
        rows.append(row)
    return _clean_standardize_features(np.asarray(rows, dtype=float))


def _predictive_score(history: np.ndarray, lag: int) -> np.ndarray:
    values = np.asarray(history, dtype=float)
    if len(values) <= int(lag):
        return np.zeros((values.shape[1], values.shape[1]), dtype=float)
    target = _standardize_columns(values[int(lag):])
    source = _standardize_columns(values[:-int(lag)])
    scores = (target.T @ source) / max(len(target), 1)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(scores, 0.0)
    return scores


def _feature_matrix(
    history: np.ndarray,
    series_ids: list[str],
    feature_source: str,
) -> np.ndarray:
    if feature_source == "error":
        return _error_feature_matrix(history)
    if feature_source == "hierarchy":
        return _hierarchy_feature_matrix(series_ids)
    if feature_source == "hybrid":
        return np.column_stack([
            _error_feature_matrix(history),
            _hierarchy_feature_matrix(series_ids),
        ])
    if feature_source == "predictive-lag1":
        return _predictive_score(history, 1)
    if feature_source == "predictive-average":
        return 0.5 * (_predictive_score(history, 1) + _predictive_score(history, 12))
    raise ValueError(f"Unknown OOS-error feature source: {feature_source}")


def _transform_error_history(history: np.ndarray, transform: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(history, axis=0, keepdims=True)
    if transform == "center":
        scale = np.ones_like(mean)
    elif transform == "standardize":
        scale = np.std(history, axis=0, keepdims=True)
        scale = np.where(scale > 1e-12, scale, 1.0)
    else:
        raise ValueError(f"Unknown OOS-error transform: {transform}")
    return (history - mean) / scale, mean, scale


def _error_vara_task(
    error_history: np.ndarray,
    series_ids: list[str],
    strategy_dict: dict,
    target_index: int,
    target_date: str,
    history_start: str,
    history_end: str,
) -> dict:
    strategy = ErrorStrategy(**strategy_dict)
    try:
        history = np.asarray(error_history, dtype=float)
        target_timestamp = pd.Timestamp(target_date)
        history_start_timestamp = pd.Timestamp(history_start)
        history_end_timestamp = pd.Timestamp(history_end)
        if history.ndim != 2 or history.shape[1] != len(series_ids):
            raise ValueError("OOS-error history must have shape (T, n_series).")
        if len(history) < 12:
            raise ValueError("At least 12 historical rolling-origin errors are required.")
        if strategy.history_window is not None and len(history) > int(strategy.history_window):
            raise ValueError("OOS-error history exceeds the configured history window.")
        if history_start_timestamp > history_end_timestamp:
            raise ValueError("OOS-error history start must not follow its end.")
        if history_end_timestamp >= target_timestamp:
            raise ValueError("OOS-error history must end strictly before the target date.")
        model_history, mean, scale = _transform_error_history(history, strategy.transform)
        features = _feature_matrix(history, series_ids, strategy.feature_source)

        import jax.numpy as jnp
        from vara.forecast import VARAForecaster

        forecaster = VARAForecaster(
            eta_bounds=(float(strategy.eta_lower), float(strategy.eta_upper)),
            grid_size=int(strategy.grid_size),
            ridge=float(strategy.ridge),
            dense_threshold=2500,
            attention_mode=strategy.attention_mode,
            attention_top_k=strategy.attention_top_k,
        )
        forecaster.fit(jnp.asarray(model_history), jnp.asarray(features))
        if strategy.stability_cap is not None:
            alpha = np.asarray(forecaster.alpha_hat, dtype=float)
            beta = np.asarray(forecaster.beta_hat, dtype=float)
            coefficient_size = np.abs(alpha) + np.abs(beta)
            projection = np.minimum(
                1.0,
                float(strategy.stability_cap) / np.maximum(coefficient_size, 1e-12),
            )
            forecaster.alpha_hat = jnp.asarray(alpha * projection)
            forecaster.beta_hat = jnp.asarray(beta * projection)
        correction = np.asarray(forecaster.predict(h=1))[0] * scale[0] + mean[0]
        diagnostics = forecaster.diagnostics()
        projected_stationarity = float(
            np.max(
                np.abs(np.asarray(forecaster.alpha_hat, dtype=float))
                + np.abs(np.asarray(forecaster.beta_hat, dtype=float))
            )
        )
        return {
            "ok": True,
            "strategy_id": strategy.strategy_id,
            "target_index": int(target_index),
            "date": str(target_timestamp.date()),
            "history_start": str(history_start_timestamp.date()),
            "history_end": str(history_end_timestamp.date()),
            "series_ids": series_ids,
            "correction": correction,
            "eta_hat": float(diagnostics["eta_hat"]),
            "eta_on_boundary": bool(diagnostics["eta_on_boundary"]),
            "stationarity_proxy": projected_stationarity,
            "n_train": int(len(history)),
        }
    except Exception as exc:
        return {
            "ok": False,
            "strategy_id": strategy.strategy_id,
            "target_index": int(target_index),
            "date": str(target_date),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def _result_frame(result: dict) -> pd.DataFrame:
    n_series = len(result["series_ids"])
    return pd.DataFrame({
        "strategy_id": result["strategy_id"],
        "target_index": int(result["target_index"]),
        "date": result["date"],
        "history_start": result["history_start"],
        "history_end": result["history_end"],
        "series_id": result["series_ids"],
        "vara_correction": np.asarray(result["correction"], dtype=float),
        "eta_hat": np.full(n_series, float(result["eta_hat"])),
        "eta_on_boundary": np.full(n_series, bool(result["eta_on_boundary"])),
        "stationarity_proxy": np.full(n_series, float(result["stationarity_proxy"])),
        "n_train": np.full(n_series, int(result["n_train"])),
    })


def _completed_keys(frame: pd.DataFrame, n_series: int) -> set[tuple[str, int]]:
    if frame.empty:
        return set()
    counts = frame.groupby(["strategy_id", "target_index"], observed=True).size()
    return {
        (str(strategy_id), int(target_index))
        for (strategy_id, target_index), count in counts.items()
        if int(count) == int(n_series)
    }


def run_sweep(
    *,
    purpose_forecasts_path: Path,
    output_dir: Path,
    strategies: list[ErrorStrategy],
    validation_start: str,
    validation_end: str,
    ray_num_cpus: int,
    max_cpus_per_task: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(purpose_forecasts_path)
    required = {"date", "series_id", "actual", "direct_auto_ets"}
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"OOS-error forecast panel is missing columns: {missing}")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["series_id"] = panel["series_id"].astype(str)
    if panel.duplicated(["date", "series_id"]).any():
        raise ValueError("OOS-error forecast panel contains duplicate date/series rows.")
    series_ids = sorted(panel["series_id"].astype(str).unique())
    actual = panel.pivot(index="date", columns="series_id", values="actual").sort_index().loc[:, series_ids]
    baseline = panel.pivot(index="date", columns="series_id", values="direct_auto_ets").sort_index().loc[:, series_ids]
    if actual.isna().any().any() or baseline.isna().any().any():
        raise ValueError("OOS-error forecast panel must be complete for every date and series.")
    errors = actual.to_numpy(dtype=float) - baseline.to_numpy(dtype=float)
    dates = actual.index.to_numpy()
    target_mask = (actual.index >= pd.Timestamp(validation_start)) & (actual.index <= pd.Timestamp(validation_end))
    target_indices = np.flatnonzero(target_mask).tolist()
    if not target_indices:
        raise ValueError("No OOS-error VARA targets fall in the requested validation range.")

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "origin_results.csv"
    failures_path = output_dir / "failures.csv"
    checkpoint = pd.read_csv(checkpoint_path) if checkpoint_path.exists() else pd.DataFrame()
    completed = _completed_keys(checkpoint, len(series_ids))

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
    remote_task = ray.remote(num_cpus=int(max_cpus_per_task))(_error_vara_task)
    try:
        series_ref = ray.put(series_ids)
        history_refs = {}
        futures = []
        submitted_at = {}
        for strategy in strategies:
            strategy_dict = asdict(strategy)
            for target_index in target_indices:
                if (strategy.strategy_id, int(target_index)) in completed:
                    continue
                start = 0
                if strategy.history_window is not None:
                    start = max(0, int(target_index) - int(strategy.history_window))
                history_key = (int(start), int(target_index))
                if history_key not in history_refs:
                    history_refs[history_key] = ray.put(
                        np.asarray(errors[start:int(target_index)], dtype=float).copy()
                    )
                history_dates = dates[start:int(target_index)]
                if len(history_dates) == 0:
                    raise ValueError("OOS-error VARA target has no prior error history.")
                ref = remote_task.remote(
                    history_refs[history_key],
                    series_ref,
                    strategy_dict,
                    int(target_index),
                    str(pd.Timestamp(dates[int(target_index)]).date()),
                    str(pd.Timestamp(history_dates[0]).date()),
                    str(pd.Timestamp(history_dates[-1]).date()),
                )
                futures.append(ref)
                submitted_at[ref] = time.time()
        print(
            f"Submitted {len(futures)} OOS-error VARA tasks for {len(strategies)} strategies "
            f"and {len(target_indices)} target months; resumed={len(completed)}.",
            flush=True,
        )
        pending = list(futures)
        done = 0
        failures = []
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=60.0)
            if not ready:
                oldest = time.time() - min(submitted_at.values()) if submitted_at else 0.0
                print(
                    f"OOS-error sweep: done={done}/{len(futures)}, pending={len(pending)}, "
                    f"oldest_sec={oldest:.1f}",
                    flush=True,
                )
                continue
            result = ray.get(ready[0])
            submitted_at.pop(ready[0], None)
            done += 1
            if result["ok"]:
                _result_frame(result).to_csv(
                    checkpoint_path,
                    mode="a",
                    header=not checkpoint_path.exists() or checkpoint_path.stat().st_size == 0,
                    index=False,
                )
            else:
                failures.append(result)
                pd.DataFrame([result]).to_csv(
                    failures_path,
                    mode="a",
                    header=not failures_path.exists() or failures_path.stat().st_size == 0,
                    index=False,
                )
            if done % 25 == 0 or not pending:
                print(f"Collected {done}/{len(futures)}; failures={len(failures)}.", flush=True)
    finally:
        if started_here:
            ray.shutdown()
    results = pd.read_csv(checkpoint_path) if checkpoint_path.exists() else pd.DataFrame()
    failures = pd.read_csv(failures_path) if failures_path.exists() else pd.DataFrame()
    return results, failures


def _fit_combination(train: pd.DataFrame) -> np.ndarray:
    design = train.loc[:, ["purpose_correction", "vara_correction"]].to_numpy(dtype=float)
    target = train["actual"].to_numpy(dtype=float) - train["direct_auto_ets"].to_numpy(dtype=float)
    return np.asarray(lsq_linear(design, target, bounds=(0.0, 1.0)).x, dtype=float)


def score_results(
    results: pd.DataFrame,
    purpose_forecasts_path: Path,
    strategies: list[ErrorStrategy],
    output_dir: Path,
    selection_end: str | None = None,
) -> pd.DataFrame:
    if results.empty:
        raise ValueError("No OOS-error VARA results are available to score.")
    results = results.copy()
    results["date"] = pd.to_datetime(results["date"])
    if selection_end is not None:
        results = results[results["date"] <= pd.Timestamp(selection_end)].copy()
        if results.empty:
            raise ValueError("No OOS-error VARA results remain before selection_end.")
    purpose = pd.read_csv(purpose_forecasts_path)
    purpose["date"] = pd.to_datetime(purpose["date"])
    merged = results.merge(
        purpose.loc[:, [
            "date",
            "series_id",
            "year",
            "actual",
            "direct_auto_ets",
            "purpose_sum_auto_ets",
        ]],
        on=["date", "series_id"],
        how="inner",
        validate="many_to_one",
    )
    merged["purpose_correction"] = merged["purpose_sum_auto_ets"] - merged["direct_auto_ets"]

    ranking_rows = []
    fold_rows = []
    coefficient_rows = []
    prediction_rows = []
    for strategy in strategies:
        frame = merged[merged["strategy_id"] == strategy.strategy_id].copy()
        if frame.empty:
            continue
        raw_error = (
            np.maximum(frame["direct_auto_ets"] + frame["vara_correction"], 0.0)
            - frame["actual"]
        ).to_numpy(dtype=float)
        baseline_error = (frame["direct_auto_ets"] - frame["actual"]).to_numpy(dtype=float)

        nested_errors = []
        nested_baseline_errors = []
        nested_purpose_errors = []
        for test_year in sorted(frame["year"].unique())[1:]:
            train = frame[frame["year"] < int(test_year)]
            test = frame[frame["year"] == int(test_year)].copy()
            coefficients = _fit_combination(train)
            forecast = np.maximum(
                test["direct_auto_ets"].to_numpy(dtype=float)
                + test.loc[:, ["purpose_correction", "vara_correction"]].to_numpy(dtype=float)
                @ coefficients,
                0.0,
            )
            purpose_only = np.maximum(
                test["direct_auto_ets"].to_numpy(dtype=float)
                + test["purpose_correction"].to_numpy(dtype=float) * coefficients[0],
                0.0,
            )
            actual = test["actual"].to_numpy(dtype=float)
            errors = forecast - actual
            nested_errors.extend(errors)
            nested_baseline_errors.extend(test["direct_auto_ets"].to_numpy(dtype=float) - actual)
            nested_purpose_errors.extend(purpose_only - actual)
            fold_rows.append({
                "strategy_id": strategy.strategy_id,
                "test_year": int(test_year),
                "rmse": float(np.sqrt(np.mean(errors**2))),
                "mae": float(np.mean(np.abs(errors))),
                "bias": float(np.mean(errors)),
            })
            coefficient_rows.append({
                "strategy_id": strategy.strategy_id,
                "test_year": int(test_year),
                "purpose_weight": float(coefficients[0]),
                "vara_weight": float(coefficients[1]),
                "n_training_observations": int(len(train)),
            })
            prediction = test.loc[:, [
                "strategy_id",
                "date",
                "series_id",
                "year",
                "actual",
                "direct_auto_ets",
                "purpose_sum_auto_ets",
                "vara_correction",
            ]].copy()
            prediction["forecast"] = forecast
            prediction_rows.append(prediction)

        nested_errors = np.asarray(nested_errors, dtype=float)
        nested_baseline_errors = np.asarray(nested_baseline_errors, dtype=float)
        nested_purpose_errors = np.asarray(nested_purpose_errors, dtype=float)
        baseline_rmse = float(np.sqrt(np.mean(nested_baseline_errors**2)))
        nested_rmse = float(np.sqrt(np.mean(nested_errors**2)))
        purpose_rmse = float(np.sqrt(np.mean(nested_purpose_errors**2)))
        origins = frame.drop_duplicates(["strategy_id", "target_index"])
        ranking_rows.append({
            "strategy_id": strategy.strategy_id,
            "raw_vara_rmse": float(np.sqrt(np.mean(raw_error**2))),
            "raw_vara_improvement_vs_auto_ets_pct": (
                100.0
                * (float(np.sqrt(np.mean(baseline_error**2))) - float(np.sqrt(np.mean(raw_error**2))))
                / float(np.sqrt(np.mean(baseline_error**2)))
            ),
            "nested_auto_ets_rmse": baseline_rmse,
            "nested_purpose_rmse": purpose_rmse,
            "nested_purpose_vara_rmse": nested_rmse,
            "nested_improvement_vs_auto_ets_pct": 100.0 * (baseline_rmse - nested_rmse) / baseline_rmse,
            "incremental_improvement_vs_purpose_pct": 100.0 * (purpose_rmse - nested_rmse) / purpose_rmse,
            "eta_hat_mean": float(origins["eta_hat"].mean()),
            "eta_boundary_rate": float(origins["eta_on_boundary"].astype(float).mean()),
            "stationarity_proxy_max": float(origins["stationarity_proxy"].max()),
        })

    ranking = pd.DataFrame(ranking_rows).sort_values("nested_purpose_vara_rmse").reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(output_dir / "ranking.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output_dir / "nested_fold_metrics.csv", index=False)
    pd.DataFrame(coefficient_rows).to_csv(output_dir / "nested_coefficients.csv", index=False)
    if prediction_rows:
        pd.concat(prediction_rows, ignore_index=True).to_csv(
            output_dir / "nested_predictions.csv", index=False
        )
    print("OOS-error VARA ranking:", flush=True)
    print(ranking.to_string(index=False), flush=True)
    return ranking


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-panel-path",
        "--purpose-forecasts-path",
        dest="purpose_forecasts_path",
        type=Path,
        default=Path("outputs/tourism_purpose_autoets_rolling_2011_2016/rolling_forecasts.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_oos_error_vara_search"),
    )
    parser.add_argument("--strategy-ids", type=str, default=None)
    parser.add_argument("--validation-start", type=str, default="2012-01-01")
    parser.add_argument("--validation-end", type=str, default="2015-12-01")
    parser.add_argument(
        "--selection-end",
        type=str,
        default=None,
        help="Last origin permitted in model ranking; defaults to validation-end.",
    )
    parser.add_argument("--ray-num-cpus", type=int, default=104)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    parser.add_argument(
        "--skip-scoring",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save OOS-error VARA corrections without running the all-purpose "
            "purpose/VARA strategy ranking."
        ),
    )
    args = parser.parse_args()

    strategies = _select_strategies(args.strategy_ids)
    selection_end = args.selection_end or args.validation_end
    if pd.Timestamp(selection_end) > pd.Timestamp(args.validation_end):
        raise ValueError("selection-end cannot follow validation-end.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "purpose_forecasts_path": str(args.purpose_forecasts_path),
        "output_dir": str(args.output_dir),
        "validation_start": args.validation_start,
        "validation_end": args.validation_end,
        "selection_end": selection_end,
        "ray_num_cpus": int(args.ray_num_cpus),
        "max_cpus_per_task": int(args.max_cpus_per_task),
        "skip_scoring": bool(args.skip_scoring),
        "strategies": [asdict(strategy) for strategy in strategies],
        "leakage_control": "each VARA worker receives OOS errors through t-1 only",
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    results, failures = run_sweep(
        purpose_forecasts_path=args.purpose_forecasts_path,
        output_dir=args.output_dir,
        strategies=strategies,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
    )
    if not failures.empty:
        print(f"OOS-error VARA failures: {len(failures)}", flush=True)
    if not args.skip_scoring:
        score_results(
            results,
            args.purpose_forecasts_path,
            strategies,
            args.output_dir,
            selection_end=selection_end,
        )


if __name__ == "__main__":
    main()
