#!/usr/bin/env python3
"""Evaluate VARA--AutoETS using fitted AutoETS in-sample residuals.

This is a single, prespecified ablation of the final-2016 Tourism experiment.
It keeps the current AutoETS and VARA settings fixed and changes only the error
history supplied to VARA: the rolling out-of-sample errors are replaced by the
last 24 fitted residuals from the origin-specific AutoETS fit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

try:
    from _bootstrap import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT  # noqa: F401

from scripts.build_tourism_table6_zza import (
    DISPLAY_NAMES,
    NON_AUTOETS_MODELS,
    RANK_SCORE_END,
    RANK_SCORE_START,
    ZZA_BENCHMARK_MODELS,
    ZZA_VARA_MODEL_NAMES,
)
from scripts.search_tourism_oos_error_vara import (
    ErrorStrategy,
    _error_vara_task,
)
from vara.forecast_baselines import auto_ets_baseline_panel, auto_ets_config


MODEL_NAME = "vara_auto_ets_insample"
METHOD_NAME = "VARA–AutoETS (in-sample errors)"
BASELINE_MODEL_NAME = "direct_auto_ets"
TABLE6_OTHER_MODELS = tuple(
    model for model in DISPLAY_NAMES if model != MODEL_NAME
)


def fixed_insample_strategy() -> ErrorStrategy:
    """Return the frozen VARA settings used for the error-source ablation."""
    return ErrorStrategy(
        strategy_id="predictive_avg_topk10_w24_stable05_insample",
        feature_source="predictive-average",
        attention_mode="score-topk",
        attention_top_k=10,
        transform="standardize",
        history_window=24,
        ridge=0.01,
        grid_size=7,
        eta_lower=-2.0,
        eta_upper=2.0,
        stability_cap=0.5,
    )


def _all_series_ids(columns: list[str], series_depth: int) -> list[str]:
    series_ids = sorted(
        str(column)
        for column in columns
        if str(column).endswith("All")
        and len(str(column)[:-3]) == int(series_depth)
    )
    if not series_ids:
        raise ValueError(
            f"No all-purpose Tourism series found at depth {series_depth}."
        )
    return series_ids


def _fit_origin(
    values: np.ndarray,
    dates: list[str],
    series_ids: list[str],
    origin_index: int,
    strategy_dict: dict,
    *,
    season_length: int,
    model_specification: str,
    response_normalization: str,
) -> dict:
    """Fit AutoETS and VARA at one origin without using the target outcome."""
    target_index = int(origin_index)
    target_date = pd.Timestamp(dates[target_index])
    history = np.asarray(values[:target_index], dtype=float)
    history_dates = pd.to_datetime(dates[:target_index])
    if history.shape != (target_index, len(series_ids)):
        raise ValueError("Origin history has an unexpected shape.")
    if len(history_dates) != len(history):
        raise ValueError("Origin history dates do not align with observations.")
    if history_dates[-1] >= target_date:
        raise ValueError("AutoETS training history must end before its target.")

    strategy = ErrorStrategy(**strategy_dict)
    if strategy.history_window is None:
        raise ValueError("The in-sample comparison requires a fixed residual window.")
    window = int(strategy.history_window)
    if len(history) < window:
        raise ValueError(
            f"Only {len(history)} training months are available; need {window}."
        )

    baseline = auto_ets_baseline_panel(
        history,
        horizon=1,
        seasonal_lag=int(season_length),
        series_ids=series_ids,
        model_specification=model_specification,
        damped=None,
        phi=None,
        response_normalization=response_normalization,
    )
    residuals = np.asarray(baseline["residuals"], dtype=float)
    if residuals.shape != history.shape or not np.isfinite(residuals).all():
        raise ValueError("AutoETS returned an invalid in-sample residual panel.")
    residual_history = residuals[-window:].copy()
    residual_dates = history_dates[-window:]

    vara_result = _error_vara_task(
        residual_history,
        series_ids,
        strategy_dict,
        target_index,
        str(target_date.date()),
        str(residual_dates[0].date()),
        str(residual_dates[-1].date()),
    )
    if not vara_result.get("ok", False):
        raise RuntimeError(
            "In-sample-error VARA failed at "
            f"{target_date.date()}: {vara_result.get('error_type')}: "
            f"{vara_result.get('error')}"
        )

    baseline_forecast = np.maximum(
        np.asarray(baseline["forecast"][0], dtype=float),
        0.0,
    )
    correction = np.asarray(vara_result["correction"], dtype=float)
    forecast = np.maximum(baseline_forecast + correction, 0.0)
    if not (
        baseline_forecast.shape
        == correction.shape
        == forecast.shape
        == (len(series_ids),)
    ):
        raise ValueError("Origin forecasts have an unexpected shape.")
    if not np.isfinite(
        np.column_stack([baseline_forecast, correction, forecast])
    ).all():
        raise ValueError("Origin forecasts contain non-finite values.")

    return {
        "origin_index": target_index,
        "date": str(target_date.date()),
        "history_start": str(history_dates[0].date()),
        "history_end": str(history_dates[-1].date()),
        "residual_history_start": str(residual_dates[0].date()),
        "residual_history_end": str(residual_dates[-1].date()),
        "n_autoets_train": int(len(history)),
        "n_vara_train": int(window),
        "baseline_forecast": baseline_forecast,
        "vara_correction": correction,
        "forecast": forecast,
        "eta_hat": float(vara_result["eta_hat"]),
        "eta_on_boundary": bool(vara_result["eta_on_boundary"]),
        "stationarity_proxy": float(vara_result["stationarity_proxy"]),
    }


def _run_origins(
    values: np.ndarray,
    dates: list[str],
    series_ids: list[str],
    origin_indices: list[int],
    strategy: ErrorStrategy,
    *,
    season_length: int,
    model_specification: str,
    response_normalization: str,
    ray_num_cpus: int,
    max_cpus_per_task: int,
) -> list[dict]:
    kwargs = {
        "season_length": int(season_length),
        "model_specification": str(model_specification),
        "response_normalization": str(response_normalization),
    }
    strategy_dict = asdict(strategy)
    if int(ray_num_cpus) <= 1:
        results = []
        for completed, origin_index in enumerate(origin_indices, start=1):
            results.append(_fit_origin(
                values,
                dates,
                series_ids,
                origin_index,
                strategy_dict,
                **kwargs,
            ))
            print(
                "Completed in-sample-error VARA origins: "
                f"{completed}/{len(origin_indices)}.",
                flush=True,
            )
        return results

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
        remote = ray.remote(num_cpus=int(max_cpus_per_task))(_fit_origin)
        values_ref = ray.put(np.asarray(values, dtype=float))
        dates_ref = ray.put(list(dates))
        series_ref = ray.put(list(series_ids))
        futures = [
            remote.remote(
                values_ref,
                dates_ref,
                series_ref,
                int(origin_index),
                strategy_dict,
                **kwargs,
            )
            for origin_index in origin_indices
        ]
        print(
            f"Submitted {len(futures)} in-sample-error VARA origins.",
            flush=True,
        )
        results: list[dict] = []
        pending = list(futures)
        while pending:
            ready, pending = ray.wait(
                pending,
                num_returns=min(2, len(pending)),
                timeout=30.0,
            )
            if not ready:
                print(
                    "Waiting for in-sample-error VARA origins: "
                    f"completed={len(results)}, pending={len(pending)}.",
                    flush=True,
                )
                continue
            results.extend(ray.get(ready))
            print(
                "Collected in-sample-error VARA origins: "
                f"{len(results)}/{len(futures)}.",
                flush=True,
            )
        return sorted(results, key=lambda item: int(item["origin_index"]))
    finally:
        if started_here:
            ray.shutdown()


def _prediction_frame(
    origin_results: list[dict],
    values: np.ndarray,
    series_ids: list[str],
) -> pd.DataFrame:
    rows = []
    for result in sorted(
        origin_results,
        key=lambda item: int(item["origin_index"]),
    ):
        origin_index = int(result["origin_index"])
        actual = np.asarray(values[origin_index], dtype=float)
        for series_index, series_id in enumerate(series_ids):
            rows.append({
                "date": pd.Timestamp(result["date"]),
                "series_id": series_id,
                "actual": float(actual[series_index]),
                "direct_auto_ets": float(
                    result["baseline_forecast"][series_index]
                ),
                "vara_correction": float(
                    result["vara_correction"][series_index]
                ),
                MODEL_NAME: float(result["forecast"][series_index]),
                "autoets_history_start": result["history_start"],
                "autoets_history_end": result["history_end"],
                "autoets_n_train": int(result["n_autoets_train"]),
                "vara_error_source": "AutoETS fitted in-sample residuals",
                "vara_history_start": result["residual_history_start"],
                "vara_history_end": result["residual_history_end"],
                "vara_n_train": int(result["n_vara_train"]),
                "eta_hat": float(result["eta_hat"]),
                "eta_on_boundary": bool(result["eta_on_boundary"]),
                "stationarity_proxy": float(
                    result["stationarity_proxy"]
                ),
            })
    frame = pd.DataFrame(rows).sort_values(
        ["date", "series_id"]
    ).reset_index(drop=True)
    if frame.duplicated(["date", "series_id"]).any():
        raise ValueError("Predictions contain duplicate date/series rows.")
    return frame


def calculate_metrics(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pooled and monthly original-scale RMSE and MAE."""
    methods = {
        "auto_ets": "direct_auto_ets",
        MODEL_NAME: MODEL_NAME,
    }
    origin_rows = []
    summary_rows = []
    for model, forecast_column in methods.items():
        error = (
            predictions[forecast_column].to_numpy(dtype=float)
            - predictions["actual"].to_numpy(dtype=float)
        )
        summary_rows.append({
            "model": model,
            "method": (
                "AutoETS" if model == "auto_ets" else METHOD_NAME
            ),
            "rmse": float(np.sqrt(np.mean(np.square(error)))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "n_origins": int(predictions["date"].nunique()),
            "n_series": int(predictions["series_id"].nunique()),
        })
        for date, group in predictions.groupby("date", sort=True):
            origin_error = (
                group[forecast_column].to_numpy(dtype=float)
                - group["actual"].to_numpy(dtype=float)
            )
            origin_rows.append({
                "model": model,
                "date": pd.Timestamp(date),
                "rmse": float(
                    np.sqrt(np.mean(np.square(origin_error)))
                ),
                "mae": float(np.mean(np.abs(origin_error))),
                "bias": float(np.mean(origin_error)),
                "n_series": int(len(group)),
            })
    summary = pd.DataFrame(summary_rows)
    autoets_rmse = float(
        summary.loc[summary["model"].eq("auto_ets"), "rmse"].iloc[0]
    )
    summary["rmse_improvement_vs_autoets_pct"] = (
        100.0 * (autoets_rmse - summary["rmse"]) / autoets_rmse
    )
    origins = pd.DataFrame(origin_rows).sort_values(
        ["model", "date"]
    ).reset_index(drop=True)
    return summary, origins


def _select_origin_mae(
    frame: pd.DataFrame,
    models: tuple[str, ...],
    *,
    source: str,
    rank_score_start: pd.Timestamp = RANK_SCORE_START,
    rank_score_end: pd.Timestamp = RANK_SCORE_END,
) -> pd.DataFrame:
    required = {"model", "date", "mae", "rmse"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    selected = frame[frame["model"].astype(str).isin(models)].copy()
    missing_models = sorted(
        set(models).difference(selected["model"].astype(str).unique())
    )
    if missing_models:
        raise ValueError(f"{source} is missing models: {missing_models}")
    selected["model"] = selected["model"].astype(str)
    selected["date"] = pd.to_datetime(selected["date"])
    selected = selected[selected["date"].between(
        pd.Timestamp(rank_score_start),
        pd.Timestamp(rank_score_end),
    )].copy()
    if selected.duplicated(["model", "date"]).any():
        raise ValueError(f"{source} contains duplicate model/date rows.")
    if not np.isfinite(
        selected.loc[:, ["mae", "rmse"]].to_numpy(dtype=float)
    ).all():
        raise ValueError(f"{source} contains non-finite origin metrics.")
    return selected.loc[:, ["model", "date", "mae", "rmse"]]


def load_table6_origin_mae(
    standard_origin_metrics_path: Path,
    autoets_origin_metrics_path: Path,
    vara_origin_metrics_path: Path,
    *,
    rank_score_start: pd.Timestamp = RANK_SCORE_START,
    rank_score_end: pd.Timestamp = RANK_SCORE_END,
) -> pd.DataFrame:
    """Load the exact eleven monthly-error panels used in current Table 6."""
    standard = _select_origin_mae(
        pd.read_csv(standard_origin_metrics_path),
        NON_AUTOETS_MODELS,
        source="non-AutoETS Table 6 origin metrics",
        rank_score_start=rank_score_start,
        rank_score_end=rank_score_end,
    )
    autoets = _select_origin_mae(
        pd.read_csv(autoets_origin_metrics_path),
        ZZA_BENCHMARK_MODELS,
        source="AutoETS Table 6 origin metrics",
        rank_score_start=rank_score_start,
        rank_score_end=rank_score_end,
    )
    vara = _select_origin_mae(
        pd.read_csv(vara_origin_metrics_path),
        tuple(ZZA_VARA_MODEL_NAMES),
        source="VARA Table 6 origin metrics",
        rank_score_start=rank_score_start,
        rank_score_end=rank_score_end,
    )
    vara["model"] = vara["model"].map(ZZA_VARA_MODEL_NAMES)
    combined = pd.concat([standard, autoets, vara], ignore_index=True)
    if set(combined["model"]) != set(TABLE6_OTHER_MODELS):
        raise ValueError(
            "The loaded monthly errors do not match the ten retained "
            "Table 6 methods."
        )
    return combined


def calculate_strict_rank_scores(
    origin_metrics: pd.DataFrame,
    *,
    rank_score_start: pd.Timestamp = RANK_SCORE_START,
    rank_score_end: pd.Timestamp = RANK_SCORE_END,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute strict monthly absolute-error win scores for any method set."""
    required = {"model", "date", "mae"}
    missing = sorted(required.difference(origin_metrics.columns))
    if missing:
        raise ValueError(f"RankScore input is missing columns: {missing}")
    frame = origin_metrics.loc[:, ["model", "date", "mae"]].copy()
    frame["model"] = frame["model"].astype(str)
    frame["date"] = pd.to_datetime(frame["date"])
    if frame.duplicated(["model", "date"]).any():
        raise ValueError("RankScore input contains duplicate model/date rows.")
    if not np.isfinite(frame["mae"].to_numpy(dtype=float)).all():
        raise ValueError("RankScore absolute errors must be finite.")

    wide = frame.pivot(index="date", columns="model", values="mae").sort_index()
    if wide.isna().any().any():
        raise ValueError("Every RankScore method must cover the same origins.")
    expected_dates = tuple(pd.date_range(
        pd.Timestamp(rank_score_start),
        pd.Timestamp(rank_score_end),
        freq="MS",
    ))
    if tuple(wide.index) != expected_dates:
        raise ValueError(
            "RankScore requires every monthly origin from "
            f"{pd.Timestamp(rank_score_start).date()} through "
            f"{pd.Timestamp(rank_score_end).date()}."
        )
    values = wide.to_numpy(dtype=float)
    n_comparators = values.shape[1] - 1
    if n_comparators < 1:
        raise ValueError("RankScore requires at least two methods.")
    wins = (values[:, :, None] < values[:, None, :]).sum(axis=2)
    scores = 100.0 * wins / float(n_comparators)
    summary = pd.DataFrame({
        "model": wide.columns,
        "rank_score": scores.mean(axis=0),
        "mean_monthly_absolute_error_wins": wins.mean(axis=0),
        "n_rank_score_origins": len(wide),
        "n_rank_score_comparators": n_comparators,
    })
    audit = (
        wide.rename_axis(columns="model")
        .stack()
        .rename("absolute_prediction_error")
        .reset_index()
    )
    audit["absolute_error_wins"] = wins.reshape(-1)
    audit["rank_score_s"] = scores.reshape(-1)
    return summary, audit


def validate_reference_autoets(
    predictions: pd.DataFrame,
    reference_path: Path,
    *,
    tolerance: float = 1e-8,
    evaluation_start: pd.Timestamp = RANK_SCORE_START,
    evaluation_end: pd.Timestamp = RANK_SCORE_END,
) -> float:
    """Require the refitted baseline to match the saved Table 6 AutoETS panel."""
    reference = pd.read_csv(reference_path)
    required = {"date", "series_id", "direct_auto_ets"}
    missing = sorted(required.difference(reference.columns))
    if missing:
        raise ValueError(f"Reference AutoETS panel is missing columns: {missing}")
    reference["date"] = pd.to_datetime(reference["date"])
    if "response_normalization" in reference:
        selected_normalization = reference.loc[
            reference["date"].between(
                pd.Timestamp(evaluation_start),
                pd.Timestamp(evaluation_end),
            ),
            "response_normalization",
        ]
        if not selected_normalization.eq("training_zscore").all():
            raise ValueError(
                "Reference AutoETS forecasts do not use training z-scores."
            )
    merged = predictions.loc[:, [
        "date",
        "series_id",
        "direct_auto_ets",
    ]].merge(
        reference.loc[:, [
            "date",
            "series_id",
            "direct_auto_ets",
        ]],
        on=["date", "series_id"],
        suffixes=("_new", "_reference"),
        how="left",
        validate="one_to_one",
    )
    if merged["direct_auto_ets_reference"].isna().any():
        raise ValueError("Some refitted AutoETS forecasts lack a reference row.")
    max_difference = float(np.max(np.abs(
        merged["direct_auto_ets_new"].to_numpy(dtype=float)
        - merged["direct_auto_ets_reference"].to_numpy(dtype=float)
    )))
    if max_difference > float(tolerance):
        raise ValueError(
            "Refitted AutoETS does not match the current Table 6 baseline; "
            f"maximum absolute difference is {max_difference}."
        )
    return max_difference


def run_evaluation(
    *,
    data_path: Path,
    reference_autoets_path: Path,
    standard_origin_metrics_path: Path,
    autoets_origin_metrics_path: Path,
    vara_origin_metrics_path: Path,
    table6_path: Path,
    output_dir: Path,
    series_depth: int,
    expected_n_series: int,
    season_length: int,
    model_specification: str,
    response_normalization: str,
    final_start: str,
    final_end: str,
    ray_num_cpus: int,
    max_cpus_per_task: int,
) -> dict:
    started = time.time()
    data = pd.read_csv(data_path)
    if "date" not in data:
        raise ValueError("Tourism data require a date column.")
    data["date"] = pd.to_datetime(data["date"])
    if data["date"].duplicated().any() or not data["date"].is_monotonic_increasing:
        raise ValueError("Tourism dates must be unique and increasing.")
    series_ids = _all_series_ids(list(data.columns), int(series_depth))
    if len(series_ids) != int(expected_n_series):
        raise ValueError(
            f"Expected {expected_n_series} series, found {len(series_ids)}."
        )
    values = data.loc[:, series_ids].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Tourism observations contain non-finite values.")
    start_date = pd.Timestamp(final_start)
    end_date = pd.Timestamp(final_end)
    origin_indices = [
        index
        for index, date in enumerate(data["date"])
        if start_date <= date <= end_date
    ]
    expected_dates = tuple(pd.date_range(start_date, end_date, freq="MS"))
    observed_dates = tuple(data.loc[origin_indices, "date"])
    if observed_dates != expected_dates:
        raise ValueError("The requested evaluation period is not a complete month grid.")

    strategy = fixed_insample_strategy()
    origin_results = _run_origins(
        values,
        [str(date.date()) for date in data["date"]],
        series_ids,
        origin_indices,
        strategy,
        season_length=season_length,
        model_specification=model_specification,
        response_normalization=response_normalization,
        ray_num_cpus=ray_num_cpus,
        max_cpus_per_task=max_cpus_per_task,
    )
    predictions = _prediction_frame(origin_results, values, series_ids)
    summary, origin_metrics = calculate_metrics(predictions)
    origin_wide = origin_metrics.pivot(
        index="date",
        columns="model",
        values=["rmse", "mae"],
    )
    monthly_rmse_wins = int(
        (
            origin_wide["rmse"][MODEL_NAME]
            < origin_wide["rmse"]["auto_ets"]
        ).sum()
    )
    monthly_mae_wins = int(
        (
            origin_wide["mae"][MODEL_NAME]
            < origin_wide["mae"]["auto_ets"]
        ).sum()
    )
    summary["monthly_rmse_wins_vs_autoets"] = np.where(
        summary["model"].eq(MODEL_NAME),
        monthly_rmse_wins,
        np.nan,
    )
    summary["monthly_mae_wins_vs_autoets"] = np.where(
        summary["model"].eq(MODEL_NAME),
        monthly_mae_wins,
        np.nan,
    )
    max_autoets_difference = validate_reference_autoets(
        predictions,
        reference_autoets_path,
        evaluation_start=start_date,
        evaluation_end=end_date,
    )

    table6_origins = load_table6_origin_mae(
        standard_origin_metrics_path,
        autoets_origin_metrics_path,
        vara_origin_metrics_path,
        rank_score_start=start_date,
        rank_score_end=end_date,
    )
    candidate_origins = origin_metrics[
        origin_metrics["model"].eq(MODEL_NAME)
    ].loc[:, ["model", "date", "mae", "rmse"]]
    replacement_origin_metrics = pd.concat(
        [table6_origins, candidate_origins],
        ignore_index=True,
    )
    replacement_rank_scores, replacement_rank_audit = (
        calculate_strict_rank_scores(
            replacement_origin_metrics,
            rank_score_start=start_date,
            rank_score_end=end_date,
        )
    )
    candidate_rank_score = float(
        replacement_rank_scores.loc[
            replacement_rank_scores["model"].eq(MODEL_NAME),
            "rank_score",
        ].iloc[0]
    )
    summary["rank_score_table6_replacement"] = np.where(
        summary["model"].eq(MODEL_NAME),
        candidate_rank_score,
        np.nan,
    )

    table6 = pd.read_csv(table6_path)
    current_labels = table6.loc[
        ~table6["model"].isin({MODEL_NAME, "vara_auto_ets_o"}),
        ["model", "method"],
    ].drop_duplicates("model")
    current_rmse = (
        table6_origins.assign(squared_error=table6_origins["rmse"].pow(2.0))
        .groupby("model", as_index=False, observed=True)["squared_error"]
        .mean()
        .rename(columns={"squared_error": "rmse"})
    )
    current_rmse["rmse"] = np.sqrt(
        current_rmse["rmse"].to_numpy(dtype=float)
    )
    current = current_labels.merge(
        current_rmse,
        on="model",
        how="inner",
        validate="one_to_one",
    )
    if len(current) != len(TABLE6_OTHER_MODELS):
        raise ValueError(
            "The comparison table does not contain all ten retained methods."
        )
    candidate_rmse = float(
        summary.loc[summary["model"].eq(MODEL_NAME), "rmse"].iloc[0]
    )
    comparison = pd.concat([
        current,
        pd.DataFrame([{
            "model": MODEL_NAME,
            "method": METHOD_NAME,
            "rmse": candidate_rmse,
        }]),
    ], ignore_index=True)
    comparison = comparison.merge(
        replacement_rank_scores,
        on="model",
        how="left",
        validate="one_to_one",
    )
    comparison["rank"] = (
        comparison["rank_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    comparison = comparison.sort_values(
        ["rank", "rmse", "model"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    origin_metrics.to_csv(output_dir / "metrics_by_origin.csv", index=False)
    comparison.to_csv(
        output_dir / "comparison_with_table6.csv",
        index=False,
    )
    replacement_rank_audit.to_csv(
        output_dir / "rankscore_table6_replacement_audit.csv",
        index=False,
    )

    config = {
        "data_path": str(data_path),
        "reference_autoets_path": str(reference_autoets_path),
        "output_dir": str(output_dir),
        "model": MODEL_NAME,
        "method": METHOD_NAME,
        "final_start": str(start_date.date()),
        "final_end": str(end_date.date()),
        "n_origins": len(origin_indices),
        "n_series": len(series_ids),
        "autoets_config": {
            **auto_ets_config(
                season_length,
                model_specification=model_specification,
                damped=None,
                phi=None,
            ),
            "response_normalization": response_normalization,
            "training_window": "expanding",
            "refit_every_origin": True,
        },
        "vara_strategy": asdict(strategy),
        "correction_weight": 1.0,
        "forecast_formula": (
            "max(training-normalized AutoETS forecast + VARA forecast of "
            "AutoETS fitted in-sample residuals, 0)"
        ),
        "residual_scale": "original visitor-night scale",
        "comparison_change": (
            "only the error source changes from rolling out-of-sample errors "
            "to origin-specific fitted AutoETS in-sample residuals"
        ),
        "hyperparameter_search": False,
        "target_leakage_control": (
            "each AutoETS and VARA fit uses observations through target t-1; "
            "the target outcome is used only for evaluation"
        ),
        "insample_residual_interpretation": (
            "residuals for earlier training dates are fitted using the full "
            "origin-specific training sample through t-1"
        ),
        "reference_autoets_max_absolute_difference": max_autoets_difference,
        "rank_score_comparators": 10,
        "monthly_rmse_wins_vs_autoets": monthly_rmse_wins,
        "monthly_mae_wins_vs_autoets": monthly_mae_wins,
        "execution": {
            "ray_num_cpus": int(ray_num_cpus),
            "max_cpus_per_task": int(max_cpus_per_task),
        },
        "elapsed_seconds": float(time.time() - started),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    print(comparison.loc[:, [
        "rank",
        "method",
        "rank_score",
        "rmse",
    ]].to_string(index=False), flush=True)
    print(
        "\nCandidate RankScore in the replacement Table 6 set: "
        f"{candidate_rank_score:.2f}",
        flush=True,
    )
    return {
        "predictions": predictions,
        "summary": summary,
        "origin_metrics": origin_metrics,
        "comparison": comparison,
        "config": config,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/tourism/tourism.csv"),
    )
    parser.add_argument(
        "--reference-autoets-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_autoets_zza_train_zscore_extended/"
            "rolling_forecasts.csv"
        ),
    )
    parser.add_argument(
        "--standard-origin-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_standardized_run/rolling-origin/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--autoets-origin-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_zza_train_zscore_run/rolling-origin/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--vara-origin-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_reliability_vara_autoets_zza_train_zscore/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--table6-path",
        type=Path,
        default=Path(
            "outputs/tourism_table6_zza_train_zscore/table6_zza_full.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "outputs/tourism_76_insample_error_vara_autoets"
        ),
    )
    parser.add_argument("--series-depth", type=int, default=3)
    parser.add_argument("--expected-n-series", type=int, default=76)
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--model-specification", default="ZZA")
    parser.add_argument(
        "--response-normalization",
        choices=("none", "training_zscore"),
        default="training_zscore",
    )
    parser.add_argument("--final-start", default="2016-01-01")
    parser.add_argument("--final-end", default="2016-12-01")
    parser.add_argument("--ray-num-cpus", type=int, default=1)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    args = parser.parse_args()
    run_evaluation(
        data_path=args.data_path,
        reference_autoets_path=args.reference_autoets_path,
        standard_origin_metrics_path=args.standard_origin_metrics_path,
        autoets_origin_metrics_path=args.autoets_origin_metrics_path,
        vara_origin_metrics_path=args.vara_origin_metrics_path,
        table6_path=args.table6_path,
        output_dir=args.output_dir,
        series_depth=args.series_depth,
        expected_n_series=args.expected_n_series,
        season_length=args.season_length,
        model_specification=args.model_specification,
        response_normalization=args.response_normalization,
        final_start=args.final_start,
        final_end=args.final_end,
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
    )


if __name__ == "__main__":
    main()
