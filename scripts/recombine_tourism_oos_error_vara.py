#!/usr/bin/env python3
"""Recombine Tourism AutoETS forecasts with leakage-safe OOS-error VARA corrections.

The combination weights are re-estimated at every rolling origin using only
previously observed forecast errors.  The no-VARA comparator is fitted
independently so that the incremental VARA contribution is measured fairly.
Shared-lambda mode applies one coefficient to the sum of the purpose and VARA
corrections, optionally estimating that coefficient by Tourism group.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from vara.forecast_metrics import forecast_metric_tables
from vara.forecast_significance import panel_diebold_mariano_tests


AUTO_ETS = "auto_ets"
PURPOSE_BLEND = "rolling_purpose_blend"
PURPOSE_VARA = "rolling_purpose_oos_vara"
RAW_VARA = "raw_oos_vara"
RANK_SCORE_METRICS = ("rmse", "mae", "mase", "rmsse")
SEPARATE_WEIGHTS = "separate-weights"
SHARED_LAMBDA = "shared-lambda"
COMBINATION_MODES = (SEPARATE_WEIGHTS, SHARED_LAMBDA)


def add_method_rank_scores(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rank every method using aggregate and scale-normalized point accuracy.

    The rank score is a normalized Borda score based on the mean rank across
    RMSE, MAE, MASE, and RMSSE.  It ranges from zero to 100 and is relative to
    the methods in ``comparison``; higher is better.
    """
    required = {
        "model",
        *RANK_SCORE_METRICS,
        "mape_pct",
        "smape_pct",
        "bias",
        "oos_r2",
        "oos_r2_zero",
    }
    missing = sorted(required.difference(comparison.columns))
    if missing:
        raise ValueError(f"Rank-score input is missing columns: {missing}")
    if comparison.empty:
        raise ValueError("At least one method is required to compute rank scores.")
    if comparison["model"].duplicated().any():
        duplicates = sorted(
            comparison.loc[comparison["model"].duplicated(False), "model"]
            .astype(str)
            .unique()
        )
        raise ValueError(f"Rank-score input has duplicate methods: {duplicates}")

    ranked = comparison.copy()
    numeric_columns = [
        *RANK_SCORE_METRICS,
        "mape_pct",
        "smape_pct",
        "bias",
        "oos_r2",
        "oos_r2_zero",
    ]
    numeric = ranked.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Rank-score metrics must all be finite.")
    ranked.loc[:, numeric_columns] = numeric

    loss_metrics = (*RANK_SCORE_METRICS, "mape_pct", "smape_pct")
    for metric in loss_metrics:
        ranked[f"rank_{metric}"] = ranked[metric].rank(
            method="average",
            ascending=True,
        )
    ranked["abs_bias"] = ranked["bias"].abs()
    ranked["rank_abs_bias"] = ranked["abs_bias"].rank(
        method="average",
        ascending=True,
    )
    for metric in ("oos_r2", "oos_r2_zero"):
        ranked[f"rank_{metric}"] = ranked[metric].rank(
            method="average",
            ascending=False,
        )

    score_rank_columns = [f"rank_{metric}" for metric in RANK_SCORE_METRICS]
    ranked["mean_accuracy_rank"] = ranked[score_rank_columns].mean(axis=1)
    n_methods = len(ranked)
    if n_methods == 1:
        ranked["rank_score"] = 100.0
    else:
        ranked["rank_score"] = 100.0 * (
            n_methods - ranked["mean_accuracy_rank"]
        ) / (n_methods - 1)
    ranked["overall_rank"] = (
        ranked["rank_score"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return ranked.sort_values(
        ["rank_score", "rmse", "model"],
        ascending=[False, True, True],
    ).reset_index(drop=True)


def load_strategy_panel(
    purpose_forecasts_path: Path,
    vara_results_path: Path,
    strategy_id: str,
) -> pd.DataFrame:
    purpose = pd.read_csv(purpose_forecasts_path)
    results = pd.read_csv(vara_results_path)
    required_purpose = {
        "date",
        "series_id",
        "actual",
        "direct_auto_ets",
        "purpose_sum_auto_ets",
        "autoets_history_end",
        "autoets_n_train",
    }
    required_results = {
        "date",
        "series_id",
        "strategy_id",
        "vara_correction",
        "history_start",
        "history_end",
        "n_train",
    }
    missing_purpose = sorted(required_purpose.difference(purpose.columns))
    missing_results = sorted(required_results.difference(results.columns))
    if missing_purpose:
        raise ValueError(f"Purpose forecasts are missing columns: {missing_purpose}")
    if missing_results:
        raise ValueError(f"VARA results are missing columns: {missing_results}")

    purpose["date"] = pd.to_datetime(purpose["date"])
    purpose["autoets_history_end"] = pd.to_datetime(purpose["autoets_history_end"])
    results["date"] = pd.to_datetime(results["date"])
    results["history_start"] = pd.to_datetime(results["history_start"])
    results["history_end"] = pd.to_datetime(results["history_end"])
    if not (purpose["autoets_history_end"] < purpose["date"]).all():
        raise ValueError("AutoETS history must end strictly before every forecast date.")
    if not (results["history_end"] < results["date"]).all():
        raise ValueError("VARA error history must end strictly before every forecast date.")
    if not (results["history_start"] <= results["history_end"]).all():
        raise ValueError("VARA history_start must not follow history_end.")
    results = results[results["strategy_id"].astype(str) == str(strategy_id)].copy()
    if results.empty:
        raise ValueError(f"No VARA results found for strategy {strategy_id!r}.")
    if results.duplicated(["date", "series_id"]).any():
        raise ValueError(f"VARA strategy {strategy_id!r} has duplicate date/series rows.")

    purpose_columns = sorted(required_purpose)
    if "group" in purpose.columns:
        purpose_columns.append("group")
    panel = results.loc[:, [
        "date",
        "series_id",
        "vara_correction",
        "history_start",
        "history_end",
        "n_train",
    ]].merge(
        purpose.loc[:, purpose_columns],
        on=["date", "series_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(panel) != len(results):
        raise ValueError("Some VARA corrections do not match the purpose forecast panel.")
    panel["year"] = panel["date"].dt.year.astype(int)
    panel["month"] = panel["date"].dt.month.astype(int)
    if "group" not in panel.columns:
        panel["group"] = panel["series_id"].astype(str).str[0]
    panel["purpose_correction"] = (
        panel["purpose_sum_auto_ets"] - panel["direct_auto_ets"]
    )
    numeric = [
        "actual",
        "direct_auto_ets",
        "purpose_sum_auto_ets",
        "purpose_correction",
        "vara_correction",
    ]
    if not np.isfinite(panel[numeric].to_numpy(dtype=float)).all():
        raise ValueError("The strategy panel contains non-finite forecast values.")
    return panel.sort_values(["date", "series_id"]).reset_index(drop=True)


def _fit_combination(
    train: pd.DataFrame,
    *,
    include_vara: bool,
    vara_weight_cap: float,
) -> dict[str, float]:
    columns = ["purpose_correction"]
    lower = [0.0]
    upper = [1.0]
    if include_vara:
        columns.append("vara_correction")
        lower.append(0.0)
        upper.append(float(vara_weight_cap))
    design = train.loc[:, columns].to_numpy(dtype=float)
    target = (
        train["actual"].to_numpy(dtype=float)
        - train["direct_auto_ets"].to_numpy(dtype=float)
    )
    fitted = lsq_linear(
        design,
        target,
        bounds=(np.asarray(lower), np.asarray(upper)),
        lsmr_tol="auto",
    )
    if not fitted.success:
        raise RuntimeError(f"Combination fit failed: {fitted.message}")
    return {column: float(value) for column, value in zip(columns, fitted.x, strict=True)}


def _fit_shared_lambda(train: pd.DataFrame, *, upper_bound: float) -> float:
    correction = (
        train["purpose_correction"].to_numpy(dtype=float)
        + train["vara_correction"].to_numpy(dtype=float)
    )
    target = (
        train["actual"].to_numpy(dtype=float)
        - train["direct_auto_ets"].to_numpy(dtype=float)
    )
    fitted = lsq_linear(
        correction[:, None],
        target,
        bounds=(np.asarray([0.0]), np.asarray([float(upper_bound)])),
        lsmr_tol="auto",
    )
    if not fitted.success:
        raise RuntimeError(f"Shared-lambda fit failed: {fitted.message}")
    return float(fitted.x[0])


def rolling_recombine(
    panel: pd.DataFrame,
    target_dates: list[pd.Timestamp] | pd.DatetimeIndex,
    *,
    calibration_window_months: int,
    vara_weight_cap: float,
    minimum_training_months: int = 12,
    combination_mode: str = SEPARATE_WEIGHTS,
    combination_group_col: str | None = None,
    shared_lambda_cap: float = 0.5,
    group_lambda_global_weight: float = 0.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if int(calibration_window_months) < int(minimum_training_months):
        raise ValueError("calibration_window_months must cover minimum_training_months.")
    if not 0.0 <= float(vara_weight_cap) <= 1.0:
        raise ValueError("vara_weight_cap must be between zero and one.")
    if str(combination_mode) not in COMBINATION_MODES:
        raise ValueError(
            f"combination_mode must be one of {COMBINATION_MODES}, got {combination_mode!r}."
        )
    if not 0.0 <= float(shared_lambda_cap) <= 1.0:
        raise ValueError("shared_lambda_cap must be between zero and one.")
    if not 0.0 <= float(group_lambda_global_weight) <= 1.0:
        raise ValueError("group_lambda_global_weight must be between zero and one.")
    if combination_group_col is not None and str(combination_mode) != SHARED_LAMBDA:
        raise ValueError("combination_group_col is only valid in shared-lambda mode.")
    if combination_group_col is not None and combination_group_col not in panel.columns:
        raise ValueError(
            f"Combination group column {combination_group_col!r} is not present in the panel."
        )
    if float(group_lambda_global_weight) > 0.0 and combination_group_col is None:
        raise ValueError(
            "group_lambda_global_weight requires a shared-lambda combination group column."
        )

    predictions = []
    coefficient_rows = []
    for target_date in sorted(pd.Timestamp(value) for value in target_dates):
        previous_dates = sorted(panel.loc[panel["date"] < target_date, "date"].unique())
        previous_dates = previous_dates[-int(calibration_window_months):]
        if len(previous_dates) < int(minimum_training_months):
            raise ValueError(
                f"Only {len(previous_dates)} prior months are available for {target_date.date()}; "
                f"need at least {minimum_training_months}."
            )
        train = panel[panel["date"].isin(previous_dates)]
        test = panel[panel["date"] == target_date].copy()
        if test.empty:
            raise ValueError(f"No strategy forecasts are available for {target_date.date()}.")

        purpose_coefficients = _fit_combination(
            train,
            include_vara=False,
            vara_weight_cap=vara_weight_cap,
        )
        baseline = test["direct_auto_ets"].to_numpy(dtype=float)
        purpose = test["purpose_correction"].to_numpy(dtype=float)
        vara = test["vara_correction"].to_numpy(dtype=float)
        test[AUTO_ETS] = np.maximum(baseline, 0.0)
        test[PURPOSE_BLEND] = np.maximum(
            baseline + purpose_coefficients["purpose_correction"] * purpose,
            0.0,
        )
        if str(combination_mode) == SEPARATE_WEIGHTS:
            full_coefficients = _fit_combination(
                train,
                include_vara=True,
                vara_weight_cap=vara_weight_cap,
            )
            test[PURPOSE_VARA] = np.maximum(
                baseline
                + full_coefficients["purpose_correction"] * purpose
                + full_coefficients["vara_correction"] * vara,
                0.0,
            )
            coefficient_rows.append({
                "date": target_date,
                "training_start": pd.Timestamp(previous_dates[0]),
                "training_end": pd.Timestamp(previous_dates[-1]),
                "n_training_months": int(len(previous_dates)),
                "n_training_observations": int(len(train)),
                "combination_mode": SEPARATE_WEIGHTS,
                "combination_group_col": None,
                "combination_group": "all",
                "purpose_only_weight": purpose_coefficients["purpose_correction"],
                "full_purpose_weight": full_coefficients["purpose_correction"],
                "full_vara_weight": full_coefficients["vara_correction"],
                "shared_lambda": np.nan,
                "group_raw_lambda": np.nan,
                "global_shared_lambda": np.nan,
                "group_lambda_global_weight": np.nan,
            })
        else:
            test[PURPOSE_VARA] = np.nan
            global_shared_lambda = _fit_shared_lambda(
                train,
                upper_bound=shared_lambda_cap,
            )
            if combination_group_col is None:
                grouped_indices = [("all", test.index)]
            else:
                grouped_indices = list(test.groupby(combination_group_col, sort=True).groups.items())
            for group_value, test_indices in grouped_indices:
                group_train = train
                if combination_group_col is not None:
                    group_train = train[
                        train[combination_group_col] == group_value
                    ]
                if group_train.empty:
                    raise ValueError(
                        f"No calibration rows are available for group {group_value!r} "
                        f"at {target_date.date()}."
                    )
                group_raw_lambda = _fit_shared_lambda(
                    group_train,
                    upper_bound=shared_lambda_cap,
                )
                shared_lambda = (
                    float(group_lambda_global_weight) * global_shared_lambda
                    + (1.0 - float(group_lambda_global_weight)) * group_raw_lambda
                )
                grouped_baseline = test.loc[test_indices, "direct_auto_ets"].to_numpy(dtype=float)
                grouped_correction = (
                    test.loc[test_indices, "purpose_correction"].to_numpy(dtype=float)
                    + test.loc[test_indices, "vara_correction"].to_numpy(dtype=float)
                )
                test.loc[test_indices, PURPOSE_VARA] = np.maximum(
                    grouped_baseline + shared_lambda * grouped_correction,
                    0.0,
                )
                coefficient_rows.append({
                    "date": target_date,
                    "training_start": pd.Timestamp(previous_dates[0]),
                    "training_end": pd.Timestamp(previous_dates[-1]),
                    "n_training_months": int(len(previous_dates)),
                    "n_training_observations": int(len(group_train)),
                    "combination_mode": SHARED_LAMBDA,
                    "combination_group_col": combination_group_col,
                    "combination_group": str(group_value),
                    "purpose_only_weight": purpose_coefficients["purpose_correction"],
                    "full_purpose_weight": shared_lambda,
                    "full_vara_weight": shared_lambda,
                    "shared_lambda": shared_lambda,
                    "group_raw_lambda": group_raw_lambda,
                    "global_shared_lambda": global_shared_lambda,
                    "group_lambda_global_weight": float(group_lambda_global_weight),
                })
            if test[PURPOSE_VARA].isna().any():
                raise RuntimeError(f"Shared-lambda forecasts are incomplete for {target_date.date()}.")
        test[RAW_VARA] = np.maximum(baseline + vara, 0.0)
        predictions.append(test)
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(coefficient_rows)


def _point_metrics(frame: pd.DataFrame, forecast_column: str, period: str) -> dict:
    error = (
        frame[forecast_column].to_numpy(dtype=float)
        - frame["actual"].to_numpy(dtype=float)
    )
    return {
        "period": period,
        "model": forecast_column,
        "n_observations": int(len(frame)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def historical_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    models = (PURPOSE_VARA, PURPOSE_BLEND, AUTO_ETS, RAW_VARA)
    rows = []
    for year in sorted(predictions["year"].unique()):
        fold = predictions[predictions["year"] == int(year)]
        rows.extend(_point_metrics(fold, model, str(year)) for model in models)
    rows.extend(_point_metrics(predictions, model, "pooled") for model in models)
    metrics = pd.DataFrame(rows)
    auto_by_period = (
        metrics[metrics["model"] == AUTO_ETS]
        .set_index("period")["rmse"]
        .to_dict()
    )
    purpose_by_period = (
        metrics[metrics["model"] == PURPOSE_BLEND]
        .set_index("period")["rmse"]
        .to_dict()
    )
    auto_rmse = metrics["period"].map(auto_by_period).to_numpy(dtype=float)
    purpose_rmse = metrics["period"].map(purpose_by_period).to_numpy(dtype=float)
    metrics["rmse_improvement_vs_auto_ets_pct"] = 100.0 * (
        auto_rmse - metrics["rmse"]
    ) / auto_rmse
    metrics["rmse_improvement_vs_purpose_blend_pct"] = 100.0 * (
        purpose_rmse - metrics["rmse"]
    ) / purpose_rmse
    return metrics


def build_benchmark_comparison(
    summary: pd.DataFrame,
    benchmark_metrics_path: Path,
) -> pd.DataFrame:
    benchmark = pd.read_csv(benchmark_metrics_path)
    required = {
        "model",
        *RANK_SCORE_METRICS,
        "mape_pct",
        "smape_pct",
        "bias",
        "oos_r2",
        "oos_r2_zero",
    }
    missing = sorted(required.difference(benchmark.columns))
    if missing:
        raise ValueError(f"Benchmark metrics are missing columns: {missing}")
    benchmark = benchmark.copy()
    benchmark["source"] = "benchmark"
    additions = summary[summary["model"] != AUTO_ETS].copy()
    additions["source"] = "oos_error_vara_recombination"
    comparison = pd.concat([additions, benchmark], ignore_index=True, sort=False)
    auto_rmse = float(
        comparison.loc[comparison["model"] == AUTO_ETS, "rmse"].iloc[0]
    )
    seasonal_naive_rmse = float(
        comparison.loc[comparison["model"] == "seasonal_naive", "rmse"].iloc[0]
    )
    comparison["rmse_improvement_vs_auto_ets_pct"] = 100.0 * (
        auto_rmse - comparison["rmse"]
    ) / auto_rmse
    comparison["rmse_improvement_vs_seasonal_naive_pct"] = 100.0 * (
        seasonal_naive_rmse - comparison["rmse"]
    ) / seasonal_naive_rmse
    return add_method_rank_scores(comparison)


def _wide(
    frame: pd.DataFrame,
    value_column: str,
    series_ids: list[str],
) -> np.ndarray:
    return (
        frame.pivot(index="date", columns="series_id", values=value_column)
        .sort_index()
        .loc[:, series_ids]
        .to_numpy(dtype=float)
    )


def load_dm_forecast_panel(
    final_predictions: pd.DataFrame,
    benchmark_forecasts_path: Path,
    method_names: list[str],
) -> tuple[dict[str, np.ndarray], np.ndarray, list[pd.Timestamp], list[str]]:
    """Load and strictly align all forecasts used in the Tourism DM tests."""
    keys = ["date", "series_id"]
    local_models = [PURPOSE_VARA, PURPOSE_BLEND, AUTO_ETS, RAW_VARA]
    required_local = {*keys, "actual", *local_models}
    missing_local = sorted(required_local.difference(final_predictions.columns))
    if missing_local:
        raise ValueError(f"Final predictions are missing DM columns: {missing_local}")
    if len(method_names) != len(set(method_names)):
        raise ValueError("DM method names must be unique.")

    local = final_predictions.loc[:, [*keys, "actual", *local_models]].copy()
    local["date"] = pd.to_datetime(local["date"])
    local["series_id"] = local["series_id"].astype(str)
    if local.duplicated(keys).any():
        raise ValueError("Final predictions have duplicate date/series rows.")

    benchmark = pd.read_csv(benchmark_forecasts_path)
    required_benchmark = {*keys, "train_end"}
    missing_benchmark = sorted(required_benchmark.difference(benchmark.columns))
    if missing_benchmark:
        raise ValueError(
            f"Benchmark forecasts are missing DM columns: {missing_benchmark}"
        )
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    benchmark["train_end"] = pd.to_datetime(benchmark["train_end"])
    benchmark["series_id"] = benchmark["series_id"].astype(str)
    if benchmark.duplicated(keys).any():
        raise ValueError("Benchmark forecasts have duplicate date/series rows.")
    if not (benchmark["train_end"] < benchmark["date"]).all():
        raise ValueError("Benchmark training must end strictly before each forecast date.")

    benchmark_models = [
        model for model in method_names if model in benchmark.columns
    ]
    available_models = set(local_models).union(benchmark_models)
    missing_methods = sorted(set(method_names).difference(available_models))
    if missing_methods:
        raise ValueError(f"Matched forecasts are missing methods: {missing_methods}")
    benchmark = benchmark.loc[:, [*keys, *benchmark_models]]
    merged = local.merge(
        benchmark,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_local", "_benchmark"),
    )
    unmatched = merged["_merge"] != "both"
    if unmatched.any():
        raise ValueError(
            f"Local and benchmark forecasts have {int(unmatched.sum())} unmatched rows."
        )
    merged = merged.drop(columns="_merge")

    overlap = set(local_models).intersection(benchmark_models)
    for model in overlap:
        local_column = f"{model}_local"
        benchmark_column = f"{model}_benchmark"
        if not np.allclose(
            merged[local_column].to_numpy(dtype=float),
            merged[benchmark_column].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-8,
        ):
            raise ValueError(f"Local and benchmark {model!r} forecasts do not match.")
        merged[model] = merged[local_column]

    series_ids = sorted(merged["series_id"].unique())
    dates = sorted(pd.Timestamp(value) for value in merged["date"].unique())
    if len(merged) != len(series_ids) * len(dates):
        raise ValueError("DM forecasts do not form a complete date-by-series panel.")
    actual = _wide(merged, "actual", series_ids)
    forecasts = {
        model: _wide(merged, model, series_ids)
        for model in method_names
    }
    return forecasts, actual, dates, series_ids


def build_leakage_audit(
    panel: pd.DataFrame,
    historical_coefficients: pd.DataFrame,
    final_coefficients: pd.DataFrame,
    *,
    validation_end_year: int,
    final_year: int,
) -> dict:
    coefficient_frames = [historical_coefficients, final_coefficients]
    checks = {
        "autoets_history_precedes_target": bool(
            (panel["autoets_history_end"] < panel["date"]).all()
        ),
        "vara_history_precedes_target": bool(
            (panel["history_end"] < panel["date"]).all()
        ),
        "vara_history_start_not_after_end": bool(
            (panel["history_start"] <= panel["history_end"]).all()
        ),
        "combination_training_precedes_target": bool(all(
            (frame["training_end"] < frame["date"]).all()
            for frame in coefficient_frames
        )),
        "validation_precedes_final_test": bool(
            int(validation_end_year) < int(final_year)
        ),
    }
    return {
        "all_checks_passed": bool(all(checks.values())),
        "checks": checks,
        "forecast_protocol": "monthly rolling-origin one-step-ahead",
        "information_available_at_origin": "observations dated t-1 or earlier",
        "n_forecast_rows": int(len(panel)),
        "n_series": int(panel["series_id"].nunique()),
        "max_vara_training_months": int(panel["n_train"].max()),
        "max_combination_training_months": int(max(
            frame["n_training_months"].max() for frame in coefficient_frames
        )),
        "validation_end_year": int(validation_end_year),
        "final_test_year": int(final_year),
    }


def run_recombination(
    *,
    data_path: Path,
    purpose_forecasts_path: Path,
    vara_results_path: Path,
    strategy_id: str,
    output_dir: Path,
    validation_start_year: int,
    validation_end_year: int,
    final_year: int,
    calibration_window_months: int,
    vara_weight_cap: float,
    benchmark_metrics_path: Path | None,
    benchmark_forecasts_path: Path | None = None,
    dm_alpha: float = 0.05,
    combination_mode: str = SHARED_LAMBDA,
    combination_group_col: str | None = None,
    shared_lambda_cap: float = 0.5,
    group_lambda_global_weight: float = 0.0,
) -> dict:
    if int(validation_start_year) > int(validation_end_year):
        raise ValueError("validation_start_year must not exceed validation_end_year.")
    if int(validation_end_year) >= int(final_year):
        raise ValueError("Validation years must end strictly before the final test year.")
    panel = load_strategy_panel(
        purpose_forecasts_path,
        vara_results_path,
        strategy_id,
    )
    historical_dates = sorted(
        panel.loc[
            panel["year"].between(int(validation_start_year), int(validation_end_year)),
            "date",
        ].unique()
    )
    final_dates = sorted(panel.loc[panel["year"] == int(final_year), "date"].unique())
    if not historical_dates:
        raise ValueError("No historical validation dates are available.")
    if len(final_dates) != 12:
        raise ValueError(
            f"Expected 12 final-year VARA origins for {final_year}, found {len(final_dates)}."
        )

    historical, historical_coefficients = rolling_recombine(
        panel,
        historical_dates,
        calibration_window_months=calibration_window_months,
        vara_weight_cap=vara_weight_cap,
        combination_mode=combination_mode,
        combination_group_col=combination_group_col,
        shared_lambda_cap=shared_lambda_cap,
        group_lambda_global_weight=group_lambda_global_weight,
    )
    historical_metrics = historical_metric_table(historical)
    final, final_coefficients = rolling_recombine(
        panel,
        final_dates,
        calibration_window_months=calibration_window_months,
        vara_weight_cap=vara_weight_cap,
        combination_mode=combination_mode,
        combination_group_col=combination_group_col,
        shared_lambda_cap=shared_lambda_cap,
        group_lambda_global_weight=group_lambda_global_weight,
    )
    leakage_audit = build_leakage_audit(
        panel,
        historical_coefficients,
        final_coefficients,
        validation_end_year=validation_end_year,
        final_year=final_year,
    )
    if not leakage_audit["all_checks_passed"]:
        raise RuntimeError(f"Information-leakage audit failed: {leakage_audit['checks']}")

    series_ids = sorted(final["series_id"].astype(str).unique())
    dates = sorted(final["date"].unique())
    models = (PURPOSE_VARA, PURPOSE_BLEND, AUTO_ETS, RAW_VARA)
    forecasts = {model: _wide(final, model, series_ids) for model in models}
    test = _wide(final, "actual", series_ids)
    raw = pd.read_csv(data_path)
    raw["date"] = pd.to_datetime(raw["date"])
    train = (
        raw[raw["date"].dt.year < int(final_year)]
        .sort_values("date")
        .loc[:, series_ids]
        .to_numpy(dtype=float)
    )
    by_series, by_origin, summary = forecast_metric_tables(
        forecasts,
        test,
        train,
        dates,
        series_ids,
        season_length=12,
        forecast_mode="rolling-origin",
        horizon=12,
        summary_context={
            "data": str(data_path),
            "test_year": int(final_year),
            "strategy_id": str(strategy_id),
            "combination_mode": str(combination_mode),
            "combination_group_col": combination_group_col,
            "shared_lambda_cap": float(shared_lambda_cap),
            "group_lambda_global_weight": float(group_lambda_global_weight),
            "forecast_lead": 1,
            "n_rolling_origins": int(len(dates)),
        },
    )
    auto_rmse = float(summary.loc[summary["model"] == AUTO_ETS, "rmse"].iloc[0])
    purpose_rmse = float(summary.loc[summary["model"] == PURPOSE_BLEND, "rmse"].iloc[0])
    summary["rmse_improvement_vs_auto_ets_pct"] = 100.0 * (
        auto_rmse - summary["rmse"]
    ) / auto_rmse
    summary["rmse_improvement_vs_purpose_blend_pct"] = 100.0 * (
        purpose_rmse - summary["rmse"]
    ) / purpose_rmse
    summary = summary.sort_values("rmse").reset_index(drop=True)
    summary["rank_rmse"] = np.arange(1, len(summary) + 1)
    benchmark_comparison = None
    if benchmark_metrics_path is not None:
        benchmark_comparison = build_benchmark_comparison(
            summary,
            benchmark_metrics_path,
        )
    dm_tests = None
    if benchmark_forecasts_path is not None:
        if benchmark_comparison is None:
            raise ValueError(
                "benchmark_metrics_path is required when DM tests are requested."
            )
        dm_forecasts, dm_actual, dm_dates, dm_series_ids = load_dm_forecast_panel(
            final,
            benchmark_forecasts_path,
            benchmark_comparison["model"].astype(str).tolist(),
        )
        dm_tests = panel_diebold_mariano_tests(
            dm_forecasts,
            dm_actual,
            dm_dates,
            focal_model=PURPOSE_VARA,
            forecast_horizon=1,
            alpha=dm_alpha,
        )
        comparator_ranks = benchmark_comparison.loc[:, [
            "model",
            "overall_rank",
            "rank_score",
        ]].rename(columns={
            "model": "comparator_model",
            "overall_rank": "comparator_overall_rank",
            "rank_score": "comparator_rank_score",
        })
        dm_tests = dm_tests.merge(
            comparator_ranks,
            on="comparator_model",
            how="left",
            validate="many_to_one",
        )
        loss_order = pd.Categorical(
            dm_tests["loss"],
            categories=["squared_error", "absolute_error"],
            ordered=True,
        )
        dm_tests = (
            dm_tests.assign(_loss_order=loss_order)
            .sort_values(["_loss_order", "comparator_overall_rank", "comparator_model"])
            .drop(columns="_loss_order")
            .reset_index(drop=True)
        )
        leakage_audit["checks"]["benchmark_training_precedes_target"] = True
        leakage_audit["all_checks_passed"] = bool(
            all(leakage_audit["checks"].values())
        )
        leakage_audit["dm_n_origins"] = int(len(dm_dates))
        leakage_audit["dm_n_series"] = int(len(dm_series_ids))

    output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_dir / "strategy_panel.csv", index=False)
    historical.to_csv(output_dir / "historical_predictions.csv", index=False)
    historical_coefficients.to_csv(
        output_dir / "historical_calibration_coefficients.csv", index=False
    )
    historical_metrics.to_csv(output_dir / "historical_metrics.csv", index=False)
    final.to_csv(output_dir / "final_predictions_long.csv", index=False)
    final_coefficients.to_csv(output_dir / "final_calibration_coefficients.csv", index=False)
    by_series.to_csv(output_dir / "metrics_by_series.csv", index=False)
    by_origin.to_csv(output_dir / "metrics_by_origin.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    if benchmark_comparison is not None:
        benchmark_comparison.to_csv(
            output_dir / "benchmark_comparison.csv", index=False
        )
    if dm_tests is not None:
        dm_tests.to_csv(output_dir / "diebold_mariano_tests.csv", index=False)
    forecast = pd.DataFrame(forecasts[PURPOSE_VARA], columns=series_ids)
    forecast.insert(0, "date", dates)
    forecast.to_csv(output_dir / "forecast.csv", index=False)

    pooled = historical_metrics[historical_metrics["period"] == "pooled"]
    selection = {
        "strategy_id": str(strategy_id),
        "combination": PURPOSE_VARA,
        "selection_status": "locked before final-year VARA forecasts",
        "selection_rule": (
            "minimum historical incremental VARA RMSE gain of 0.5%; "
            "then minimum pooled historical RMSE"
        ),
        "validation_years": list(
            range(int(validation_start_year), int(validation_end_year) + 1)
        ),
        "final_test_year": int(final_year),
        "forecast_protocol": "monthly rolling-origin one-step-ahead",
        "calibration_window_months": int(calibration_window_months),
        "vara_weight_cap": float(vara_weight_cap),
        "combination_mode": str(combination_mode),
        "combination_group_col": combination_group_col,
        "shared_lambda_cap": float(shared_lambda_cap),
        "group_lambda_global_weight": float(group_lambda_global_weight),
        "historical_pooled_metrics": pooled.to_dict(orient="records"),
        "purpose_forecasts_path": str(purpose_forecasts_path),
        "vara_results_path": str(vara_results_path),
        "leakage_control": {
            "autoets_information_end": "t-1",
            "vara_error_information_end": "t-1",
            "combination_weight_information_end": "t-1",
            "validation_end_before_final_test": True,
        },
        "benchmark_metrics_path": (
            None if benchmark_metrics_path is None else str(benchmark_metrics_path)
        ),
        "benchmark_forecasts_path": (
            None
            if benchmark_forecasts_path is None
            else str(benchmark_forecasts_path)
        ),
        "benchmark_rank_score": (
            None
            if benchmark_comparison is None
            else {
                "metrics": list(RANK_SCORE_METRICS),
                "metric_rank_method": "average ranks; lower metric values are better",
                "mean_accuracy_rank": "mean of the four metric ranks",
                "rank_score": (
                    "100 * (n_methods - mean_accuracy_rank) / (n_methods - 1); "
                    "higher is better"
                ),
                "excluded_from_score": {
                    "mape_pct_and_smape_pct": (
                        "reported as diagnostic ranks but excluded because near-zero "
                        "actuals can dominate percentage errors"
                    ),
                    "bias": "reported as absolute-bias rank but is not an accuracy loss",
                    "oos_r2_and_oos_r2_zero": (
                        "reported as diagnostic ranks but excluded because they are "
                        "monotone transformations of squared error on a common panel"
                    ),
                },
            }
        ),
        "diebold_mariano_tests": (
            None
            if dm_tests is None
            else {
                "focal_model": PURPOSE_VARA,
                "losses": ["squared_error", "absolute_error"],
                "panel_aggregation": (
                    "mean loss across all series within each monthly origin"
                ),
                "n_origins": int(dm_tests["n_origins"].iloc[0]),
                "n_series": int(dm_tests["n_series"].iloc[0]),
                "forecast_horizon": 1,
                "hac_lag": 0,
                "small_sample_correction": "Harvey-Leybourne-Newbold",
                "reference_distribution": "Student t with n_origins - 1 degrees of freedom",
                "raw_dm_reference_distribution": (
                    "standard normal; reported as a diagnostic"
                ),
                "multiple_testing": (
                    "Holm adjustment within each loss across all comparator methods"
                ),
                "alpha": float(dm_alpha),
                "direction": (
                    "negative statistic and positive metric improvement favor focal_model"
                ),
            }
        ),
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    (output_dir / "leakage_audit.json").write_text(
        json.dumps(leakage_audit, indent=2) + "\n"
    )

    print("Historical rolling-origin metrics:", flush=True)
    print(pooled.sort_values("rmse").to_string(index=False), flush=True)
    print("\nFinal-year metrics:", flush=True)
    print(summary.to_string(index=False), flush=True)
    if benchmark_comparison is not None:
        print("\nFinal-year benchmark comparison:", flush=True)
        print(
            benchmark_comparison.loc[:, [
                "overall_rank",
                "rank_score",
                "mean_accuracy_rank",
                "rank_rmse",
                "rank_mae",
                "rank_mase",
                "rank_rmsse",
                "model",
                "rmse",
                "mae",
                "mase",
                "rmsse",
                "oos_r2",
                "rmse_improvement_vs_auto_ets_pct",
                "rmse_improvement_vs_seasonal_naive_pct",
            ]].to_string(index=False),
            flush=True,
        )
    if dm_tests is not None:
        for loss, label in (
            ("squared_error", "Squared-error DM tests (RMSE)"),
            ("absolute_error", "Absolute-error DM tests (MAE)"),
        ):
            print(f"\n{label}:", flush=True)
            print(
                dm_tests.loc[dm_tests["loss"] == loss, [
                    "comparator_overall_rank",
                    "comparator_model",
                    "focal_metric",
                    "comparator_metric",
                    "metric_improvement_pct",
                    "dm_statistic",
                    "p_value_two_sided",
                    "p_value_focal_better",
                    "p_value_focal_better_holm",
                    "significant_improvement_holm",
                ]].to_string(index=False),
                flush=True,
            )
        for loss, label in (
            ("squared_error", "Raw asymptotic-normal DM tests (RMSE)"),
            ("absolute_error", "Raw asymptotic-normal DM tests (MAE)"),
        ):
            print(f"\n{label}:", flush=True)
            print(
                dm_tests.loc[dm_tests["loss"] == loss, [
                    "comparator_overall_rank",
                    "comparator_model",
                    "metric_improvement_pct",
                    "dm_statistic_raw",
                    "p_value_raw_two_sided",
                    "p_value_raw_focal_better",
                    "p_value_raw_two_sided_holm",
                    "p_value_raw_focal_better_holm",
                    "significant_improvement_raw_holm",
                ]].to_string(index=False),
                flush=True,
            )
    return {
        "panel": panel,
        "historical_metrics": historical_metrics,
        "final_summary": summary,
        "dm_tests": dm_tests,
        "selection": selection,
        "leakage_audit": leakage_audit,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/tourism/tourism.csv"))
    parser.add_argument(
        "--purpose-forecasts-path",
        type=Path,
        default=Path(
            "outputs/tourism_purpose_autoets_leakage_safe_2011_2016/"
            "rolling_forecasts.csv"
        ),
    )
    parser.add_argument(
        "--vara-results-path",
        type=Path,
        default=Path(
            "outputs/tourism_oos_error_vara_leakage_safe_search/origin_results.csv"
        ),
    )
    parser.add_argument(
        "--strategy-id",
        type=str,
        default="predictive_avg_topk10_w24_stable05",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_oos_error_vara_shared_lambda"),
    )
    parser.add_argument("--validation-start-year", type=int, default=2013)
    parser.add_argument("--validation-end-year", type=int, default=2015)
    parser.add_argument("--final-year", type=int, default=2016)
    parser.add_argument("--calibration-window-months", type=int, default=24)
    parser.add_argument("--vara-weight-cap", type=float, default=0.5)
    parser.add_argument(
        "--combination-mode",
        choices=COMBINATION_MODES,
        default=SHARED_LAMBDA,
        help=(
            "Use the existing separate purpose/VARA weights or one coefficient "
            "shared by the purpose and VARA corrections."
        ),
    )
    parser.add_argument(
        "--combination-group-col",
        type=str,
        default=None,
        help=(
            "In shared-lambda mode, fit one lambda per value of this panel column; "
            "omit it to pool all series."
        ),
    )
    parser.add_argument("--shared-lambda-cap", type=float, default=0.5)
    parser.add_argument(
        "--group-lambda-global-weight",
        type=float,
        default=0.0,
        help=(
            "For grouped shared lambdas, convex weight on the pooled estimate; "
            "zero gives independent group fits and one gives complete pooling."
        ),
    )
    parser.add_argument(
        "--benchmark-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_seasonal_naive_standardized/"
            "rolling-origin/metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--benchmark-forecasts-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_seasonal_naive_standardized/"
            "rolling-origin/benchmark_forecasts_long.csv"
        ),
    )
    parser.add_argument("--dm-alpha", type=float, default=0.05)
    args = parser.parse_args()
    run_recombination(
        data_path=args.data_path,
        purpose_forecasts_path=args.purpose_forecasts_path,
        vara_results_path=args.vara_results_path,
        strategy_id=args.strategy_id,
        output_dir=args.output_dir,
        validation_start_year=args.validation_start_year,
        validation_end_year=args.validation_end_year,
        final_year=args.final_year,
        calibration_window_months=args.calibration_window_months,
        vara_weight_cap=args.vara_weight_cap,
        benchmark_metrics_path=args.benchmark_metrics_path,
        benchmark_forecasts_path=args.benchmark_forecasts_path,
        dm_alpha=args.dm_alpha,
        combination_mode=args.combination_mode,
        combination_group_col=args.combination_group_col,
        shared_lambda_cap=args.shared_lambda_cap,
        group_lambda_global_weight=args.group_lambda_global_weight,
    )


if __name__ == "__main__":
    main()
