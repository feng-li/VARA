#!/usr/bin/env python3
"""Run a multi-head hard-group VARA correction on saved DataMine artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vara.datamine import (
    estimate_global_lambda,
    evaluate_risk_forecasts,
    load_panel,
    relative_panel_split,
    rolling_group_lambdas,
    rolling_group_vara_corrections,
    write_json,
)
from vara.forecast_significance import diebold_mariano_test


def _parse_columns(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def _r2(error: np.ndarray, correction: np.ndarray, lambdas: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(error) & np.isfinite(correction) & np.isfinite(lambdas)
    baseline_sse = float(np.sum(np.square(error[valid])))
    final_sse = float(np.sum(np.square(error[valid] - correction[valid] * lambdas[valid])))
    return {
        "cells": int(valid.sum()),
        "baseline_rmse": float(np.sqrt(baseline_sse / valid.sum())),
        "vara_rmse": float(np.sqrt(final_sse / valid.sum())),
        "oos_r2_vs_baseline": 1.0 - final_sse / baseline_sse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_output", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--group-cols", type=_parse_columns, default=("numerator", "denominator"))
    parser.add_argument("--include-global-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--error-window-months", type=int, default=240)
    parser.add_argument("--refit-frequency-months", type=int, default=12)
    parser.add_argument("--response-clip", type=float, default=8.0)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--lambda-window-months", type=int, default=24)
    parser.add_argument("--lambda-ridge", type=float, default=0.001)
    parser.add_argument("--full", action="store_true")
    args = parser.parse_args()

    source_contract = json.loads(
        (args.source_output / "forecasts" / "locked_test_config.json").read_text(encoding="utf-8")
    )
    panel = load_panel(args.source_output / "processed")
    split = relative_panel_split(
        panel.dates,
        validation_months=int(source_contract["validation_months"]),
        test_months=int(source_contract["test_months"]),
    )
    target = np.load(args.source_output / "baseline" / "target.npy", mmap_mode="r")
    baseline = np.load(args.source_output / "baseline" / "baseline_forecast.npy", mmap_mode="r")
    error = np.load(args.source_output / "baseline" / "baseline_error.npy", mmap_mode="r")
    evaluation = (
        np.concatenate((split.validation_indices, split.test_indices))
        if args.full
        else split.validation_indices
    )
    result = rolling_group_vara_corrections(
        baseline_errors=error,
        evaluation_indices=evaluation,
        metadata=panel.metadata,
        group_cols=args.group_cols,
        include_global_head=args.include_global_head,
        error_window_months=args.error_window_months,
        response_clip=args.response_clip,
        refit_frequency_months=args.refit_frequency_months,
        ridge=args.ridge,
    )
    lambda_training = split.validation_indices[-int(args.lambda_window_months):]
    lambda_info = estimate_global_lambda(
        np.asarray(error)[lambda_training],
        result.correction[lambda_training],
        ridge=args.lambda_ridge,
    )
    fixed_lambda = float(lambda_info["lambda"])
    final = np.asarray(baseline, dtype=float) + fixed_lambda * result.correction
    rolling_lambdas, _ = rolling_group_lambdas(
        error,
        result.correction,
        panel.metadata,
        split.validation_indices,
        window_months=args.lambda_window_months,
        min_history_months=args.lambda_window_months,
        ridge=args.lambda_ridge,
    )
    holdout = split.validation_indices[-int(args.lambda_window_months):]
    validation_report = {
        "fixed_lambda_full_validation": _r2(
            np.asarray(error)[split.validation_indices],
            result.correction[split.validation_indices],
            np.full_like(result.correction[split.validation_indices], fixed_lambda),
        ),
        "rolling_lambda_final_validation_block": _r2(
            np.asarray(error)[holdout],
            result.correction[holdout],
            rolling_lambdas[holdout],
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    forecast_dir = args.output_dir / "forecasts"
    metric_dir = args.output_dir / "metrics"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    np.save(forecast_dir / "vara_correction.npy", result.correction.astype(np.float32), allow_pickle=False)
    np.save(forecast_dir / "vara_final_forecast.npy", final.astype(np.float32), allow_pickle=False)
    result.fit_summary.to_csv(forecast_dir / "fit_summary.csv", index=False)
    write_json(forecast_dir / "lambda.json", lambda_info)
    write_json(metric_dir / "validation_metrics.json", validation_report)
    contract = {
        "source_output": args.source_output.as_posix(),
        "group_cols": list(args.group_cols),
        "include_global_head": args.include_global_head,
        "error_window_months": args.error_window_months,
        "refit_frequency_months": args.refit_frequency_months,
        "response_clip": args.response_clip,
        "ridge": args.ridge,
        "lambda_window_months": args.lambda_window_months,
        "lambda_ridge": args.lambda_ridge,
        "lambda": fixed_lambda,
        "forecast_scope": "validation_and_test" if args.full else "validation",
        "target_month_excluded_from_all_fits": True,
    }
    write_json(forecast_dir / "contract.json", contract)
    print(json.dumps({"contract": contract, "validation": validation_report}, indent=2))

    if args.full:
        metrics, monthly, calibration = evaluate_risk_forecasts(
            target=target,
            returns=panel.returns,
            dates=panel.dates,
            forecasts={"baseline": baseline, "group_vara": final},
            validation_indices=split.validation_indices,
            test_indices=split.test_indices,
            benchmark_name="baseline",
            epsilon=float(source_contract["epsilon"]),
            variance_floor=float(source_contract["variance_floor"]),
        )
        focal = monthly.loc[monthly["model"].eq("group_vara")]
        comparator = monthly.loc[monthly["model"].eq("baseline")]
        dm_rows = []
        for loss in ("squared_error", "qlike"):
            matched = focal[["date", loss]].merge(
                comparator[["date", loss]], on="date", suffixes=("_focal", "_baseline")
            ).dropna()
            dm_rows.append({
                "loss": loss,
                **diebold_mariano_test(
                    matched[f"{loss}_focal"],
                    matched[f"{loss}_baseline"],
                    forecast_horizon=1,
                ),
            })
        metrics.to_csv(metric_dir / "metrics.csv", index=False)
        monthly.to_csv(metric_dir / "monthly_losses.csv", index=False)
        pd.DataFrame(dm_rows).to_csv(metric_dir / "diebold_mariano_raw.csv", index=False)
        write_json(metric_dir / "variance_calibration.json", calibration)
        print(metrics.to_string(index=False))
        print(pd.DataFrame(dm_rows).to_string(index=False))


if __name__ == "__main__":
    main()
