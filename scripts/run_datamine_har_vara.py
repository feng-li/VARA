#!/usr/bin/env python3
"""Run a HAR-lag VARA correction on saved DataMine artifacts."""

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
    rolling_har_vara_corrections,
    write_json,
)
from vara.forecast_significance import diebold_mariano_test


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
    parser.add_argument("--feature-snapshot", default="snapshot_200712")
    parser.add_argument("--error-window-months", type=int, default=240)
    parser.add_argument("--refit-frequency-months", type=int, default=12)
    parser.add_argument("--response-clip", type=float, default=8.0)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--attention-top-k", type=int, default=500)
    parser.add_argument("--attention-cache-mode", choices=("exact", "cluster"), default="exact")
    parser.add_argument("--attention-batch-size", type=int, default=512)
    parser.add_argument("--eta", type=float, default=-1e-4)
    parser.add_argument("--lambda-window-months", type=int, default=24)
    parser.add_argument("--lambda-ridge", type=float, default=0.001)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--reuse-saved-correction", action="store_true")
    args = parser.parse_args()

    source_contract = json.loads(
        (args.source_output / "forecasts" / "locked_test_config.json").read_text(encoding="utf-8")
    )
    baseline_manifest = json.loads(
        (args.source_output / "baseline" / "baseline_manifest.json").read_text(encoding="utf-8")
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
    forecast_dir = args.output_dir / "forecasts"
    metric_dir = args.output_dir / "metrics"
    if args.reuse_saved_correction:
        correction_path = forecast_dir / "vara_correction.npy"
        if not correction_path.exists():
            raise FileNotFoundError(correction_path)
        correction = np.load(correction_path, mmap_mode="r")
        result = None
    else:
        features = np.load(
            args.source_output / "features" / args.feature_snapshot / "Z.npy",
            mmap_mode="r",
        )
        result = rolling_har_vara_corrections(
            baseline_errors=error,
            evaluation_indices=evaluation,
            feature_matrix=features,
            error_window_months=args.error_window_months,
            response_clip=args.response_clip,
            refit_frequency_months=args.refit_frequency_months,
            ridge=args.ridge,
            attention_top_k=args.attention_top_k,
            attention_cache_mode=args.attention_cache_mode,
            attention_batch_size=args.attention_batch_size,
            eta=args.eta,
        )
        correction = result.correction
    lambda_training = split.validation_indices[-int(args.lambda_window_months):]
    lambda_info = estimate_global_lambda(
        np.asarray(error)[lambda_training],
        np.asarray(correction)[lambda_training],
        ridge=args.lambda_ridge,
    )
    fixed_lambda = float(lambda_info["lambda"])
    final = np.asarray(baseline, dtype=float) + fixed_lambda * np.asarray(correction)
    rolling_lambdas, _ = rolling_group_lambdas(
        error,
        correction,
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
            np.asarray(correction)[split.validation_indices],
            np.full_like(np.asarray(correction)[split.validation_indices], fixed_lambda),
        ),
        "rolling_lambda_final_validation_block": _r2(
            np.asarray(error)[holdout],
            np.asarray(correction)[holdout],
            rolling_lambdas[holdout],
        ),
    }
    forecast_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    if result is not None:
        np.save(
            forecast_dir / "vara_correction.npy",
            np.asarray(correction, dtype=np.float32),
            allow_pickle=False,
        )
        result.fit_summary.to_csv(forecast_dir / "fit_summary.csv", index=False)
    np.save(forecast_dir / "vara_final_forecast.npy", final.astype(np.float32), allow_pickle=False)
    write_json(forecast_dir / "lambda.json", lambda_info)
    write_json(metric_dir / "validation_metrics.json", validation_report)
    contract = {
        "source_output": args.source_output.as_posix(),
        "feature_snapshot": args.feature_snapshot,
        "component_count": len(panel.component_ids),
        "har_windows": [1, 3, 12],
        "error_window_months": args.error_window_months,
        "refit_frequency_months": args.refit_frequency_months,
        "response_clip": args.response_clip,
        "ridge": args.ridge,
        "attention_top_k": args.attention_top_k,
        "attention_cache_mode": args.attention_cache_mode,
        "eta": args.eta,
        "lambda_window_months": args.lambda_window_months,
        "lambda_ridge": args.lambda_ridge,
        "lambda": fixed_lambda,
        "lambda_estimation_period": {
            "start": panel.dates[int(lambda_training[0])].date().isoformat(),
            "end": panel.dates[int(lambda_training[-1])].date().isoformat(),
            "months": int(len(lambda_training)),
        },
        "split": {
            "universe_selection_end": panel.dates[
                int(split.validation_indices[0]) - 1
            ].date().isoformat(),
            "validation_start": panel.dates[int(split.validation_indices[0])].date().isoformat(),
            "validation_end": panel.dates[int(split.validation_indices[-1])].date().isoformat(),
            "test_start": panel.dates[int(split.test_indices[0])].date().isoformat(),
            "test_end": panel.dates[int(split.test_indices[-1])].date().isoformat(),
        },
        "benchmarks": baseline_manifest.get("benchmarks", {}),
        "forecast_scope": "validation_and_test" if args.full else "validation",
        "target_month_excluded_from_all_fits": True,
        "test_configuration_locked_before_har_vara_metric_evaluation": bool(args.full),
        "post_selection_warning": (
            "Earlier candidate test results were viewed during development; use a new outer "
            "holdout for confirmatory inference."
        ),
        "reused_saved_correction": bool(args.reuse_saved_correction),
    }
    write_json(forecast_dir / "contract.json", contract)
    print(json.dumps({"contract": contract, "validation": validation_report}, indent=2))

    if args.full:
        forecasts = {"baseline": baseline, "har_vara": final}
        for model_name, filename in (
            ("ar1", "ar1_forecast.npy"),
            ("garch11", "garch11_forecast.npy"),
        ):
            path = args.source_output / "baseline" / filename
            if path.exists():
                forecasts[model_name] = np.load(path, mmap_mode="r")
        metrics, monthly, calibration = evaluate_risk_forecasts(
            target=target,
            returns=panel.returns,
            dates=panel.dates,
            forecasts=forecasts,
            validation_indices=split.validation_indices,
            test_indices=split.test_indices,
            benchmark_name="baseline",
            epsilon=float(source_contract["epsilon"]),
            variance_floor=float(source_contract["variance_floor"]),
        )
        dm_rows = []
        focal = monthly.loc[monthly["model"].eq("har_vara")]
        for comparator_name in ("baseline", "ar1", "garch11"):
            if comparator_name not in forecasts:
                continue
            comparator = monthly.loc[monthly["model"].eq(comparator_name)]
            for loss in ("squared_error", "gaussian_return_crps"):
                matched = focal[["date", loss]].merge(
                    comparator[["date", loss]],
                    on="date",
                    suffixes=("_focal", "_comparator"),
                ).dropna()
                dm_rows.append({
                    "focal_model": "har_vara",
                    "comparator_model": comparator_name,
                    "loss": loss,
                    **diebold_mariano_test(
                        matched[f"{loss}_focal"],
                        matched[f"{loss}_comparator"],
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
