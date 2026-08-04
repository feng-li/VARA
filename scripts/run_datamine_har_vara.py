#!/usr/bin/env python3
"""Run a HAR-lag VARA correction on saved DataMine artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd

if __package__:
    from .datamine_paper_eta import (
        APPLICATION_CONTRACT_SCHEMA,
        DEFAULT_PAPER_ETA_BOUNDS,
        DEFAULT_PAPER_ETA_GRID_SIZE,
        DEFAULT_PAPER_ETA_LOCAL_TOLERANCE,
        PaperEtaSpecification,
        attention_support_name,
        file_fingerprints,
    )
else:
    from datamine_paper_eta import (  # type: ignore[no-redef]
        APPLICATION_CONTRACT_SCHEMA,
        DEFAULT_PAPER_ETA_BOUNDS,
        DEFAULT_PAPER_ETA_GRID_SIZE,
        DEFAULT_PAPER_ETA_LOCAL_TOLERANCE,
        PaperEtaSpecification,
        attention_support_name,
        file_fingerprints,
    )

from vara.datamine import (
    evaluate_risk_forecasts,
    load_panel,
    relative_panel_split,
    rolling_har_vara_corrections,
    write_json,
)
from vara.datamine_vara_runner import (
    RESPONSE_STANDARDIZATION_CHOICES,
    _rolling_refit_blocks,
    _validate_ray_resources,
    combine_base_vara_forecasts,
    combine_rolling_vara_family,
    fit_rolling_vara_family,
    rolling_har_vara_corrections_ray,
)
from vara.forecast_significance import diebold_mariano_test


HAR_VARA_LAMBDA_MODEL = "har_vara"
HAR_VARA_UNIT_MODEL = "har_vara_unit"
HAR_VARA_NO_LAG_LAMBDA_MODEL = "har_vara_no_lag"
HAR_VARA_NO_LAG_UNIT_MODEL = "har_vara_no_lag_unit"


def _print_eta_refit(
    row: dict[str, object],
    eta_specification: PaperEtaSpecification,
) -> None:
    """Emit one progress line distinguishing paper eta from the core slope."""
    effective_hat = float(row["eta_hat"])
    effective_grid_best = float(row["eta_grid_best"])
    print(
        "eta refit"
        f" index={int(row['refit_index'])}"
        f" paper_eta_hat={eta_specification.paper_eta(effective_hat):.6f}"
        f" effective_slope={effective_hat:.6f}"
        f" paper_grid_best={eta_specification.paper_eta(effective_grid_best):.6f}"
        f" boundary={bool(row['eta_on_boundary'])}",
        flush=True,
    )


def combine_har_vara_forecasts(
    *,
    har_forecast: np.ndarray,
    correction: np.ndarray,
    selected_lambda: float | np.ndarray,
) -> dict[str, np.ndarray]:
    """Compatibility wrapper around the shared VARA forecast combiner."""
    unit, weighted = combine_base_vara_forecasts(
        base_forecast=har_forecast,
        correction=correction,
        lambda_weights=selected_lambda,
    )
    return {HAR_VARA_UNIT_MODEL: unit, HAR_VARA_LAMBDA_MODEL: weighted}


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
    parser.add_argument(
        "--response-standardization",
        choices=RESPONSE_STANDARDIZATION_CHOICES,
        default="median_mad",
        help=(
            "Componentwise rolling VARA-response scaler: median/MAD or "
            "training-window mean/population-standard-deviation z-scores."
        ),
    )
    parser.add_argument("--response-clip", type=float, default=8.0)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--attention-top-k", type=int, default=500)
    parser.add_argument("--attention-cache-mode", choices=("exact", "cluster"), default="exact")
    parser.add_argument("--attention-batch-size", type=int, default=512)
    parser.add_argument(
        "--eta-lower",
        type=float,
        default=DEFAULT_PAPER_ETA_BOUNDS[0],
        help="Lower bound for paper eta in exp{eta * log(p) * score}.",
    )
    parser.add_argument(
        "--eta-upper",
        type=float,
        default=DEFAULT_PAPER_ETA_BOUNDS[1],
        help="Upper bound for paper eta in exp{eta * log(p) * score}.",
    )
    parser.add_argument(
        "--eta-grid-size",
        type=int,
        default=DEFAULT_PAPER_ETA_GRID_SIZE,
    )
    parser.add_argument(
        "--eta-local-tolerance",
        type=float,
        default=DEFAULT_PAPER_ETA_LOCAL_TOLERANCE,
        help="Local-refinement tolerance on the paper-eta scale.",
    )
    parser.add_argument(
        "--zero-own-lag-coefficients",
        action="store_true",
        help="Fix own-error HAR coefficients at zero and fit only peer-error terms.",
    )
    parser.add_argument("--lambda-window-months", type=int, default=24)
    parser.add_argument("--lambda-ridge", type=float, default=0.001)
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=1,
        help="Total Ray CPU budget; values greater than one parallelize refit blocks.",
    )
    parser.add_argument(
        "--max-cpus-per-task",
        type=int,
        default=1,
        help="Logical Ray CPUs reserved for each rolling-refit block.",
    )
    parser.add_argument(
        "--ray-debug",
        action="store_true",
        help="Print each Ray refit block as it is collected.",
    )
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--reuse-saved-correction", action="store_true")
    args = parser.parse_args()
    _validate_ray_resources(args.ray_num_cpus, args.max_cpus_per_task)

    source_contract = json.loads(
        (args.source_output / "forecasts" / "locked_test_config.json").read_text(encoding="utf-8")
    )
    baseline_manifest = json.loads(
        (args.source_output / "baseline" / "baseline_manifest.json").read_text(encoding="utf-8")
    )
    source_output = args.source_output.resolve().as_posix()
    correction_input_fingerprints = file_fingerprints({
        "baseline_error": args.source_output / "baseline" / "baseline_error.npy",
        "component_order": args.source_output / "processed" / "component_order.csv",
        "dates": args.source_output / "processed" / "dates.npy",
        "feature_matrix": (
            args.source_output / "features" / args.feature_snapshot / "Z.npy"
        ),
        "locked_test_config": (
            args.source_output / "forecasts" / "locked_test_config.json"
        ),
    })
    panel = load_panel(args.source_output / "processed")
    eta_specification = PaperEtaSpecification(
        component_count=len(panel.component_ids),
        lower=args.eta_lower,
        upper=args.eta_upper,
        grid_size=args.eta_grid_size,
        local_tolerance=args.eta_local_tolerance,
    )
    eta_estimation = eta_specification.contract(
        attention_support=attention_support_name(args.attention_cache_mode)
    )
    split = relative_panel_split(
        panel.dates,
        validation_months=int(source_contract["validation_months"]),
        test_months=int(source_contract["test_months"]),
    )
    split_contract = {
        "universe_selection_end": panel.dates[
            int(split.validation_indices[0]) - 1
        ].date().isoformat(),
        "validation_start": panel.dates[int(split.validation_indices[0])].date().isoformat(),
        "validation_end": panel.dates[int(split.validation_indices[-1])].date().isoformat(),
        "test_start": panel.dates[int(split.test_indices[0])].date().isoformat(),
        "test_end": panel.dates[int(split.test_indices[-1])].date().isoformat(),
    }
    target = np.load(args.source_output / "baseline" / "target.npy", mmap_mode="r")
    baseline = np.load(args.source_output / "baseline" / "baseline_forecast.npy", mmap_mode="r")
    error = np.load(args.source_output / "baseline" / "baseline_error.npy", mmap_mode="r")
    expected_panel_shape = (len(panel.dates), len(panel.component_ids))
    if target.shape != expected_panel_shape or baseline.shape != expected_panel_shape:
        raise ValueError("Target and HAR forecast arrays do not match the processed panel.")
    if error.shape != expected_panel_shape:
        raise ValueError("HAR error array does not match the processed panel.")
    evaluation = (
        np.concatenate((split.validation_indices, split.test_indices))
        if args.full
        else split.validation_indices
    )
    forecast_dir = args.output_dir / "forecasts"
    metric_dir = args.output_dir / "metrics"
    requested_forecast_scope = "validation_and_test" if args.full else "validation"
    rolling_refit_execution: dict[str, Any]
    if args.reuse_saved_correction:
        correction_path = forecast_dir / "vara_correction.npy"
        if not correction_path.exists():
            raise FileNotFoundError(correction_path)
        contract_path = forecast_dir / "contract.json"
        if not contract_path.exists():
            raise FileNotFoundError(contract_path)
        saved_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if saved_contract.get("eta_estimation") != eta_estimation:
            raise ValueError("Saved correction does not match the requested eta estimator.")
        for field, expected in (
            ("application_contract_schema", APPLICATION_CONTRACT_SCHEMA),
            ("source_output", source_output),
            ("correction_input_fingerprints", correction_input_fingerprints),
            ("feature_snapshot", str(args.feature_snapshot)),
            ("component_count", len(panel.component_ids)),
            ("error_window_months", int(args.error_window_months)),
            ("refit_frequency_months", int(args.refit_frequency_months)),
            ("response_standardization", str(args.response_standardization)),
            ("response_clip", float(args.response_clip)),
            ("ridge", float(args.ridge)),
            ("attention_top_k", int(args.attention_top_k)),
            ("attention_cache_mode", str(args.attention_cache_mode)),
            ("attention_batch_size", int(args.attention_batch_size)),
            ("forecast_scope", requested_forecast_scope),
            ("split", split_contract),
        ):
            if saved_contract.get(field) != expected:
                raise ValueError(
                    f"Saved correction has {field}={saved_contract.get(field)!r}; "
                    f"requested {expected!r}."
                )
        if bool(saved_contract.get("zero_own_lag_coefficients", False)) != bool(
            args.zero_own_lag_coefficients
        ):
            raise ValueError(
                "Saved correction and --zero-own-lag-coefficients specification differ."
            )
        correction = np.load(correction_path, mmap_mode="r")
        if correction.shape != error.shape:
            raise ValueError("Saved correction and baseline-error panels have different shapes.")
        fit_summary_path = forecast_dir / "fit_summary.csv"
        if not fit_summary_path.exists():
            raise FileNotFoundError(fit_summary_path)
        fit_summary = pd.read_csv(fit_summary_path)
        eta_specification.validate_persisted_fit_summary(fit_summary)
        refit_path = forecast_dir / "vara_refit_index.npy"
        if not refit_path.exists():
            raise FileNotFoundError(refit_path)
        refit_index = np.load(refit_path, mmap_mode="r")
        if refit_index.shape != (error.shape[0],):
            raise ValueError("Saved VARA refit indices have the wrong shape.")
        eta_specification.validate_refit_coverage(fit_summary, refit_index, evaluation)
        rolling_refit_execution = dict(
            saved_contract.get(
                "rolling_refit_execution",
                {
                    "backend": "legacy_unspecified",
                    "ray_used": False,
                    "task_unit": "rolling_refit_block",
                    "task_count": int(
                        len(
                            _rolling_refit_blocks(
                                evaluation,
                                args.refit_frequency_months,
                            )
                        )
                    ),
                },
            )
        )
        vara_family = combine_rolling_vara_family(
            base_forecast=baseline,
            base_error=error,
            correction=correction,
            refit_index=refit_index,
            core_fit_summary=fit_summary,
            correction_execution=rolling_refit_execution,
            metadata=panel.metadata,
            evaluation_indices=evaluation,
            lambda_window_months=args.lambda_window_months,
            lambda_ridge=args.lambda_ridge,
        )
    else:
        features = np.load(
            args.source_output / "features" / args.feature_snapshot / "Z.npy",
            mmap_mode="r",
        )
        if features.ndim != 2 or features.shape[0] != len(panel.component_ids):
            raise ValueError("Feature rows do not match the processed panel components.")
        vara_family = fit_rolling_vara_family(
            base_forecast=baseline,
            base_error=error,
            metadata=panel.metadata,
            evaluation_indices=evaluation,
            feature_matrix=features,
            error_window_months=args.error_window_months,
            response_standardization=args.response_standardization,
            response_clip=args.response_clip,
            refit_frequency_months=args.refit_frequency_months,
            ridge=args.ridge,
            attention_top_k=args.attention_top_k,
            attention_cache_mode=args.attention_cache_mode,
            attention_batch_size=args.attention_batch_size,
            eta_bounds=eta_specification.effective_bounds,
            eta_grid_size=args.eta_grid_size,
            eta_local_tolerance=eta_specification.effective_local_tolerance,
            lambda_window_months=args.lambda_window_months,
            lambda_ridge=args.lambda_ridge,
            zero_own_lag_coefficients=args.zero_own_lag_coefficients,
            ray_num_cpus=args.ray_num_cpus,
            max_cpus_per_task=args.max_cpus_per_task,
            ray_debug=args.ray_debug,
            progress_callback=lambda row: _print_eta_refit(row, eta_specification),
        )
        fit_summary = eta_specification.convert_core_fit_summary(
            vara_family.core_fit_summary
        )
    correction = vara_family.correction
    refit_index = vara_family.refit_index
    rolling_refit_execution = vara_family.correction_execution
    rolling_lambdas = vara_family.rolling_lambdas
    lambda_table = vara_family.lambda_refit_fits
    lambda_weights = vara_family.lambda_weights
    unit = vara_family.unit_forecast
    final = vara_family.lambda_forecast
    global_lambda_table = lambda_table.loc[
        lambda_table["group"].eq("__global__")
    ].copy()
    lambda_refits = lambda_table.loc[
        :, ["refit_index", "lambda_history_sufficient"]
    ].drop_duplicates()
    fitted_lambda_refits = int(lambda_refits["lambda_history_sufficient"].sum())
    holdout = split.validation_indices[-int(args.lambda_window_months):]

    def lambda_period_summary(indices: np.ndarray) -> dict[str, float | int]:
        values = lambda_weights[np.asarray(indices, dtype=np.int64), 0]
        return {
            "origins": int(len(values)),
            "minimum": float(np.min(values)),
            "median": float(np.median(values)),
            "mean": float(np.mean(values)),
            "maximum": float(np.max(values)),
        }

    lambda_info = {
        "method": "rolling_refit_block_pooled_bounded_lse",
        "window_months": int(args.lambda_window_months),
        "minimum_history_months": int(args.lambda_window_months),
        "refit_frequency_months": int(args.refit_frequency_months),
        "ridge": float(args.lambda_ridge),
        "bounds": [0.0, 1.0],
        "target_month_excluded": True,
        "held_fixed_within_refit_block": True,
        "calibration_source": "preceding_genuine_oos_error_correction_pairs",
        "insufficient_history_lambda": 0.0,
        "refits": int(len(lambda_refits)),
        "fitted_refits": fitted_lambda_refits,
        "warmup_refits": int(len(lambda_refits) - fitted_lambda_refits),
        "validation": lambda_period_summary(split.validation_indices),
        "test": lambda_period_summary(split.test_indices) if args.full else None,
    }
    validation_report = {
        "unit_full_validation": _r2(
            np.asarray(error)[split.validation_indices],
            np.asarray(correction)[split.validation_indices],
            np.ones_like(np.asarray(correction)[split.validation_indices]),
        ),
        "rolling_lambda_full_validation": _r2(
            np.asarray(error)[split.validation_indices],
            np.asarray(correction)[split.validation_indices],
            rolling_lambdas[split.validation_indices],
        ),
        "rolling_lambda_final_validation_block": _r2(
            np.asarray(error)[holdout],
            np.asarray(correction)[holdout],
            rolling_lambdas[holdout],
        ),
    }
    forecast_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    if not args.reuse_saved_correction:
        np.save(
            forecast_dir / "vara_correction.npy",
            np.asarray(correction, dtype=np.float32),
            allow_pickle=False,
        )
        np.save(
            forecast_dir / "vara_refit_index.npy",
            np.asarray(refit_index, dtype=np.int64),
            allow_pickle=False,
        )
    lambda_fit_summary = global_lambda_table.rename(columns={
        "lambda": "lambda_hat",
        "lambda_unbounded": "lambda_unbounded_hat",
        "ridge": "lambda_ridge",
    }).drop(columns=["group", "series", "fallback_global"], errors="ignore")
    fit_summary = fit_summary.drop(
        columns=[
            column for column in lambda_fit_summary.columns
            if column != "refit_index"
        ],
        errors="ignore",
    ).merge(
        lambda_fit_summary,
        on="refit_index",
        how="left",
        validate="one_to_one",
    )
    eta_specification.validate_refit_coverage(fit_summary, refit_index, evaluation)
    fit_summary.to_csv(forecast_dir / "fit_summary.csv", index=False)
    np.save(forecast_dir / "vara_final_forecast.npy", final.astype(np.float32), allow_pickle=False)
    np.save(forecast_dir / "vara_unit_forecast.npy", unit.astype(np.float32), allow_pickle=False)
    np.save(
        forecast_dir / "lambda_by_origin.npy",
        lambda_weights[:, 0].astype(np.float32),
        allow_pickle=False,
    )
    lambda_origins = pd.DataFrame({
        "target_index": evaluation,
        "target_date": panel.dates[evaluation],
        "refit_index": np.asarray(refit_index, dtype=np.int64)[evaluation],
        "lambda": lambda_weights[evaluation, 0],
    })
    if not lambda_table.empty:
        lambda_origin_fit = global_lambda_table.drop(
            columns=["group", "series", "fallback_global", "lambda"],
            errors="ignore",
        )
        lambda_origins = lambda_origins.merge(
            lambda_origin_fit,
            on="refit_index",
            how="left",
        )
    lambda_origins.to_csv(forecast_dir / "lambda_by_origin.csv", index=False)
    lambda_table.to_csv(forecast_dir / "lambda_refit_fits.csv", index=False)
    write_json(forecast_dir / "lambda.json", lambda_info)
    write_json(metric_dir / "validation_metrics.json", validation_report)
    contract = {
        "application_contract_schema": APPLICATION_CONTRACT_SCHEMA,
        "source_output": source_output,
        "correction_input_fingerprints": correction_input_fingerprints,
        "feature_snapshot": args.feature_snapshot,
        "component_count": len(panel.component_ids),
        "har_windows": [1, 3, 12],
        "error_window_months": args.error_window_months,
        "refit_frequency_months": args.refit_frequency_months,
        "response_standardization": args.response_standardization,
        "response_clip": args.response_clip,
        "ridge": args.ridge,
        "attention_top_k": args.attention_top_k,
        "attention_cache_mode": args.attention_cache_mode,
        "attention_batch_size": args.attention_batch_size,
        "eta_estimation": eta_estimation,
        "eta_refit_summary": eta_specification.refit_summary(fit_summary),
        "rolling_refit_execution": rolling_refit_execution,
        "zero_own_lag_coefficients": bool(args.zero_own_lag_coefficients),
        "lambda_window_months": args.lambda_window_months,
        "lambda_ridge": args.lambda_ridge,
        "lambda_estimation": lambda_info,
        "unit_weight": 1.0,
        "split": split_contract,
        "benchmarks": baseline_manifest.get("benchmarks", {}),
        "forecast_scope": requested_forecast_scope,
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
        unit_model_name = (
            HAR_VARA_NO_LAG_UNIT_MODEL
            if args.zero_own_lag_coefficients
            else HAR_VARA_UNIT_MODEL
        )
        lambda_model_name = (
            HAR_VARA_NO_LAG_LAMBDA_MODEL
            if args.zero_own_lag_coefficients
            else HAR_VARA_LAMBDA_MODEL
        )
        forecasts = {
            "baseline": baseline,
            unit_model_name: unit,
            lambda_model_name: final,
        }
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
        comparator_map = {
            unit_model_name: ("baseline", lambda_model_name, "ar1", "garch11"),
            lambda_model_name: ("baseline", unit_model_name, "ar1", "garch11"),
        }
        for focal_name, comparator_names in comparator_map.items():
            focal = monthly.loc[monthly["model"].eq(focal_name)]
            for comparator_name in comparator_names:
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
                        "focal_model": focal_name,
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
