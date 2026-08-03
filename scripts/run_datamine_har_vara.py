#!/usr/bin/env python3
"""Run a HAR-lag VARA correction on saved DataMine artifacts."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

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
    RollingHARVARAResult,
    estimate_har_vara_eta,
    evaluate_risk_forecasts,
    fit_robust_panel_scaler,
    load_panel,
    relative_panel_split,
    rolling_har_vara_corrections,
    rolling_refit_group_lambdas,
    write_json,
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


def _rolling_refit_blocks(
    evaluation_indices: Sequence[int],
    refit_frequency_months: int,
) -> list[np.ndarray]:
    """Partition evaluation positions exactly where the sequential loop refits."""
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    frequency = int(refit_frequency_months)
    if evaluation.size == 0 or np.any(np.diff(evaluation) <= 0):
        raise ValueError("evaluation_indices must be a nonempty increasing sequence.")
    if frequency < 1:
        raise ValueError("refit_frequency_months must be positive.")
    return [
        evaluation[start : start + frequency].copy()
        for start in range(0, len(evaluation), frequency)
    ]


def _validate_ray_resources(ray_num_cpus: int, max_cpus_per_task: int) -> None:
    """Validate the logical CPU budget used by the optional Ray backend."""
    if int(ray_num_cpus) < 1:
        raise ValueError("ray_num_cpus must be at least one.")
    if int(max_cpus_per_task) < 1:
        raise ValueError("max_cpus_per_task must be at least one.")
    if int(max_cpus_per_task) > int(ray_num_cpus):
        raise ValueError("max_cpus_per_task cannot exceed ray_num_cpus.")


def _build_shared_attention_cache_arrays(
    feature_matrix: np.ndarray,
    *,
    attention_top_k: int,
    attention_cache_mode: str,
    attention_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build one fixed-smallest attention graph for all Ray refit workers."""
    from vara.attention import build_fixed_smallest_topk_attention_cache

    Z = np.asarray(feature_matrix, dtype=float)
    if str(attention_cache_mode) != "exact":
        raise ValueError(
            "Ray HAR-VARA rolling refits require attention_cache_mode='exact'."
        )
    cache = build_fixed_smallest_topk_attention_cache(
        Z,
        top_k=int(attention_top_k),
        batch_size=int(attention_batch_size),
    )
    indices = cache.indices_pos
    scores = cache.scores_pos
    return (
        np.ascontiguousarray(np.asarray(indices), dtype=np.int32),
        np.ascontiguousarray(np.asarray(scores), dtype=np.float64),
    )


def _fit_har_vara_refit_block(
    baseline_errors: np.ndarray,
    feature_matrix: np.ndarray,
    attention_indices: np.ndarray,
    attention_scores: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    block_position: int,
    error_window_months: int,
    response_clip: float | None,
    ridge: float,
    attention_top_k: int,
    attention_batch_size: int,
    eta_bounds: tuple[float, float],
    eta_grid_size: int,
    eta_local_tolerance: float,
    zero_own_lag_coefficients: bool,
) -> dict[str, Any]:
    """Fit one independent rolling-refit block and return only its target rows."""
    from vara.attention import TopKAttentionCache, build_cached_topk_attention_csr

    try:
        errors = np.asarray(baseline_errors)
        Z = np.asarray(feature_matrix, dtype=float)
        evaluation = np.asarray(evaluation_indices, dtype=np.int64)
        if evaluation.size == 0 or np.any(np.diff(evaluation) <= 0):
            raise ValueError("Ray refit block must contain increasing target indices.")
        if evaluation[0] < 1 or evaluation[-1] >= len(errors):
            raise ValueError("Ray refit block contains an invalid target index.")
        if errors.ndim != 2 or Z.ndim != 2 or errors.shape[1] != Z.shape[0]:
            raise ValueError("baseline_errors and feature_matrix have inconsistent shapes.")

        indices = np.asarray(attention_indices, dtype=np.int32)
        scores = np.asarray(attention_scores, dtype=np.float64)
        if indices.shape != scores.shape or indices.shape[0] != errors.shape[1]:
            raise ValueError("Shared attention cache arrays have inconsistent shapes.")
        cache = TopKAttentionCache(
            indices_pos=indices,
            scores_pos=scores,
            indices_neg=indices,
            scores_neg=scores,
            mode="exact-fixed-smallest",
        )

        windows = (1, 3, 12)
        origin_index = int(evaluation[0]) - 1
        start = max(0, origin_index - int(error_window_months) + 1)
        history = np.asarray(errors[start : origin_index + 1], dtype=float)
        history = history[np.isfinite(history).any(axis=1)]
        if history.shape[0] < 24:
            raise ValueError(
                f"HAR-VARA refit at index {origin_index} has only "
                f"{history.shape[0]} rows."
            )
        scaler = fit_robust_panel_scaler(history, clip=response_clip)
        response = scaler.transform(history)
        imputed_cells = int((~np.isfinite(response)).sum())
        response = np.where(np.isfinite(response), response, 0.0)
        eta_hat, eta_diagnostics, profile_fit = estimate_har_vara_eta(
            response=response,
            feature_matrix=Z,
            attention_cache=cache,
            attention_top_k=int(attention_top_k),
            attention_batch_size=int(attention_batch_size),
            ridge=float(ridge),
            eta_bounds=(float(eta_bounds[0]), float(eta_bounds[1])),
            eta_grid_size=int(eta_grid_size),
            eta_local_tolerance=float(eta_local_tolerance),
            windows=windows,
            zero_own_lag_coefficients=bool(zero_own_lag_coefficients),
        )
        coefficients = np.asarray(profile_fit["coefficients"], dtype=float)
        regressor_names = list(profile_fit["regressor_names"])
        row: dict[str, Any] = {
            "refit_index": origin_index,
            "error_window_start_index": start,
            "error_rows": int(history.shape[0]),
            "imputed_error_cells": imputed_cells,
            "imputed_error_fraction": imputed_cells / response.size,
            "response_scale_floor": scaler.scale_floor,
            "residual_variance_mean": float(
                np.mean(profile_fit["residual_variance"])
            ),
            "attention_top_k": int(attention_top_k),
            "own_lag_coefficients_fixed_zero": bool(zero_own_lag_coefficients),
        }
        row.update(eta_diagnostics)
        if bool(zero_own_lag_coefficients):
            for window in windows:
                row[f"own_mean_{window}_coefficient_abs_mean"] = 0.0
                row[f"own_mean_{window}_coefficient_abs_max"] = 0.0
        for index, name in enumerate(regressor_names):
            row[f"{name}_coefficient_abs_mean"] = float(
                np.mean(np.abs(coefficients[:, index]))
            )
            row[f"{name}_coefficient_abs_max"] = float(
                np.max(np.abs(coefficients[:, index]))
            )

        attention_operator = build_cached_topk_attention_csr(cache, float(eta_hat))
        calculation_dtype = np.result_type(scores.dtype, np.float32)
        standardized = np.empty((len(evaluation), errors.shape[1]), dtype=float)
        correction = np.empty_like(standardized)
        for target_position, target_index in enumerate(evaluation):
            target_origin = int(target_index) - 1
            recent = np.asarray(
                errors[max(0, target_origin - 11) : target_origin + 1],
                dtype=float,
            )
            recent = scaler.transform(recent)
            recent = np.where(np.isfinite(recent), recent, 0.0)
            if len(recent) < 12:
                raise ValueError(
                    "HAR-VARA prediction requires at least 12 lagged error months."
                )
            own_regressors = [
                np.mean(recent[-window:], axis=0) for window in windows
            ]
            own_matrix = np.stack(own_regressors, axis=0)
            peer_matrix = np.asarray(
                attention_operator
                @ own_matrix.astype(calculation_dtype, copy=False).T
            ).T
            peer_regressors = [
                peer_matrix[index] for index in range(len(windows))
            ]
            prediction_regressors = (
                peer_regressors
                if bool(zero_own_lag_coefficients)
                else own_regressors + peer_regressors
            )
            zhat = sum(
                prediction_regressors[index] * coefficients[:, index]
                for index in range(len(prediction_regressors))
            )
            standardized[target_position] = zhat
            correction[target_position] = scaler.inverse(zhat)

        return {
            "block_position": int(block_position),
            "evaluation_indices": evaluation,
            "standardized_forecast": np.ascontiguousarray(standardized),
            "correction": np.ascontiguousarray(correction),
            "refit_index": np.full(len(evaluation), origin_index, dtype=np.int64),
            "fit_row": row,
        }
    except Exception as exc:
        raise RuntimeError(
            f"HAR-VARA Ray refit block {int(block_position)} failed."
        ) from exc


def _assemble_har_vara_refit_blocks(
    block_results: Sequence[dict[str, Any]],
    *,
    panel_shape: tuple[int, int],
    evaluation_indices: Sequence[int],
) -> RollingHARVARAResult:
    """Merge asynchronously completed Ray blocks into chronological panels."""
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    ordered = sorted(block_results, key=lambda item: int(item["block_position"]))
    standardized = np.full(panel_shape, np.nan, dtype=float)
    correction = np.full(panel_shape, np.nan, dtype=float)
    refit_index = np.full(panel_shape[0], -1, dtype=np.int64)
    fit_rows: list[dict[str, Any]] = []
    observed: list[np.ndarray] = []
    for expected_position, item in enumerate(ordered):
        if int(item["block_position"]) != expected_position:
            raise ValueError("Ray refit block positions are incomplete or duplicated.")
        indices = np.asarray(item["evaluation_indices"], dtype=np.int64)
        expected_shape = (len(indices), panel_shape[1])
        block_standardized = np.asarray(item["standardized_forecast"], dtype=float)
        block_correction = np.asarray(item["correction"], dtype=float)
        block_refit = np.asarray(item["refit_index"], dtype=np.int64)
        if block_standardized.shape != expected_shape or block_correction.shape != expected_shape:
            raise ValueError("Ray refit block forecast arrays have the wrong shape.")
        if block_refit.shape != (len(indices),):
            raise ValueError("Ray refit block assignments have the wrong shape.")
        standardized[indices] = block_standardized
        correction[indices] = block_correction
        refit_index[indices] = block_refit
        fit_rows.append(dict(item["fit_row"]))
        observed.append(indices)
    merged_indices = np.concatenate(observed) if observed else np.empty(0, dtype=np.int64)
    if not np.array_equal(merged_indices, evaluation):
        raise ValueError("Ray refit blocks do not exactly cover evaluation_indices.")
    fit_summary = pd.DataFrame(fit_rows).sort_values("refit_index").reset_index(drop=True)
    return RollingHARVARAResult(
        standardized_forecast=standardized,
        correction=correction,
        refit_index=refit_index,
        fit_summary=fit_summary,
    )


def rolling_har_vara_corrections_ray(
    *,
    baseline_errors: np.ndarray,
    evaluation_indices: Sequence[int],
    feature_matrix: np.ndarray,
    error_window_months: int,
    response_clip: float | None,
    refit_frequency_months: int,
    ridge: float,
    attention_top_k: int,
    attention_cache_mode: str,
    attention_batch_size: int,
    eta_bounds: tuple[float, float],
    eta_grid_size: int,
    eta_local_tolerance: float,
    zero_own_lag_coefficients: bool,
    ray_num_cpus: int,
    max_cpus_per_task: int,
    ray_debug: bool = False,
    progress_callback: Any | None = None,
    ray_module: Any | None = None,
) -> tuple[RollingHARVARAResult, dict[str, Any]]:
    """Run independent rolling-refit blocks as Ray tasks."""
    _validate_ray_resources(ray_num_cpus, max_cpus_per_task)
    if int(ray_num_cpus) <= 1:
        raise ValueError("Ray rolling refits require ray_num_cpus greater than one.")
    errors = np.asarray(baseline_errors)
    Z = np.asarray(feature_matrix, dtype=float)
    if errors.ndim != 2 or Z.ndim != 2 or errors.shape[1] != Z.shape[0]:
        raise ValueError("baseline_errors and feature_matrix have inconsistent shapes.")
    blocks = _rolling_refit_blocks(evaluation_indices, refit_frequency_months)
    ray = ray_module
    if ray is None:
        import ray as ray_runtime

        ray = ray_runtime

    started_here = False
    pending: list[Any] = []
    started_at = time.monotonic()
    worker_runtime_env = {
        "env_vars": {
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "OMP_NUM_THREADS": str(int(max_cpus_per_task)),
            "OPENBLAS_NUM_THREADS": str(int(max_cpus_per_task)),
            "MKL_NUM_THREADS": str(int(max_cpus_per_task)),
            "NUMEXPR_NUM_THREADS": str(int(max_cpus_per_task)),
        }
    }
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=int(ray_num_cpus),
            _node_ip_address="127.0.0.1",
            runtime_env=worker_runtime_env,
        )
        started_here = True
    try:
        available = ray.available_resources()
        if float(available.get("CPU", 0.0)) < float(max_cpus_per_task):
            raise RuntimeError(
                f"Ray has no available CPUs after initialization: {available}. "
                "Stop stale Ray workers or reduce other Ray jobs."
            )
        print(
            "Ray HAR-VARA rolling refits: "
            f"blocks={len(blocks)} ray_num_cpus={int(ray_num_cpus)} "
            f"max_cpus_per_task={int(max_cpus_per_task)}",
            flush=True,
        )
        attention_indices, attention_scores = _build_shared_attention_cache_arrays(
            Z,
            attention_top_k=int(attention_top_k),
            attention_cache_mode=str(attention_cache_mode),
            attention_batch_size=int(attention_batch_size),
        )
        errors_ref = ray.put(errors)
        features_ref = ray.put(Z)
        attention_indices_ref = ray.put(attention_indices)
        attention_scores_ref = ray.put(attention_scores)
        remote_refit = ray.remote(
            num_cpus=int(max_cpus_per_task),
            runtime_env=worker_runtime_env,
        )(_fit_har_vara_refit_block)
        futures: list[Any] = []
        submitted_at: dict[Any, float] = {}
        max_inflight = max(1, int(ray_num_cpus) // int(max_cpus_per_task))
        next_block_position = 0

        def submit_block(block_position: int) -> None:
            block = blocks[block_position]
            future = remote_refit.remote(
                errors_ref,
                features_ref,
                attention_indices_ref,
                attention_scores_ref,
                block,
                block_position=int(block_position),
                error_window_months=int(error_window_months),
                response_clip=response_clip,
                ridge=float(ridge),
                attention_top_k=int(attention_top_k),
                attention_batch_size=int(attention_batch_size),
                eta_bounds=(float(eta_bounds[0]), float(eta_bounds[1])),
                eta_grid_size=int(eta_grid_size),
                eta_local_tolerance=float(eta_local_tolerance),
                zero_own_lag_coefficients=bool(zero_own_lag_coefficients),
            )
            futures.append(future)
            pending.append(future)
            submitted_at[future] = time.monotonic()

        while next_block_position < len(blocks) and len(pending) < max_inflight:
            submit_block(next_block_position)
            next_block_position += 1
        completed: list[dict[str, Any]] = []
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=60.0)
            if not ready:
                slowest = time.monotonic() - min(
                    submitted_at[future] for future in pending
                )
                print(
                    "Ray HAR-VARA wait: "
                    f"done={len(completed)}/{len(blocks)} pending={len(pending)} "
                    f"slowest_pending_sec={slowest:.1f} "
                    f"available={ray.available_resources()}",
                    flush=True,
                )
                continue
            for future in ready:
                item = ray.get(future)
                completed.append(item)
                submitted_at.pop(future, None)
                if ray_debug:
                    print(
                        "Ray HAR-VARA collected: "
                        f"block={int(item['block_position']) + 1}/{len(blocks)} "
                        f"refit_index={int(item['fit_row']['refit_index'])}",
                        flush=True,
                    )
            while next_block_position < len(blocks) and len(pending) < max_inflight:
                submit_block(next_block_position)
                next_block_position += 1
        result = _assemble_har_vara_refit_blocks(
            completed,
            panel_shape=errors.shape,
            evaluation_indices=evaluation_indices,
        )
        if progress_callback is not None:
            for row in result.fit_summary.to_dict(orient="records"):
                progress_callback(dict(row))
        execution = {
            "backend": "ray",
            "ray_used": True,
            "ray_version": str(getattr(ray, "__version__", "unknown")),
            "ray_num_cpus": int(ray_num_cpus),
            "max_cpus_per_task": int(max_cpus_per_task),
            "task_unit": "rolling_refit_block",
            "task_count": int(len(blocks)),
            "elapsed_seconds": float(time.monotonic() - started_at),
        }
        return result, execution
    except Exception:
        cancel = getattr(ray, "cancel", None)
        if cancel is not None:
            for future in pending:
                try:
                    cancel(future, force=True)
                except Exception:
                    pass
        raise
    finally:
        if started_here:
            ray.shutdown()


def combine_har_vara_forecasts(
    *,
    har_forecast: np.ndarray,
    correction: np.ndarray,
    selected_lambda: float | np.ndarray,
) -> dict[str, np.ndarray]:
    """Return unit-weight and rolling-LSE HAR--VARA forecasts."""
    base = np.asarray(har_forecast, dtype=float)
    adjustment = np.asarray(correction, dtype=float)
    if base.shape != adjustment.shape:
        raise ValueError("har_forecast and correction must have the same shape.")
    lambda_values = np.asarray(selected_lambda, dtype=float)
    if lambda_values.ndim > 0 and lambda_values.shape != base.shape:
        raise ValueError("selected_lambda must be scalar or match the forecast shape.")
    return {
        HAR_VARA_UNIT_MODEL: base + adjustment,
        HAR_VARA_LAMBDA_MODEL: base + lambda_values * adjustment,
    }


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
        result = None
    else:
        features = np.load(
            args.source_output / "features" / args.feature_snapshot / "Z.npy",
            mmap_mode="r",
        )
        if features.ndim != 2 or features.shape[0] != len(panel.component_ids):
            raise ValueError("Feature rows do not match the processed panel components.")
        correction_kwargs = {
            "baseline_errors": error,
            "evaluation_indices": evaluation,
            "feature_matrix": features,
            "error_window_months": args.error_window_months,
            "response_clip": args.response_clip,
            "refit_frequency_months": args.refit_frequency_months,
            "ridge": args.ridge,
            "attention_top_k": args.attention_top_k,
            "attention_cache_mode": args.attention_cache_mode,
            "attention_batch_size": args.attention_batch_size,
            "eta_bounds": eta_specification.effective_bounds,
            "eta_grid_size": args.eta_grid_size,
            "eta_local_tolerance": eta_specification.effective_local_tolerance,
            "zero_own_lag_coefficients": args.zero_own_lag_coefficients,
            "progress_callback": lambda row: _print_eta_refit(
                row,
                eta_specification,
            ),
        }
        if int(args.ray_num_cpus) > 1:
            result, rolling_refit_execution = rolling_har_vara_corrections_ray(
                **correction_kwargs,
                ray_num_cpus=args.ray_num_cpus,
                max_cpus_per_task=args.max_cpus_per_task,
                ray_debug=args.ray_debug,
            )
        else:
            sequential_started_at = time.monotonic()
            result = rolling_har_vara_corrections(**correction_kwargs)
            rolling_refit_execution = {
                "backend": "sequential",
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
                "elapsed_seconds": float(time.monotonic() - sequential_started_at),
            }
        correction = np.asarray(result.correction, dtype=np.float32)
        refit_index = result.refit_index
        fit_summary = eta_specification.convert_core_fit_summary(result.fit_summary)
    rolling_lambdas, lambda_table = rolling_refit_group_lambdas(
        error,
        correction,
        panel.metadata,
        evaluation,
        refit_index,
        window_months=args.lambda_window_months,
        min_history_months=args.lambda_window_months,
        ridge=args.lambda_ridge,
    )
    global_lambda_table = lambda_table.loc[
        lambda_table["group"].eq("__global__")
    ].copy()
    lambda_refits = lambda_table.loc[
        :, ["refit_index", "lambda_history_sufficient"]
    ].drop_duplicates()
    fitted_lambda_refits = int(lambda_refits["lambda_history_sufficient"].sum())
    lambda_weights = np.nan_to_num(rolling_lambdas, nan=0.0)
    vara_forecasts = combine_har_vara_forecasts(
        har_forecast=baseline,
        correction=correction,
        selected_lambda=lambda_weights,
    )
    unit = vara_forecasts[HAR_VARA_UNIT_MODEL]
    final = vara_forecasts[HAR_VARA_LAMBDA_MODEL]
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
    if result is not None:
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
