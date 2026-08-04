"""Shared rolling VARA orchestration for DataMine forecasting runners.

This module deliberately starts after a model-specific base forecast and its
genuine one-step error panel have been constructed.  Base-model estimation,
calibration, application-specific contracts, and artifact naming remain in the
individual runners.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd

from .datamine import (
    RollingHARVARAResult,
    estimate_har_vara_eta,
    fit_panel_scaler,
    rolling_har_vara_corrections,
    rolling_refit_group_lambdas,
)

RESPONSE_STANDARDIZATION_CHOICES = ("median_mad", "training_zscore")


@dataclass
class RollingVARAFamilyResult:
    """Common outputs for a base, unit-VARA, and rolling-lambda VARA family."""

    correction: np.ndarray
    refit_index: np.ndarray
    core_fit_summary: pd.DataFrame
    rolling_lambdas: np.ndarray
    lambda_weights: np.ndarray
    lambda_refit_fits: pd.DataFrame
    unit_forecast: np.ndarray
    lambda_forecast: np.ndarray
    correction_execution: dict[str, Any]


def combine_base_vara_forecasts(
    *,
    base_forecast: np.ndarray,
    correction: np.ndarray,
    lambda_weights: float | np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unit- and lambda-weighted VARA forecasts for one base model."""
    base = np.asarray(base_forecast, dtype=float)
    adjustment = np.asarray(correction, dtype=float)
    if base.shape != adjustment.shape:
        raise ValueError("base_forecast and correction must have the same shape.")
    weights = np.asarray(lambda_weights, dtype=float)
    if weights.ndim > 0 and weights.shape != base.shape:
        raise ValueError("lambda_weights must be scalar or match the forecast shape.")
    return base + adjustment, base + weights * adjustment


def combine_rolling_vara_family(
    *,
    base_forecast: np.ndarray,
    base_error: np.ndarray,
    correction: np.ndarray,
    refit_index: np.ndarray,
    core_fit_summary: pd.DataFrame,
    correction_execution: dict[str, Any],
    metadata: pd.DataFrame,
    evaluation_indices: Sequence[int],
    lambda_window_months: int,
    lambda_ridge: float,
) -> RollingVARAFamilyResult:
    """Estimate rolling LSE weights and combine one base forecast with VARA.

    This is also the common continuation point when a runner reuses a saved
    correction instead of fitting it again.
    """
    base = np.asarray(base_forecast, dtype=float)
    error = np.asarray(base_error)
    adjustment = np.asarray(correction)
    refits = np.asarray(refit_index, dtype=np.int64)
    if base.ndim != 2 or error.ndim != 2 or base.shape != error.shape:
        raise ValueError("base_forecast and base_error must have the same 2D shape.")
    if adjustment.shape != base.shape:
        raise ValueError("correction must have the same shape as base_forecast.")
    if refits.shape != (base.shape[0],):
        raise ValueError("refit_index must have one entry per panel row.")

    rolling_lambdas, lambda_table = rolling_refit_group_lambdas(
        error,
        adjustment,
        metadata,
        evaluation_indices,
        refits,
        window_months=lambda_window_months,
        min_history_months=lambda_window_months,
        ridge=lambda_ridge,
    )
    lambda_weights = np.nan_to_num(rolling_lambdas, nan=0.0)
    unit_forecast, lambda_forecast = combine_base_vara_forecasts(
        base_forecast=base,
        correction=adjustment,
        lambda_weights=lambda_weights,
    )
    return RollingVARAFamilyResult(
        correction=adjustment,
        refit_index=refits,
        core_fit_summary=core_fit_summary,
        rolling_lambdas=rolling_lambdas,
        lambda_weights=lambda_weights,
        lambda_refit_fits=lambda_table,
        unit_forecast=unit_forecast,
        lambda_forecast=lambda_forecast,
        correction_execution=dict(correction_execution),
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
    from .attention import build_fixed_smallest_topk_attention_cache

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
    response_standardization: str,
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
    from .attention import TopKAttentionCache, build_cached_topk_attention_csr

    try:
        errors = np.asarray(baseline_errors)
        Z = np.asarray(feature_matrix, dtype=float)
        evaluation = np.asarray(evaluation_indices, dtype=np.int64)
        if evaluation.size == 0 or np.any(np.diff(evaluation) <= 0):
            raise ValueError("Ray refit block must contain increasing target indices.")
        if evaluation[0] < 1 or evaluation[-1] >= len(errors):
            raise ValueError("Ray refit block contains an invalid target index.")
        if errors.ndim != 2 or Z.ndim != 2 or errors.shape[1] != Z.shape[0]:
            raise ValueError(
                "baseline_errors and feature_matrix have inconsistent shapes."
            )

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
        scaler = fit_panel_scaler(
            history,
            method=response_standardization,
            clip=response_clip,
        )
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
        if (
            block_standardized.shape != expected_shape
            or block_correction.shape != expected_shape
        ):
            raise ValueError("Ray refit block forecast arrays have the wrong shape.")
        if block_refit.shape != (len(indices),):
            raise ValueError("Ray refit block assignments have the wrong shape.")
        standardized[indices] = block_standardized
        correction[indices] = block_correction
        refit_index[indices] = block_refit
        fit_rows.append(dict(item["fit_row"]))
        observed.append(indices)
    merged_indices = (
        np.concatenate(observed) if observed else np.empty(0, dtype=np.int64)
    )
    if not np.array_equal(merged_indices, evaluation):
        raise ValueError("Ray refit blocks do not exactly cover evaluation_indices.")
    fit_summary = (
        pd.DataFrame(fit_rows).sort_values("refit_index").reset_index(drop=True)
    )
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
    response_standardization: str = "median_mad",
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
                response_standardization=str(response_standardization),
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


def fit_rolling_vara_family(
    *,
    base_forecast: np.ndarray,
    base_error: np.ndarray,
    metadata: pd.DataFrame,
    evaluation_indices: Sequence[int],
    feature_matrix: np.ndarray,
    error_window_months: int,
    response_standardization: str,
    response_clip: float | None,
    refit_frequency_months: int,
    ridge: float,
    attention_top_k: int,
    attention_cache_mode: str,
    attention_batch_size: int,
    eta_bounds: tuple[float, float],
    eta_grid_size: int,
    eta_local_tolerance: float,
    lambda_window_months: int,
    lambda_ridge: float,
    zero_own_lag_coefficients: bool = False,
    ray_num_cpus: int = 1,
    max_cpus_per_task: int = 1,
    ray_debug: bool = False,
    ray_module: Any | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> RollingVARAFamilyResult:
    """Fit one common rolling VARA correction and its bounded-LSE family.

    ``base_error`` must follow the application convention ``target - base``.
    It is supplied explicitly so existing persisted error panels and their
    floating-point representation remain authoritative.
    """
    _validate_ray_resources(ray_num_cpus, max_cpus_per_task)
    base = np.asarray(base_forecast, dtype=float)
    error = np.asarray(base_error)
    if base.ndim != 2 or error.ndim != 2 or base.shape != error.shape:
        raise ValueError("base_forecast and base_error must have the same 2D shape.")

    correction_kwargs = {
        "baseline_errors": error,
        "evaluation_indices": evaluation_indices,
        "feature_matrix": feature_matrix,
        "error_window_months": error_window_months,
        "response_standardization": response_standardization,
        "response_clip": response_clip,
        "refit_frequency_months": refit_frequency_months,
        "ridge": ridge,
        "attention_top_k": attention_top_k,
        "attention_cache_mode": attention_cache_mode,
        "attention_batch_size": attention_batch_size,
        "eta_bounds": eta_bounds,
        "eta_grid_size": eta_grid_size,
        "eta_local_tolerance": eta_local_tolerance,
        "zero_own_lag_coefficients": zero_own_lag_coefficients,
        "progress_callback": progress_callback,
    }
    if int(ray_num_cpus) > 1:
        correction_result, correction_execution = (
            rolling_har_vara_corrections_ray(
                **correction_kwargs,
                ray_num_cpus=ray_num_cpus,
                max_cpus_per_task=max_cpus_per_task,
                ray_debug=ray_debug,
                ray_module=ray_module,
            )
        )
    else:
        started_at = time.monotonic()
        correction_result = rolling_har_vara_corrections(**correction_kwargs)
        correction_execution = {
            "backend": "sequential",
            "ray_used": False,
            "task_unit": "rolling_refit_block",
            "task_count": int(
                len(
                    _rolling_refit_blocks(
                        evaluation_indices,
                        refit_frequency_months,
                    )
                )
            ),
            "elapsed_seconds": float(time.monotonic() - started_at),
        }

    return combine_rolling_vara_family(
        base_forecast=base,
        base_error=error,
        correction=np.asarray(correction_result.correction, dtype=np.float32),
        refit_index=np.asarray(correction_result.refit_index, dtype=np.int64),
        core_fit_summary=correction_result.fit_summary,
        correction_execution=correction_execution,
        metadata=metadata,
        evaluation_indices=evaluation_indices,
        lambda_window_months=lambda_window_months,
        lambda_ridge=lambda_ridge,
    )
