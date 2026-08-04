#!/usr/bin/env python3
"""Fit a pooled LightGBM log-risk base and its rolling VARA corrections.

The base model is one global LightGBM regressor pooled across components.  Its
only predictors are the same causal own-series summaries used by HAR: lag 1
and trailing 3- and 12-month means.  Formula features and component IDs are
excluded from the base so that the subsequent VARA step remains the only
cross-series, formula-aware mechanism.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/vara-matplotlib-cache")

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
    from .run_datamine_garch_vara import (
        _correction_summary,
        _print_eta_refit,
        _save_float32_array,
        _save_int64_array,
        load_validated_har_vara_forecast,
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
    from run_datamine_garch_vara import (  # type: ignore[no-redef]
        _correction_summary,
        _print_eta_refit,
        _save_float32_array,
        _save_int64_array,
        load_validated_har_vara_forecast,
    )

from vara.datamine import (
    evaluate_risk_forecasts,
    load_panel,
    relative_panel_split,
    write_json,
)
from vara.datamine_vara_runner import (
    _rolling_refit_blocks,
    _validate_ray_resources,
    combine_base_vara_forecasts,
    fit_rolling_vara_family,
)
from vara.forecast_significance import diebold_mariano_test


POOLED_LIGHTGBM_MODEL = "pooled_lightgbm"
POOLED_LIGHTGBM_VARA_UNIT_MODEL = "pooled_lightgbm_vara_unit"
POOLED_LIGHTGBM_VARA_LAMBDA_MODEL = "pooled_lightgbm_vara_lambda"
LIGHTGBM_FEATURE_WINDOWS = (1, 3, 12)
LIGHTGBM_FEATURE_NAMES = ("lag_1", "mean_3", "mean_12")
LIGHTGBM_RESPONSE_NORMALIZATIONS = ("none", "training_zscore")
LIGHTGBM_NORMALIZATION_SCALE_FLOOR = 1e-12


def _validate_lightgbm_response_normalization(method: str) -> str:
    """Return a validated LightGBM response-normalization identifier."""
    parsed = str(method)
    if parsed not in LIGHTGBM_RESPONSE_NORMALIZATIONS:
        raise ValueError(
            "lightgbm_response_normalization must be one of "
            f"{LIGHTGBM_RESPONSE_NORMALIZATIONS}."
        )
    return parsed


def fit_finite_componentwise_training_normalization(
    history: np.ndarray,
) -> dict[str, np.ndarray]:
    """Fit finite-only per-component population z-score statistics.

    Accounting-signal histories may contain missing component-month cells, so
    the balanced-panel Tourism helper cannot be used directly.  Components
    without a finite observation receive center zero; nonfinite or effectively
    constant scales receive the deterministic unit-scale fallback.
    """
    values = np.asarray(history, dtype=float)
    if values.ndim != 2:
        raise ValueError("LightGBM training normalization expects a 2D panel.")
    finite = np.isfinite(values)
    finite_count = finite.sum(axis=0, dtype=np.int64)
    total = np.where(finite, values, 0.0).sum(axis=0)
    mean = np.divide(
        total,
        finite_count,
        out=np.zeros(values.shape[1], dtype=float),
        where=finite_count > 0,
    )
    centered = np.where(finite, values - mean[None, :], 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=0),
        finite_count,
        out=np.zeros(values.shape[1], dtype=float),
        where=finite_count > 0,
    )
    variance = np.maximum(variance, 0.0)
    raw_scale = np.sqrt(variance)
    scale_fallback = (
        ~np.isfinite(raw_scale)
        | (raw_scale <= float(LIGHTGBM_NORMALIZATION_SCALE_FLOOR))
    )
    scale = np.where(scale_fallback, 1.0, raw_scale)
    return {
        "mean": np.asarray(mean, dtype=np.float64),
        "variance": np.asarray(variance, dtype=np.float64),
        "scale": np.asarray(scale, dtype=np.float64),
        "finite_count": np.asarray(finite_count, dtype=np.int64),
        "scale_fallback": np.asarray(scale_fallback, dtype=bool),
    }


def fixed_lightgbm_parameters(
    *,
    seed: int,
    num_threads: int,
    learning_rate: float,
    num_leaves: int,
    max_depth: int,
    min_data_in_leaf: int,
    lambda_l2: float,
    max_bin: int,
) -> dict[str, Any]:
    """Return the complete prespecified deterministic LightGBM configuration."""
    if int(num_threads) < 1:
        raise ValueError("num_threads must be positive.")
    if not 0.0 < float(learning_rate) <= 1.0:
        raise ValueError("learning_rate must lie in (0, 1].")
    if int(num_leaves) < 2:
        raise ValueError("num_leaves must be at least two.")
    if int(max_depth) < 1:
        raise ValueError("max_depth must be positive.")
    if int(min_data_in_leaf) < 1:
        raise ValueError("min_data_in_leaf must be positive.")
    if float(lambda_l2) < 0.0:
        raise ValueError("lambda_l2 must be nonnegative.")
    if int(max_bin) < 3:
        raise ValueError("max_bin must be at least three.")
    return {
        "objective": "regression_l2",
        "metric": "rmse",
        "boosting_type": "gbdt",
        "learning_rate": float(learning_rate),
        "num_leaves": int(num_leaves),
        "max_depth": int(max_depth),
        "min_data_in_leaf": int(min_data_in_leaf),
        "lambda_l1": 0.0,
        "lambda_l2": float(lambda_l2),
        "max_bin": int(max_bin),
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "deterministic": True,
        "force_col_wise": True,
        "device_type": "cpu",
        "num_threads": int(num_threads),
        "seed": int(seed),
        "data_random_seed": int(seed),
        "feature_fraction_seed": int(seed),
        "bagging_seed": int(seed),
        "verbosity": -1,
    }


def build_causal_har_features(
    target: np.ndarray,
    *,
    windows: Sequence[int] = LIGHTGBM_FEATURE_WINDOWS,
) -> np.ndarray:
    """Build time-by-component causal rolling features.

    Feature row ``t`` uses target observations strictly before ``t``.  Missing
    values propagate, matching the existing HAR information-set rule.
    """
    values = np.asarray(target)
    if values.ndim != 2:
        raise ValueError("target must have shape (time, components).")
    parsed_windows = tuple(int(window) for window in windows)
    if not parsed_windows or any(window < 1 for window in parsed_windows):
        raise ValueError("windows must contain positive integers.")
    features = np.full(
        (values.shape[0], values.shape[1], len(parsed_windows)),
        np.nan,
        dtype=np.float32,
    )
    for target_index in range(max(parsed_windows), values.shape[0]):
        for feature_index, window in enumerate(parsed_windows):
            features[target_index, :, feature_index] = np.mean(
                np.asarray(values[target_index - window : target_index], dtype=float),
                axis=0,
            ).astype(np.float32)
    return features


def lightgbm_base_evaluation_indices(
    *,
    validation_indices: Sequence[int],
    test_indices: Sequence[int],
    error_window_months: int,
    refit_frequency_months: int,
) -> np.ndarray:
    """Return the contiguous base-forecast span needed by the first VARA fit."""
    validation = np.asarray(validation_indices, dtype=np.int64)
    test = np.asarray(test_indices, dtype=np.int64)
    if validation.size == 0 or test.size == 0:
        raise ValueError("validation_indices and test_indices must be nonempty.")
    if np.any(np.diff(validation) != 1) or np.any(np.diff(test) != 1):
        raise ValueError("validation and test indices must each be contiguous.")
    if int(test[0]) != int(validation[-1]) + 1:
        raise ValueError("validation and test periods must be adjacent.")
    history = int(error_window_months)
    frequency = int(refit_frequency_months)
    if history < 1 or frequency < 1:
        raise ValueError("window and refit frequency must be positive.")
    if history % frequency != 0:
        raise ValueError(
            "error_window_months must be divisible by refit_frequency_months "
            "so the first validation origin starts a base-model refit block."
        )
    start = int(validation[0]) - history
    if start <= max(LIGHTGBM_FEATURE_WINDOWS):
        raise ValueError("The panel has insufficient pre-validation LightGBM history.")
    return np.arange(start, int(test[-1]) + 1, dtype=np.int64)


def _fit_lightgbm_refit_block(
    target: np.ndarray,
    causal_features: np.ndarray,
    evaluation_indices: np.ndarray,
    *,
    block_position: int,
    training_window_months: int,
    num_boost_round: int,
    lightgbm_parameters: Mapping[str, Any],
    response_normalization: str = "none",
) -> dict[str, Any]:
    """Fit one pooled LightGBM using only labels before its forecast block."""
    try:
        import lightgbm as lgb

        started_at = time.monotonic()
        values = np.asarray(target)
        features = np.asarray(causal_features)
        evaluation = np.asarray(evaluation_indices, dtype=np.int64)
        if values.ndim != 2:
            raise ValueError("target must have shape (time, components).")
        if features.shape != (
            values.shape[0],
            values.shape[1],
            len(LIGHTGBM_FEATURE_NAMES),
        ):
            raise ValueError("causal_features have the wrong shape.")
        if evaluation.size == 0 or np.any(np.diff(evaluation) != 1):
            raise ValueError("A LightGBM refit block must be nonempty and contiguous.")
        if evaluation[0] <= max(LIGHTGBM_FEATURE_WINDOWS):
            raise ValueError("LightGBM evaluation begins before causal features exist.")
        if evaluation[-1] >= values.shape[0]:
            raise ValueError("LightGBM evaluation exceeds the target panel.")
        window = int(training_window_months)
        rounds = int(num_boost_round)
        if window < 24:
            raise ValueError("training_window_months must be at least 24.")
        if rounds < 1:
            raise ValueError("num_boost_round must be positive.")
        normalization_method = _validate_lightgbm_response_normalization(
            response_normalization
        )

        origin_index = int(evaluation[0]) - 1
        training_start_index = origin_index - window + 1
        if training_start_index <= max(LIGHTGBM_FEATURE_WINDOWS):
            raise ValueError("Insufficient causal history for the LightGBM training window.")
        training_indices = np.arange(
            training_start_index,
            origin_index + 1,
            dtype=np.int64,
        )
        training_panel = np.asarray(values[training_indices], dtype=float)
        normalization = fit_finite_componentwise_training_normalization(
            training_panel
        )
        training_feature_panel = np.asarray(features[training_indices], dtype=float)
        if normalization_method == "training_zscore":
            center = normalization["mean"]
            scale = normalization["scale"]
            model_training_y = (
                training_panel - center[None, :]
            ) / scale[None, :]
            model_training_x = (
                training_feature_panel - center[None, :, None]
            ) / scale[None, :, None]
        else:
            model_training_y = training_panel
            model_training_x = training_feature_panel

        training_x = model_training_x.reshape(
            -1,
            len(LIGHTGBM_FEATURE_NAMES),
        )
        training_y = model_training_y.reshape(-1)
        training_valid = np.isfinite(training_y) & np.isfinite(training_x).all(axis=1)
        if not training_valid.any():
            raise ValueError("The pooled LightGBM training sample has no finite rows.")
        training_x = np.ascontiguousarray(training_x[training_valid], dtype=np.float32)
        training_y = np.ascontiguousarray(training_y[training_valid], dtype=np.float32)

        parameters = dict(lightgbm_parameters)
        dataset = lgb.Dataset(
            training_x,
            label=training_y,
            feature_name=list(LIGHTGBM_FEATURE_NAMES),
            free_raw_data=True,
        )
        booster = lgb.train(
            parameters,
            dataset,
            num_boost_round=rounds,
        )
        block_feature_panel = np.asarray(features[evaluation], dtype=float)
        if normalization_method == "training_zscore":
            block_feature_panel = (
                block_feature_panel - normalization["mean"][None, :, None]
            ) / normalization["scale"][None, :, None]
        block_x = block_feature_panel.reshape(-1, len(LIGHTGBM_FEATURE_NAMES))
        prediction_valid = np.isfinite(block_x).all(axis=1)
        model_scale_prediction = np.full(
            block_x.shape[0],
            np.nan,
            dtype=np.float32,
        )
        if prediction_valid.any():
            model_scale_prediction[prediction_valid] = np.asarray(
                booster.predict(
                    np.ascontiguousarray(block_x[prediction_valid], dtype=np.float32),
                    num_iteration=booster.current_iteration(),
                ),
                dtype=np.float32,
            )
        model_scale_prediction = model_scale_prediction.reshape(
            len(evaluation),
            values.shape[1],
        )
        if normalization_method == "training_zscore":
            prediction = (
                model_scale_prediction.astype(float)
                * normalization["scale"][None, :]
                + normalization["mean"][None, :]
            ).astype(np.float32)
        else:
            prediction = model_scale_prediction
        importance = booster.feature_importance(importance_type="gain")
        finite_count = normalization["finite_count"]
        fallback = normalization["scale_fallback"]
        fit_row: dict[str, Any] = {
            "refit_index": origin_index,
            "training_start_index": training_start_index,
            "training_end_index": origin_index,
            "training_months": int(len(training_indices)),
            "training_cells": int(training_valid.sum()),
            "training_candidate_cells": int(training_valid.size),
            "training_observed_fraction": float(training_valid.mean()),
            "forecast_start_index": int(evaluation[0]),
            "forecast_end_index": int(evaluation[-1]),
            "forecast_cells": int(np.isfinite(prediction).sum()),
            "boosting_rounds": int(booster.current_iteration()),
            "lightgbm_version": str(lgb.__version__),
            "response_normalization": normalization_method,
            "normalization_scope": "per_refit_block_per_component",
            "normalization_statistics_frozen_within_block": True,
            "normalization_variance_ddof": 0,
            "normalization_finite_count_minimum": int(np.min(finite_count)),
            "normalization_finite_count_median": float(np.median(finite_count)),
            "normalization_finite_count_maximum": int(np.max(finite_count)),
            "normalization_empty_components": int((finite_count == 0).sum()),
            "normalization_unit_scale_fallback_components": int(fallback.sum()),
            "normalization_scale_minimum": float(np.min(normalization["scale"])),
            "normalization_scale_median": float(np.median(normalization["scale"])),
            "normalization_scale_maximum": float(np.max(normalization["scale"])),
            "forecast_returned_to_original_log_risk_scale": True,
            "elapsed_seconds": float(time.monotonic() - started_at),
        }
        for name, value in zip(LIGHTGBM_FEATURE_NAMES, importance, strict=True):
            fit_row[f"{name}_gain_importance"] = float(value)
        return {
            "block_position": int(block_position),
            "evaluation_indices": evaluation,
            "forecast": np.ascontiguousarray(prediction, dtype=np.float32),
            "refit_index": np.full(len(evaluation), origin_index, dtype=np.int64),
            "fit_row": fit_row,
            "normalization_statistics": {
                "refit_index": int(origin_index),
                "mean": np.ascontiguousarray(normalization["mean"], dtype=np.float32),
                "variance": np.ascontiguousarray(
                    normalization["variance"],
                    dtype=np.float32,
                ),
                "scale": np.ascontiguousarray(normalization["scale"], dtype=np.float32),
                "finite_count": np.ascontiguousarray(
                    normalization["finite_count"],
                    dtype=np.int64,
                ),
                "scale_fallback": np.ascontiguousarray(
                    normalization["scale_fallback"],
                    dtype=bool,
                ),
            },
        }
    except Exception as exc:
        raise RuntimeError(
            f"Pooled LightGBM Ray refit block {int(block_position)} failed."
        ) from exc


def _assemble_lightgbm_refit_blocks(
    block_results: Sequence[dict[str, Any]],
    *,
    panel_shape: tuple[int, int],
    evaluation_indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    """Merge asynchronously completed LightGBM blocks with exact coverage checks."""
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    ordered = sorted(block_results, key=lambda item: int(item["block_position"]))
    forecast = np.full(panel_shape, np.nan, dtype=np.float32)
    refit_index = np.full(panel_shape[0], -1, dtype=np.int64)
    fit_rows: list[dict[str, Any]] = []
    normalization_rows: list[dict[str, Any]] = []
    observed: list[np.ndarray] = []
    for expected_position, item in enumerate(ordered):
        if int(item["block_position"]) != expected_position:
            raise ValueError("LightGBM block positions are incomplete or duplicated.")
        indices = np.asarray(item["evaluation_indices"], dtype=np.int64)
        block_forecast = np.asarray(item["forecast"], dtype=np.float32)
        block_refit = np.asarray(item["refit_index"], dtype=np.int64)
        if block_forecast.shape != (len(indices), panel_shape[1]):
            raise ValueError("A LightGBM block forecast has the wrong shape.")
        if block_refit.shape != (len(indices),):
            raise ValueError("A LightGBM block refit assignment has the wrong shape.")
        forecast[indices] = block_forecast
        refit_index[indices] = block_refit
        fit_rows.append(dict(item["fit_row"]))
        normalization = dict(item["normalization_statistics"])
        if int(normalization["refit_index"]) != int(item["fit_row"]["refit_index"]):
            raise ValueError("LightGBM normalization statistics have the wrong refit index.")
        for name in ("mean", "variance", "scale", "finite_count", "scale_fallback"):
            if np.asarray(normalization[name]).shape != (panel_shape[1],):
                raise ValueError(
                    f"LightGBM normalization statistic {name!r} has the wrong shape."
                )
        normalization_rows.append(normalization)
        observed.append(indices)
    merged = np.concatenate(observed) if observed else np.empty(0, dtype=np.int64)
    if not np.array_equal(merged, evaluation):
        raise ValueError("LightGBM refit blocks do not exactly cover evaluation_indices.")
    fit_summary = pd.DataFrame(fit_rows).sort_values("refit_index").reset_index(drop=True)
    normalization_audit = {
        "refit_index": np.asarray(
            [row["refit_index"] for row in normalization_rows],
            dtype=np.int64,
        ),
        "mean": np.stack(
            [np.asarray(row["mean"], dtype=np.float32) for row in normalization_rows]
        ),
        "variance": np.stack(
            [np.asarray(row["variance"], dtype=np.float32) for row in normalization_rows]
        ),
        "scale": np.stack(
            [np.asarray(row["scale"], dtype=np.float32) for row in normalization_rows]
        ),
        "finite_count": np.stack(
            [np.asarray(row["finite_count"], dtype=np.int64) for row in normalization_rows]
        ),
        "scale_fallback": np.stack(
            [np.asarray(row["scale_fallback"], dtype=bool) for row in normalization_rows]
        ),
    }
    if not np.array_equal(
        normalization_audit["refit_index"],
        fit_summary["refit_index"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("LightGBM normalization and fit-summary refits are out of order.")
    return forecast, refit_index, fit_summary, normalization_audit


def rolling_pooled_lightgbm_forecast(
    *,
    target: np.ndarray,
    causal_features: np.ndarray,
    evaluation_indices: Sequence[int],
    training_window_months: int,
    refit_frequency_months: int,
    num_boost_round: int,
    lightgbm_parameters: Mapping[str, Any],
    ray_num_cpus: int,
    max_cpus_per_task: int,
    response_normalization: str = "none",
    ray_debug: bool = False,
    ray_module: Any | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    pd.DataFrame,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    """Run pooled annual LightGBM refits serially or as bounded Ray tasks."""
    _validate_ray_resources(ray_num_cpus, max_cpus_per_task)
    values = np.asarray(target)
    features = np.asarray(causal_features)
    if values.ndim != 2 or features.shape[:2] != values.shape:
        raise ValueError("target and causal_features have inconsistent shapes.")
    normalization_method = _validate_lightgbm_response_normalization(
        response_normalization
    )
    blocks = _rolling_refit_blocks(evaluation_indices, refit_frequency_months)
    started_at = time.monotonic()
    if int(ray_num_cpus) == 1:
        completed = [
            _fit_lightgbm_refit_block(
                values,
                features,
                block,
                block_position=block_position,
                training_window_months=training_window_months,
                num_boost_round=num_boost_round,
                lightgbm_parameters=lightgbm_parameters,
                response_normalization=normalization_method,
            )
            for block_position, block in enumerate(blocks)
        ]
        forecast, refit_index, fit_summary, normalization_audit = (
            _assemble_lightgbm_refit_blocks(
                completed,
                panel_shape=values.shape,
                evaluation_indices=evaluation_indices,
            )
        )
        return (
            forecast,
            refit_index,
            fit_summary,
            {
                "backend": "sequential",
                "ray_used": False,
                "ray_num_cpus": 1,
                "max_cpus_per_task": 1,
                "task_unit": "rolling_refit_block",
                "task_count": int(len(blocks)),
                "response_normalization": normalization_method,
                "elapsed_seconds": float(time.monotonic() - started_at),
            },
            normalization_audit,
        )

    ray = ray_module
    if ray is None:
        import ray as ray_runtime

        ray = ray_runtime
    worker_runtime_env = {
        "env_vars": {
            "JAX_PLATFORMS": "cpu",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "MPLCONFIGDIR": "/tmp/vara-matplotlib-cache",
            "OMP_NUM_THREADS": str(int(max_cpus_per_task)),
            "OPENBLAS_NUM_THREADS": str(int(max_cpus_per_task)),
            "MKL_NUM_THREADS": str(int(max_cpus_per_task)),
            "NUMEXPR_NUM_THREADS": str(int(max_cpus_per_task)),
        }
    }
    started_here = False
    pending: list[Any] = []
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
                f"Ray has no available CPUs after initialization: {available}."
            )
        print(
            "Ray pooled LightGBM rolling refits: "
            f"blocks={len(blocks)} ray_num_cpus={int(ray_num_cpus)} "
            f"max_cpus_per_task={int(max_cpus_per_task)}",
            flush=True,
        )
        target_ref = ray.put(values)
        features_ref = ray.put(features)
        remote_fit = ray.remote(
            num_cpus=int(max_cpus_per_task),
            runtime_env=worker_runtime_env,
        )(_fit_lightgbm_refit_block)
        max_inflight = max(1, int(ray_num_cpus) // int(max_cpus_per_task))
        submitted_at: dict[Any, float] = {}
        next_block_position = 0

        def submit_block(block_position: int) -> None:
            future = remote_fit.remote(
                target_ref,
                features_ref,
                blocks[block_position],
                block_position=int(block_position),
                training_window_months=int(training_window_months),
                num_boost_round=int(num_boost_round),
                lightgbm_parameters=dict(lightgbm_parameters),
                response_normalization=normalization_method,
            )
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
                    "Ray pooled LightGBM wait: "
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
                if ray_debug or len(completed) == len(blocks) or len(completed) % 5 == 0:
                    print(
                        "Ray pooled LightGBM collected: "
                        f"done={len(completed)}/{len(blocks)} "
                        f"refit_index={int(item['fit_row']['refit_index'])}",
                        flush=True,
                    )
            while next_block_position < len(blocks) and len(pending) < max_inflight:
                submit_block(next_block_position)
                next_block_position += 1
        forecast, refit_index, fit_summary, normalization_audit = (
            _assemble_lightgbm_refit_blocks(
                completed,
                panel_shape=values.shape,
                evaluation_indices=evaluation_indices,
            )
        )
        return (
            forecast,
            refit_index,
            fit_summary,
            {
                "backend": "ray",
                "ray_used": True,
                "ray_version": str(getattr(ray, "__version__", "unknown")),
                "ray_num_cpus": int(ray_num_cpus),
                "max_cpus_per_task": int(max_cpus_per_task),
                "task_unit": "rolling_refit_block",
                "task_count": int(len(blocks)),
                "response_normalization": normalization_method,
                "elapsed_seconds": float(time.monotonic() - started_at),
            },
            normalization_audit,
        )
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


def combine_lightgbm_vara_forecasts(
    *,
    base_forecast: np.ndarray,
    correction: np.ndarray,
    selected_lambda: float | np.ndarray,
) -> dict[str, np.ndarray]:
    """Compatibility wrapper around the shared VARA forecast combiner."""
    unit, weighted = combine_base_vara_forecasts(
        base_forecast=base_forecast,
        correction=correction,
        lambda_weights=selected_lambda,
    )
    return {
        POOLED_LIGHTGBM_MODEL: np.asarray(base_forecast, dtype=float),
        POOLED_LIGHTGBM_VARA_UNIT_MODEL: unit,
        POOLED_LIGHTGBM_VARA_LAMBDA_MODEL: weighted,
    }


def _dm_table(monthly: pd.DataFrame) -> pd.DataFrame:
    comparisons = (
        (POOLED_LIGHTGBM_VARA_UNIT_MODEL, POOLED_LIGHTGBM_MODEL),
        (POOLED_LIGHTGBM_VARA_LAMBDA_MODEL, POOLED_LIGHTGBM_MODEL),
        (POOLED_LIGHTGBM_VARA_LAMBDA_MODEL, "har_ridge"),
        (POOLED_LIGHTGBM_VARA_LAMBDA_MODEL, "har_vara_lambda"),
    )
    rows: list[dict[str, Any]] = []
    available = set(monthly["model"].astype(str))
    for focal_name, comparator_name in comparisons:
        if focal_name not in available or comparator_name not in available:
            continue
        focal = monthly.loc[monthly["model"].eq(focal_name)]
        comparator = monthly.loc[monthly["model"].eq(comparator_name)]
        for loss in ("squared_error", "gaussian_return_crps"):
            matched = focal[["date", loss]].merge(
                comparator[["date", loss]],
                on="date",
                suffixes=("_focal", "_comparator"),
                validate="one_to_one",
            ).dropna()
            rows.append(
                {
                    "focal_model": focal_name,
                    "comparator_model": comparator_name,
                    "loss": loss,
                    **diebold_mariano_test(
                        matched[f"{loss}_focal"],
                        matched[f"{loss}_comparator"],
                        forecast_horizon=1,
                    ),
                }
            )
    return pd.DataFrame(rows)


def _split_contract(panel: Any, split: Any) -> dict[str, str]:
    return {
        "universe_selection_end": panel.dates[
            int(split.validation_indices[0]) - 1
        ].date().isoformat(),
        "validation_start": panel.dates[
            int(split.validation_indices[0])
        ].date().isoformat(),
        "validation_end": panel.dates[
            int(split.validation_indices[-1])
        ].date().isoformat(),
        "test_start": panel.dates[int(split.test_indices[0])].date().isoformat(),
        "test_end": panel.dates[int(split.test_indices[-1])].date().isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_output", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--har-vara-output", type=Path)
    parser.add_argument("--feature-snapshot", default="snapshot_200712")
    parser.add_argument("--lightgbm-training-window-months", type=int, default=240)
    parser.add_argument("--lightgbm-num-boost-round", type=int, default=100)
    parser.add_argument("--lightgbm-learning-rate", type=float, default=0.05)
    parser.add_argument("--lightgbm-num-leaves", type=int, default=15)
    parser.add_argument("--lightgbm-max-depth", type=int, default=4)
    parser.add_argument("--lightgbm-min-data-in-leaf", type=int, default=5000)
    parser.add_argument("--lightgbm-lambda-l2", type=float, default=1.0)
    parser.add_argument("--lightgbm-max-bin", type=int, default=63)
    parser.add_argument(
        "--lightgbm-response-normalization",
        choices=LIGHTGBM_RESPONSE_NORMALIZATIONS,
        default="none",
        help=(
            "Optional finite-only per-component z-score fitted on each "
            "LightGBM refit's training window. Statistics remain fixed for "
            "that refit block and forecasts are returned to log-risk units."
        ),
    )
    parser.add_argument("--seed", type=int, default=202407)
    parser.add_argument("--error-window-months", type=int, default=240)
    parser.add_argument("--refit-frequency-months", type=int, default=12)
    parser.add_argument("--response-clip", type=float, default=8.0)
    parser.add_argument(
        "--response-standardization",
        choices=("median_mad", "training_zscore"),
        default="median_mad",
        help=(
            "Standardization fitted separately on each trailing VARA residual "
            "window. Forecasts and corrections are returned to original "
            "log-risk units."
        ),
    )
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--attention-top-k", type=int, default=500)
    parser.add_argument("--attention-cache-mode", choices=("exact", "cluster"), default="exact")
    parser.add_argument("--attention-batch-size", type=int, default=512)
    parser.add_argument("--eta-lower", type=float, default=DEFAULT_PAPER_ETA_BOUNDS[0])
    parser.add_argument("--eta-upper", type=float, default=DEFAULT_PAPER_ETA_BOUNDS[1])
    parser.add_argument("--eta-grid-size", type=int, default=DEFAULT_PAPER_ETA_GRID_SIZE)
    parser.add_argument(
        "--eta-local-tolerance",
        type=float,
        default=DEFAULT_PAPER_ETA_LOCAL_TOLERANCE,
    )
    parser.add_argument("--lambda-window-months", type=int, default=24)
    parser.add_argument("--lambda-ridge", type=float, default=0.001)
    parser.add_argument("--ray-num-cpus", type=int, default=1)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    parser.add_argument("--ray-debug", action="store_true")
    args = parser.parse_args()
    _validate_ray_resources(args.ray_num_cpus, args.max_cpus_per_task)

    import lightgbm as lgb

    source_contract = json.loads(
        (args.source_output / "forecasts" / "locked_test_config.json").read_text(
            encoding="utf-8"
        )
    )
    source_output = args.source_output.resolve().as_posix()
    panel = load_panel(args.source_output / "processed")
    split = relative_panel_split(
        panel.dates,
        validation_months=int(source_contract["validation_months"]),
        test_months=int(source_contract["test_months"]),
    )
    split_contract = _split_contract(panel, split)
    target = np.load(args.source_output / "baseline" / "target.npy", mmap_mode="r")
    har = np.load(
        args.source_output / "baseline" / "baseline_forecast.npy",
        mmap_mode="r",
    )
    ar1 = np.load(args.source_output / "baseline" / "ar1_forecast.npy", mmap_mode="r")
    expected_shape = (len(panel.dates), len(panel.component_ids))
    if any(values.shape != expected_shape for values in (target, har, ar1)):
        raise ValueError("Risk target and benchmark arrays do not match the panel.")

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
    lightgbm_parameters = fixed_lightgbm_parameters(
        seed=args.seed,
        num_threads=args.max_cpus_per_task,
        learning_rate=args.lightgbm_learning_rate,
        num_leaves=args.lightgbm_num_leaves,
        max_depth=args.lightgbm_max_depth,
        min_data_in_leaf=args.lightgbm_min_data_in_leaf,
        lambda_l2=args.lightgbm_lambda_l2,
        max_bin=args.lightgbm_max_bin,
    )
    base_evaluation = lightgbm_base_evaluation_indices(
        validation_indices=split.validation_indices,
        test_indices=split.test_indices,
        error_window_months=args.error_window_months,
        refit_frequency_months=args.refit_frequency_months,
    )
    vara_evaluation = np.concatenate((split.validation_indices, split.test_indices))

    input_fingerprints = file_fingerprints(
        {
            "component_order": args.source_output / "processed" / "component_order.csv",
            "dates": args.source_output / "processed" / "dates.npy",
            "locked_test_config": (
                args.source_output / "forecasts" / "locked_test_config.json"
            ),
            "target": args.source_output / "baseline" / "target.npy",
            "vara_feature_matrix": (
                args.source_output / "features" / args.feature_snapshot / "Z.npy"
            ),
        }
    )
    causal_features = build_causal_har_features(target)

    ray_module: Any | None = None
    started_ray_here = False
    if int(args.ray_num_cpus) > 1:
        import ray

        ray_module = ray
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                num_cpus=int(args.ray_num_cpus),
                _node_ip_address="127.0.0.1",
                runtime_env={
                    "env_vars": {
                        "JAX_PLATFORMS": "cpu",
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        "MPLCONFIGDIR": "/tmp/vara-matplotlib-cache",
                        "OMP_NUM_THREADS": str(int(args.max_cpus_per_task)),
                        "OPENBLAS_NUM_THREADS": str(int(args.max_cpus_per_task)),
                        "MKL_NUM_THREADS": str(int(args.max_cpus_per_task)),
                        "NUMEXPR_NUM_THREADS": str(int(args.max_cpus_per_task)),
                    }
                },
            )
            started_ray_here = True

    try:
        (
            base_forecast,
            base_refit_index,
            base_fit_summary,
            base_execution,
            base_normalization_audit,
        ) = rolling_pooled_lightgbm_forecast(
            target=target,
            causal_features=causal_features,
            evaluation_indices=base_evaluation,
            training_window_months=args.lightgbm_training_window_months,
            refit_frequency_months=args.refit_frequency_months,
            num_boost_round=args.lightgbm_num_boost_round,
            lightgbm_parameters=lightgbm_parameters,
            ray_num_cpus=args.ray_num_cpus,
            max_cpus_per_task=args.max_cpus_per_task,
            response_normalization=args.lightgbm_response_normalization,
            ray_debug=args.ray_debug,
            ray_module=ray_module,
        )
        base_error = np.asarray(target, dtype=float) - np.asarray(
            base_forecast,
            dtype=float,
        )
        features = np.load(
            args.source_output / "features" / args.feature_snapshot / "Z.npy",
            mmap_mode="r",
        )
        if features.shape[0] != len(panel.component_ids):
            raise ValueError("VARA feature rows do not match panel components.")
        vara_family = fit_rolling_vara_family(
            base_forecast=base_forecast,
            base_error=base_error,
            metadata=panel.metadata,
            evaluation_indices=vara_evaluation,
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
            ray_num_cpus=args.ray_num_cpus,
            max_cpus_per_task=args.max_cpus_per_task,
            ray_debug=args.ray_debug,
            ray_module=ray_module,
            progress_callback=lambda row: _print_eta_refit(row, eta_specification),
        )
    finally:
        if started_ray_here and ray_module is not None:
            ray_module.shutdown()

    correction = vara_family.correction
    vara_refit_index = vara_family.refit_index
    vara_fit_summary = eta_specification.convert_core_fit_summary(
        vara_family.core_fit_summary
    )
    correction_execution = vara_family.correction_execution
    rolling_lambdas = vara_family.rolling_lambdas
    lambda_table = vara_family.lambda_refit_fits
    lambda_weights = vara_family.lambda_weights
    forecasts = {
        POOLED_LIGHTGBM_MODEL: np.asarray(base_forecast, dtype=float),
        POOLED_LIGHTGBM_VARA_UNIT_MODEL: vara_family.unit_forecast,
        POOLED_LIGHTGBM_VARA_LAMBDA_MODEL: vara_family.lambda_forecast,
    }

    forecast_dir = args.output_dir / "forecasts"
    metric_dir = args.output_dir / "metrics"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    _save_float32_array(
        forecast_dir / "pooled_lightgbm_forecast.npy",
        base_forecast,
    )
    _save_float32_array(
        forecast_dir / "pooled_lightgbm_error.npy",
        base_error,
    )
    _save_int64_array(
        forecast_dir / "pooled_lightgbm_refit_index.npy",
        base_refit_index,
    )
    base_fit_summary.to_csv(
        forecast_dir / "pooled_lightgbm_fit_summary.csv",
        index=False,
    )
    _save_int64_array(
        forecast_dir / "pooled_lightgbm_normalization_refit_index.npy",
        base_normalization_audit["refit_index"],
    )
    _save_float32_array(
        forecast_dir / "pooled_lightgbm_training_mean_by_refit.npy",
        base_normalization_audit["mean"],
    )
    _save_float32_array(
        forecast_dir / "pooled_lightgbm_training_variance_by_refit.npy",
        base_normalization_audit["variance"],
    )
    _save_float32_array(
        forecast_dir / "pooled_lightgbm_training_scale_by_refit.npy",
        base_normalization_audit["scale"],
    )
    _save_int64_array(
        forecast_dir / "pooled_lightgbm_training_finite_count_by_refit.npy",
        base_normalization_audit["finite_count"],
    )
    _save_int64_array(
        forecast_dir / "pooled_lightgbm_training_scale_fallback_by_refit.npy",
        base_normalization_audit["scale_fallback"].astype(np.int64),
    )
    _save_float32_array(forecast_dir / "vara_correction.npy", correction)
    _save_int64_array(forecast_dir / "vara_refit_index.npy", vara_refit_index)
    _save_float32_array(
        forecast_dir / "pooled_lightgbm_vara_unit_forecast.npy",
        forecasts[POOLED_LIGHTGBM_VARA_UNIT_MODEL],
    )
    _save_float32_array(
        forecast_dir / "pooled_lightgbm_vara_lambda_forecast.npy",
        forecasts[POOLED_LIGHTGBM_VARA_LAMBDA_MODEL],
    )
    np.save(
        forecast_dir / "lambda_by_origin.npy",
        lambda_weights[:, 0].astype(np.float32),
        allow_pickle=False,
    )

    global_lambda_table = lambda_table.loc[
        lambda_table["group"].eq("__global__")
    ].copy()
    lambda_fit_summary = global_lambda_table.rename(
        columns={
            "lambda": "lambda_hat",
            "lambda_unbounded": "lambda_unbounded_hat",
            "ridge": "lambda_ridge",
        }
    ).drop(columns=["group", "series", "fallback_global"], errors="ignore")
    vara_fit_summary = vara_fit_summary.merge(
        lambda_fit_summary,
        on="refit_index",
        how="left",
        validate="one_to_one",
    )
    eta_specification.validate_refit_coverage(
        vara_fit_summary,
        vara_refit_index,
        vara_evaluation,
    )
    vara_fit_summary.to_csv(forecast_dir / "fit_summary.csv", index=False)
    lambda_table.to_csv(forecast_dir / "lambda_refit_fits.csv", index=False)
    lambda_origins = pd.DataFrame(
        {
            "target_index": vara_evaluation,
            "target_date": panel.dates[vara_evaluation],
            "refit_index": np.asarray(vara_refit_index)[vara_evaluation],
            "lambda": lambda_weights[vara_evaluation, 0],
        }
    )
    lambda_origins.to_csv(forecast_dir / "lambda_by_origin.csv", index=False)

    lambda_refits = lambda_table.loc[
        :, ["refit_index", "lambda_history_sufficient"]
    ].drop_duplicates()

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
        "fitted_refits": int(lambda_refits["lambda_history_sufficient"].sum()),
        "validation": lambda_period_summary(split.validation_indices),
        "test": lambda_period_summary(split.test_indices),
    }
    write_json(forecast_dir / "lambda.json", lambda_info)

    har_vara_forecast: np.ndarray | None = None
    if args.har_vara_output is not None:
        har_input_fingerprints = file_fingerprints(
            {
                "baseline_error": args.source_output / "baseline" / "baseline_error.npy",
                "component_order": args.source_output / "processed" / "component_order.csv",
                "dates": args.source_output / "processed" / "dates.npy",
                "feature_matrix": (
                    args.source_output / "features" / args.feature_snapshot / "Z.npy"
                ),
                "locked_test_config": (
                    args.source_output / "forecasts" / "locked_test_config.json"
                ),
            }
        )
        har_vara_forecast = load_validated_har_vara_forecast(
            args.har_vara_output,
            expected_shape=expected_shape,
            required_contract_fields={
                "application_contract_schema": APPLICATION_CONTRACT_SCHEMA,
                "source_output": source_output,
                "correction_input_fingerprints": har_input_fingerprints,
                "component_count": len(panel.component_ids),
                "feature_snapshot": str(args.feature_snapshot),
                "har_windows": [1, 3, 12],
                "error_window_months": int(args.error_window_months),
                "refit_frequency_months": int(args.refit_frequency_months),
                "response_clip": float(args.response_clip),
                "response_standardization": str(args.response_standardization),
                "ridge": float(args.ridge),
                "attention_top_k": int(args.attention_top_k),
                "attention_cache_mode": str(args.attention_cache_mode),
                "attention_batch_size": int(args.attention_batch_size),
                "eta_estimation": eta_estimation,
                "zero_own_lag_coefficients": False,
                "lambda_window_months": int(args.lambda_window_months),
                "lambda_ridge": float(args.lambda_ridge),
                "split": split_contract,
                "forecast_scope": "validation_and_test",
            },
        )

    evaluation_forecasts: dict[str, np.ndarray] = {
        **forecasts,
        "har_ridge": har,
        "ar1_ridge": ar1,
    }
    if har_vara_forecast is not None:
        evaluation_forecasts["har_vara_lambda"] = har_vara_forecast
    metrics, monthly, calibration = evaluate_risk_forecasts(
        target=target,
        returns=panel.returns,
        dates=panel.dates,
        forecasts=evaluation_forecasts,
        validation_indices=split.validation_indices,
        test_indices=split.test_indices,
        benchmark_name=POOLED_LIGHTGBM_MODEL,
        epsilon=float(source_contract["epsilon"]),
        variance_floor=float(source_contract["variance_floor"]),
    )
    dm = _dm_table(monthly)
    metrics.to_csv(metric_dir / "metrics.csv", index=False)
    monthly.to_csv(metric_dir / "monthly_losses.csv", index=False)
    dm.to_csv(metric_dir / "diebold_mariano_raw.csv", index=False)
    write_json(metric_dir / "variance_calibration.json", calibration)
    validation_report = {
        "unit_full_validation": _correction_summary(
            base_error[split.validation_indices],
            correction[split.validation_indices],
            1.0,
        ),
        "rolling_lambda_full_validation": _correction_summary(
            base_error[split.validation_indices],
            correction[split.validation_indices],
            rolling_lambdas[split.validation_indices],
        ),
    }
    write_json(metric_dir / "validation_metrics.json", validation_report)

    contract = {
        "application_contract_schema": APPLICATION_CONTRACT_SCHEMA,
        "source_output": source_output,
        "base_variant": "pooled_lightgbm",
        "component_count": len(panel.component_ids),
        "forecast_scope": "validation_and_test",
        "base_forecast_scope": {
            "start": panel.dates[int(base_evaluation[0])].date().isoformat(),
            "end": panel.dates[int(base_evaluation[-1])].date().isoformat(),
            "months": int(len(base_evaluation)),
            "purpose": "240 pre-validation genuine errors plus validation and test",
        },
        "split": split_contract,
        "input_fingerprints": input_fingerprints,
        "har_vara_output": (
            None
            if args.har_vara_output is None
            else args.har_vara_output.resolve().as_posix()
        ),
        "model_ids": {
            "base": POOLED_LIGHTGBM_MODEL,
            "unit_vara": POOLED_LIGHTGBM_VARA_UNIT_MODEL,
            "lambda_vara": POOLED_LIGHTGBM_VARA_LAMBDA_MODEL,
        },
        "forecast_artifacts": {
            "base": "pooled_lightgbm_forecast.npy",
            "base_error": "pooled_lightgbm_error.npy",
            "unit_vara": "pooled_lightgbm_vara_unit_forecast.npy",
            "lambda_vara": "pooled_lightgbm_vara_lambda_forecast.npy",
            "vara_correction": "vara_correction.npy",
            "normalization_refit_index": (
                "pooled_lightgbm_normalization_refit_index.npy"
            ),
            "normalization_training_mean": (
                "pooled_lightgbm_training_mean_by_refit.npy"
            ),
            "normalization_training_variance": (
                "pooled_lightgbm_training_variance_by_refit.npy"
            ),
            "normalization_training_scale": (
                "pooled_lightgbm_training_scale_by_refit.npy"
            ),
            "normalization_training_finite_count": (
                "pooled_lightgbm_training_finite_count_by_refit.npy"
            ),
            "normalization_training_scale_fallback": (
                "pooled_lightgbm_training_scale_fallback_by_refit.npy"
            ),
        },
        "lightgbm": {
            "version": str(lgb.__version__),
            "target": "log(epsilon + squared return)",
            "pooling": "all finite component-month training cells with equal weight",
            "formula_features_used": False,
            "component_identifiers_used": False,
            "normalization": str(args.lightgbm_response_normalization),
            "normalization_scope": "per refit block and per component",
            "normalization_statistics_source": (
                "finite target observations in the same trailing LightGBM "
                "training window through the refit origin"
            ),
            "normalization_variance_ddof": 0,
            "normalization_formula": (
                "(response_or_feature - training_mean) / training_scale"
            ),
            "normalization_feature_rule": (
                "apply each component's response mean and scale to lag_1, "
                "mean_3, and mean_12"
            ),
            "normalization_statistics_frozen_within_refit_block": True,
            "normalization_empty_component_center_fallback": 0.0,
            "normalization_constant_or_invalid_scale_fallback": 1.0,
            "normalization_scale_floor": float(
                LIGHTGBM_NORMALIZATION_SCALE_FLOOR
            ),
            "forecast_inverse_formula": (
                "training_mean + training_scale * model_scale_forecast"
            ),
            "stored_forecast_scale": "original log risk",
            "features": list(LIGHTGBM_FEATURE_NAMES),
            "feature_windows": list(LIGHTGBM_FEATURE_WINDOWS),
            "feature_information_cutoff": "strictly before each target month",
            "training_window_months": int(args.lightgbm_training_window_months),
            "refit_frequency_months": int(args.refit_frequency_months),
            "num_boost_round": int(args.lightgbm_num_boost_round),
            "hyperparameter_selection": (
                "prespecified single configuration; no validation or test search"
            ),
            "parameters": lightgbm_parameters,
            "missing_rule": "omit nonfinite training rows and nonfinite forecast features",
            "one_step_ahead": True,
            "base_refit_execution": base_execution,
        },
        "vara_response": (
            "target log risk minus the pooled LightGBM genuine one-step forecast"
        ),
        "feature_snapshot": args.feature_snapshot,
        "har_windows": [1, 3, 12],
        "error_window_months": int(args.error_window_months),
        "refit_frequency_months": int(args.refit_frequency_months),
        "response_clip": float(args.response_clip),
        "response_standardization": str(args.response_standardization),
        "ridge": float(args.ridge),
        "attention_top_k": int(args.attention_top_k),
        "attention_cache_mode": str(args.attention_cache_mode),
        "attention_batch_size": int(args.attention_batch_size),
        "eta_estimation": eta_estimation,
        "eta_refit_summary": eta_specification.refit_summary(vara_fit_summary),
        "rolling_refit_execution": correction_execution,
        "zero_own_lag_coefficients": False,
        "lambda_window_months": int(args.lambda_window_months),
        "lambda_ridge": float(args.lambda_ridge),
        "lambda_estimation": lambda_info,
        "target_month_excluded_from_base_and_vara_refits": True,
        "current_or_future_test_target_used_for_base_fit": False,
        "past_test_targets_used_for_later_base_refits": True,
        "exploratory_post_test_method": True,
        "post_selection_warning": (
            "This LightGBM base was proposed after the 2013-2022 test results were "
            "inspected; use a new outer holdout for confirmatory inference."
        ),
    }
    write_json(forecast_dir / "contract.json", contract)
    print(json.dumps({"contract": contract, "validation": validation_report}, indent=2))
    print(
        metrics.loc[
            metrics["model"].isin(
                (
                    POOLED_LIGHTGBM_MODEL,
                    POOLED_LIGHTGBM_VARA_UNIT_MODEL,
                    POOLED_LIGHTGBM_VARA_LAMBDA_MODEL,
                    "har_ridge",
                    "har_vara_lambda",
                )
            ),
            [
                "model",
                "log_risk_rmse",
                "log_risk_mae",
                "gaussian_return_crps",
                "log_risk_bias",
                "absolute_r2",
                "rank_score",
            ],
        ].sort_values("log_risk_rmse").to_string(index=False)
    )
    print(dm.to_string(index=False))


if __name__ == "__main__":
    main()
