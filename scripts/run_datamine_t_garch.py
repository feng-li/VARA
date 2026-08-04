#!/usr/bin/env python3
"""Fit and evaluate a Ray-parallel Student-t GARCH(1,1) benchmark.

Each Accounting-Signal Risk component is fitted independently using only
pre-validation returns.  Ray tasks process coarse component blocks so optimizer
jobs run in parallel without creating one scheduler task per series.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vara-matplotlib")

import numpy as np
import pandas as pd

if not __package__:
    import _bootstrap  # noqa: F401

from vara.datamine import (
    evaluate_risk_forecasts,
    load_panel,
    relative_panel_split,
    sha256_file,
    write_json,
)
from vara.datamine_garch import fit_student_t_garch11_block


STUDENT_T_GARCH_OFFSET_MODEL = "garch11_student_t_validation_offset"
GAUSSIAN_GARCH_OFFSET_MODEL = "garch11_validation_offset"


def _component_blocks(component_count: int, block_size: int) -> list[tuple[int, int]]:
    """Return half-open component blocks in frozen panel order."""
    count = int(component_count)
    size = int(block_size)
    if count < 1:
        raise ValueError("component_count must be positive.")
    if size < 1:
        raise ValueError("component_block_size must be positive.")
    return [(start, min(count, start + size)) for start in range(0, count, size)]


def _validate_ray_resources(ray_num_cpus: int, max_cpus_per_task: int) -> None:
    """Validate the logical CPU budget for component-block fits."""
    if int(ray_num_cpus) < 1:
        raise ValueError("ray_num_cpus must be at least one.")
    if int(max_cpus_per_task) < 1:
        raise ValueError("max_cpus_per_task must be at least one.")
    if int(max_cpus_per_task) > int(ray_num_cpus):
        raise ValueError("max_cpus_per_task cannot exceed ray_num_cpus.")


def _fit_student_t_garch_block_task(
    shared_returns: np.ndarray,
    block_start: int,
    block_stop: int,
    *,
    block_position: int,
    estimation_end_index: int,
    minimum_observations: int,
    return_scale: float,
    variance_floor: float,
    max_iterations: int,
    tolerance: float,
) -> dict[str, Any]:
    """Fit one independent component block and return compact artifacts."""
    try:
        start = int(block_start)
        stop = int(block_stop)
        # Slice before converting so each worker copies only its own block.
        block = np.asarray(shared_returns[:, start:stop])
        variance, parameters = fit_student_t_garch11_block(
            block,
            estimation_end_index=int(estimation_end_index),
            minimum_observations=int(minimum_observations),
            return_scale=float(return_scale),
            variance_floor=float(variance_floor),
            max_iterations=int(max_iterations),
            tolerance=float(tolerance),
        )
        parameters = parameters.copy()
        parameters["component_position"] = (
            start + parameters["block_component_position"].to_numpy(dtype=np.int64)
        )
        return {
            "block_position": int(block_position),
            "block_start": start,
            "block_stop": stop,
            "variance": np.asarray(variance, dtype=np.float32),
            "parameters": parameters.drop(columns="block_component_position"),
        }
    except Exception as exc:
        raise RuntimeError(
            f"Student-t GARCH Ray component block {int(block_position)} failed."
        ) from exc


def _merge_student_t_garch_block(
    item: dict[str, Any],
    *,
    variance: np.ndarray,
    block_count: int,
    seen_blocks: set[int],
    parameter_blocks: dict[int, pd.DataFrame],
) -> None:
    """Validate and merge one asynchronously completed component block."""
    position = int(item["block_position"])
    start = int(item["block_start"])
    stop = int(item["block_stop"])
    if position < 0 or position >= int(block_count) or position in seen_blocks:
        raise ValueError("Ray component block positions are invalid or duplicated.")
    if start < 0 or stop <= start or stop > variance.shape[1]:
        raise ValueError("Ray component block boundaries are invalid.")
    block_variance = np.asarray(item["variance"], dtype=np.float32)
    if block_variance.shape != (variance.shape[0], stop - start):
        raise ValueError("Ray component block variance has the wrong shape.")
    parameters = pd.DataFrame(item["parameters"]).copy()
    expected_positions = np.arange(start, stop, dtype=np.int64)
    if not np.array_equal(
        parameters["component_position"].to_numpy(dtype=np.int64),
        expected_positions,
    ):
        raise ValueError("Ray component block parameters are out of order.")
    variance[:, start:stop] = block_variance
    parameter_blocks[position] = parameters
    seen_blocks.add(position)


def _finalize_student_t_garch_blocks(
    *,
    variance: np.ndarray,
    blocks: Sequence[tuple[int, int]],
    seen_blocks: set[int],
    parameter_blocks: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Verify exact component coverage and restore frozen component order."""
    if seen_blocks != set(range(len(blocks))):
        raise ValueError("Student-t GARCH blocks do not cover every component.")
    parameters = pd.concat(
        [parameter_blocks[position] for position in range(len(blocks))],
        ignore_index=True,
    ).sort_values("component_position", kind="stable").reset_index(drop=True)
    expected = np.arange(variance.shape[1], dtype=np.int64)
    if not np.array_equal(
        parameters["component_position"].to_numpy(dtype=np.int64), expected
    ):
        raise ValueError("Student-t GARCH parameter rows do not match panel order.")
    return parameters


def fit_student_t_garch11_panel(
    *,
    returns: np.ndarray,
    estimation_end_index: int,
    minimum_observations: int,
    return_scale: float,
    variance_floor: float,
    max_iterations: int,
    tolerance: float,
    component_block_size: int,
    ray_num_cpus: int,
    max_cpus_per_task: int,
    ray_debug: bool = False,
    ray_module: Any | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    """Fit the full panel serially or with bounded Ray component-block tasks."""
    _validate_ray_resources(ray_num_cpus, max_cpus_per_task)
    values = np.asarray(returns)
    if values.ndim != 2:
        raise ValueError("returns must have shape (time, components).")
    end = int(estimation_end_index)
    if end < 1 or end >= values.shape[0]:
        raise ValueError("estimation_end_index must precede at least one forecast row.")
    blocks = _component_blocks(values.shape[1], component_block_size)
    variance = np.full(values.shape, np.nan, dtype=np.float32)
    seen_blocks: set[int] = set()
    parameter_blocks: dict[int, pd.DataFrame] = {}
    started_at = time.monotonic()

    task_kwargs = {
        "estimation_end_index": end,
        "minimum_observations": int(minimum_observations),
        "return_scale": float(return_scale),
        "variance_floor": float(variance_floor),
        "max_iterations": int(max_iterations),
        "tolerance": float(tolerance),
    }
    if int(ray_num_cpus) == 1:
        for block_position, (start, stop) in enumerate(blocks):
            item = _fit_student_t_garch_block_task(
                values,
                start,
                stop,
                block_position=block_position,
                **task_kwargs,
            )
            _merge_student_t_garch_block(
                item,
                variance=variance,
                block_count=len(blocks),
                seen_blocks=seen_blocks,
                parameter_blocks=parameter_blocks,
            )
        parameters = _finalize_student_t_garch_blocks(
            variance=variance,
            blocks=blocks,
            seen_blocks=seen_blocks,
            parameter_blocks=parameter_blocks,
        )
        execution = {
            "backend": "sequential",
            "ray_used": False,
            "ray_num_cpus": 1,
            "max_cpus_per_task": int(max_cpus_per_task),
            "task_unit": "component_block",
            "task_count": int(len(blocks)),
            "component_block_size": int(component_block_size),
            "elapsed_seconds": float(time.monotonic() - started_at),
        }
        return variance, parameters, execution

    ray = ray_module
    if ray is None:
        import ray as ray_runtime

        ray = ray_runtime
    worker_runtime_env = {
        "env_vars": {
            "MPLCONFIGDIR": "/tmp/vara-matplotlib",
            "OMP_NUM_THREADS": str(int(max_cpus_per_task)),
            "OPENBLAS_NUM_THREADS": str(int(max_cpus_per_task)),
            "MKL_NUM_THREADS": str(int(max_cpus_per_task)),
            "NUMEXPR_NUM_THREADS": str(int(max_cpus_per_task)),
        }
    }
    started_here = False
    pending: list[Any] = []
    submitted_at: dict[Any, float] = {}
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
            "Ray Student-t GARCH fits: "
            f"components={values.shape[1]} blocks={len(blocks)} "
            f"ray_num_cpus={int(ray_num_cpus)} "
            f"max_cpus_per_task={int(max_cpus_per_task)}",
            flush=True,
        )
        returns_ref = ray.put(values)
        remote_fit = ray.remote(
            num_cpus=int(max_cpus_per_task),
            runtime_env=worker_runtime_env,
        )(_fit_student_t_garch_block_task)
        max_inflight = max(1, int(ray_num_cpus) // int(max_cpus_per_task))
        next_block_position = 0

        def submit(block_position: int) -> None:
            start, stop = blocks[block_position]
            future = remote_fit.remote(
                returns_ref,
                start,
                stop,
                block_position=int(block_position),
                **task_kwargs,
            )
            pending.append(future)
            submitted_at[future] = time.monotonic()

        while next_block_position < len(blocks) and len(pending) < max_inflight:
            submit(next_block_position)
            next_block_position += 1
        completed = 0
        while pending:
            ready, remaining = ray.wait(pending, num_returns=1, timeout=60.0)
            pending = list(remaining)
            if not ready:
                slowest = time.monotonic() - min(
                    submitted_at[future] for future in pending
                )
                print(
                    "Ray Student-t GARCH wait: "
                    f"done={completed}/{len(blocks)} pending={len(pending)} "
                    f"slowest_pending_sec={slowest:.1f} "
                    f"available={ray.available_resources()}",
                    flush=True,
                )
                continue
            for future in ready:
                item = ray.get(future)
                submitted_at.pop(future, None)
                _merge_student_t_garch_block(
                    item,
                    variance=variance,
                    block_count=len(blocks),
                    seen_blocks=seen_blocks,
                    parameter_blocks=parameter_blocks,
                )
                completed += 1
                if ray_debug or completed == len(blocks) or completed % 10 == 0:
                    print(
                        "Ray Student-t GARCH collected: "
                        f"done={completed}/{len(blocks)} "
                        f"block={int(item['block_position']) + 1}",
                        flush=True,
                    )
            while next_block_position < len(blocks) and len(pending) < max_inflight:
                submit(next_block_position)
                next_block_position += 1
        parameters = _finalize_student_t_garch_blocks(
            variance=variance,
            blocks=blocks,
            seen_blocks=seen_blocks,
            parameter_blocks=parameter_blocks,
        )
        execution = {
            "backend": "ray",
            "ray_used": True,
            "ray_version": str(getattr(ray, "__version__", "unknown")),
            "ray_num_cpus": int(ray_num_cpus),
            "max_cpus_per_task": int(max_cpus_per_task),
            "task_unit": "component_block",
            "task_count": int(len(blocks)),
            "component_block_size": int(component_block_size),
            "maximum_inflight_tasks": int(max_inflight),
            "elapsed_seconds": float(time.monotonic() - started_at),
        }
        return variance, parameters, execution
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


def estimate_validation_offset(
    *,
    target: np.ndarray,
    forecast: np.ndarray,
    reference_forecast: np.ndarray,
    validation_indices: Sequence[int],
) -> dict[str, float | int]:
    """Estimate the same pooled validation-only log-risk offset as GARCH."""
    indices = np.asarray(validation_indices, dtype=np.int64)
    observed = np.asarray(target)[indices]
    predicted = np.asarray(forecast)[indices]
    reference = np.asarray(reference_forecast)[indices]
    if observed.shape != predicted.shape or observed.shape != reference.shape:
        raise ValueError("Offset inputs must have matching validation shapes.")
    matched = np.isfinite(observed) & np.isfinite(predicted) & np.isfinite(reference)
    if not matched.any():
        raise ValueError("No matched validation cells are available for the offset.")
    raw_error = np.asarray(predicted[matched] - observed[matched], dtype=float)
    offset = float(np.mean(raw_error))
    adjusted_error = raw_error - offset
    return {
        "offset": offset,
        "validation_cells": int(matched.sum()),
        "raw_validation_bias": float(np.mean(raw_error)),
        "adjusted_validation_bias": float(np.mean(adjusted_error)),
        "raw_validation_rmse": float(np.sqrt(np.mean(np.square(raw_error)))),
        "adjusted_validation_rmse": float(
            np.sqrt(np.mean(np.square(adjusted_error)))
        ),
    }


def _save_float32(path: Path, values: np.ndarray) -> None:
    """Atomically save a float32 NumPy artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values, dtype=np.float32), allow_pickle=False)
    temporary.replace(path)


def _parameter_summary(parameters: pd.DataFrame) -> dict[str, Any]:
    """Return compact convergence and fitted-parameter diagnostics."""
    quantile_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    summary: dict[str, Any] = {
        "series": int(len(parameters)),
        "successful_fits": int((~parameters["used_fallback"]).sum()),
        "fallback_fits": int(parameters["used_fallback"].sum()),
        "fit_status_counts": {
            str(key): int(value)
            for key, value in parameters["fit_status"].value_counts().items()
        },
    }
    for column in ("persistence", "degrees_of_freedom"):
        values = parameters[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        summary[f"{column}_quantiles"] = {
            f"q{int(100 * level):02d}": float(value)
            for level, value in zip(
                quantile_levels,
                np.quantile(values, quantile_levels),
                strict=True,
            )
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_output", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--minimum-observations", type=int, default=120)
    parser.add_argument("--return-scale", type=float, default=100.0)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--component-block-size", type=int, default=256)
    parser.add_argument("--ray-num-cpus", type=int, default=15)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    parser.add_argument("--ray-debug", action="store_true")
    args = parser.parse_args()

    source_contract_path = args.source_output / "forecasts" / "locked_test_config.json"
    source_contract = json.loads(source_contract_path.read_text(encoding="utf-8"))
    if source_contract.get("target_type") != "log_squared_return":
        raise ValueError("Student-t GARCH benchmark requires the log-risk application.")
    panel = load_panel(args.source_output / "processed")
    split = relative_panel_split(
        panel.dates,
        validation_months=int(source_contract["validation_months"]),
        test_months=int(source_contract["test_months"]),
    )
    estimation_end_index = int(split.validation_indices[0]) - 1
    variance_floor = float(source_contract["variance_floor"])
    variance, parameters, execution = fit_student_t_garch11_panel(
        returns=panel.returns,
        estimation_end_index=estimation_end_index,
        minimum_observations=int(args.minimum_observations),
        return_scale=float(args.return_scale),
        variance_floor=variance_floor,
        max_iterations=int(args.max_iterations),
        tolerance=float(args.tolerance),
        component_block_size=int(args.component_block_size),
        ray_num_cpus=int(args.ray_num_cpus),
        max_cpus_per_task=int(args.max_cpus_per_task),
        ray_debug=bool(args.ray_debug),
    )
    parameter_summary = _parameter_summary(parameters)
    execution.update({
        "successful_fits": parameter_summary["successful_fits"],
        "fallback_fits": parameter_summary["fallback_fits"],
    })

    baseline_dir = args.source_output / "baseline"
    target = np.load(baseline_dir / "target.npy", mmap_mode="r")
    har = np.load(baseline_dir / "baseline_forecast.npy", mmap_mode="r")
    ar1 = np.load(baseline_dir / "ar1_forecast.npy", mmap_mode="r")
    gaussian_raw = np.load(baseline_dir / "garch11_forecast.npy", mmap_mode="r")
    expected_shape = (len(panel.dates), len(panel.component_ids))
    if variance.shape != expected_shape or any(
        values.shape != expected_shape for values in (target, har, ar1, gaussian_raw)
    ):
        raise ValueError("Benchmark arrays do not match the processed panel shape.")

    epsilon = float(source_contract["epsilon"])
    raw_log_forecast = np.asarray(
        np.log(epsilon + np.maximum(variance, variance_floor)),
        dtype=np.float32,
    )
    student_t_offset = estimate_validation_offset(
        target=target,
        forecast=raw_log_forecast,
        reference_forecast=har,
        validation_indices=split.validation_indices,
    )
    gaussian_offset = estimate_validation_offset(
        target=target,
        forecast=gaussian_raw,
        reference_forecast=har,
        validation_indices=split.validation_indices,
    )
    adjusted_student_t = np.asarray(
        raw_log_forecast - float(student_t_offset["offset"]),
        dtype=np.float32,
    )
    adjusted_gaussian = np.asarray(
        gaussian_raw - float(gaussian_offset["offset"]),
        dtype=np.float32,
    )

    forecasts = {
        "har_ridge": har,
        "ar1_ridge": ar1,
        GAUSSIAN_GARCH_OFFSET_MODEL: adjusted_gaussian,
        STUDENT_T_GARCH_OFFSET_MODEL: adjusted_student_t,
    }
    metrics, monthly, calibration = evaluate_risk_forecasts(
        target=target,
        returns=panel.returns,
        dates=panel.dates,
        forecasts=forecasts,
        validation_indices=split.validation_indices,
        test_indices=split.test_indices,
        benchmark_name="har_ridge",
        epsilon=epsilon,
        variance_floor=variance_floor,
    )

    forecast_dir = args.output_dir / "forecasts"
    metric_dir = args.output_dir / "metrics"
    forecast_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    _save_float32(forecast_dir / "garch11_t_variance_forecast.npy", variance)
    _save_float32(forecast_dir / "garch11_t_forecast.npy", raw_log_forecast)
    _save_float32(
        forecast_dir / "garch11_t_validation_offset_forecast.npy",
        adjusted_student_t,
    )
    parameters = parameters.copy()
    parameters.insert(1, "component_id", panel.component_ids)
    parameters.insert(2, "signalid", panel.signal_ids)
    parameters.to_parquet(
        forecast_dir / "garch11_t_parameters.parquet",
        index=False,
    )
    write_json(forecast_dir / "validation_offset.json", student_t_offset)
    metrics.to_csv(metric_dir / "metrics.csv", index=False)
    monthly.to_csv(metric_dir / "monthly_losses.csv", index=False)
    write_json(metric_dir / "variance_calibration.json", calibration)

    split_contract = {
        "universe_selection_end": panel.dates[estimation_end_index].date().isoformat(),
        "validation_start": panel.dates[int(split.validation_indices[0])].date().isoformat(),
        "validation_end": panel.dates[int(split.validation_indices[-1])].date().isoformat(),
        "test_start": panel.dates[int(split.test_indices[0])].date().isoformat(),
        "test_end": panel.dates[int(split.test_indices[-1])].date().isoformat(),
    }
    contract = {
        "source_output": args.source_output.resolve().as_posix(),
        "component_count": len(panel.component_ids),
        "model": "constant-mean GARCH(1,1)",
        "innovation_distribution": "variance-standardized Student-t",
        "estimation_library": "arch",
        "parameters_estimated_per_component": [
            "mean_return",
            "omega",
            "alpha",
            "beta",
            "degrees_of_freedom",
        ],
        "parameter_estimation_end": panel.dates[estimation_end_index].date().isoformat(),
        "parameters_frozen_during_validation_and_test": True,
        "post_estimation_variance_updated_with_observed_returns": True,
        "conditional_second_moment": "mean_return^2 + conditional_variance",
        "fit_missing_value_rule": "finite pre-validation returns in chronological order",
        "forecast_missing_lag_rule": "conditional expected squared innovation",
        "minimum_observations": int(args.minimum_observations),
        "return_optimization_scale": float(args.return_scale),
        "variance_floor": variance_floor,
        "max_iterations": int(args.max_iterations),
        "optimizer_tolerance": float(args.tolerance),
        "validation_offset": student_t_offset,
        "validation_offset_rule": (
            "pooled mean raw log-risk forecast error on matched validation cells"
        ),
        "gaussian_comparator_validation_offset": gaussian_offset,
        "execution": execution,
        "parameter_summary": parameter_summary,
        "split": split_contract,
        "input_fingerprints": {
            "returns": sha256_file(args.source_output / "processed" / "returns.npy"),
            "component_order": sha256_file(
                args.source_output / "processed" / "component_order.csv"
            ),
            "locked_test_config": sha256_file(source_contract_path),
        },
        "exploratory_post_test_method": True,
    }
    write_json(forecast_dir / "contract.json", contract)

    display_columns = [
        "model",
        "log_risk_rmse",
        "log_risk_mae",
        "log_risk_bias",
        "absolute_r2",
        "oos_r2_vs_benchmark",
    ]
    print(json.dumps({"execution": execution, "parameters": parameter_summary}, indent=2))
    print(metrics[display_columns].sort_values("log_risk_rmse").to_string(index=False))


if __name__ == "__main__":
    main()
