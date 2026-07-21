#!/usr/bin/env python3
"""Fit and evaluate a bounded ensemble of saved DataMine VARA corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from vara.datamine import evaluate_risk_forecasts, load_panel, relative_panel_split, write_json
from vara.forecast_significance import diebold_mariano_test


def _fit_coefficients(
    error: np.ndarray,
    corrections: list[np.ndarray],
    indices: np.ndarray,
    *,
    ridge: float,
) -> tuple[np.ndarray, int]:
    response = np.asarray(error)[indices].reshape(-1)
    design = np.column_stack([
        np.asarray(correction)[indices].reshape(-1) for correction in corrections
    ])
    valid = np.isfinite(response) & np.isfinite(design).all(axis=1)
    y = response[valid]
    X = design[valid]
    xtx = X.T @ X
    xty = X.T @ y

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        value = 0.5 * values @ xtx @ values - values @ xty + 0.5 * ridge * (values @ values)
        gradient = xtx @ values - xty + ridge * values
        return float(value), gradient

    result = minimize(
        objective,
        np.zeros(len(corrections), dtype=float),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, 1.0)] * len(corrections),
    )
    if not result.success:
        raise RuntimeError(result.message)
    return np.asarray(result.x, dtype=float), int(valid.sum())


def _period_metrics(
    error: np.ndarray,
    corrections: list[np.ndarray],
    coefficients: np.ndarray,
    indices: np.ndarray,
) -> dict[str, float]:
    response = np.asarray(error)[indices]
    adjustment = sum(
        coefficient * np.asarray(correction)[indices]
        for coefficient, correction in zip(coefficients, corrections, strict=True)
    )
    valid = np.isfinite(response) & np.isfinite(adjustment)
    baseline_sse = float(np.sum(np.square(response[valid])))
    final_sse = float(np.sum(np.square(response[valid] - adjustment[valid])))
    return {
        "cells": int(valid.sum()),
        "baseline_rmse": float(np.sqrt(baseline_sse / valid.sum())),
        "vara_rmse": float(np.sqrt(final_sse / valid.sum())),
        "oos_r2_vs_baseline": 1.0 - final_sse / baseline_sse,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dirs", nargs="+", type=Path)
    parser.add_argument("--holdout-months", type=int, default=24)
    parser.add_argument("--ridge", type=float, default=0.001)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.output_dirs[0]
    contract = json.loads(
        (source / "forecasts" / "locked_test_config.json").read_text(encoding="utf-8")
        if (source / "forecasts" / "locked_test_config.json").exists()
        else (source / "forecasts" / "validation_candidate_config.json").read_text(encoding="utf-8")
    )
    panel = load_panel(source / "processed")
    split = relative_panel_split(
        panel.dates,
        validation_months=int(contract["validation_months"]),
        test_months=int(contract["test_months"]),
    )
    error = np.load(source / "baseline" / "baseline_error.npy", mmap_mode="r")
    corrections = [
        np.load(path / "forecasts" / "vara_correction.npy", mmap_mode="r")
        for path in args.output_dirs
    ]
    model_names = [path.name for path in args.output_dirs]
    nested_training = split.validation_indices[:-int(args.holdout_months)]
    nested_holdout = split.validation_indices[-int(args.holdout_months):]
    nested_coefficients, nested_cells = _fit_coefficients(
        error,
        corrections,
        nested_training,
        ridge=args.ridge,
    )
    nested_metrics = _period_metrics(
        error,
        corrections,
        nested_coefficients,
        nested_holdout,
    )
    final_training = split.validation_indices[-int(args.holdout_months):]
    final_coefficients, final_training_cells = _fit_coefficients(
        error,
        corrections,
        final_training,
        ridge=args.ridge,
    )
    result: dict[str, object] = {
        "models": model_names,
        "ridge": args.ridge,
        "nested_training_cells": nested_cells,
        "nested_coefficients": nested_coefficients.tolist(),
        "nested_holdout": nested_metrics,
        "final_training_cells": final_training_cells,
        "final_coefficients": final_coefficients.tolist(),
        "test_evaluated": bool(args.evaluate_test),
    }
    artifact_dir = args.output_dir or source / "metrics" / "vara_ensemble"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if args.evaluate_test:
        target = np.load(source / "baseline" / "target.npy", mmap_mode="r")
        baseline = np.load(source / "baseline" / "baseline_forecast.npy", mmap_mode="r")
        adjustment = sum(
            coefficient * np.asarray(correction)
            for coefficient, correction in zip(final_coefficients, corrections, strict=True)
        )
        ensemble = np.asarray(baseline, dtype=float) + adjustment
        metrics, monthly, calibration = evaluate_risk_forecasts(
            target=target,
            returns=panel.returns,
            dates=panel.dates,
            forecasts={"baseline": baseline, "vara_ensemble": ensemble},
            validation_indices=split.validation_indices,
            test_indices=split.test_indices,
            benchmark_name="baseline",
            epsilon=float(contract["epsilon"]),
            variance_floor=float(contract["variance_floor"]),
        )
        focal = monthly.loc[monthly["model"].eq("vara_ensemble")]
        comparator = monthly.loc[monthly["model"].eq("baseline")]
        dm_rows = []
        for loss in ("squared_error", "gaussian_return_crps"):
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
        result["test_metrics"] = metrics.to_dict(orient="records")
        result["variance_calibration"] = calibration
        metrics.to_csv(artifact_dir / "metrics.csv", index=False)
        monthly.to_csv(artifact_dir / "monthly_losses.csv", index=False)
        pd.DataFrame(dm_rows).to_csv(artifact_dir / "diebold_mariano_raw.csv", index=False)
        print(metrics.to_string(index=False))
        print(pd.DataFrame(dm_rows).to_string(index=False))
    write_json(artifact_dir / "ensemble_contract.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
