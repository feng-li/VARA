#!/usr/bin/env python3
"""Evaluate a predeclared rolling-lambda rule from saved DataMine corrections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vara.datamine import (
    evaluate_risk_forecasts,
    load_panel,
    relative_panel_split,
    rolling_group_lambdas,
    write_json,
)
from vara.forecast_significance import diebold_mariano_test


def _parse_columns(value: str) -> tuple[str, ...]:
    if str(value).strip().lower() in {"", "global", "none"}:
        return ()
    return tuple(item.strip() for item in str(value).split(",") if item.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--group-cols", type=_parse_columns, default=("denominator",))
    parser.add_argument("--window-months", type=int, default=24)
    parser.add_argument("--min-history-months", type=int, default=24)
    parser.add_argument("--min-group-series", type=int, default=20)
    parser.add_argument("--lambda-ridge", type=float, default=0.001)
    args = parser.parse_args()

    contract_path = args.output_dir / "forecasts" / "locked_test_config.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    panel = load_panel(args.output_dir / "processed")
    split = relative_panel_split(
        panel.dates,
        validation_months=int(contract["validation_months"]),
        test_months=int(contract["test_months"]),
    )
    target = np.load(args.output_dir / "baseline" / "target.npy", mmap_mode="r")
    baseline = np.load(args.output_dir / "baseline" / "baseline_forecast.npy", mmap_mode="r")
    error = np.load(args.output_dir / "baseline" / "baseline_error.npy", mmap_mode="r")
    correction = np.load(args.output_dir / "forecasts" / "vara_correction.npy", mmap_mode="r")
    evaluation = np.concatenate((split.validation_indices, split.test_indices))
    lambdas, lambda_table = rolling_group_lambdas(
        error,
        correction,
        panel.metadata,
        evaluation,
        window_months=args.window_months,
        min_history_months=args.min_history_months,
        group_cols=args.group_cols,
        ridge=args.lambda_ridge,
        min_group_series=args.min_group_series,
    )
    rolling_forecast = np.asarray(baseline, dtype=float) + np.nan_to_num(lambdas) * np.asarray(
        correction, dtype=float
    )
    forecasts = {"baseline": baseline, "vara_rolling_lambda": rolling_forecast}
    metrics, monthly, calibration = evaluate_risk_forecasts(
        target=target,
        returns=panel.returns,
        dates=panel.dates,
        forecasts=forecasts,
        validation_indices=split.validation_indices,
        test_indices=split.test_indices,
        benchmark_name="baseline",
        epsilon=float(contract["epsilon"]),
        variance_floor=float(contract["variance_floor"]),
    )
    focal = monthly.loc[monthly["model"].eq("vara_rolling_lambda")]
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
    label = "global" if not args.group_cols else "_".join(args.group_cols)
    artifact_dir = args.output_dir / "metrics" / f"rolling_lambda_{label}_{args.window_months}m"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    monthly.to_csv(artifact_dir / "monthly_losses.csv", index=False)
    pd.DataFrame(dm_rows).to_csv(artifact_dir / "diebold_mariano_raw.csv", index=False)
    lambda_table.to_parquet(artifact_dir / "lambdas.parquet", index=False)
    np.save(artifact_dir / "lambda_values.npy", lambdas.astype(np.float32), allow_pickle=False)
    write_json(artifact_dir / "contract.json", {
        "source_locked_test_config": contract_path.as_posix(),
        "group_cols": list(args.group_cols),
        "window_months": args.window_months,
        "min_history_months": args.min_history_months,
        "min_group_series": args.min_group_series,
        "lambda_ridge": args.lambda_ridge,
        "target_month_excluded_from_lambda_training": True,
        "variance_calibration": calibration,
    })
    print(metrics.to_string(index=False))
    print(pd.DataFrame(dm_rows).to_string(index=False))


if __name__ == "__main__":
    main()
