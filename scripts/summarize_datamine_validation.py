#!/usr/bin/env python3
"""Compare DataMine forecast candidates on the validation period only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from vara.forecast_significance import diebold_mariano_test


def summarize_candidate(output_dir: Path) -> dict[str, object]:
    path = output_dir / "forecasts" / "forecasts.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_parquet(
        path,
        columns=(
            "target_date",
            "actual_target",
            "baseline_forecast",
            "final_forecast",
            "lambda",
            "specification_id",
            "evaluation_period",
        ),
        filters=[("evaluation_period", "==", "validation")],
    )
    valid = np.isfinite(frame["actual_target"]) & np.isfinite(frame["baseline_forecast"]) & np.isfinite(
        frame["final_forecast"]
    )
    matched = frame.loc[valid].copy()
    matched["baseline_squared_error"] = np.square(
        matched["baseline_forecast"] - matched["actual_target"]
    )
    matched["final_squared_error"] = np.square(
        matched["final_forecast"] - matched["actual_target"]
    )
    monthly = matched.groupby("target_date", sort=True)[
        ["baseline_squared_error", "final_squared_error"]
    ].mean()
    dm = diebold_mariano_test(
        monthly["final_squared_error"],
        monthly["baseline_squared_error"],
        forecast_horizon=1,
    )
    baseline_sse = float(matched["baseline_squared_error"].sum())
    final_sse = float(matched["final_squared_error"].sum())
    return {
        "candidate": output_dir.name,
        "specification_id": str(matched["specification_id"].iloc[0]),
        "validation_months": int(len(monthly)),
        "matched_cells": int(len(matched)),
        "lambda": float(matched["lambda"].iloc[0]),
        "baseline_rmse": float(np.sqrt(baseline_sse / len(matched))),
        "vara_rmse": float(np.sqrt(final_sse / len(matched))),
        "oos_r2_vs_baseline": 1.0 - final_sse / baseline_sse,
        "dm_statistic_raw": float(dm["dm_statistic_raw"]),
        "dm_p_value_raw_two_sided": float(dm["p_value_raw_two_sided"]),
        "dm_statistic_corrected": float(dm["dm_statistic"]),
        "dm_p_value_corrected_two_sided": float(dm["p_value_two_sided"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    summary = pd.DataFrame(summarize_candidate(path) for path in args.output_dirs)
    summary = summary.sort_values("oos_r2_vs_baseline", ascending=False, ignore_index=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
