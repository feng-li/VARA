#!/usr/bin/env python3
"""Compare DataMine lambda grouping rules on a nested validation holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vara.datamine import (
    estimate_group_lambdas,
    estimate_series_lambdas,
    load_panel,
    relative_panel_split,
)
from vara.forecast_significance import diebold_mariano_test


GROUP_SPECS = (
    (),
    ("formula_type",),
    ("numerator",),
    ("denominator",),
    ("numerator", "formula_type"),
    ("denominator", "formula_type"),
)
ROLLING_GROUP_SPECS = ((), ("formula_type",), ("denominator",))
ROLLING_WINDOWS = (12, 24, 36)
SERIES_PRIOR_STRENGTHS = (0, 12, 24, 60, 120, 240)


def _contract(output_dir: Path) -> dict[str, object]:
    for name in ("validation_candidate_config.json", "locked_test_config.json"):
        path = output_dir / "forecasts" / name
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No forecast contract found under {output_dir}")


def compare_candidate(
    output_dir: Path,
    *,
    holdout_months: int,
    ridge: float,
    min_group_series: int,
) -> pd.DataFrame:
    contract = _contract(output_dir)
    panel = load_panel(output_dir / "processed")
    split = relative_panel_split(
        panel.dates,
        validation_months=int(contract["validation_months"]),
        test_months=int(contract["test_months"]),
    )
    if holdout_months < 1 or holdout_months >= len(split.validation_indices):
        raise ValueError("holdout_months must leave at least one lambda-training month.")
    training = split.validation_indices[:-int(holdout_months)]
    holdout = split.validation_indices[-int(holdout_months):]
    error = np.load(output_dir / "baseline" / "baseline_error.npy", mmap_mode="r")
    correction = np.load(output_dir / "forecasts" / "vara_correction.npy", mmap_mode="r")
    rows: list[dict[str, object]] = []

    for group_cols in GROUP_SPECS:
        lambdas, groups, global_info = estimate_group_lambdas(
            np.asarray(error)[training],
            np.asarray(correction)[training],
            panel.metadata,
            group_cols=group_cols,
            ridge=ridge,
            min_group_series=min_group_series,
        )
        holdout_error = np.asarray(error)[holdout]
        holdout_correction = np.asarray(correction)[holdout]
        valid = np.isfinite(holdout_error) & np.isfinite(holdout_correction)
        baseline_squared = np.where(valid, np.square(holdout_error), np.nan)
        final_squared = np.where(
            valid,
            np.square(holdout_error - holdout_correction * lambdas[None, :]),
            np.nan,
        )
        monthly_baseline = np.nanmean(baseline_squared, axis=1)
        monthly_final = np.nanmean(final_squared, axis=1)
        dm = diebold_mariano_test(
            monthly_final,
            monthly_baseline,
            forecast_horizon=1,
        )
        baseline_sse = float(np.nansum(baseline_squared))
        final_sse = float(np.nansum(final_squared))
        rows.append({
            "candidate": output_dir.name,
            "specification_id": str(contract["specification_id"]),
            "lambda_group_cols": "global" if not group_cols else "+".join(group_cols),
            "lambda_update_mode": "fixed",
            "lambda_window_months": int(len(training)),
            "lambda_training_start": panel.dates[int(training[0])].date().isoformat(),
            "lambda_training_end": panel.dates[int(training[-1])].date().isoformat(),
            "holdout_start": panel.dates[int(holdout[0])].date().isoformat(),
            "holdout_end": panel.dates[int(holdout[-1])].date().isoformat(),
            "holdout_months": int(len(holdout)),
            "matched_cells": int(valid.sum()),
            "oos_r2_vs_baseline": 1.0 - final_sse / baseline_sse,
            "baseline_rmse": float(np.sqrt(baseline_sse / valid.sum())),
            "vara_rmse": float(np.sqrt(final_sse / valid.sum())),
            "lambda_global": float(global_info["lambda"]),
            "lambda_series_mean": float(np.mean(lambdas)),
            "lambda_series_median": float(np.median(lambdas)),
            "lambda_series_min": float(np.min(lambdas)),
            "lambda_series_max": float(np.max(lambdas)),
            "zero_lambda_series_fraction": float(np.mean(lambdas == 0.0)),
            "groups": int(len(groups)),
            "fallback_groups": int(groups["fallback_global"].sum()),
            "dm_statistic_raw": float(dm["dm_statistic_raw"]),
            "dm_p_value_raw_two_sided": float(dm["p_value_raw_two_sided"]),
        })

    for group_cols in ROLLING_GROUP_SPECS:
        for window in ROLLING_WINDOWS:
            monthly_baseline: list[float] = []
            monthly_final: list[float] = []
            baseline_sse = 0.0
            final_sse = 0.0
            lambda_means: list[float] = []
            lambda_medians: list[float] = []
            lambda_minima: list[float] = []
            lambda_maxima: list[float] = []
            lambda_zero_fractions: list[float] = []
            fallback_groups = 0
            matched_cells = 0
            for target_index in holdout:
                prior_validation = split.validation_indices[
                    split.validation_indices < int(target_index)
                ]
                lambda_training = prior_validation[-int(window):]
                lambdas, groups, _ = estimate_group_lambdas(
                    np.asarray(error)[lambda_training],
                    np.asarray(correction)[lambda_training],
                    panel.metadata,
                    group_cols=group_cols,
                    ridge=ridge,
                    min_group_series=min_group_series,
                )
                target_error = np.asarray(error)[int(target_index)]
                target_correction = np.asarray(correction)[int(target_index)]
                valid = np.isfinite(target_error) & np.isfinite(target_correction)
                baseline_squared = np.square(target_error[valid])
                final_squared = np.square(
                    target_error[valid] - target_correction[valid] * lambdas[valid]
                )
                monthly_baseline.append(float(np.mean(baseline_squared)))
                monthly_final.append(float(np.mean(final_squared)))
                baseline_sse += float(np.sum(baseline_squared))
                final_sse += float(np.sum(final_squared))
                matched_cells += int(valid.sum())
                lambda_means.append(float(np.mean(lambdas)))
                lambda_medians.append(float(np.median(lambdas)))
                lambda_minima.append(float(np.min(lambdas)))
                lambda_maxima.append(float(np.max(lambdas)))
                lambda_zero_fractions.append(float(np.mean(lambdas == 0.0)))
                fallback_groups += int(groups["fallback_global"].sum())
            baseline_losses = np.asarray(monthly_baseline)
            final_losses = np.asarray(monthly_final)
            dm = diebold_mariano_test(
                final_losses,
                baseline_losses,
                forecast_horizon=1,
            )
            rows.append({
                "candidate": output_dir.name,
                "specification_id": str(contract["specification_id"]),
                "lambda_group_cols": "global" if not group_cols else "+".join(group_cols),
                "lambda_update_mode": "rolling",
                "lambda_window_months": int(window),
                "lambda_training_start": "rolling",
                "lambda_training_end": "target_minus_one",
                "holdout_start": panel.dates[int(holdout[0])].date().isoformat(),
                "holdout_end": panel.dates[int(holdout[-1])].date().isoformat(),
                "holdout_months": int(len(holdout)),
                "matched_cells": matched_cells,
                "oos_r2_vs_baseline": 1.0 - final_sse / baseline_sse,
                "baseline_rmse": float(np.sqrt(baseline_sse / matched_cells)),
                "vara_rmse": float(np.sqrt(final_sse / matched_cells)),
                "lambda_global": np.nan,
                "lambda_series_mean": float(np.mean(lambda_means)),
                "lambda_series_median": float(np.mean(lambda_medians)),
                "lambda_series_min": float(np.min(lambda_minima)),
                "lambda_series_max": float(np.max(lambda_maxima)),
                "zero_lambda_series_fraction": float(np.mean(lambda_zero_fractions)),
                "groups": 1 if not group_cols else int(panel.metadata.groupby(list(group_cols)).ngroups),
                "fallback_groups": fallback_groups,
                "dm_statistic_raw": float(dm["dm_statistic_raw"]),
                "dm_p_value_raw_two_sided": float(dm["p_value_raw_two_sided"]),
            })

    for prior_strength in SERIES_PRIOR_STRENGTHS:
        lambdas, info = estimate_series_lambdas(
            np.asarray(error)[training],
            np.asarray(correction)[training],
            ridge=ridge,
            prior_strength_months=float(prior_strength),
        )
        holdout_error = np.asarray(error)[holdout]
        holdout_correction = np.asarray(correction)[holdout]
        valid = np.isfinite(holdout_error) & np.isfinite(holdout_correction)
        baseline_squared = np.where(valid, np.square(holdout_error), np.nan)
        final_squared = np.where(
            valid,
            np.square(holdout_error - holdout_correction * lambdas[None, :]),
            np.nan,
        )
        monthly_baseline = np.nanmean(baseline_squared, axis=1)
        monthly_final = np.nanmean(final_squared, axis=1)
        dm = diebold_mariano_test(monthly_final, monthly_baseline, forecast_horizon=1)
        baseline_sse = float(np.nansum(baseline_squared))
        final_sse = float(np.nansum(final_squared))
        rows.append({
            "candidate": output_dir.name,
            "specification_id": str(contract["specification_id"]),
            "lambda_group_cols": "series",
            "lambda_update_mode": "series_shrunk",
            "lambda_window_months": int(len(training)),
            "series_prior_strength_months": int(prior_strength),
            "lambda_training_start": panel.dates[int(training[0])].date().isoformat(),
            "lambda_training_end": panel.dates[int(training[-1])].date().isoformat(),
            "holdout_start": panel.dates[int(holdout[0])].date().isoformat(),
            "holdout_end": panel.dates[int(holdout[-1])].date().isoformat(),
            "holdout_months": int(len(holdout)),
            "matched_cells": int(valid.sum()),
            "oos_r2_vs_baseline": 1.0 - final_sse / baseline_sse,
            "baseline_rmse": float(np.sqrt(baseline_sse / valid.sum())),
            "vara_rmse": float(np.sqrt(final_sse / valid.sum())),
            "lambda_global": float(info["lambda"]),
            "lambda_series_mean": float(np.mean(lambdas)),
            "lambda_series_median": float(np.median(lambdas)),
            "lambda_series_min": float(np.min(lambdas)),
            "lambda_series_max": float(np.max(lambdas)),
            "zero_lambda_series_fraction": float(np.mean(lambdas == 0.0)),
            "groups": int(len(lambdas)),
            "fallback_groups": 0,
            "dm_statistic_raw": float(dm["dm_statistic_raw"]),
            "dm_p_value_raw_two_sided": float(dm["p_value_raw_two_sided"]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dirs", nargs="+", type=Path)
    parser.add_argument("--holdout-months", type=int, default=24)
    parser.add_argument("--lambda-ridge", type=float, default=0.001)
    parser.add_argument("--min-group-series", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = pd.concat([
        compare_candidate(
            path,
            holdout_months=args.holdout_months,
            ridge=args.lambda_ridge,
            min_group_series=args.min_group_series,
        )
        for path in args.output_dirs
    ], ignore_index=True)
    result = result.sort_values("oos_r2_vs_baseline", ascending=False, ignore_index=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
