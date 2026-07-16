#!/usr/bin/env python3
"""Evaluate direct and purpose-aggregated AutoETS forecasts for Tourism.

The Tourism hierarchy contains an exact identity for every geographic series::

    All = Hol + Vis + Bus + Oth

This script fits AutoETS independently to the direct ``All`` series and the
four purpose components at rolling one-step origins.  It saves the resulting
forecast panel so later VARA experiments can use purpose-level residuals
without refitting the baselines.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from vara.forecast_baselines import auto_ets_baseline_panel


PURPOSES = ("Hol", "Vis", "Bus", "Oth")


def _detailed_geo_codes(columns: Iterable[str], depth: int) -> list[str]:
    codes = [str(column)[:-3] for column in columns if str(column).endswith("All")]
    return sorted(code for code in codes if len(code) == int(depth))


def _validate_purpose_identity(wide: pd.DataFrame, geo_codes: list[str]) -> float:
    errors = []
    for code in geo_codes:
        component_sum = sum(wide[f"{code}{purpose}"].to_numpy(dtype=float) for purpose in PURPOSES)
        errors.append(wide[f"{code}All"].to_numpy(dtype=float) - component_sum)
    max_abs_error = float(np.max(np.abs(np.concatenate(errors))))
    if max_abs_error > 1e-8:
        raise ValueError(
            "Tourism purpose components do not sum to All; "
            f"maximum absolute identity error is {max_abs_error:.6g}."
        )
    return max_abs_error


def _forecast_origin_task(
    history: np.ndarray,
    series_ids: list[str],
    n_geo: int,
    origin_index: int,
    target_date: str,
    history_end: str,
    season_length: int,
    clip_nonnegative: bool,
) -> dict:
    history = np.asarray(history, dtype=float)
    target_timestamp = pd.Timestamp(target_date)
    history_end_timestamp = pd.Timestamp(history_end)
    if history.ndim != 2 or history.shape[1] != len(series_ids):
        raise ValueError("AutoETS history must have shape (T, n_series).")
    if len(history) != int(origin_index):
        raise ValueError("AutoETS history length must equal the target origin index.")
    if history_end_timestamp >= target_timestamp:
        raise ValueError("AutoETS history must end strictly before the target date.")
    baseline = auto_ets_baseline_panel(
        history,
        horizon=1,
        seasonal_lag=int(season_length),
        series_ids=series_ids,
    )
    forecast = np.asarray(baseline["forecast"][0], dtype=float)
    if clip_nonnegative:
        forecast = np.maximum(forecast, 0.0)

    direct = forecast[:n_geo]
    components = forecast[n_geo:].reshape(len(PURPOSES), n_geo)
    return {
        "origin_index": int(origin_index),
        "date": str(target_timestamp.date()),
        "history_end": str(history_end_timestamp.date()),
        "n_train": int(len(history)),
        "direct_auto_ets": direct,
        "purpose_sum_auto_ets": components.sum(axis=0),
        "purpose_forecasts": components,
    }


def _closed_form_weight(frame: pd.DataFrame) -> float:
    correction = (
        frame["purpose_sum_auto_ets"].to_numpy(dtype=float)
        - frame["direct_auto_ets"].to_numpy(dtype=float)
    )
    target = (
        frame["actual"].to_numpy(dtype=float)
        - frame["direct_auto_ets"].to_numpy(dtype=float)
    )
    denominator = float(correction @ correction)
    if denominator <= 1e-12:
        return 0.0
    return float(np.clip((correction @ target) / denominator, 0.0, 1.0))


def _learn_weights(
    train: pd.DataFrame,
    group_columns: tuple[str, ...],
) -> pd.DataFrame:
    if not group_columns:
        return pd.DataFrame({"weight": [_closed_form_weight(train)]})
    rows = []
    grouper = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for group_key, group in train.groupby(grouper, sort=True, observed=True):
        keys = (group_key,) if len(group_columns) == 1 else tuple(group_key)
        row = {column: value for column, value in zip(group_columns, keys, strict=True)}
        row.update({"weight": _closed_form_weight(group), "n_train": int(len(group))})
        rows.append(row)
    return pd.DataFrame(rows)


def _apply_blend(
    frame: pd.DataFrame,
    weights: pd.DataFrame,
    group_columns: tuple[str, ...],
    output_column: str,
) -> pd.DataFrame:
    out = frame.copy()
    if group_columns:
        out = out.merge(weights, on=list(group_columns), how="left", validate="many_to_one")
    else:
        out["weight"] = float(weights["weight"].iloc[0])
    if out["weight"].isna().any():
        raise ValueError(f"Missing blend weights while constructing {output_column}.")
    out[output_column] = out["direct_auto_ets"] + out["weight"] * (
        out["purpose_sum_auto_ets"] - out["direct_auto_ets"]
    )
    return out.drop(columns="weight")


def _metric_row(frame: pd.DataFrame, method: str, prediction_column: str, period: str) -> dict:
    error = frame[prediction_column].to_numpy(dtype=float) - frame["actual"].to_numpy(dtype=float)
    return {
        "period": period,
        "method": method,
        "n_observations": int(error.size),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def _evaluate_methods(frame: pd.DataFrame, period: str) -> list[dict]:
    methods = {
        "direct_auto_ets": "direct_auto_ets",
        "purpose_sum_auto_ets": "purpose_sum_auto_ets",
        "global_blend": "global_blend",
        "group_blend": "group_blend",
        "month_blend": "month_blend",
        "group_month_blend": "group_month_blend",
    }
    return [_metric_row(frame, method, column, period) for method, column in methods.items()]


def _fit_and_apply_all_blends(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    specifications = {
        "global_blend": (),
        "group_blend": ("group",),
        "month_blend": ("month",),
        "group_month_blend": ("group", "month"),
    }
    out = test.copy()
    learned = {}
    for output_column, group_columns in specifications.items():
        weights = _learn_weights(train, group_columns)
        learned[output_column] = weights
        out = _apply_blend(out, weights, group_columns, output_column)
    return out, learned


def _nested_historical_evaluation(frame: pd.DataFrame, first_year: int, final_year: int) -> pd.DataFrame:
    rows = []
    for year in range(int(first_year) + 1, int(final_year)):
        train = frame[frame["year"] < year]
        test = frame[frame["year"] == year]
        if train.empty or test.empty:
            continue
        scored, _ = _fit_and_apply_all_blends(train, test)
        rows.extend(_evaluate_methods(scored, str(year)))
    return pd.DataFrame(rows)


def run_evaluation(
    *,
    data_path: Path,
    output_dir: Path,
    first_year: int,
    final_year: int,
    series_depth: int,
    season_length: int,
    ray_num_cpus: int,
    max_cpus_per_task: int,
    clip_nonnegative: bool,
) -> dict:
    wide = pd.read_csv(data_path)
    if "date" not in wide.columns:
        raise ValueError(f"Expected a date column in {data_path}.")
    wide["date"] = pd.to_datetime(wide["date"])
    geo_codes = _detailed_geo_codes(wide.columns, series_depth)
    if not geo_codes:
        raise ValueError(f"No Tourism All series found at depth {series_depth}.")

    all_ids = [f"{code}All" for code in geo_codes]
    component_ids = [f"{code}{purpose}" for purpose in PURPOSES for code in geo_codes]
    series_ids = [*all_ids, *component_ids]
    values = wide.loc[:, series_ids].to_numpy(dtype=float)
    dates = wide["date"].to_numpy()
    origin_indices = [
        i
        for i, date in enumerate(wide["date"])
        if int(first_year) <= int(date.year) <= int(final_year)
    ]
    if not origin_indices:
        raise ValueError("No forecast origins fall within the requested year range.")
    if min(origin_indices) < 1:
        raise ValueError("A rolling AutoETS origin requires prior history.")
    identity_error = _validate_purpose_identity(
        wide.iloc[: min(origin_indices)],
        geo_codes,
    )

    start = time.time()
    if int(ray_num_cpus) > 1:
        import ray

        started_here = False
        if not ray.is_initialized():
            ray.init(
                num_cpus=int(ray_num_cpus),
                include_dashboard=False,
                ignore_reinit_error=True,
                _node_ip_address="127.0.0.1",
            )
            started_here = True
        try:
            remote_task = ray.remote(num_cpus=int(max_cpus_per_task))(_forecast_origin_task)
            series_ids_ref = ray.put(series_ids)
            futures = []
            for origin_index in origin_indices:
                history_ref = ray.put(np.asarray(values[:origin_index], dtype=float).copy())
                futures.append(remote_task.remote(
                    history_ref,
                    series_ids_ref,
                    len(geo_codes),
                    origin_index,
                    str(pd.Timestamp(dates[origin_index]).date()),
                    str(pd.Timestamp(dates[origin_index - 1]).date()),
                    int(season_length),
                    bool(clip_nonnegative),
                ))
            print(f"Submitted {len(futures)} rolling AutoETS origins to Ray.", flush=True)
            results = []
            pending = list(futures)
            while pending:
                ready, pending = ray.wait(pending, num_returns=1, timeout=60.0)
                if not ready:
                    print(
                        f"Waiting for AutoETS origins: done={len(results)}/{len(futures)}, "
                        f"pending={len(pending)}",
                        flush=True,
                    )
                    continue
                results.append(ray.get(ready[0]))
                if len(results) % 6 == 0 or not pending:
                    print(f"Collected {len(results)}/{len(futures)} origins.", flush=True)
        finally:
            if started_here:
                ray.shutdown()
    else:
        results = [
            _forecast_origin_task(
                np.asarray(values[:origin_index], dtype=float).copy(),
                series_ids,
                len(geo_codes),
                origin_index,
                str(pd.Timestamp(dates[origin_index]).date()),
                str(pd.Timestamp(dates[origin_index - 1]).date()),
                int(season_length),
                bool(clip_nonnegative),
            )
            for origin_index in origin_indices
        ]

    records = []
    for result in sorted(results, key=lambda item: item["origin_index"]):
        date = pd.Timestamp(result["date"])
        origin_index = int(result["origin_index"])
        for j, code in enumerate(geo_codes):
            row = {
                "date": date,
                "year": int(date.year),
                "month": int(date.month),
                "series_id": f"{code}All",
                "group": code[0],
                "actual": float(values[origin_index, j]),
                "direct_auto_ets": float(result["direct_auto_ets"][j]),
                "purpose_sum_auto_ets": float(result["purpose_sum_auto_ets"][j]),
                "autoets_history_end": result["history_end"],
                "autoets_n_train": int(result["n_train"]),
            }
            for k, purpose in enumerate(PURPOSES):
                row[f"forecast_{purpose.lower()}"] = float(result["purpose_forecasts"][k, j])
            records.append(row)
    forecasts = pd.DataFrame(records)

    historical = forecasts[forecasts["year"] < int(final_year)]
    final = forecasts[forecasts["year"] == int(final_year)]
    final_scored, final_weights = _fit_and_apply_all_blends(historical, final)
    nested = _nested_historical_evaluation(forecasts, first_year, final_year)

    summary_rows = []
    historical_scored, _ = _fit_and_apply_all_blends(historical, historical)
    summary_rows.extend(_evaluate_methods(historical_scored, f"{first_year}-{final_year - 1}_in_sample_weights"))
    summary_rows.extend(_evaluate_methods(final_scored, str(final_year)))
    summary = pd.DataFrame(summary_rows)
    direct_rmse = summary.groupby("period", observed=True).apply(
        lambda group: float(group.loc[group["method"] == "direct_auto_ets", "rmse"].iloc[0]),
        include_groups=False,
    )
    summary["rmse_improvement_vs_direct_pct"] = [
        100.0 * (direct_rmse.loc[period] - rmse) / direct_rmse.loc[period]
        for period, rmse in zip(summary["period"], summary["rmse"], strict=True)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output_dir / "rolling_forecasts.csv", index=False)
    final_scored.to_csv(output_dir / "final_scored_forecasts.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    nested.to_csv(output_dir / "nested_historical_metrics.csv", index=False)
    for name, weights in final_weights.items():
        weights.to_csv(output_dir / f"weights_{name}.csv", index=False)

    config = {
        "data_path": str(data_path),
        "output_dir": str(output_dir),
        "first_year": int(first_year),
        "final_year": int(final_year),
        "series_depth": int(series_depth),
        "season_length": int(season_length),
        "purposes": list(PURPOSES),
        "n_geo_series": len(geo_codes),
        "n_forecast_origins": len(origin_indices),
        "clip_nonnegative": bool(clip_nonnegative),
        "ray_num_cpus": int(ray_num_cpus),
        "max_cpus_per_task": int(max_cpus_per_task),
        "leakage_control": "each AutoETS worker receives observations through t-1 only",
        "purpose_identity_max_abs_error": identity_error,
        "purpose_identity_validation_scope": "history before the first forecast origin",
        "elapsed_seconds": float(time.time() - start),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    return {
        "forecasts": forecasts,
        "final_scored": final_scored,
        "summary": summary,
        "nested": nested,
        "weights": final_weights,
        "config": config,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/tourism/tourism.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_purpose_autoets_rolling"),
    )
    parser.add_argument("--first-year", type=int, default=2011)
    parser.add_argument("--final-year", type=int, default=2016)
    parser.add_argument("--series-depth", type=int, default=3)
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--ray-num-cpus", type=int, default=1)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    parser.add_argument("--clip-nonnegative", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_evaluation(
        data_path=args.data_path,
        output_dir=args.output_dir,
        first_year=args.first_year,
        final_year=args.final_year,
        series_depth=args.series_depth,
        season_length=args.season_length,
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
        clip_nonnegative=args.clip_nonnegative,
    )


if __name__ == "__main__":
    main()
