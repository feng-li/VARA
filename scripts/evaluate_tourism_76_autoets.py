#!/usr/bin/env python3
"""Generate long-history rolling AutoETS forecasts for 76 Tourism totals."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from _bootstrap import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT  # noqa: F401
from vara.forecast_baselines import auto_ets_baseline_panel


def _all_series_ids(columns: list[str], series_depth: int) -> list[str]:
    ids = sorted(
        str(column)
        for column in columns
        if str(column).endswith("All")
        and len(str(column)[:-3]) == int(series_depth)
    )
    if not ids:
        raise ValueError(
            f"No all-purpose Tourism series found at depth {series_depth}."
        )
    return ids


def _forecast_origin(
    history: np.ndarray,
    series_ids: list[str],
    origin_index: int,
    target_date: str,
    history_end: str,
    season_length: int,
    clip_nonnegative: bool,
) -> dict:
    values = np.asarray(history, dtype=float)
    target = pd.Timestamp(target_date)
    information_end = pd.Timestamp(history_end)
    if values.ndim != 2 or values.shape[1] != len(series_ids):
        raise ValueError("AutoETS history must have shape (T, 76).")
    if len(values) != int(origin_index):
        raise ValueError("AutoETS history length must equal the origin index.")
    if information_end >= target:
        raise ValueError("AutoETS history must end before the target date.")
    result = auto_ets_baseline_panel(
        values,
        horizon=1,
        seasonal_lag=int(season_length),
        series_ids=series_ids,
    )
    forecast = np.asarray(result["forecast"][0], dtype=float)
    if forecast.shape != (len(series_ids),):
        raise ValueError("AutoETS returned an unexpected forecast shape.")
    if clip_nonnegative:
        forecast = np.maximum(forecast, 0.0)
    return {
        "origin_index": int(origin_index),
        "date": str(target.date()),
        "history_end": str(information_end.date()),
        "n_train": int(len(values)),
        "forecast": forecast,
    }


def run_backtest(
    *,
    data_path: Path,
    output_path: Path,
    first_year: int,
    final_year: int,
    series_depth: int,
    expected_n_series: int,
    season_length: int,
    ray_num_cpus: int,
    max_cpus_per_task: int,
    clip_nonnegative: bool,
) -> pd.DataFrame:
    if int(first_year) > int(final_year):
        raise ValueError("first_year must not exceed final_year.")
    data = pd.read_csv(data_path)
    if "date" not in data.columns:
        raise ValueError(f"Expected a date column in {data_path}.")
    data["date"] = pd.to_datetime(data["date"])
    if data["date"].duplicated().any() or not data["date"].is_monotonic_increasing:
        raise ValueError("Tourism dates must be unique and increasing.")
    series_ids = _all_series_ids(list(data.columns), series_depth)
    if len(series_ids) != int(expected_n_series):
        raise ValueError(
            f"Expected {expected_n_series} all-purpose series, found {len(series_ids)}."
        )
    values = data.loc[:, series_ids].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Tourism observations contain non-finite values.")
    origin_indices = [
        index
        for index, date in enumerate(data["date"])
        if int(first_year) <= int(date.year) <= int(final_year)
    ]
    if not origin_indices or min(origin_indices) < 2:
        raise ValueError("The requested origins need at least two prior observations.")

    dates = data["date"].to_numpy()
    started = time.time()
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
            remote = ray.remote(num_cpus=int(max_cpus_per_task))(_forecast_origin)
            series_ref = ray.put(series_ids)
            futures = []
            for origin_index in origin_indices:
                history_ref = ray.put(values[:origin_index].copy())
                futures.append(remote.remote(
                    history_ref,
                    series_ref,
                    origin_index,
                    str(pd.Timestamp(dates[origin_index]).date()),
                    str(pd.Timestamp(dates[origin_index - 1]).date()),
                    int(season_length),
                    bool(clip_nonnegative),
                ))
            print(
                f"Submitted {len(futures)} rolling 76-series AutoETS origins.",
                flush=True,
            )
            results = []
            pending = list(futures)
            while pending:
                ready, pending = ray.wait(
                    pending,
                    num_returns=min(12, len(pending)),
                    timeout=60.0,
                )
                results.extend(ray.get(ready))
                print(
                    f"Collected {len(results)}/{len(futures)} AutoETS origins.",
                    flush=True,
                )
        finally:
            if started_here:
                ray.shutdown()
    else:
        results = [
            _forecast_origin(
                values[:origin_index].copy(),
                series_ids,
                origin_index,
                str(pd.Timestamp(dates[origin_index]).date()),
                str(pd.Timestamp(dates[origin_index - 1]).date()),
                int(season_length),
                bool(clip_nonnegative),
            )
            for origin_index in origin_indices
        ]

    rows = []
    for result in sorted(results, key=lambda item: item["origin_index"]):
        date = pd.Timestamp(result["date"])
        origin_index = int(result["origin_index"])
        for column_index, series_id in enumerate(series_ids):
            rows.append({
                "date": date,
                "year": int(date.year),
                "month": int(date.month),
                "series_id": series_id,
                "group": series_id[0],
                "actual": float(values[origin_index, column_index]),
                "direct_auto_ets": float(result["forecast"][column_index]),
                "autoets_history_end": result["history_end"],
                "autoets_n_train": int(result["n_train"]),
            })
    forecasts = pd.DataFrame(rows).sort_values(
        ["date", "series_id"]
    ).reset_index(drop=True)
    if forecasts["series_id"].nunique() != int(expected_n_series):
        raise RuntimeError("The output does not contain exactly 76 series.")
    history_end = pd.to_datetime(forecasts["autoets_history_end"])
    if not (history_end < forecasts["date"]).all():
        raise RuntimeError("Saved AutoETS histories do not precede their targets.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecasts.to_csv(output_path, index=False)
    metadata = {
        "data_path": str(data_path),
        "output_path": str(output_path),
        "first_year": int(first_year),
        "final_year": int(final_year),
        "series_depth": int(series_depth),
        "n_series": int(expected_n_series),
        "n_origins": int(forecasts["date"].nunique()),
        "n_rows": int(len(forecasts)),
        "purpose_component_series_used": False,
        "leakage_control": "each worker receives observations through t-1 only",
        "elapsed_seconds": float(time.time() - started),
    }
    output_path.with_suffix(output_path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(json.dumps(metadata, indent=2), flush=True)
    return forecasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/tourism/tourism.csv"))
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/tourism_76_autoets_long/rolling_forecasts.csv"),
    )
    parser.add_argument("--first-year", type=int, default=2001)
    parser.add_argument("--final-year", type=int, default=2016)
    parser.add_argument("--series-depth", type=int, default=3)
    parser.add_argument("--expected-n-series", type=int, default=76)
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--ray-num-cpus", type=int, default=104)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    parser.add_argument(
        "--clip-nonnegative",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()
    run_backtest(
        data_path=args.data_path,
        output_path=args.output_path,
        first_year=args.first_year,
        final_year=args.final_year,
        series_depth=args.series_depth,
        expected_n_series=args.expected_n_series,
        season_length=args.season_length,
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
        clip_nonnegative=args.clip_nonnegative,
    )


if __name__ == "__main__":
    main()
