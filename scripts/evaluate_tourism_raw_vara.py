#!/usr/bin/env python3
"""Evaluate direct VARA forecasts of the raw Tourism panel without AutoETS."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

try:
    from _bootstrap import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT  # noqa: F401

from scripts.search_tourism_oos_error_vara import (
    ErrorStrategy,
    _error_vara_task,
)
from vara.forecast_metrics import add_composite_rank_score, forecast_metric_tables


MODEL = "vara_raw_direct"
RANK_SCORE_METRICS = ("rmse", "mae", "mase", "rmsse")
DEFAULT_STRATEGY = ErrorStrategy(
    strategy_id="raw_direct_predictive_avg_topk10_w24_stable05",
    feature_source="predictive-average",
    attention_mode="score-topk",
    attention_top_k=10,
    transform="standardize",
    history_window=24,
    ridge=0.01,
    grid_size=7,
    eta_lower=-2.0,
    eta_upper=2.0,
    stability_cap=0.5,
)


def depth_three_all_series(columns: list[str]) -> list[str]:
    """Select the 76 depth-three all-purpose Tourism series."""
    return sorted(
        column
        for column in columns
        if column.endswith("All") and len(column[: -len("All")]) == 3
    )


def _wide(
    frame: pd.DataFrame,
    column: str,
    dates: list[pd.Timestamp],
    series_ids: list[str],
) -> np.ndarray:
    return (
        frame.pivot(index="date", columns="series_id", values=column)
        .reindex(index=dates, columns=series_ids)
        .to_numpy(dtype=float)
    )


def build_raw_vara_predictions(
    raw_data: pd.DataFrame,
    *,
    final_start: str,
    final_end: str,
    strategy: ErrorStrategy = DEFAULT_STRATEGY,
    series_ids: list[str] | None = None,
    task_runner: Callable[..., dict] = _error_vara_task,
) -> pd.DataFrame:
    """Fit direct raw-level VARA separately at each rolling origin."""
    if "date" not in raw_data:
        raise ValueError("Tourism data must contain a date column.")
    frame = raw_data.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("Tourism data contains duplicate dates.")

    selected = (
        depth_three_all_series(list(frame.columns))
        if series_ids is None
        else list(series_ids)
    )
    if not selected:
        raise ValueError("No raw Tourism series were selected.")
    missing = sorted(set(selected).difference(frame.columns))
    if missing:
        raise ValueError(f"Tourism data is missing selected series: {missing}")
    values = frame.loc[:, selected].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Raw Tourism panel contains non-finite values.")

    target_mask = frame["date"].between(
        pd.Timestamp(final_start),
        pd.Timestamp(final_end),
    )
    target_indices = np.flatnonzero(target_mask.to_numpy())
    if target_indices.size == 0:
        raise ValueError("No raw Tourism targets fall in the final period.")

    rows = []
    for target_index in target_indices:
        target_date = pd.Timestamp(frame.loc[target_index, "date"])
        history_window = strategy.history_window
        history_start_index = (
            0
            if history_window is None
            else max(0, int(target_index) - int(history_window))
        )
        history = values[history_start_index:int(target_index)]
        history_dates = frame.loc[
            history_start_index:int(target_index) - 1,
            "date",
        ]
        if history_window is not None and len(history) != int(history_window):
            raise ValueError(
                f"Target {target_date.date()} has {len(history)} raw history rows; "
                f"expected {history_window}."
            )
        if history_dates.empty:
            raise ValueError(f"Target {target_date.date()} has no prior history.")
        result = task_runner(
            error_history=history,
            series_ids=selected,
            strategy_dict=asdict(strategy),
            target_index=int(target_index),
            target_date=str(target_date.date()),
            history_start=str(pd.Timestamp(history_dates.iloc[0]).date()),
            history_end=str(pd.Timestamp(history_dates.iloc[-1]).date()),
        )
        if not bool(result.get("ok", False)):
            raise RuntimeError(
                f"Direct raw VARA failed at {target_date.date()}: "
                f"{result.get('error_type', 'Error')}: {result.get('error', '')}"
            )
        direct_unclipped = np.asarray(result["correction"], dtype=float)
        if direct_unclipped.shape != (len(selected),):
            raise ValueError(
                "Direct raw VARA forecast does not match the selected series count."
            )
        direct = np.maximum(direct_unclipped, 0.0)
        rows.append(pd.DataFrame({
            "date": target_date,
            "series_id": selected,
            "actual_original": values[int(target_index)],
            "vara_raw_direct_unclipped": direct_unclipped,
            MODEL: direct,
            "history_start": pd.Timestamp(result["history_start"]),
            "history_end": pd.Timestamp(result["history_end"]),
            "n_train": int(result["n_train"]),
            "eta_hat": float(result["eta_hat"]),
            "eta_on_boundary": bool(result["eta_on_boundary"]),
            "stationarity_proxy": float(result["stationarity_proxy"]),
            "response_transform": "none",
            "evaluation_scale": "original",
            "autoets_used": False,
            "strategy_id": strategy.strategy_id,
        }))
        print(
            f"Direct raw VARA origin {target_date.date()} complete "
            f"({len(history)} prior months).",
            flush=True,
        )

    predictions = pd.concat(rows, ignore_index=True)
    if not (predictions["history_end"] < predictions["date"]).all():
        raise RuntimeError("Direct raw VARA history includes a target or future date.")
    return predictions.sort_values(["date", "series_id"]).reset_index(drop=True)


def raw_vara_metric_tables(
    predictions: pd.DataFrame,
    raw_data: pd.DataFrame,
    *,
    final_start: str,
    season_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute paper metrics for direct raw VARA in original units."""
    raw = raw_data.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    dates = sorted(pd.to_datetime(predictions["date"].unique()))
    series_ids = sorted(predictions["series_id"].astype(str).unique())
    forecast = _wide(predictions, MODEL, dates, series_ids)
    actual = _wide(predictions, "actual_original", dates, series_ids)
    raw_actual = (
        raw.set_index("date")
        .reindex(index=dates)
        .loc[:, series_ids]
        .to_numpy(dtype=float)
    )
    if not np.allclose(actual, raw_actual, rtol=1e-10, atol=1e-8):
        raise ValueError("Direct raw VARA actuals do not match the Tourism panel.")
    train = (
        raw[raw["date"] < pd.Timestamp(final_start)]
        .sort_values("date")
        .loc[:, series_ids]
        .to_numpy(dtype=float)
    )
    return forecast_metric_tables(
        {MODEL: forecast},
        actual,
        train,
        dates,
        series_ids,
        season_length=int(season_length),
        forecast_mode="rolling-origin",
        horizon=len(dates),
        summary_context={
            "period": "final",
            "response_transform": "none",
            "evaluation_scale": "original",
            "autoets_used": False,
        },
    )


def add_raw_vara_to_comparison(
    raw_vara_summary: pd.DataFrame,
    existing_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Append direct raw VARA and recompute the relative paper RankScore."""
    base = existing_comparison[
        existing_comparison["model"].astype(str) != MODEL
    ].copy()
    raw_vara = raw_vara_summary.copy()
    raw_vara["source"] = "direct_raw_vara"
    comparison = pd.concat([base, raw_vara], ignore_index=True, sort=False)
    ranked = add_composite_rank_score(
        comparison,
        lower_is_better=RANK_SCORE_METRICS,
    )
    autoets_rmse = float(
        ranked.loc[ranked["model"] == "auto_ets", "rmse"].iloc[0]
    )
    ranked["rmse_improvement_vs_auto_ets_pct"] = (
        100.0 * (autoets_rmse - ranked["rmse"]) / autoets_rmse
    )
    return ranked.sort_values(
        ["overall_rank", "rmse", "model"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def run_evaluation(
    *,
    data_path: Path,
    comparison_path: Path,
    output_dir: Path,
    final_start: str,
    final_end: str,
    season_length: int,
    expected_n_series: int,
    history_window: int,
) -> pd.DataFrame:
    raw = pd.read_csv(data_path)
    strategy = replace(DEFAULT_STRATEGY, history_window=int(history_window))
    predictions = build_raw_vara_predictions(
        raw,
        final_start=final_start,
        final_end=final_end,
        strategy=strategy,
    )
    if predictions["series_id"].nunique() != int(expected_n_series):
        raise ValueError(
            f"Expected {expected_n_series} raw series, found "
            f"{predictions['series_id'].nunique()}."
        )
    by_series, by_origin, summary = raw_vara_metric_tables(
        predictions,
        raw,
        final_start=final_start,
        season_length=season_length,
    )
    comparison = add_raw_vara_to_comparison(
        summary,
        pd.read_csv(comparison_path),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "final_predictions.csv", index=False)
    by_series.to_csv(output_dir / "metrics_by_series.csv", index=False)
    by_origin.to_csv(output_dir / "metrics_by_origin.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    comparison.to_csv(output_dir / "paper_comparison.csv", index=False)
    predictions.loc[:, [
        "date",
        "history_start",
        "history_end",
        "n_train",
        "eta_hat",
        "eta_on_boundary",
        "stationarity_proxy",
        "strategy_id",
    ]].drop_duplicates("date").to_csv(
        output_dir / "origin_diagnostics.csv",
        index=False,
    )
    metadata = {
        "model": MODEL,
        "response": "original visitor-night levels",
        "response_transform": "none",
        "internal_fit_transform": strategy.transform,
        "forecast_scenario": "direct",
        "autoets_used": False,
        "forecast_formula": "max(direct raw-level VARA forecast, 0)",
        "strategy": asdict(strategy),
        "final_start": final_start,
        "final_end": final_end,
        "metric_scale": "original visitor nights",
        "rank_score_metrics": list(RANK_SCORE_METRICS),
        "comparison_method_count": int(len(comparison)),
        "n_origins": int(predictions["date"].nunique()),
        "n_series": int(predictions["series_id"].nunique()),
        "n_cells": int(len(predictions)),
        "leakage_control": "each raw VARA history ends before its forecast target",
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("Direct raw VARA original-scale metrics:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nComparison with recomputed RankScore:", flush=True)
    print(
        comparison.loc[:, [
            "overall_rank",
            "model",
            "rank_score",
            "rmse",
            "mae",
            "mase",
            "rmsse",
        ]].to_string(index=False),
        flush=True,
    )
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/tourism/tourism.csv"),
    )
    parser.add_argument(
        "--comparison-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_log1p_raw_oos_vara/paper_comparison.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_76_raw_direct_vara"),
    )
    parser.add_argument("--final-start", default="2016-01-01")
    parser.add_argument("--final-end", default="2016-12-01")
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--expected-n-series", type=int, default=76)
    parser.add_argument("--history-window", type=int, default=24)
    args = parser.parse_args()
    run_evaluation(
        data_path=args.data_path,
        comparison_path=args.comparison_path,
        output_dir=args.output_dir,
        final_start=args.final_start,
        final_end=args.final_end,
        season_length=args.season_length,
        expected_n_series=args.expected_n_series,
        history_window=args.history_window,
    )


if __name__ == "__main__":
    main()
