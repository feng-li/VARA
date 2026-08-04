#!/usr/bin/env python3
"""Evaluate a unit-weight VARA correction for a saved Tourism error window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from _bootstrap import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT  # noqa: F401

from vara.forecast_metrics import forecast_metric_tables


MODEL = "raw_oos_vara"
DIRECT_MODEL = "direct_auto_ets"


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


def evaluate_unit_weight_window(
    *,
    data_path: Path,
    forecast_panel_path: Path,
    vara_results_path: Path,
    strategy_id: str,
    history_window_months: int,
    evaluation_start: str,
    evaluation_end: str,
    output_dir: Path,
) -> pd.DataFrame:
    """Evaluate direct AutoETS plus one times the saved VARA correction."""
    if int(history_window_months) < 12:
        raise ValueError("history_window_months must be at least 12.")

    forecast_metadata_path = forecast_panel_path.with_suffix(
        forecast_panel_path.suffix + ".metadata.json"
    )
    forecast_metadata = (
        json.loads(forecast_metadata_path.read_text())
        if forecast_metadata_path.exists()
        else {}
    )
    baseline = pd.read_csv(forecast_panel_path)
    required_baseline = {"date", "series_id", "actual", "direct_auto_ets"}
    missing = sorted(required_baseline.difference(baseline.columns))
    if missing:
        raise ValueError(f"AutoETS forecast panel is missing columns: {missing}")
    baseline["date"] = pd.to_datetime(baseline["date"])
    baseline["series_id"] = baseline["series_id"].astype(str)
    if baseline.duplicated(["date", "series_id"]).any():
        raise ValueError("AutoETS forecast panel contains duplicate rows.")
    response_transform = str(
        forecast_metadata.get(
            "response_transform",
            (
                baseline["response_transform"].iloc[0]
                if "response_transform" in baseline
                else "none"
            ),
        )
    )
    response_normalization = str(
        forecast_metadata.get(
            "response_normalization",
            (
                baseline["response_normalization"].iloc[0]
                if "response_normalization" in baseline
                else "none"
            ),
        )
    )
    if response_transform != "none":
        raise ValueError(
            "The level-scale unit-weight evaluator requires an untransformed "
            "AutoETS response."
        )

    corrections = pd.read_csv(vara_results_path)
    required_corrections = {
        "strategy_id",
        "date",
        "history_start",
        "history_end",
        "series_id",
        "vara_correction",
        "n_train",
    }
    missing = sorted(required_corrections.difference(corrections.columns))
    if missing:
        raise ValueError(f"VARA result panel is missing columns: {missing}")
    corrections = corrections[
        corrections["strategy_id"].astype(str).eq(str(strategy_id))
    ].copy()
    if corrections.empty:
        raise ValueError(f"No VARA rows use strategy {strategy_id!r}.")
    for column in ("date", "history_start", "history_end"):
        corrections[column] = pd.to_datetime(corrections[column])
    corrections["series_id"] = corrections["series_id"].astype(str)
    if corrections.duplicated(["date", "series_id"]).any():
        raise ValueError("VARA result panel contains duplicate rows.")
    if not corrections["n_train"].eq(int(history_window_months)).all():
        observed = sorted(corrections["n_train"].astype(int).unique())
        raise ValueError(
            f"VARA history lengths are {observed}, not "
            f"{history_window_months} months."
        )
    history_by_origin = corrections.groupby("date", observed=True).agg(
        history_start_count=("history_start", "nunique"),
        history_end_count=("history_end", "nunique"),
        history_start=("history_start", "first"),
        history_end=("history_end", "first"),
    )
    if not (
        history_by_origin["history_start_count"].eq(1)
        & history_by_origin["history_end_count"].eq(1)
    ).all():
        raise ValueError("Series at the same origin use different VARA histories.")
    observed_spans = (
        12
        * (
            history_by_origin["history_end"].dt.year
            - history_by_origin["history_start"].dt.year
        )
        + (
            history_by_origin["history_end"].dt.month
            - history_by_origin["history_start"].dt.month
        )
        + 1
    )
    if not observed_spans.eq(int(history_window_months)).all():
        raise ValueError("VARA history dates do not span the requested window.")
    if not (corrections["history_end"] < corrections["date"]).all():
        raise ValueError("A VARA correction includes its target or a future month.")

    start = pd.Timestamp(evaluation_start)
    end = pd.Timestamp(evaluation_end)
    evaluation_dates = list(pd.date_range(start, end, freq="MS"))
    corrections = corrections[corrections["date"].isin(evaluation_dates)].copy()
    baseline_evaluation = baseline[baseline["date"].isin(evaluation_dates)].copy()
    panel = baseline_evaluation.merge(
        corrections.loc[:, [
            "strategy_id",
            "date",
            "series_id",
            "history_start",
            "history_end",
            "n_train",
            "vara_correction",
        ]],
        on=["date", "series_id"],
        how="inner",
        validate="one_to_one",
    )
    series_ids = sorted(baseline_evaluation["series_id"].unique())
    expected_rows = len(evaluation_dates) * len(series_ids)
    if len(panel) != expected_rows:
        raise ValueError(
            f"Matched evaluation panel has {len(panel)} rows; "
            f"expected {expected_rows}."
        )
    panel["forecast"] = np.maximum(
        panel["direct_auto_ets"].to_numpy(dtype=float)
        + panel["vara_correction"].to_numpy(dtype=float),
        0.0,
    )
    panel["error"] = (
        panel["forecast"].to_numpy(dtype=float)
        - panel["actual"].to_numpy(dtype=float)
    )

    raw = pd.read_csv(data_path)
    raw["date"] = pd.to_datetime(raw["date"])
    missing_series = sorted(set(series_ids).difference(raw.columns))
    if missing_series:
        raise ValueError(f"Raw Tourism data are missing series: {missing_series}")
    train = (
        raw[raw["date"] < start]
        .sort_values("date")
        .loc[:, series_ids]
        .to_numpy(dtype=float)
    )
    actual = _wide(panel, "actual", evaluation_dates, series_ids)
    forecasts = {
        MODEL: _wide(panel, "forecast", evaluation_dates, series_ids),
        DIRECT_MODEL: _wide(panel, "direct_auto_ets", evaluation_dates, series_ids),
    }
    by_series, by_origin, summary = forecast_metric_tables(
        forecasts,
        actual,
        train,
        evaluation_dates,
        series_ids,
        season_length=12,
        forecast_mode="rolling-origin",
        horizon=len(evaluation_dates),
        summary_context={
            "strategy_id": strategy_id,
            "history_window_months": int(history_window_months),
            "correction_weight": 1.0,
            "response_transform": response_transform,
            "response_normalization": response_normalization,
            "evaluation_scale": "original visitor nights",
        },
    )
    direct_rmse = float(
        summary.loc[summary["model"].eq(DIRECT_MODEL), "rmse"].iloc[0]
    )
    summary["rmse_improvement_vs_autoets_pct"] = (
        100.0 * (direct_rmse - summary["rmse"]) / direct_rmse
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    panel.sort_values(["date", "series_id"]).to_csv(
        output_dir / "unit_weight_predictions.csv",
        index=False,
    )
    by_series.to_csv(output_dir / "metrics_by_series.csv", index=False)
    by_origin.to_csv(output_dir / "metrics_by_origin.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    metadata = {
        "method": "VARA--AutoETS(ZZA)_O",
        "autoets_specification": forecast_metadata.get(
            "autoets_config",
            {},
        ).get("model_specification", "unknown"),
        "correction_weight": 1.0,
        "history_window_months": int(history_window_months),
        "history_window_policy": "fixed trailing window",
        "forecast_panel_path": str(forecast_panel_path),
        "vara_results_path": str(vara_results_path),
        "strategy_id": strategy_id,
        "evaluation_start": str(start.date()),
        "evaluation_end": str(end.date()),
        "n_origins": len(evaluation_dates),
        "n_series": len(series_ids),
        "evaluation_scale": "original visitor nights",
        "response_transform": response_transform,
        "response_normalization": response_normalization,
        "normalization_statistics": (
            "saved per-origin per-series training mean and population variance"
            if response_normalization == "training_zscore"
            else "not applied"
        ),
        "leakage_control": (
            "AutoETS normalization uses only its origin-specific training "
            "slice, and each VARA correction uses exactly the requested "
            "residual months ending before its target"
        ),
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/tourism/tourism.csv"),
    )
    parser.add_argument(
        "--forecast-panel-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_autoets_zza_extended/rolling_forecasts.csv"
        ),
    )
    parser.add_argument("--vara-results-path", type=Path, required=True)
    parser.add_argument("--strategy-id", type=str, required=True)
    parser.add_argument("--history-window-months", type=int, required=True)
    parser.add_argument("--evaluation-start", default="2016-01-01")
    parser.add_argument("--evaluation-end", default="2016-12-01")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = evaluate_unit_weight_window(
        data_path=args.data_path,
        forecast_panel_path=args.forecast_panel_path,
        vara_results_path=args.vara_results_path,
        strategy_id=args.strategy_id,
        history_window_months=args.history_window_months,
        evaluation_start=args.evaluation_start,
        evaluation_end=args.evaluation_end,
        output_dir=args.output_dir,
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
