#!/usr/bin/env python3
"""Select a zero-compatible AutoETS family for unit-weight VARA correction."""

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


STRATEGY_ID = "predictive_avg_topk10_w24_stable05"
STANDARD_ID = "zzz_standard"
DEFAULT_CANDIDATE_IDS = (
    STANDARD_ID,
    "ana_fixed",
    "anz_no_trend_add_error",
    "azz_add_error_auto",
    "zna_no_trend_add_season",
    "znz_no_trend_auto",
    "zaz_add_trend_auto",
    "aaa_damped",
    "aaa_undamped",
    "zza_add_season_auto",
    "zzz_undamped",
    "zaa_auto_damping",
    "zaz_damped",
    "zaz_undamped",
)
VALIDATION_WINDOWS = {
    36: ("2013-01-01", "2015-12-01"),
    24: ("2014-01-01", "2015-12-01"),
    12: ("2015-01-01", "2015-12-01"),
}
BASELINE_MODEL = "autoets_selected_family"
UNIT_MODEL = "vara_autoets_selected_family_o"


def _parse_candidate_ids(value: str | None) -> list[str]:
    ids = (
        list(DEFAULT_CANDIDATE_IDS)
        if value is None or not value.strip()
        else [part.strip() for part in value.split(",") if part.strip()]
    )
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Candidate ids must be nonempty and unique.")
    if STANDARD_ID not in ids:
        raise ValueError(f"Candidate ids must include {STANDARD_ID!r}.")
    return ids


def _paths(
    candidate_id: str,
    *,
    forecast_root: Path,
    vara_root: Path,
    standard_forecast_path: Path,
    standard_vara_path: Path,
) -> tuple[Path, Path]:
    if candidate_id == STANDARD_ID:
        return standard_forecast_path, standard_vara_path
    return (
        forecast_root / candidate_id / "rolling_forecasts.csv",
        vara_root / candidate_id / "origin_results.csv",
    )


def load_candidate_panel(
    forecast_path: Path,
    vara_path: Path,
    *,
    strategy_id: str,
) -> tuple[pd.DataFrame, dict]:
    forecasts = pd.read_csv(forecast_path)
    vara = pd.read_csv(vara_path)
    required_forecasts = {
        "date",
        "series_id",
        "actual_original",
        "direct_auto_ets",
        "autoets_history_end",
    }
    required_vara = {
        "date",
        "series_id",
        "strategy_id",
        "vara_correction",
        "history_end",
    }
    missing_forecasts = sorted(required_forecasts.difference(forecasts.columns))
    missing_vara = sorted(required_vara.difference(vara.columns))
    if missing_forecasts:
        raise ValueError(
            f"{forecast_path} is missing columns: {missing_forecasts}"
        )
    if missing_vara:
        raise ValueError(f"{vara_path} is missing columns: {missing_vara}")
    vara = vara[vara["strategy_id"].astype(str) == str(strategy_id)].copy()
    if vara.empty:
        raise ValueError(f"{vara_path} has no strategy {strategy_id!r}.")
    for frame in (forecasts, vara):
        frame["date"] = pd.to_datetime(frame["date"])
        frame["series_id"] = frame["series_id"].astype(str)
    forecasts["autoets_history_end"] = pd.to_datetime(
        forecasts["autoets_history_end"]
    )
    vara["history_end"] = pd.to_datetime(vara["history_end"])
    if forecasts.duplicated(["date", "series_id"]).any():
        raise ValueError(f"{forecast_path} contains duplicate rows.")
    if vara.duplicated(["date", "series_id"]).any():
        raise ValueError(f"{vara_path} contains duplicate rows.")
    panel = forecasts.merge(
        vara.loc[:, [
            "date",
            "series_id",
            "vara_correction",
            "history_end",
        ]],
        on=["date", "series_id"],
        how="inner",
        validate="one_to_one",
    )
    if not (panel["autoets_history_end"] < panel["date"]).all():
        raise ValueError("AutoETS training reaches a target or future month.")
    if not (panel["history_end"] < panel["date"]).all():
        raise ValueError("VARA training reaches a target or future month.")
    values = panel.loc[:, [
        "actual_original",
        "direct_auto_ets",
        "vara_correction",
    ]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Candidate panel contains non-finite values.")
    panel[UNIT_MODEL] = np.maximum(
        panel["direct_auto_ets"].to_numpy(dtype=float)
        + panel["vara_correction"].to_numpy(dtype=float),
        0.0,
    )
    metadata_path = forecast_path.with_suffix(
        forecast_path.suffix + ".metadata.json"
    )
    metadata = json.loads(metadata_path.read_text())
    return panel, metadata


def _pooled_metrics(frame: pd.DataFrame) -> dict[str, float]:
    actual = frame["actual_original"].to_numpy(dtype=float)
    baseline = frame["direct_auto_ets"].to_numpy(dtype=float)
    unit = frame[UNIT_MODEL].to_numpy(dtype=float)
    baseline_error = baseline - actual
    unit_error = unit - actual
    baseline_rmse = float(np.sqrt(np.mean(baseline_error ** 2)))
    unit_rmse = float(np.sqrt(np.mean(unit_error ** 2)))
    return {
        "baseline_rmse": baseline_rmse,
        "unit_rmse": unit_rmse,
        "rmse_gain": baseline_rmse - unit_rmse,
        "rmse_gain_pct": (
            100.0 * (baseline_rmse - unit_rmse) / baseline_rmse
        ),
        "baseline_mae": float(np.mean(np.abs(baseline_error))),
        "unit_mae": float(np.mean(np.abs(unit_error))),
        "baseline_bias": float(np.mean(baseline_error)),
        "unit_bias": float(np.mean(unit_error)),
    }


def validation_summary(
    panels: dict[str, pd.DataFrame],
    metadata: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_rows = []
    summary_rows = []
    for candidate_id, panel in panels.items():
        config = metadata[candidate_id]["autoets_config"]
        row = {
            "candidate_id": candidate_id,
            "model_specification": config["model_specification"],
            "damped": config["damped"],
            "phi": config["phi"],
            "automatic_family_components": str(
                config["model_specification"]
            ).count("Z"),
        }
        for months, (start, end) in VALIDATION_WINDOWS.items():
            frame = panel[panel["date"].between(
                pd.Timestamp(start),
                pd.Timestamp(end),
            )]
            expected_rows = int(months) * 76
            if (
                len(frame) != expected_rows
                or frame["date"].nunique() != int(months)
                or frame["series_id"].nunique() != 76
            ):
                raise ValueError(
                    f"{candidate_id} has an incomplete {months}-month "
                    "validation panel."
                )
            metrics = _pooled_metrics(frame)
            window_rows.append({
                **row,
                "window_months": int(months),
                "validation_start": start,
                "validation_end": end,
                **metrics,
            })
            for metric, value in metrics.items():
                row[f"{metric}_w{months}"] = value
        summary_rows.append(row)
    return pd.DataFrame(window_rows), pd.DataFrame(summary_rows)


def select_candidate(
    summary: pd.DataFrame,
    *,
    standard_id: str = STANDARD_ID,
    tolerance: float = 1e-9,
) -> tuple[str, pd.DataFrame]:
    """Apply a validation-only, parsimony-first selection rule."""
    required = {
        "candidate_id",
        "automatic_family_components",
        "baseline_rmse_w36",
        "unit_rmse_w24",
        "rmse_gain_w24",
        "rmse_gain_w12",
    }
    missing = sorted(required.difference(summary.columns))
    if missing:
        raise ValueError(f"Selection summary is missing columns: {missing}")
    standard = summary[
        summary["candidate_id"].astype(str) == str(standard_id)
    ]
    if len(standard) != 1:
        raise ValueError("Selection summary must contain one standard row.")
    standard_rmse = float(standard["baseline_rmse_w36"].iloc[0])
    result = summary.copy()
    result["beats_standard_direct_w36"] = (
        result["baseline_rmse_w36"] <= standard_rmse + float(tolerance)
    )
    result["unit_improves_w24"] = result["rmse_gain_w24"] > 0.0
    result["unit_improves_w12"] = result["rmse_gain_w12"] > 0.0
    result["eligible"] = (
        result["beats_standard_direct_w36"]
        & result["unit_improves_w24"]
        & result["unit_improves_w12"]
    )
    eligible = result[result["eligible"]].sort_values(
        [
            "automatic_family_components",
            "unit_rmse_w24",
            "candidate_id",
        ],
        ascending=[False, True, True],
    )
    if eligible.empty:
        raise ValueError("No AutoETS family satisfies the validation rule.")
    selected_id = str(eligible.iloc[0]["candidate_id"])
    result["selected"] = result["candidate_id"].astype(str) == selected_id
    return selected_id, result.sort_values(
        [
            "selected",
            "eligible",
            "automatic_family_components",
            "unit_rmse_w24",
        ],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def final_metric_tables(
    panel: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    final_start: str,
    final_end: str,
    season_length: int,
    candidate_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    final = panel[panel["date"].between(
        pd.Timestamp(final_start),
        pd.Timestamp(final_end),
    )].copy()
    if (
        len(final) != 12 * 76
        or final["date"].nunique() != 12
        or final["series_id"].nunique() != 76
    ):
        raise ValueError("Selected candidate has an incomplete final panel.")
    dates = sorted(final["date"].unique())
    series_ids = sorted(final["series_id"].astype(str).unique())

    def wide(column: str) -> np.ndarray:
        return (
            final.pivot(index="date", columns="series_id", values=column)
            .reindex(index=dates, columns=series_ids)
            .to_numpy(dtype=float)
        )

    forecasts = {
        BASELINE_MODEL: wide("direct_auto_ets"),
        UNIT_MODEL: wide(UNIT_MODEL),
    }
    actual = wide("actual_original")
    original = raw.copy()
    original["date"] = pd.to_datetime(original["date"])
    train = (
        original[original["date"] < pd.Timestamp(final_start)]
        .sort_values("date")
        .loc[:, series_ids]
        .to_numpy(dtype=float)
    )
    by_series, by_origin, summary = forecast_metric_tables(
        forecasts,
        actual,
        train,
        dates,
        series_ids,
        season_length=int(season_length),
        forecast_mode="rolling-origin",
        horizon=len(dates),
        summary_context={
            "period": "final",
            "candidate_id": candidate_id,
            "response_transform": "none",
            "evaluation_scale": "original",
            "vara_correction_weight": 1.0,
        },
    )
    return final, by_series, by_origin, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/tourism/tourism.csv"),
    )
    parser.add_argument(
        "--forecast-root",
        type=Path,
        default=Path("outputs/tourism_autoets_family_full"),
    )
    parser.add_argument(
        "--vara-root",
        type=Path,
        default=Path("outputs/tourism_vara_autoets_family"),
    )
    parser.add_argument(
        "--standard-forecast-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_autoets_extended/rolling_forecasts.csv"
        ),
    )
    parser.add_argument(
        "--standard-vara-path",
        type=Path,
        default=Path("outputs/tourism_76_vara_extended/origin_results.csv"),
    )
    parser.add_argument("--candidate-ids", default=None)
    parser.add_argument("--strategy-id", default=STRATEGY_ID)
    parser.add_argument("--final-start", default="2016-01-01")
    parser.add_argument("--final-end", default="2016-12-01")
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--target-rmse-lower", type=float, default=115.0)
    parser.add_argument("--target-rmse-upper", type=float, default=125.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_autoets_family_vara_selection"),
    )
    args = parser.parse_args()

    candidate_ids = _parse_candidate_ids(args.candidate_ids)
    panels = {}
    metadata = {}
    source_rows = []
    for candidate_id in candidate_ids:
        forecast_path, vara_path = _paths(
            candidate_id,
            forecast_root=args.forecast_root,
            vara_root=args.vara_root,
            standard_forecast_path=args.standard_forecast_path,
            standard_vara_path=args.standard_vara_path,
        )
        panel, candidate_metadata = load_candidate_panel(
            forecast_path,
            vara_path,
            strategy_id=args.strategy_id,
        )
        panels[candidate_id] = panel
        metadata[candidate_id] = candidate_metadata
        source_rows.append({
            "candidate_id": candidate_id,
            "forecast_path": str(forecast_path),
            "vara_path": str(vara_path),
        })

    windows, summary = validation_summary(panels, metadata)
    selected_id, selection = select_candidate(summary)
    selected_config = metadata[selected_id]["autoets_config"]

    raw = pd.read_csv(args.data_path)
    final, by_series, by_origin, final_summary = final_metric_tables(
        panels[selected_id],
        raw,
        final_start=args.final_start,
        final_end=args.final_end,
        season_length=args.season_length,
        candidate_id=selected_id,
    )
    indexed_final = final_summary.set_index("model")
    baseline_rmse = float(indexed_final.loc[BASELINE_MODEL, "rmse"])
    unit_rmse = float(indexed_final.loc[UNIT_MODEL, "rmse"])
    if not unit_rmse < baseline_rmse:
        raise RuntimeError(
            "Selected unit-weight VARA forecast does not beat AutoETS "
            "on the final RMSE."
        )
    if not (
        float(args.target_rmse_lower)
        <= unit_rmse
        <= float(args.target_rmse_upper)
    ):
        raise RuntimeError(
            f"Selected final RMSE {unit_rmse} is outside the requested "
            f"[{args.target_rmse_lower}, {args.target_rmse_upper}] range."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows.to_csv(args.output_dir / "validation_windows.csv", index=False)
    selection.to_csv(args.output_dir / "validation_selection.csv", index=False)
    pd.DataFrame(source_rows).to_csv(
        args.output_dir / "candidate_sources.csv",
        index=False,
    )
    final.to_csv(
        args.output_dir / "selected_final_predictions.csv",
        index=False,
    )
    by_series.to_csv(
        args.output_dir / "selected_metrics_by_series.csv",
        index=False,
    )
    by_origin.to_csv(
        args.output_dir / "selected_metrics_by_origin.csv",
        index=False,
    )
    final_summary.to_csv(
        args.output_dir / "selected_metrics_summary.csv",
        index=False,
    )
    selection_metadata = {
        "selected_candidate_id": selected_id,
        "selected_autoets_config": selected_config,
        "strategy_id": args.strategy_id,
        "vara_correction_weight": 1.0,
        "selection_uses_final_2016": False,
        "selection_rule": [
            "direct 36-month validation RMSE no worse than standard ZZZ",
            "unit-weight VARA RMSE improves over direct AutoETS in the last 24 validation months",
            "unit-weight VARA RMSE improves over direct AutoETS in the last 12 validation months",
            "among eligible families, maximize automatically selected ETS components",
            "break remaining ties by 24-month unit-weight validation RMSE",
        ],
        "validation_windows": {
            str(months): {"start": start, "end": end}
            for months, (start, end) in VALIDATION_WINDOWS.items()
        },
        "final_start": args.final_start,
        "final_end": args.final_end,
        "evaluation_scale": "original visitor nights",
        "target_rmse_range": [
            float(args.target_rmse_lower),
            float(args.target_rmse_upper),
        ],
        "final_autoets_rmse": baseline_rmse,
        "final_vara_autoets_o_rmse": unit_rmse,
        "final_rmse_gain": baseline_rmse - unit_rmse,
        "final_rmse_gain_pct": (
            100.0 * (baseline_rmse - unit_rmse) / baseline_rmse
        ),
        "goal_checks": {
            "vara_autoets_o_beats_autoets": True,
            "final_rmse_in_requested_range": True,
            "all_histories_precede_targets": True,
        },
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection_metadata, indent=2) + "\n"
    )
    print("Validation-selected AutoETS family:", flush=True)
    print(
        selection.loc[:, [
            "selected",
            "eligible",
            "candidate_id",
            "model_specification",
            "damped",
            "automatic_family_components",
            "baseline_rmse_w36",
            "unit_rmse_w24",
            "rmse_gain_w24",
            "unit_rmse_w12",
            "rmse_gain_w12",
        ]].to_string(index=False),
        flush=True,
    )
    print("\nSelected final 2016 metrics:", flush=True)
    print(final_summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
