#!/usr/bin/env python3
"""Compare Tourism AutoETS and unit-VARA forecasts with training normalization."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DIRECT_MODEL = "direct_auto_ets"
UNIT_MODEL = "raw_oos_vara"
LOG1P_DIRECT_MODEL = "log1p_auto_ets"
LOG1P_UNIT_MODEL = "log1p_raw_oos_vara"


def _select_pair(
    path: Path,
    *,
    response_transform: str,
    response_normalization: str,
    direct_model: str = DIRECT_MODEL,
    unit_model: str = UNIT_MODEL,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"model", "rmse", "mae", "bias"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing comparison columns: {missing}")
    selected = frame[
        frame["model"].astype(str).isin((direct_model, unit_model))
    ].copy()
    if set(selected["model"].astype(str)) != {direct_model, unit_model}:
        raise ValueError(f"{path} does not contain direct and unit-VARA rows.")
    if selected["model"].duplicated().any():
        raise ValueError(f"{path} contains duplicate comparison models.")
    selected["response_transform"] = response_transform
    selected["response_normalization"] = response_normalization
    selected["method"] = selected["model"].map({
        direct_model: "AutoETS(ZZA)",
        unit_model: "VARA–AutoETS(ZZA)_O",
    })
    selected["model"] = selected["model"].map({
        direct_model: DIRECT_MODEL,
        unit_model: UNIT_MODEL,
    })
    return selected.loc[:, [
        "method",
        "model",
        "response_transform",
        "response_normalization",
        "rmse",
        "mae",
        "bias",
    ]]


def build_comparison(
    *,
    level_metrics_path: Path,
    normalized_level_metrics_path: Path,
    log1p_metrics_path: Path,
    normalized_log1p_metrics_path: Path,
) -> pd.DataFrame:
    """Build an original-scale comparison across transform/normalization choices."""
    comparison = pd.concat([
        _select_pair(
            level_metrics_path,
            response_transform="none",
            response_normalization="none",
        ),
        _select_pair(
            normalized_level_metrics_path,
            response_transform="none",
            response_normalization="training_zscore",
        ),
        _select_pair(
            log1p_metrics_path,
            response_transform="log1p",
            response_normalization="none",
            direct_model=LOG1P_DIRECT_MODEL,
            unit_model=LOG1P_UNIT_MODEL,
        ),
        _select_pair(
            normalized_log1p_metrics_path,
            response_transform="log1p",
            response_normalization="training_zscore",
            direct_model=LOG1P_DIRECT_MODEL,
            unit_model=LOG1P_UNIT_MODEL,
        ),
    ], ignore_index=True)
    baselines = (
        comparison[
            comparison["response_normalization"].eq("none")
        ]
        .loc[:, ["model", "response_transform", "rmse"]]
        .rename(columns={"rmse": "rmse_without_normalization"})
    )
    comparison = comparison.merge(
        baselines,
        on=["model", "response_transform"],
        how="left",
        validate="many_to_one",
    )
    comparison["rmse_change_vs_without_normalization"] = (
        comparison["rmse"] - comparison["rmse_without_normalization"]
    )
    comparison["rmse_change_pct_vs_without_normalization"] = (
        100.0
        * comparison["rmse_change_vs_without_normalization"]
        / comparison["rmse_without_normalization"]
    )
    return comparison.sort_values([
        "response_transform",
        "model",
        "response_normalization",
    ]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_reliability_vara_autoets_zza/"
            "metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--normalized-level-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_vara_autoets_zza_train_zscore_w24_unit/"
            "metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--log1p-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_log1p_vara_autoets_zza_w24/"
            "metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--normalized-log1p-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_log1p_train_zscore_vara_autoets_zza_w24/"
            "metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_zza_training_normalization_comparison.csv"
        ),
    )
    args = parser.parse_args()
    comparison = build_comparison(
        level_metrics_path=args.level_metrics_path,
        normalized_level_metrics_path=args.normalized_level_metrics_path,
        log1p_metrics_path=args.log1p_metrics_path,
        normalized_log1p_metrics_path=args.normalized_log1p_metrics_path,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_path, index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
