#!/usr/bin/env python3
"""Add the fixed ETS(A,N,N) sensitivity run to the Tourism comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from _bootstrap import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT  # noqa: F401

from vara.forecast_metrics import add_composite_rank_score


RANK_SCORE_METRICS = ("rmse", "mae", "mase", "rmsse")
ANN_MODEL_NAMES = {
    "reliability_gated_vara_auto_ets_oos": "vara_auto_ets_ann_lambda",
    "raw_oos_vara": "vara_auto_ets_ann_o",
    "direct_auto_ets": "auto_ets_ann",
}
DERIVED_COLUMNS = {
    "mean_accuracy_rank",
    "rank_score",
    "overall_rank",
    "rmse_improvement_vs_paper_pct",
    "rmse_improvement_vs_auto_ets_pct",
    "rmse_improvement_vs_raw_vara_pct",
    "rmse_improvement_vs_auto_ets_ann_pct",
}


def build_ann_comparison(
    base_comparison: pd.DataFrame,
    ann_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Append the three ANN methods and recompute relative RankScore."""
    base = base_comparison.copy()
    derived = [
        column
        for column in base
        if column in DERIVED_COLUMNS or column.startswith("rank_")
    ]
    base = base.drop(columns=derived)

    ann = ann_summary[
        ann_summary["model"].astype(str).isin(ANN_MODEL_NAMES)
    ].copy()
    missing = sorted(set(ANN_MODEL_NAMES).difference(ann["model"].astype(str)))
    if missing:
        raise ValueError(f"ANN metric summary is missing methods: {missing}")
    ann["model"] = ann["model"].map(ANN_MODEL_NAMES)
    ann["source"] = "ann_sensitivity"
    ann["autoets_model_specification"] = "ANN"

    comparison = pd.concat([base, ann], ignore_index=True, sort=False)
    if comparison["model"].duplicated().any():
        duplicates = sorted(
            comparison.loc[
                comparison["model"].duplicated(False),
                "model",
            ].astype(str).unique()
        )
        raise ValueError(f"ANN comparison contains duplicate methods: {duplicates}")
    ranked = add_composite_rank_score(
        comparison,
        lower_is_better=RANK_SCORE_METRICS,
    )
    standard_rmse = float(
        ranked.loc[ranked["model"] == "auto_ets", "rmse"].iloc[0]
    )
    ann_rmse = float(
        ranked.loc[ranked["model"] == "auto_ets_ann", "rmse"].iloc[0]
    )
    ranked["rmse_improvement_vs_auto_ets_pct"] = (
        100.0 * (standard_rmse - ranked["rmse"]) / standard_rmse
    )
    ranked["rmse_improvement_vs_auto_ets_ann_pct"] = (
        100.0 * (ann_rmse - ranked["rmse"]) / ann_rmse
    )
    return ranked.sort_values(
        ["overall_rank", "rmse", "model"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-comparison-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_raw_direct_vara/paper_comparison.csv"
        ),
    )
    parser.add_argument(
        "--ann-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_reliability_vara_autoets_ann/"
            "metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_76_autoets_ann_comparison"),
    )
    args = parser.parse_args()

    comparison = build_ann_comparison(
        pd.read_csv(args.base_comparison_path),
        pd.read_csv(args.ann_metrics_path),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "paper_comparison.csv"
    comparison.to_csv(output_path, index=False)
    metadata = {
        "base_comparison_path": str(args.base_comparison_path),
        "ann_metrics_path": str(args.ann_metrics_path),
        "output_path": str(output_path),
        "autoets_model_specification": "ANN",
        "autoets_family": "ETS(A,N,N)",
        "response_transform": "none",
        "evaluation_scale": "original visitor nights",
        "rank_score_metrics": list(RANK_SCORE_METRICS),
        "method_count": int(len(comparison)),
    }
    (args.output_dir / "comparison.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
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


if __name__ == "__main__":
    main()
