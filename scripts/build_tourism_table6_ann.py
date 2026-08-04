#!/usr/bin/env python3
"""Build the full Tourism Table 6 with ANN in every AutoETS-based method."""

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

from vara.forecast_metrics import add_composite_rank_score


RANK_SCORE_METRICS = ("rmse", "mae", "mase", "rmsse")
NON_AUTOETS_MODELS = (
    "seasonal_naive",
    "auto_arima",
    "deepar",
    "nbeats",
    "ar1",
    "var1",
)
ANN_BENCHMARK_MODELS = (
    "auto_ets",
    "ar1_auto_ets",
    "var1_auto_ets",
)
ANN_VARA_MODEL_NAMES = {
    "reliability_gated_vara_auto_ets_oos": "vara_auto_ets_lambda",
    "raw_oos_vara": "vara_auto_ets_o",
}
DISPLAY_NAMES = {
    "vara_auto_ets_lambda": "VARA–AutoETS(ANN)_lambda",
    "auto_ets": "AutoETS(ANN)",
    "ar1_auto_ets": "AR(1)–AutoETS(ANN)",
    "vara_auto_ets_o": "VARA–AutoETS(ANN)_O",
    "auto_arima": "AutoARIMA",
    "nbeats": "N-BEATS",
    "var1_auto_ets": "VAR(1)–AutoETS(ANN)",
    "deepar": "DeepAR",
    "seasonal_naive": "Seasonal naive",
    "ar1": "AR(1)",
    "var1": "VAR(1)",
}
LATEX_NAMES = {
    **DISPLAY_NAMES,
    "vara_auto_ets_lambda": r"VARA--AutoETS(ANN)$_\lambda$",
    "ar1_auto_ets": r"AR(1)--AutoETS(ANN)",
    "vara_auto_ets_o": r"VARA--AutoETS(ANN)$_O$",
    "var1_auto_ets": r"VAR(1)--AutoETS(ANN)",
}
TABLE_METRICS = (
    "rank_score",
    "rmse",
    "mae",
    "mase",
    "rmsse",
    "oos_r2",
    "crps",
    "bias",
)


def _select_exact(
    frame: pd.DataFrame,
    model_names: tuple[str, ...],
    *,
    source: str,
) -> pd.DataFrame:
    if "model" not in frame:
        raise ValueError(f"{source} metrics have no model column.")
    if frame["model"].duplicated().any():
        raise ValueError(f"{source} metrics contain duplicate model rows.")
    indexed = frame.set_index("model", drop=False)
    missing = sorted(set(model_names).difference(indexed.index.astype(str)))
    if missing:
        raise ValueError(f"{source} metrics are missing models: {missing}")
    selected = indexed.loc[list(model_names)].copy().reset_index(drop=True)
    selected["source"] = source
    return selected


def build_table6_ann(
    standard_benchmark_summary: pd.DataFrame,
    ann_benchmark_summary: pd.DataFrame,
    ann_vara_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return the exact 11-method ANN sensitivity version of Table 6."""
    unchanged = _select_exact(
        standard_benchmark_summary,
        NON_AUTOETS_MODELS,
        source="unchanged_non_autoets_benchmark",
    )
    ann_benchmarks = _select_exact(
        ann_benchmark_summary,
        ANN_BENCHMARK_MODELS,
        source="ann_benchmark",
    )
    ann_vara = _select_exact(
        ann_vara_summary,
        tuple(ANN_VARA_MODEL_NAMES),
        source="ann_vara",
    )
    ann_vara["model"] = ann_vara["model"].map(ANN_VARA_MODEL_NAMES)

    comparison = pd.concat(
        [unchanged, ann_benchmarks, ann_vara],
        ignore_index=True,
        sort=False,
    )
    expected = set(DISPLAY_NAMES)
    observed = set(comparison["model"].astype(str))
    if observed != expected or len(comparison) != len(expected):
        raise ValueError(
            "ANN Table 6 method set does not match the prespecified 11 methods."
        )
    missing_metrics = sorted(set(TABLE_METRICS[1:]).difference(comparison.columns))
    if missing_metrics:
        raise ValueError(f"ANN Table 6 metrics are missing columns: {missing_metrics}")
    core_values = comparison.loc[:, RANK_SCORE_METRICS].to_numpy(dtype=float)
    if not np.isfinite(core_values).all():
        raise ValueError("ANN Table 6 RankScore metrics must all be finite.")

    ranked = add_composite_rank_score(
        comparison,
        lower_is_better=RANK_SCORE_METRICS,
    )
    ranked["method"] = ranked["model"].map(DISPLAY_NAMES)
    ranked["rank"] = ranked["overall_rank"].astype(int)
    ranked["autoets_family"] = np.where(
        ranked["model"].isin({
            *ANN_BENCHMARK_MODELS,
            *ANN_VARA_MODEL_NAMES.values(),
        }),
        "ANN",
        "not_applicable",
    )
    return ranked.sort_values(
        ["rank", "rmse", "model"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def validate_ann_configuration(
    ann_benchmark_config: dict,
    ann_autoets_metadata: dict,
) -> None:
    """Verify both ANN forecast sources use the same fixed family."""
    configurations = {
        "ANN benchmark": ann_benchmark_config.get("autoets_config", {}),
        "ANN rolling panel": ann_autoets_metadata.get("autoets_config", {}),
    }
    for label, config in configurations.items():
        if config.get("model_specification") != "ANN":
            raise ValueError(f"{label} does not use model specification ANN.")
        if config.get("damped") is not False:
            raise ValueError(f"{label} does not explicitly forbid damping.")
        if config.get("phi") is not None:
            raise ValueError(f"{label} unexpectedly fixes phi.")
        if config.get("selection_criterion") is not None:
            raise ValueError(f"{label} unexpectedly performs family selection.")


def validate_direct_ann_forecasts(
    ann_benchmark_forecasts: pd.DataFrame,
    ann_rolling_forecasts: pd.DataFrame,
    *,
    final_start: str,
    final_end: str,
) -> None:
    """Require the two independently generated direct ANN panels to agree."""
    benchmark = ann_benchmark_forecasts.copy()
    rolling = ann_rolling_forecasts.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    rolling["date"] = pd.to_datetime(rolling["date"])
    rolling = rolling[rolling["date"].between(
        pd.Timestamp(final_start),
        pd.Timestamp(final_end),
    )].copy()
    required_rolling = {"date", "series_id", "direct_auto_ets"}
    missing = sorted(required_rolling.difference(rolling.columns))
    if missing:
        raise ValueError(f"ANN rolling forecasts are missing columns: {missing}")
    series_ids = sorted(rolling["series_id"].astype(str).unique())
    rolling_wide = (
        rolling.pivot(
            index="date",
            columns="series_id",
            values="direct_auto_ets",
        )
        .sort_index()
        .loc[:, series_ids]
    )
    benchmark_wide = (
        benchmark.set_index("date")
        .sort_index()
        .loc[rolling_wide.index, series_ids]
    )
    if not np.allclose(
        benchmark_wide.to_numpy(dtype=float),
        rolling_wide.to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
    ):
        difference = float(np.max(np.abs(
            benchmark_wide.to_numpy(dtype=float)
            - rolling_wide.to_numpy(dtype=float)
        )))
        raise ValueError(
            "Direct ANN benchmark and rolling panels disagree; "
            f"maximum absolute difference is {difference}."
        )


def _bold(value: str, enabled: bool) -> str:
    return rf"\textbf{{{value}}}" if enabled else value


def render_latex(table: pd.DataFrame) -> str:
    """Render a standalone LaTeX fragment without editing the manuscript."""
    best = {
        "rank_score": float(table["rank_score"].max()),
        "rmse": float(table["rmse"].min()),
        "mae": float(table["mae"].min()),
        "mase": float(table["mase"].min()),
        "rmsse": float(table["rmsse"].min()),
        "oos_r2": float(table["oos_r2"].max()),
        "crps": float(table["crps"].min(skipna=True)),
        "bias": float(table.loc[table["bias"].abs().idxmin(), "bias"]),
    }
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        (
            r"\caption{Final 2016 monthly rolling-origin accuracy for the "
            r"Tourism application, using fixed ETS(A,N,N) in every "
            r"AutoETS-based method. Overall rank is determined by the rank "
            r"score based on RMSE, MAE, MASE, and RMSSE; CRPS is not included "
            r"in this score.}"
        ),
        r"\label{tab:tourism_forecasting_results_ann}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rlrrrrrrrr}",
        r"\toprule",
        (
            r"Rank & Method & RankScore & RMSE & MAE & MASE & RMSSE & "
            r"$R^2_{oos}$ & CRPS & Bias \\"
        ),
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        values = {
            "rank_score": f"{row.rank_score:.2f}",
            "rmse": f"{row.rmse:.2f}",
            "mae": f"{row.mae:.2f}",
            "mase": f"{row.mase:.3f}",
            "rmsse": f"{row.rmsse:.3f}",
            "oos_r2": f"{row.oos_r2:.3f}",
            "crps": "--" if not np.isfinite(row.crps) else f"{row.crps:.2f}",
            "bias": f"{row.bias:.2f}",
        }
        for metric, formatted in tuple(values.items()):
            is_best = (
                np.isfinite(getattr(row, metric))
                and np.isclose(float(getattr(row, metric)), best[metric])
            )
            values[metric] = _bold(formatted, is_best)
        lines.append(
            f"{row.rank} & {LATEX_NAMES[row.model]} & "
            f"{values['rank_score']} & {values['rmse']} & {values['mae']} & "
            f"{values['mase']} & {values['rmsse']} & {values['oos_r2']} & "
            f"{values['crps']} & {values['bias']} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def render_markdown(table: pd.DataFrame) -> str:
    headers = (
        "Rank",
        "Method",
        "RankScore",
        "RMSE",
        "MAE",
        "MASE",
        "RMSSE",
        "R2_oos",
        "CRPS",
        "Bias",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---:"] + ["---"] + ["---:"] * 8) + "|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join([
                str(row.rank),
                row.method,
                f"{row.rank_score:.2f}",
                f"{row.rmse:.2f}",
                f"{row.mae:.2f}",
                f"{row.mase:.3f}",
                f"{row.rmsse:.3f}",
                f"{row.oos_r2:.3f}",
                "--" if not np.isfinite(row.crps) else f"{row.crps:.2f}",
                f"{row.bias:.2f}",
            ])
            + " |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standard-benchmark-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_standardized_run/rolling-origin/"
            "metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--ann-benchmark-dir",
        type=Path,
        default=Path("outputs/tourism_benchmark_ann_run/rolling-origin"),
    )
    parser.add_argument(
        "--ann-vara-dir",
        type=Path,
        default=Path("outputs/tourism_76_reliability_vara_autoets_ann"),
    )
    parser.add_argument(
        "--ann-rolling-forecasts-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_autoets_ann_extended/rolling_forecasts.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_table6_ann"),
    )
    parser.add_argument("--final-start", default="2016-01-01")
    parser.add_argument("--final-end", default="2016-12-01")
    args = parser.parse_args()

    ann_benchmark_metrics_path = args.ann_benchmark_dir / "metrics_summary.csv"
    ann_vara_metrics_path = args.ann_vara_dir / "metrics_summary.csv"
    ann_benchmark_config_path = (
        args.ann_benchmark_dir / "config_benchmark.json"
    )
    ann_autoets_metadata_path = args.ann_rolling_forecasts_path.with_suffix(
        args.ann_rolling_forecasts_path.suffix + ".metadata.json"
    )
    validate_ann_configuration(
        json.loads(ann_benchmark_config_path.read_text()),
        json.loads(ann_autoets_metadata_path.read_text()),
    )
    validate_direct_ann_forecasts(
        pd.read_csv(args.ann_benchmark_dir / "forecast_auto_ets.csv"),
        pd.read_csv(args.ann_rolling_forecasts_path),
        final_start=args.final_start,
        final_end=args.final_end,
    )
    comparison = build_table6_ann(
        pd.read_csv(args.standard_benchmark_metrics_path),
        pd.read_csv(ann_benchmark_metrics_path),
        pd.read_csv(ann_vara_metrics_path),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "table6_ann_full.csv", index=False)
    presentation_columns = [
        "rank",
        "method",
        "rank_score",
        "rmse",
        "mae",
        "mase",
        "rmsse",
        "oos_r2",
        "crps",
        "bias",
    ]
    presentation = comparison.loc[:, presentation_columns].copy()
    presentation.to_csv(args.output_dir / "table6_ann.csv", index=False)
    (args.output_dir / "table6_ann.tex").write_text(render_latex(comparison))
    markdown = render_markdown(comparison)
    (args.output_dir / "table6_ann.md").write_text(markdown)
    metadata = {
        "standard_benchmark_metrics_path": str(
            args.standard_benchmark_metrics_path
        ),
        "ann_benchmark_metrics_path": str(ann_benchmark_metrics_path),
        "ann_vara_metrics_path": str(ann_vara_metrics_path),
        "ann_rolling_forecasts_path": str(args.ann_rolling_forecasts_path),
        "autoets_model_specification": "ANN",
        "autoets_family": "ETS(A,N,N)",
        "response_transform": "none",
        "evaluation_scale": "original visitor nights",
        "final_start": args.final_start,
        "final_end": args.final_end,
        "n_origins": 12,
        "n_series": 76,
        "method_count": int(len(comparison)),
        "rank_score_metrics": list(RANK_SCORE_METRICS),
        "direct_ann_panels_match": True,
        "manuscript_modified": False,
    }
    (args.output_dir / "table6_ann.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(markdown, end="", flush=True)


if __name__ == "__main__":
    main()
