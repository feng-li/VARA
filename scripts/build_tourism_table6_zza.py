#!/usr/bin/env python3
"""Build normalized Tourism Table 6 with ZZA in every AutoETS-based method."""

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

NON_AUTOETS_MODELS = (
    "seasonal_naive",
    "auto_arima",
    "deepar",
    "nbeats",
    "ar1",
    "var1",
)
ZZA_BENCHMARK_MODELS = (
    "auto_ets",
    "ar1_auto_ets",
    "var1_auto_ets",
)
ZZA_VARA_MODEL_NAMES = {
    "reliability_gated_vara_auto_ets_oos": "vara_auto_ets_lambda",
}
INSAMPLE_VARA_MODEL = "vara_auto_ets_insample"
DISPLAY_NAMES = {
    "vara_auto_ets_lambda": "VARA–AutoETS_λ",
    "auto_ets": "AutoETS",
    "ar1_auto_ets": "AR(1)–AutoETS",
    INSAMPLE_VARA_MODEL: "VARA–AutoETS",
    "auto_arima": "AutoARIMA",
    "nbeats": "N-BEATS",
    "var1_auto_ets": "VAR(1)–AutoETS",
    "deepar": "DeepAR",
    "seasonal_naive": "Seasonal naive",
    "ar1": "AR(1)",
    "var1": "VAR(1)",
}
LATEX_NAMES = {
    **DISPLAY_NAMES,
    "vara_auto_ets_lambda": r"VARA--AutoETS$_\lambda$",
    "ar1_auto_ets": r"AR(1)--AutoETS",
    INSAMPLE_VARA_MODEL: r"VARA--AutoETS",
    "var1_auto_ets": r"VAR(1)--AutoETS",
}
TABLE_METRICS = ("rank_score", "rmse")
RANK_SCORE_START = pd.Timestamp("2016-01-01")
RANK_SCORE_END = pd.Timestamp("2016-12-01")


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


def _select_origin_exact(
    frame: pd.DataFrame,
    model_names: tuple[str, ...],
    *,
    source: str,
) -> pd.DataFrame:
    required = {"model", "date", "mae"}
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(
            f"{source} origin metrics are missing columns: {missing_columns}"
        )
    selected = frame[frame["model"].astype(str).isin(model_names)].copy()
    missing_models = sorted(
        set(model_names).difference(selected["model"].astype(str).unique())
    )
    if missing_models:
        raise ValueError(
            f"{source} origin metrics are missing models: {missing_models}"
        )
    selected["model"] = selected["model"].astype(str)
    selected["date"] = pd.to_datetime(selected["date"])
    if selected.duplicated(["model", "date"]).any():
        raise ValueError(f"{source} origin metrics contain duplicate rows.")
    if not np.isfinite(selected["mae"].to_numpy(dtype=float)).all():
        raise ValueError(
            f"{source} origin absolute prediction errors must all be finite."
        )
    selected["source"] = source
    return selected.loc[:, ["model", "date", "mae", "source"]]


def calculate_origin_absolute_error_rank_scores(
    standard_benchmark_origins: pd.DataFrame,
    zza_benchmark_origins: pd.DataFrame,
    zza_vara_origins: pd.DataFrame,
    insample_vara_origins: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply RankScore(s) to monthly cross-series absolute prediction error."""
    unchanged = _select_origin_exact(
        standard_benchmark_origins,
        NON_AUTOETS_MODELS,
        source="unchanged_non_autoets_benchmark",
    )
    zza_benchmarks = _select_origin_exact(
        zza_benchmark_origins,
        ZZA_BENCHMARK_MODELS,
        source="zza_benchmark",
    )
    zza_vara = _select_origin_exact(
        zza_vara_origins,
        tuple(ZZA_VARA_MODEL_NAMES),
        source="zza_vara",
    )
    zza_vara["model"] = zza_vara["model"].map(ZZA_VARA_MODEL_NAMES)
    insample_vara = _select_origin_exact(
        insample_vara_origins,
        (INSAMPLE_VARA_MODEL,),
        source="insample_error_vara",
    )
    origins = pd.concat(
        [unchanged, zza_benchmarks, zza_vara, insample_vara],
        ignore_index=True,
    )
    expected = set(DISPLAY_NAMES)
    observed = set(origins["model"])
    if observed != expected:
        raise ValueError(
            "Origin absolute-error method set does not match the prespecified "
            "11 methods."
        )
    dates_by_model = origins.groupby("model", observed=True)["date"].agg(
        lambda values: tuple(sorted(pd.Timestamp(value) for value in values))
    )
    reference_dates = dates_by_model.iloc[0]
    if any(dates != reference_dates for dates in dates_by_model):
        raise ValueError(
            "Origin absolute-error methods do not share the same target dates."
        )
    expected_dates = tuple(pd.date_range(
        RANK_SCORE_START,
        RANK_SCORE_END,
        freq="MS",
    ))
    if reference_dates != expected_dates:
        raise ValueError(
            "RankScore requires exactly the 12 origins from January through "
            "December 2016."
        )

    wide = origins.pivot(index="date", columns="model", values="mae").sort_index()
    wide = wide.loc[:, sorted(expected)]
    values = wide.to_numpy(dtype=float)
    n_methods = values.shape[1]
    n_comparators = n_methods - 1
    if n_comparators != 10:
        raise ValueError(
            f"RankScore requires 10 comparators, found {n_comparators}."
        )
    # Entry (s,m) counts strict absolute-error wins for method m at origin s.
    wins = (values[:, :, None] < values[:, None, :]).sum(axis=2)
    scores = 100.0 * wins / float(n_comparators)
    score_wide = pd.DataFrame(scores, index=wide.index, columns=wide.columns)
    mean_scores = score_wide.mean(axis=0)
    summary = pd.DataFrame({
        "model": mean_scores.index,
        "rank_score": mean_scores.to_numpy(dtype=float),
        "mean_monthly_absolute_error_wins": wins.mean(axis=0),
        "n_rank_score_origins": len(wide),
        "n_rank_score_comparators": n_comparators,
    })

    audit = (
        wide.rename_axis(columns="model")
        .stack()
        .rename("absolute_prediction_error")
        .reset_index()
    )
    audit["absolute_error_wins"] = wins.reshape(-1)
    audit["rank_score_s"] = scores.reshape(-1)
    audit["method"] = audit["model"].map(DISPLAY_NAMES)
    return summary, audit


def build_table6_zza(
    standard_benchmark_summary: pd.DataFrame,
    zza_benchmark_summary: pd.DataFrame,
    zza_vara_summary: pd.DataFrame,
    insample_vara_summary: pd.DataFrame,
    standard_benchmark_origins: pd.DataFrame,
    zza_benchmark_origins: pd.DataFrame,
    zza_vara_origins: pd.DataFrame,
    insample_vara_origins: pd.DataFrame,
) -> pd.DataFrame:
    """Return the exact 11-method ZZA version of Tourism Table 6."""
    unchanged = _select_exact(
        standard_benchmark_summary,
        NON_AUTOETS_MODELS,
        source="unchanged_non_autoets_benchmark",
    )
    zza_benchmarks = _select_exact(
        zza_benchmark_summary,
        ZZA_BENCHMARK_MODELS,
        source="zza_benchmark",
    )
    zza_vara = _select_exact(
        zza_vara_summary,
        tuple(ZZA_VARA_MODEL_NAMES),
        source="zza_vara",
    )
    zza_vara["model"] = zza_vara["model"].map(ZZA_VARA_MODEL_NAMES)
    insample_vara = _select_exact(
        insample_vara_summary,
        (INSAMPLE_VARA_MODEL,),
        source="insample_error_vara",
    )

    comparison = pd.concat(
        [unchanged, zza_benchmarks, zza_vara, insample_vara],
        ignore_index=True,
        sort=False,
    )
    expected = set(DISPLAY_NAMES)
    observed = set(comparison["model"].astype(str))
    if observed != expected or len(comparison) != len(expected):
        raise ValueError(
            "ZZA Table 6 method set does not match the prespecified 11 methods."
        )
    missing_metrics = sorted(set(TABLE_METRICS[1:]).difference(comparison.columns))
    if missing_metrics:
        raise ValueError(f"ZZA Table 6 metrics are missing columns: {missing_metrics}")
    origin_scores, _ = calculate_origin_absolute_error_rank_scores(
        standard_benchmark_origins,
        zza_benchmark_origins,
        zza_vara_origins,
        insample_vara_origins,
    )
    ranked = comparison.drop(
        columns=["rank_score", "overall_rank", "rank"],
        errors="ignore",
    ).merge(origin_scores, on="model", how="left", validate="one_to_one")
    if ranked["rank_score"].isna().any():
        raise ValueError(
            "Some Table 6 methods lack an absolute-error RankScore."
        )
    ranked["overall_rank"] = (
        ranked["rank_score"].rank(method="min", ascending=False).astype(int)
    )
    ranked["method"] = ranked["model"].map(DISPLAY_NAMES)
    ranked["rank"] = ranked["overall_rank"]
    ranked["autoets_family"] = np.where(
        ranked["model"].isin({
            *ZZA_BENCHMARK_MODELS,
            *ZZA_VARA_MODEL_NAMES.values(),
            INSAMPLE_VARA_MODEL,
        }),
        "ZZA",
        "not_applicable",
    )
    ranked["response_normalization"] = np.where(
        ranked["model"].isin({
            *ZZA_BENCHMARK_MODELS,
            *ZZA_VARA_MODEL_NAMES.values(),
            INSAMPLE_VARA_MODEL,
        }),
        "training_zscore",
        "not_applicable",
    )
    return ranked.sort_values(
        ["rmse", "model"],
        ascending=[True, True],
    ).reset_index(drop=True)


def validate_zza_configuration(
    benchmark_config: dict,
    rolling_metadata: dict,
) -> None:
    """Verify both forecast sources use ZZA and training-only z-scores."""
    configurations = {
        "ZZA benchmark": benchmark_config.get("autoets_config", {}),
        "ZZA rolling panel": rolling_metadata.get("autoets_config", {}),
    }
    for label, config in configurations.items():
        if config.get("model_specification") != "ZZA":
            raise ValueError(f"{label} does not use model specification ZZA.")
        if config.get("damped") is not None:
            raise ValueError(f"{label} unexpectedly fixes damping.")
        if config.get("phi") is not None:
            raise ValueError(f"{label} unexpectedly fixes phi.")
        if config.get("selection_criterion") != "AICc":
            raise ValueError(f"{label} does not use AICc family selection.")
        if int(config.get("season_length", -1)) != 12:
            raise ValueError(f"{label} does not use monthly season length 12.")
    if rolling_metadata.get("response_transform") != "none":
        raise ValueError("ZZA rolling forecasts unexpectedly transform the response.")
    benchmark_normalization = benchmark_config.get(
        "autoets_config",
        {},
    ).get("response_normalization")
    rolling_normalization = rolling_metadata.get("response_normalization")
    for label, normalization in {
        "ZZA benchmark": benchmark_normalization,
        "ZZA rolling panel": rolling_normalization,
    }.items():
        if normalization != "training_zscore":
            raise ValueError(
                f"{label} does not use per-origin training z-score normalization."
            )


def validate_direct_zza_forecasts(
    benchmark_forecasts: pd.DataFrame,
    rolling_forecasts: pd.DataFrame,
    *,
    final_start: str,
    final_end: str,
) -> None:
    """Require independently generated direct ZZA panels to agree."""
    benchmark = benchmark_forecasts.copy()
    rolling = rolling_forecasts.copy()
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    rolling["date"] = pd.to_datetime(rolling["date"])
    rolling = rolling[rolling["date"].between(
        pd.Timestamp(final_start),
        pd.Timestamp(final_end),
    )].copy()
    required_rolling = {"date", "series_id", "direct_auto_ets"}
    missing = sorted(required_rolling.difference(rolling.columns))
    if missing:
        raise ValueError(f"ZZA rolling forecasts are missing columns: {missing}")
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
        atol=1e-8,
    ):
        difference = float(np.max(np.abs(
            benchmark_wide.to_numpy(dtype=float)
            - rolling_wide.to_numpy(dtype=float)
        )))
        raise ValueError(
            "Direct ZZA benchmark and rolling panels disagree; "
            f"maximum absolute difference is {difference}."
        )


def validate_vara_calibration(
    selection: dict,
    final_predictions: pd.DataFrame,
) -> None:
    """Check validation/test separation and the full final calibration window."""
    if selection.get("validation_end") != "2015-12-01":
        raise ValueError("ZZA VARA selection does not stop before final 2016.")
    if selection.get("final_start") != "2016-01-01":
        raise ValueError("ZZA VARA final evaluation does not begin in 2016.")
    config = selection.get("config", {})
    if int(config.get("calibration_window_months", -1)) != 36:
        raise ValueError("ZZA VARA does not retain the 36-month calibration window.")
    if int(config.get("calibration_min_months", -1)) != 24:
        raise ValueError("ZZA VARA does not use the prespecified 24-month warm-up.")
    final = final_predictions.copy()
    if not final["n_calibration_months"].eq(36).all():
        raise ValueError("A final ZZA VARA forecast lacks 36 calibration months.")
    final["date"] = pd.to_datetime(final["date"])
    final["calibration_end"] = pd.to_datetime(final["calibration_end"])
    if not (final["calibration_end"] < final["date"]).all():
        raise ValueError("ZZA VARA calibration includes a target or future month.")


def validate_insample_vara_config(config: dict) -> None:
    """Verify the prespecified in-sample-error ablation configuration."""
    if config.get("model") != INSAMPLE_VARA_MODEL:
        raise ValueError("The in-sample VARA output has an unexpected model id.")
    autoets = config.get("autoets_config", {})
    if autoets.get("model_specification") != "ZZA":
        raise ValueError("The in-sample VARA baseline does not use ZZA.")
    if autoets.get("response_normalization") != "training_zscore":
        raise ValueError(
            "The in-sample VARA baseline does not use training z-scores."
        )
    strategy = config.get("vara_strategy", {})
    expected = {
        "feature_source": "predictive-average",
        "attention_mode": "score-topk",
        "attention_top_k": 10,
        "transform": "standardize",
        "history_window": 24,
        "stability_cap": 0.5,
    }
    for key, value in expected.items():
        if strategy.get(key) != value:
            raise ValueError(
                f"The in-sample VARA setting {key!r} is "
                f"{strategy.get(key)!r}, expected {value!r}."
            )
    if float(config.get("correction_weight", np.nan)) != 1.0:
        raise ValueError("The in-sample VARA correction weight is not one.")
    if config.get("hyperparameter_search") is not False:
        raise ValueError("The in-sample VARA ablation unexpectedly tunes settings.")
    if int(config.get("n_origins", -1)) != 12:
        raise ValueError("The in-sample VARA output does not cover 12 origins.")
    if int(config.get("n_series", -1)) != 76:
        raise ValueError("The in-sample VARA output does not cover 76 series.")


def _bold(value: str, enabled: bool) -> str:
    return rf"\textbf{{{value}}}" if enabled else value


def render_latex(table: pd.DataFrame) -> str:
    """Render a standalone LaTeX fragment without editing the manuscript."""
    best = {
        "rank_score": float(table["rank_score"].max()),
        "rmse": float(table["rmse"].min()),
    }
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        (
            r"\caption{Final 2016 monthly rolling-origin accuracy for the "
            r"Tourism application, using automatic error and trend selection "
            r"with additive seasonality in every "
            r"AutoETS-based method. Each AutoETS-related series is z-score "
            r"normalized using only its rolling training sample and forecasts "
            r"are returned to the original visitor-night scale. RankScore is the 2016 average of the "
            r"monthly pairwise absolute-error win percentage defined by "
            r"$\operatorname{RankScore}(s)$ in the text. Methods are ordered "
            r"from lowest to highest RMSE.}"
        ),
        r"\label{tab:tourism_forecasting_results}",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Method & RMSE & RankScore \\",
        r"\midrule",
    ]
    for row in table.itertuples(index=False):
        values = {
            "rank_score": f"{row.rank_score:.2f}",
            "rmse": f"{row.rmse:.2f}",
        }
        for metric, formatted in tuple(values.items()):
            is_best = (
                np.isfinite(getattr(row, metric))
                and np.isclose(float(getattr(row, metric)), best[metric])
            )
            values[metric] = _bold(formatted, is_best)
        lines.append(
            f"{LATEX_NAMES[row.model]} & {values['rmse']} & "
            f"{values['rank_score']} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def render_markdown(table: pd.DataFrame) -> str:
    headers = (
        "Method",
        "RMSE",
        "RankScore",
    )
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---", "---:", "---:"]) + "|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join([
                row.method,
                f"{row.rmse:.2f}",
                f"{row.rank_score:.2f}",
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
        "--zza-benchmark-dir",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_zza_train_zscore_run/rolling-origin"
        ),
    )
    parser.add_argument(
        "--zza-vara-dir",
        type=Path,
        default=Path(
            "outputs/tourism_76_reliability_vara_autoets_zza_train_zscore"
        ),
    )
    parser.add_argument(
        "--insample-vara-dir",
        type=Path,
        default=Path(
            "outputs/tourism_76_insample_error_vara_autoets"
        ),
    )
    parser.add_argument(
        "--zza-rolling-forecasts-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_autoets_zza_train_zscore_extended/"
            "rolling_forecasts.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_table6_zza_train_zscore"),
    )
    parser.add_argument("--final-start", default="2016-01-01")
    parser.add_argument("--final-end", default="2016-12-01")
    args = parser.parse_args()

    benchmark_metrics_path = args.zza_benchmark_dir / "metrics_summary.csv"
    vara_metrics_path = args.zza_vara_dir / "metrics_summary.csv"
    insample_metrics_path = args.insample_vara_dir / "metrics_summary.csv"
    standard_origin_metrics_path = (
        args.standard_benchmark_metrics_path.parent / "metrics_by_origin.csv"
    )
    benchmark_origin_metrics_path = (
        args.zza_benchmark_dir / "metrics_by_origin.csv"
    )
    vara_origin_metrics_path = args.zza_vara_dir / "metrics_by_origin.csv"
    insample_origin_metrics_path = (
        args.insample_vara_dir / "metrics_by_origin.csv"
    )
    benchmark_config_path = args.zza_benchmark_dir / "config_benchmark.json"
    rolling_metadata_path = args.zza_rolling_forecasts_path.with_suffix(
        args.zza_rolling_forecasts_path.suffix + ".metadata.json"
    )
    vara_selection_path = args.zza_vara_dir / "selection.json"
    validate_zza_configuration(
        json.loads(benchmark_config_path.read_text()),
        json.loads(rolling_metadata_path.read_text()),
    )
    validate_direct_zza_forecasts(
        pd.read_csv(args.zza_benchmark_dir / "forecast_auto_ets.csv"),
        pd.read_csv(args.zza_rolling_forecasts_path),
        final_start=args.final_start,
        final_end=args.final_end,
    )
    vara_selection = json.loads(vara_selection_path.read_text())
    validate_vara_calibration(
        vara_selection,
        pd.read_csv(args.zza_vara_dir / "final_predictions.csv"),
    )
    validate_insample_vara_config(
        json.loads((args.insample_vara_dir / "config.json").read_text())
    )
    comparison = build_table6_zza(
        pd.read_csv(args.standard_benchmark_metrics_path),
        pd.read_csv(benchmark_metrics_path),
        pd.read_csv(vara_metrics_path),
        pd.read_csv(insample_metrics_path),
        pd.read_csv(standard_origin_metrics_path),
        pd.read_csv(benchmark_origin_metrics_path),
        pd.read_csv(vara_origin_metrics_path),
        pd.read_csv(insample_origin_metrics_path),
    )
    _, rankscore_audit = calculate_origin_absolute_error_rank_scores(
        pd.read_csv(standard_origin_metrics_path),
        pd.read_csv(benchmark_origin_metrics_path),
        pd.read_csv(vara_origin_metrics_path),
        pd.read_csv(insample_origin_metrics_path),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_dir / "table6_zza_full.csv", index=False)
    rankscore_audit.to_csv(
        args.output_dir / "table6_zza_rankscore_by_origin.csv",
        index=False,
    )
    presentation_columns = [
        "method",
        "rmse",
        "rank_score",
    ]
    presentation = comparison.loc[:, presentation_columns].copy()
    presentation.to_csv(args.output_dir / "table6_zza.csv", index=False)
    (args.output_dir / "table6_zza.tex").write_text(render_latex(comparison))
    markdown = render_markdown(comparison)
    (args.output_dir / "table6_zza.md").write_text(markdown)
    metadata = {
        "standard_benchmark_metrics_path": str(
            args.standard_benchmark_metrics_path
        ),
        "zza_benchmark_metrics_path": str(benchmark_metrics_path),
        "zza_vara_metrics_path": str(vara_metrics_path),
        "insample_vara_metrics_path": str(insample_metrics_path),
        "standard_benchmark_origin_metrics_path": str(
            standard_origin_metrics_path
        ),
        "zza_benchmark_origin_metrics_path": str(
            benchmark_origin_metrics_path
        ),
        "zza_vara_origin_metrics_path": str(vara_origin_metrics_path),
        "insample_vara_origin_metrics_path": str(
            insample_origin_metrics_path
        ),
        "zza_rolling_forecasts_path": str(args.zza_rolling_forecasts_path),
        "autoets_model_specification": "ZZA",
        "autoets_interpretation": (
            "automatic error, automatic trend and damping, additive seasonality"
        ),
        "response_transform": "none",
        "response_normalization": "training_zscore",
        "normalization_scope": "per rolling origin and per series",
        "normalization_statistics_source": (
            "training observations through t-1 only"
        ),
        "normalization_variance_ddof": 0,
        "forecast_inverse_formula": (
            "training_mean + sqrt(training_variance) * normalized_forecast"
        ),
        "evaluation_scale": "original visitor nights",
        "validation_start": vara_selection["validation_start"],
        "validation_end": vara_selection["validation_end"],
        "final_start": args.final_start,
        "final_end": args.final_end,
        "n_origins": 12,
        "n_series": 76,
        "method_count": int(len(comparison)),
        "rank_score_metric": "monthly cross-series mean absolute error",
        "rank_score_definition": (
            "For each 2016 origin s and method m, 100 times the number of the "
            "other 10 methods with larger absolute prediction error, divided "
            "by 10; then average over s=217,...,228."
        ),
        "rank_score_strict_inequality": True,
        "rank_score_comparators": 10,
        "display_columns": presentation_columns,
        "direct_zza_panels_match": True,
        "final_calibration_months": 36,
        "calibration_warmup_min_months": 24,
        "manuscript_modified_by_script": False,
    }
    (args.output_dir / "table6_zza.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(markdown, end="", flush=True)


if __name__ == "__main__":
    main()
