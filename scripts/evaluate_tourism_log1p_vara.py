#!/usr/bin/env python3
"""Evaluate log1p VARA--AutoETS corrections on the original Tourism scale."""

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

from vara.forecast_metrics import add_composite_rank_score, forecast_metric_tables


RAW_MODEL = "log1p_raw_oos_vara"
TUNED_MODEL = "log1p_tuned_oos_vara"
AUTOETS_MODEL = "log1p_auto_ets"
# Backward-compatible name used by the first uncalibrated comparison.
MODEL = RAW_MODEL
RANK_SCORE_METRICS = ("rmse", "mae", "mase", "rmsse")
PAPER_MODEL_NAMES = {
    "reliability_gated_vara_auto_ets_oos": "vara_auto_ets_lambda",
    "raw_oos_vara": "vara_auto_ets_o",
    "direct_auto_ets": "auto_ets",
}


def _validate_unique(frame: pd.DataFrame, keys: list[str], label: str) -> None:
    if frame.duplicated(keys).any():
        raise ValueError(f"{label} contains duplicate {'/'.join(keys)} rows.")


def build_log1p_predictions(
    autoets_forecasts: pd.DataFrame,
    vara_results: pd.DataFrame,
    *,
    strategy_id: str,
    final_start: str,
    final_end: str,
    calibration_window_months: int | None = None,
    lambda_cap: float = 1.25,
) -> pd.DataFrame:
    """Build unit-weight and optionally rolling-lambda log1p forecasts."""
    required_autoets = {
        "date",
        "series_id",
        "actual",
        "direct_auto_ets",
        "autoets_history_end",
    }
    required_vara = {
        "date",
        "series_id",
        "strategy_id",
        "vara_correction",
        "history_start",
        "history_end",
    }
    missing_autoets = sorted(required_autoets.difference(autoets_forecasts.columns))
    missing_vara = sorted(required_vara.difference(vara_results.columns))
    if missing_autoets:
        raise ValueError(f"Log1p AutoETS forecasts are missing columns: {missing_autoets}")
    if missing_vara:
        raise ValueError(f"Log1p VARA results are missing columns: {missing_vara}")

    autoets = autoets_forecasts.copy()
    results = vara_results[
        vara_results["strategy_id"].astype(str) == str(strategy_id)
    ].copy()
    if results.empty:
        raise ValueError(f"No VARA results found for strategy {strategy_id!r}.")

    for frame in (autoets, results):
        frame["date"] = pd.to_datetime(frame["date"])
        frame["series_id"] = frame["series_id"].astype(str)
    autoets["autoets_history_end"] = pd.to_datetime(autoets["autoets_history_end"])
    results["history_start"] = pd.to_datetime(results["history_start"])
    results["history_end"] = pd.to_datetime(results["history_end"])
    _validate_unique(autoets, ["date", "series_id"], "Log1p AutoETS forecasts")
    _validate_unique(results, ["date", "series_id"], "Log1p VARA results")

    keep_autoets = [
        "date",
        "series_id",
        "actual",
        "direct_auto_ets",
        "autoets_history_end",
    ]
    if "actual_original" in autoets:
        keep_autoets.append("actual_original")
    if "response_transform" in autoets:
        keep_autoets.append("response_transform")
    for normalization_column in (
        "response_normalization",
        "direct_auto_ets_normalized",
        "autoets_training_mean",
        "autoets_training_variance",
        "autoets_training_scale",
        "autoets_clip_nonnegative",
    ):
        if normalization_column in autoets:
            keep_autoets.append(normalization_column)
    merged = autoets.loc[:, keep_autoets].merge(
        results.loc[:, [
            "date",
            "series_id",
            "strategy_id",
            "vara_correction",
            "history_start",
            "history_end",
        ]],
        on=["date", "series_id"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("No aligned log1p forecasts are available.")
    if "response_transform" in merged and not (
        merged["response_transform"].astype(str) == "log1p"
    ).all():
        raise ValueError("AutoETS forecast panel was not generated in log1p space.")
    if (
        "response_normalization" in merged
        and merged["response_normalization"].astype(str).eq(
            "training_zscore"
        ).all()
    ):
        required_normalization = {
            "direct_auto_ets_normalized",
            "autoets_training_mean",
            "autoets_training_variance",
            "autoets_training_scale",
        }
        missing_normalization = sorted(
            required_normalization.difference(merged.columns)
        )
        if missing_normalization:
            raise ValueError(
                "Normalized AutoETS forecasts are missing columns: "
                f"{missing_normalization}"
            )
        reconstructed = (
            merged["autoets_training_mean"].to_numpy(dtype=float)
            + merged["autoets_training_scale"].to_numpy(dtype=float)
            * merged["direct_auto_ets_normalized"].to_numpy(dtype=float)
        )
        if (
            "autoets_clip_nonnegative" in merged
            and merged["autoets_clip_nonnegative"].astype(bool).all()
        ):
            reconstructed = np.maximum(reconstructed, 0.0)
        if not np.allclose(
            reconstructed,
            merged["direct_auto_ets"].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-8,
        ):
            raise ValueError(
                "AutoETS forecasts do not match the saved training-statistic "
                "inverse transformation."
            )
    if not (merged["autoets_history_end"] < merged["date"]).all():
        raise ValueError("AutoETS history must end before every target month.")
    if not (merged["history_end"] < merged["date"]).all():
        raise ValueError("VARA error history must end before every target month.")

    numeric = merged.loc[:, [
        "actual",
        "direct_auto_ets",
        "vara_correction",
    ]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Log1p predictions contain non-finite model-scale values.")
    implied_actual = np.expm1(merged["actual"].to_numpy(dtype=float))
    if "actual_original" in merged:
        original_actual = merged["actual_original"].to_numpy(dtype=float)
        if not np.allclose(implied_actual, original_actual, rtol=1e-10, atol=1e-8):
            raise ValueError("Saved original observations do not match expm1(actual).")
    else:
        original_actual = implied_actual

    merged["actual_original"] = original_actual
    merged["raw_forecast_log1p"] = (
        merged["direct_auto_ets"].to_numpy(dtype=float)
        + merged["vara_correction"].to_numpy(dtype=float)
    )
    merged[RAW_MODEL] = np.maximum(
        np.expm1(merged["raw_forecast_log1p"].to_numpy(dtype=float)),
        0.0,
    )
    merged["direct_auto_ets_original"] = np.maximum(
        np.expm1(merged["direct_auto_ets"].to_numpy(dtype=float)),
        0.0,
    )
    merged[AUTOETS_MODEL] = merged["direct_auto_ets_original"]
    if calibration_window_months is not None:
        return add_rolling_tuned_lambda(
            merged,
            final_start=final_start,
            final_end=final_end,
            calibration_window_months=calibration_window_months,
            lambda_cap=lambda_cap,
        )
    final = merged[merged["date"].between(
        pd.Timestamp(final_start),
        pd.Timestamp(final_end),
    )].copy()
    if final.empty:
        raise ValueError("No aligned log1p forecasts fall in the final period.")
    return final.sort_values(["date", "series_id"]).reset_index(drop=True)


def add_rolling_tuned_lambda(
    panel: pd.DataFrame,
    *,
    final_start: str,
    final_end: str,
    calibration_window_months: int,
    lambda_cap: float,
) -> pd.DataFrame:
    """Fit one leakage-safe correction weight at each rolling origin.

    The weight minimizes squared AutoETS error on the log1p scale over the
    preceding ``calibration_window_months`` complete origins. It has no
    intercept and is clipped to ``[0, lambda_cap]``.
    """
    required = {
        "date",
        "series_id",
        "actual",
        "direct_auto_ets",
        "vara_correction",
        RAW_MODEL,
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Lambda-tuning panel is missing columns: {missing}")
    if int(calibration_window_months) <= 0:
        raise ValueError("calibration_window_months must be positive.")
    if float(lambda_cap) <= 0.0:
        raise ValueError("lambda_cap must be positive.")

    frame = panel.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    target_dates = sorted(
        frame.loc[
            frame["date"].between(
                pd.Timestamp(final_start),
                pd.Timestamp(final_end),
            ),
            "date",
        ].unique()
    )
    if not target_dates:
        raise ValueError("No target dates are available for lambda tuning.")

    pieces = []
    for target_date in target_dates:
        prior_dates = sorted(
            frame.loc[frame["date"] < target_date, "date"].unique()
        )[-int(calibration_window_months):]
        if len(prior_dates) != int(calibration_window_months):
            raise ValueError(
                f"Target {pd.Timestamp(target_date).date()} has "
                f"{len(prior_dates)} prior origins; expected "
                f"{calibration_window_months}."
            )
        train = frame[frame["date"].isin(prior_dates)]
        test = frame[frame["date"] == target_date].copy()
        correction = train["vara_correction"].to_numpy(dtype=float)
        autoets_error = (
            train["actual"].to_numpy(dtype=float)
            - train["direct_auto_ets"].to_numpy(dtype=float)
        )
        correction_square = float(np.dot(correction, correction))
        if correction_square <= 1e-12:
            raise ValueError("VARA corrections have zero calibration variation.")
        raw_lambda = float(np.dot(correction, autoets_error) / correction_square)
        tuned_lambda = float(np.clip(raw_lambda, 0.0, float(lambda_cap)))
        test["lambda"] = tuned_lambda
        test["raw_lambda"] = raw_lambda
        test["calibration_start"] = pd.Timestamp(prior_dates[0])
        test["calibration_end"] = pd.Timestamp(prior_dates[-1])
        test["n_calibration_months"] = int(len(prior_dates))
        test["tuned_forecast_log1p"] = (
            test["direct_auto_ets"].to_numpy(dtype=float)
            + tuned_lambda * test["vara_correction"].to_numpy(dtype=float)
        )
        test[TUNED_MODEL] = np.maximum(
            np.expm1(test["tuned_forecast_log1p"].to_numpy(dtype=float)),
            0.0,
        )
        pieces.append(test)
    result = pd.concat(pieces, ignore_index=True)
    if not (result["calibration_end"] < result["date"]).all():
        raise RuntimeError("Lambda calibration includes a target or future origin.")
    return result.sort_values(["date", "series_id"]).reset_index(drop=True)


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


def original_scale_metric_tables(
    predictions: pd.DataFrame,
    original_data: pd.DataFrame,
    *,
    final_start: str,
    season_length: int,
    response_normalization: str = "none",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the paper metrics using forecasts, actuals, and scales in raw units."""
    raw = original_data.copy()
    if "date" not in raw:
        raise ValueError("Original Tourism data must contain a date column.")
    raw["date"] = pd.to_datetime(raw["date"])
    dates = sorted(pd.to_datetime(predictions["date"].unique()))
    series_ids = sorted(predictions["series_id"].astype(str).unique())
    missing_series = sorted(set(series_ids).difference(raw.columns))
    if missing_series:
        raise ValueError(f"Original Tourism data is missing series: {missing_series}")

    model_names = [
        model
        for model in (AUTOETS_MODEL, RAW_MODEL, TUNED_MODEL)
        if model in predictions.columns
    ]
    if not model_names:
        raise ValueError("Predictions contain no log1p VARA forecast columns.")
    forecasts = {
        model: _wide(predictions, model, dates, series_ids)
        for model in model_names
    }
    actual = _wide(predictions, "actual_original", dates, series_ids)
    raw_test = (
        raw.set_index("date")
        .reindex(index=dates)
        .loc[:, series_ids]
        .to_numpy(dtype=float)
    )
    if not np.allclose(actual, raw_test, rtol=1e-10, atol=1e-8):
        raise ValueError("Evaluation actuals do not match the original Tourism panel.")
    train = (
        raw[raw["date"] < pd.Timestamp(final_start)]
        .sort_values("date")
        .loc[:, series_ids]
        .to_numpy(dtype=float)
    )
    return forecast_metric_tables(
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
            "response_transform": "log1p",
            "response_normalization": response_normalization,
            "evaluation_scale": "original",
        },
    )


def build_paper_comparison(
    log1p_summary: pd.DataFrame,
    benchmark_summary: pd.DataFrame,
    paper_vara_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Add the log1p method and recompute the paper's relative RankScore."""
    benchmark = benchmark_summary.copy()
    benchmark["source"] = "paper_benchmark"
    paper_vara = paper_vara_summary[
        paper_vara_summary["model"].isin(PAPER_MODEL_NAMES)
    ].copy()
    paper_vara["model"] = paper_vara["model"].map(PAPER_MODEL_NAMES)
    paper_vara["source"] = "paper_vara"
    # Prefer the direct AutoETS metrics regenerated with the application
    # forecast panel. Fall back to the benchmark-table AutoETS row only when
    # the refreshed row is unavailable.
    if "auto_ets" in set(paper_vara["model"].astype(str)):
        benchmark = benchmark[
            benchmark["model"].astype(str) != "auto_ets"
        ].copy()
    log1p = log1p_summary.copy()
    log1p["source"] = "log1p_oos_error_vara"
    comparison = pd.concat(
        [benchmark, paper_vara, log1p],
        ignore_index=True,
        sort=False,
    )
    if comparison["model"].duplicated().any():
        duplicates = sorted(
            comparison.loc[
                comparison["model"].duplicated(False), "model"
            ].astype(str).unique()
        )
        raise ValueError(f"Paper comparison contains duplicate methods: {duplicates}")
    ranked = add_composite_rank_score(
        comparison,
        lower_is_better=RANK_SCORE_METRICS,
    )
    autoets_rmse = float(
        ranked.loc[ranked["model"] == "auto_ets", "rmse"].iloc[0]
    )
    raw_vara_rmse = float(
        ranked.loc[ranked["model"] == "vara_auto_ets_o", "rmse"].iloc[0]
    )
    ranked["rmse_improvement_vs_auto_ets_pct"] = (
        100.0 * (autoets_rmse - ranked["rmse"]) / autoets_rmse
    )
    ranked["rmse_improvement_vs_raw_vara_pct"] = (
        100.0 * (raw_vara_rmse - ranked["rmse"]) / raw_vara_rmse
    )
    return ranked.sort_values(
        ["overall_rank", "rmse", "model"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_transformation_comparison(
    log1p_summary: pd.DataFrame,
    level_summary: pd.DataFrame,
    *,
    vara_window_months: int,
    log1p_response_normalization: str = "none",
) -> pd.DataFrame:
    """Compare level and log1p AutoETS/VARA variants without reranking Table 6."""
    level_names = {
        "direct_auto_ets": ("AutoETS(ZZA)", "none", "none"),
        "raw_oos_vara": ("VARA–AutoETS(ZZA)_O", "none", "1"),
        "reliability_gated_vara_auto_ets_oos": (
            "VARA–AutoETS(ZZA)_λ",
            "none",
            "adaptive",
        ),
    }
    log1p_names = {
        AUTOETS_MODEL: ("log1p AutoETS(ZZA)", "log1p", "none"),
        RAW_MODEL: ("log1p VARA–AutoETS(ZZA)_O", "log1p", "1"),
        TUNED_MODEL: ("log1p VARA–AutoETS(ZZA)_λ", "log1p", "adaptive"),
    }
    pieces = []
    for source, names in (
        (level_summary, level_names),
        (log1p_summary, log1p_names),
    ):
        selected = source[source["model"].astype(str).isin(names)].copy()
        missing = sorted(set(names).difference(selected["model"].astype(str)))
        if missing:
            raise ValueError(
                f"Transformation comparison is missing models: {missing}"
            )
        selected["method"] = selected["model"].map(
            {model: values[0] for model, values in names.items()}
        )
        selected["response_transform"] = selected["model"].map(
            {model: values[1] for model, values in names.items()}
        )
        selected["correction_weight"] = selected["model"].map(
            {model: values[2] for model, values in names.items()}
        )
        selected["response_normalization"] = np.where(
            selected["response_transform"].eq("log1p"),
            log1p_response_normalization,
            "none",
        )
        pieces.append(selected)
    comparison = pd.concat(pieces, ignore_index=True, sort=False)
    comparison["vara_window_months"] = np.where(
        comparison["correction_weight"].eq("none"),
        np.nan,
        float(vara_window_months),
    )
    return comparison.loc[:, [
        "method",
        "response_transform",
        "response_normalization",
        "correction_weight",
        "vara_window_months",
        "rmse",
        "mae",
        "bias",
    ]].sort_values(
        ["response_transform", "correction_weight", "method"],
    ).reset_index(drop=True)


def run_evaluation(
    *,
    data_path: Path,
    autoets_forecasts_path: Path,
    vara_results_path: Path,
    benchmark_metrics_path: Path,
    paper_vara_metrics_path: Path,
    output_dir: Path,
    strategy_id: str,
    final_start: str,
    final_end: str,
    season_length: int,
    vara_window_months: int,
    calibration_window_months: int,
    lambda_cap: float,
    expected_autoets_specification: str | None = None,
    expected_response_normalization: str | None = None,
) -> pd.DataFrame:
    autoets_metadata_path = autoets_forecasts_path.with_suffix(
        autoets_forecasts_path.suffix + ".metadata.json"
    )
    if not autoets_metadata_path.exists():
        raise ValueError("The log1p AutoETS forecast metadata file is missing.")
    autoets_metadata = json.loads(autoets_metadata_path.read_text())
    autoets_specification = autoets_metadata.get(
        "autoets_config",
        {},
    ).get("model_specification")
    if (
        expected_autoets_specification is not None
        and autoets_specification != expected_autoets_specification
    ):
        raise ValueError(
            f"Expected AutoETS({expected_autoets_specification}), found "
            f"AutoETS({autoets_specification})."
        )
    if autoets_metadata.get("response_transform") != "log1p":
        raise ValueError("The AutoETS panel metadata does not specify log1p.")
    response_normalization = autoets_metadata.get(
        "response_normalization",
        "none",
    )
    if (
        expected_response_normalization is not None
        and response_normalization != expected_response_normalization
    ):
        raise ValueError(
            f"Expected response normalization "
            f"{expected_response_normalization!r}, found "
            f"{response_normalization!r}."
        )
    autoets = pd.read_csv(autoets_forecasts_path)
    vara = pd.read_csv(vara_results_path)
    if "n_train" not in vara:
        raise ValueError("VARA results do not record their training-window length.")
    chosen_vara = vara[
        vara["strategy_id"].astype(str).eq(str(strategy_id))
    ]
    if chosen_vara.empty or not chosen_vara["n_train"].eq(
        int(vara_window_months)
    ).all():
        raise ValueError(
            "VARA results do not use the requested fixed residual window."
        )
    raw = pd.read_csv(data_path)
    predictions = build_log1p_predictions(
        autoets,
        vara,
        strategy_id=strategy_id,
        final_start=final_start,
        final_end=final_end,
        calibration_window_months=calibration_window_months,
        lambda_cap=lambda_cap,
    )
    by_series, by_origin, summary = original_scale_metric_tables(
        predictions,
        raw,
        final_start=final_start,
        season_length=season_length,
        response_normalization=response_normalization,
    )
    comparison = build_paper_comparison(
        summary,
        pd.read_csv(benchmark_metrics_path),
        pd.read_csv(paper_vara_metrics_path),
    )
    transformation_comparison = build_transformation_comparison(
        summary,
        pd.read_csv(paper_vara_metrics_path),
        vara_window_months=vara_window_months,
        log1p_response_normalization=response_normalization,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "final_predictions.csv", index=False)
    by_series.to_csv(output_dir / "metrics_by_series.csv", index=False)
    by_origin.to_csv(output_dir / "metrics_by_origin.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    comparison.to_csv(output_dir / "paper_comparison.csv", index=False)
    transformation_comparison.to_csv(
        output_dir / "transformation_comparison.csv",
        index=False,
    )
    predictions.loc[:, [
        "date",
        "lambda",
        "raw_lambda",
        "calibration_start",
        "calibration_end",
        "n_calibration_months",
    ]].drop_duplicates("date").to_csv(
        output_dir / "rolling_lambdas.csv",
        index=False,
    )
    metadata = {
        "models": [AUTOETS_MODEL, RAW_MODEL, TUNED_MODEL],
        "response_transform": "log1p",
        "response_normalization": response_normalization,
        "autoets_model_specification": autoets_specification,
        "autoets_error_definition": (
            "log1p(actual) - rolling AutoETS forecast fitted to log1p history"
        ),
        "raw_vara_correction_weight": 1.0,
        "raw_forecast_formula": (
            "max(expm1(log1p_AutoETS_forecast + "
            "VARA_AutoETS_error_forecast), 0)"
        ),
        "tuned_forecast_formula": (
            "max(expm1(log1p_AutoETS_forecast + lambda_t * "
            "VARA_AutoETS_error_forecast), 0)"
        ),
        "lambda_objective": (
            "pooled squared log1p AutoETS error over prior rolling origins"
        ),
        "lambda_intercept": False,
        "lambda_bounds": [0.0, float(lambda_cap)],
        "calibration_window_months": int(calibration_window_months),
        "metric_scale": "original visitor nights",
        "rank_score_metrics": list(RANK_SCORE_METRICS),
        "rank_score_note": (
            "paper_comparison.csv retains the legacy composite sensitivity "
            "score and is not the current Table 6 absolute-error RankScore"
        ),
        "strategy_id": strategy_id,
        "vara_history_window_months": int(vara_window_months),
        "final_start": final_start,
        "final_end": final_end,
        "n_origins": int(predictions["date"].nunique()),
        "n_series": int(predictions["series_id"].nunique()),
        "n_cells": int(len(predictions)),
        "leakage_control": (
            "AutoETS normalization statistics and AutoETS/VARA histories all "
            "end before each forecast target"
        ),
    }
    (output_dir / "evaluation.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("Log1p VARA--AutoETS original-scale metrics:", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\nPaper comparison with recomputed RankScore:", flush=True)
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
    parser.add_argument("--data-path", type=Path, default=Path("data/tourism/tourism.csv"))
    parser.add_argument(
        "--autoets-forecasts-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_autoets_log1p_extended/rolling_forecasts.csv"
        ),
    )
    parser.add_argument(
        "--vara-results-path",
        type=Path,
        default=Path("outputs/tourism_76_vara_log1p_extended/origin_results.csv"),
    )
    parser.add_argument(
        "--benchmark-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_standardized_run/rolling-origin/"
            "metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--paper-vara-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_reliability_vara_autoets/metrics_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_76_log1p_raw_oos_vara"),
    )
    parser.add_argument(
        "--strategy-id",
        default="predictive_avg_topk10_w24_stable05",
    )
    parser.add_argument("--final-start", default="2016-01-01")
    parser.add_argument("--final-end", default="2016-12-01")
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--vara-window-months", type=int, default=24)
    parser.add_argument("--calibration-window-months", type=int, default=36)
    parser.add_argument("--lambda-cap", type=float, default=1.25)
    parser.add_argument("--expected-autoets-specification", default=None)
    parser.add_argument("--expected-response-normalization", default=None)
    args = parser.parse_args()
    run_evaluation(
        data_path=args.data_path,
        autoets_forecasts_path=args.autoets_forecasts_path,
        vara_results_path=args.vara_results_path,
        benchmark_metrics_path=args.benchmark_metrics_path,
        paper_vara_metrics_path=args.paper_vara_metrics_path,
        output_dir=args.output_dir,
        strategy_id=args.strategy_id,
        final_start=args.final_start,
        final_end=args.final_end,
        season_length=args.season_length,
        vara_window_months=args.vara_window_months,
        calibration_window_months=args.calibration_window_months,
        lambda_cap=args.lambda_cap,
        expected_autoets_specification=args.expected_autoets_specification,
        expected_response_normalization=args.expected_response_normalization,
    )


if __name__ == "__main__":
    main()
