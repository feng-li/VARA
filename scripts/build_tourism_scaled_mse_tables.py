#!/usr/bin/env python3
"""Build Tourism tables from training-z-score-scaled forecast errors."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from _bootstrap import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT  # noqa: F401

from scripts.build_tourism_table7_pairwise_dm import (
    LATEX_LABELS,
    METHOD_LABELS,
    _latex_p_value,
    _markdown_p_value,
)
from vara.forecast_significance import diebold_mariano_test, holm_adjust


STANDARD_MODELS = (
    "seasonal_naive",
    "auto_arima",
    "nbeats",
    "deepar",
    "ar1",
    "var1",
)
ZZA_MODELS = ("auto_ets", "ar1_auto_ets", "var1_auto_ets")
CALIBRATED_SOURCE_MODEL = "reliability_gated_vara_auto_ets_oos"
CALIBRATED_MODEL = "vara_auto_ets_lambda"
INSAMPLE_MODEL = "vara_auto_ets_insample"
MODELS = (*STANDARD_MODELS, *ZZA_MODELS, CALIBRATED_MODEL, INSAMPLE_MODEL)


@dataclass(frozen=True)
class EvaluationInputs:
    name: str
    start_date: str
    end_date: str
    standard_forecasts_path: Path
    zza_forecasts_path: Path
    calibrated_predictions_path: Path
    insample_predictions_path: Path


def _select_forecasts(
    path: Path,
    models: tuple[str, ...],
    *,
    source: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "series_id", *models}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    selected = frame.loc[:, ["date", "series_id", *models]].copy()
    selected["date"] = pd.to_datetime(selected["date"])
    selected["series_id"] = selected["series_id"].astype(str)
    if selected.duplicated(["date", "series_id"]).any():
        raise ValueError(f"{source} contains duplicate date-series rows.")
    if not np.isfinite(selected.loc[:, list(models)].to_numpy(dtype=float)).all():
        raise ValueError(f"{source} contains non-finite forecasts.")
    return selected


def load_forecast_panel(inputs: EvaluationInputs) -> pd.DataFrame:
    """Load a matched 11-method forecast panel on the original response scale."""
    standard = _select_forecasts(
        inputs.standard_forecasts_path,
        STANDARD_MODELS,
        source=f"{inputs.name} standard forecasts",
    )
    zza = _select_forecasts(
        inputs.zza_forecasts_path,
        ZZA_MODELS,
        source=f"{inputs.name} ZZA forecasts",
    )
    calibrated = pd.read_csv(inputs.calibrated_predictions_path)
    calibrated_required = {
        "date",
        "series_id",
        "actual",
        CALIBRATED_SOURCE_MODEL,
    }
    missing = sorted(calibrated_required.difference(calibrated.columns))
    if missing:
        raise ValueError(f"{inputs.name} calibrated forecasts miss: {missing}")
    calibrated = calibrated.loc[:, list(calibrated_required)].rename(
        columns={CALIBRATED_SOURCE_MODEL: CALIBRATED_MODEL}
    )
    calibrated["date"] = pd.to_datetime(calibrated["date"])
    calibrated["series_id"] = calibrated["series_id"].astype(str)

    insample = pd.read_csv(inputs.insample_predictions_path)
    insample_required = {"date", "series_id", "actual", INSAMPLE_MODEL}
    missing = sorted(insample_required.difference(insample.columns))
    if missing:
        raise ValueError(f"{inputs.name} in-sample forecasts miss: {missing}")
    insample = insample.loc[:, list(insample_required)].copy()
    insample["date"] = pd.to_datetime(insample["date"])
    insample["series_id"] = insample["series_id"].astype(str)

    panel = (
        standard.merge(
            zza,
            on=["date", "series_id"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            calibrated,
            on=["date", "series_id"],
            how="inner",
            validate="one_to_one",
        )
        .merge(
            insample,
            on=["date", "series_id"],
            how="inner",
            suffixes=("", "_insample"),
            validate="one_to_one",
        )
    )
    if not np.allclose(
        panel["actual"].to_numpy(dtype=float),
        panel["actual_insample"].to_numpy(dtype=float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError("Forecast sources disagree on Tourism actual values.")
    panel = panel.drop(columns="actual_insample")
    expected_dates = tuple(pd.date_range(
        pd.Timestamp(inputs.start_date),
        pd.Timestamp(inputs.end_date),
        freq="MS",
    ))
    observed_dates = tuple(sorted(panel["date"].unique()))
    if observed_dates != expected_dates:
        raise ValueError(f"{inputs.name} does not cover the requested month grid.")
    if panel["series_id"].nunique() != 76:
        raise ValueError(f"{inputs.name} does not contain all 76 series.")
    expected_rows = len(expected_dates) * 76
    if len(panel) != expected_rows:
        raise ValueError(
            f"{inputs.name} has {len(panel)} rows; expected {expected_rows}."
        )
    values = panel.loc[:, ["actual", *MODELS]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{inputs.name} contains non-finite values.")
    return panel.sort_values(["date", "series_id"]).reset_index(drop=True)


def add_training_zscore_errors(
    panel: pd.DataFrame,
    tourism_data: pd.DataFrame,
) -> pd.DataFrame:
    """Scale each forecast error by its origin-specific training standard deviation."""
    data = tourism_data.copy()
    if "date" not in data:
        raise ValueError("Tourism data require a date column.")
    data["date"] = pd.to_datetime(data["date"])
    if data["date"].duplicated().any() or not data["date"].is_monotonic_increasing:
        raise ValueError("Tourism dates must be unique and increasing.")
    series_ids = sorted(panel["series_id"].unique())
    missing = sorted(set(series_ids).difference(data.columns))
    if missing:
        raise ValueError(f"Tourism data are missing series: {missing}")

    statistics = []
    for target_date in sorted(panel["date"].unique()):
        history = data.loc[data["date"] < target_date, series_ids]
        if history.empty:
            raise ValueError(f"No training data precede {target_date}.")
        values = history.to_numpy(dtype=float)
        means = np.mean(values, axis=0)
        variances = np.var(values, axis=0, ddof=0)
        scales = np.sqrt(variances)
        if not np.isfinite(scales).all() or (scales <= 1e-12).any():
            raise ValueError(f"Invalid training scale at {target_date}.")
        statistics.append(pd.DataFrame({
            "date": target_date,
            "series_id": series_ids,
            "training_mean": means,
            "training_variance": variances,
            "training_scale": scales,
            "n_training_months": len(history),
            "training_end": data.loc[data["date"] < target_date, "date"].iloc[-1],
        }))
    stats = pd.concat(statistics, ignore_index=True)
    evaluated = panel.merge(
        stats,
        on=["date", "series_id"],
        how="left",
        validate="one_to_one",
    )
    if evaluated["training_scale"].isna().any():
        raise ValueError("Some forecasts lack training-only z-score statistics.")

    rows = []
    for model in MODELS:
        error = (
            evaluated[model].to_numpy(dtype=float)
            - evaluated["actual"].to_numpy(dtype=float)
        ) / evaluated["training_scale"].to_numpy(dtype=float)
        rows.append(pd.DataFrame({
            "date": evaluated["date"],
            "series_id": evaluated["series_id"],
            "model": model,
            "actual": evaluated["actual"],
            "forecast": evaluated[model],
            "training_mean": evaluated["training_mean"],
            "training_variance": evaluated["training_variance"],
            "training_scale": evaluated["training_scale"],
            "n_training_months": evaluated["n_training_months"],
            "training_end": evaluated["training_end"],
            "scaled_error": error,
            "scaled_squared_error": np.square(error),
            "scaled_absolute_error": np.abs(error),
        }))
    cells = pd.concat(rows, ignore_index=True)
    if not np.isfinite(
        cells.loc[:, [
            "scaled_error",
            "scaled_squared_error",
            "scaled_absolute_error",
        ]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Scaled evaluation errors must all be finite.")
    return cells.sort_values(["date", "series_id", "model"]).reset_index(drop=True)


def calculate_accuracy(
    cells: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate pooled scaled MSE, scaled RMSE, and strict monthly RankScore."""
    origins = (
        cells.groupby(["model", "date"], as_index=False, observed=True)
        .agg(
            mse=("scaled_squared_error", "mean"),
            mae=("scaled_absolute_error", "mean"),
            n_series=("series_id", "nunique"),
        )
    )
    origins["rmse"] = np.sqrt(origins["mse"].to_numpy(dtype=float))
    if not origins["n_series"].eq(76).all():
        raise ValueError("Every origin metric must contain all 76 series.")

    absolute = origins.pivot(index="date", columns="model", values="mae")
    absolute = absolute.sort_index().loc[:, sorted(MODELS)]
    values = absolute.to_numpy(dtype=float)
    wins = (values[:, :, None] < values[:, None, :]).sum(axis=2)
    n_comparators = values.shape[1] - 1
    if n_comparators != 10:
        raise ValueError("Tourism RankScore requires exactly 10 comparators.")
    scores = 100.0 * wins / float(n_comparators)
    rank_scores = pd.DataFrame({
        "model": absolute.columns,
        "rank_score": scores.mean(axis=0),
        "mean_monthly_absolute_error_wins": wins.mean(axis=0),
    })

    accuracy = (
        cells.groupby("model", as_index=False, observed=True)
        .agg(
            mse=("scaled_squared_error", "mean"),
            mean_scaled_absolute_error=("scaled_absolute_error", "mean"),
            n_evaluation_cells=("scaled_error", "size"),
        )
        .merge(rank_scores, on="model", how="left", validate="one_to_one")
    )
    accuracy["rmse"] = np.sqrt(accuracy["mse"].to_numpy(dtype=float))
    accuracy["method"] = accuracy["model"].map(METHOD_LABELS)
    accuracy["n_origins"] = origins["date"].nunique()
    accuracy["n_rank_score_comparators"] = n_comparators
    accuracy["response_normalization"] = np.where(
        accuracy["model"].isin({*ZZA_MODELS, CALIBRATED_MODEL, INSAMPLE_MODEL}),
        "training_zscore",
        "not_applicable",
    )
    accuracy["evaluation_scale"] = "origin_training_zscore"
    accuracy = accuracy.sort_values(["mse", "model"]).reset_index(drop=True)
    accuracy.insert(0, "method_number", np.arange(1, len(accuracy) + 1))
    accuracy["rank"] = (
        accuracy["rank_score"].rank(method="min", ascending=False).astype(int)
    )

    audit = (
        absolute.rename_axis(columns="model")
        .stack()
        .rename("scaled_absolute_prediction_error")
        .reset_index()
    )
    audit["absolute_error_wins"] = wins.reshape(-1)
    audit["rank_score_s"] = scores.reshape(-1)
    return accuracy, origins, audit


def calculate_pairwise_dm_tests(
    origins: pd.DataFrame,
    method_order: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    losses = origins.pivot(index="date", columns="model", values="mse").sort_index()
    losses = losses.loc[:, list(method_order)]
    rows = []
    for row_index in range(1, len(method_order)):
        row_model = method_order[row_index]
        for column_index in range(row_index):
            column_model = method_order[column_index]
            test = diebold_mariano_test(
                losses[row_model].to_numpy(dtype=float),
                losses[column_model].to_numpy(dtype=float),
                forecast_horizon=1,
            )
            rows.append({
                "row_number": row_index + 1,
                "row_model": row_model,
                "row_method": METHOD_LABELS[row_model],
                "column_number": column_index + 1,
                "column_model": column_model,
                "column_method": METHOD_LABELS[column_model],
                "loss": "monthly_panel_average_training_zscore_squared_error",
                "lower_mse_method": column_model,
                **test,
            })
    results = pd.DataFrame(rows)
    if len(results) != 55:
        raise ValueError("Expected all 55 unique method pairs.")
    results["p_value_raw_two_sided_holm_55"] = holm_adjust(
        results["p_value_raw_two_sided"].to_numpy(dtype=float)
    )
    results["significant_raw_10pct"] = results["p_value_raw_two_sided"] < 0.10
    results["significant_raw_5pct"] = results["p_value_raw_two_sided"] < 0.05
    results["significant_holm_5pct"] = (
        results["p_value_raw_two_sided_holm_55"] < 0.05
    )
    return results, losses


def render_accuracy_markdown(accuracy: pd.DataFrame, *, numbered: bool) -> str:
    headers = ["Method", "MSE"]
    if numbered:
        headers.insert(0, "No.")
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---:" if item != "Method" else "---" for item in headers]) + "|",
    ]
    for row in accuracy.itertuples(index=False):
        values = [row.method, f"{row.mse:.4f}"]
        if numbered:
            values.insert(0, str(row.method_number))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def render_accuracy_latex(
    accuracy: pd.DataFrame,
    *,
    numbered: bool,
    horizon_months: int,
) -> str:
    best_mse = float(accuracy["mse"].min())
    if horizon_months == 12:
        caption = (
            "Final 2016 monthly rolling-origin accuracy for the Tourism "
            "application. MSE is calculated from forecast errors divided by "
            "each series' training-only standard deviation at every origin."
        )
        label = "tab:tourism_forecasting_results"
    else:
        caption = (
            "Twenty-four-month rolling-origin robustness accuracy for the "
            "Tourism application, covering January 2015 through December "
            "2016. MSE is calculated from training-z-score-scaled errors."
        )
        label = "tab:tourism_forecasting_results_h24"
    column_spec = "clr" if numbered else "lr"
    header = "No. & Method & MSE" if numbered else "Method & MSE"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
    ]
    for row in accuracy.itertuples(index=False):
        mse = f"{row.mse:.4f}"
        if np.isclose(float(row.mse), best_mse):
            mse = rf"\textbf{{{mse}}}"
        prefix = f"({row.method_number}) & " if numbered else ""
        lines.append(prefix + f"{LATEX_LABELS[row.model]} & {mse} " + r"\\")
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def render_dm_markdown(results: pd.DataFrame, method_order: tuple[str, ...]) -> str:
    lookup = results.set_index(["row_number", "column_number"])
    headers = ["Row method", *[f"({number})" for number in range(1, 11)]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---", *(["---:"] * 10)]) + "|",
    ]
    for row_number in range(2, 12):
        model = method_order[row_number - 1]
        cells = []
        for column_number in range(1, 11):
            if column_number >= row_number:
                cells.append("")
            else:
                row = lookup.loc[(row_number, column_number)]
                cells.append(_markdown_p_value(
                    float(row["p_value_raw_two_sided"]),
                    bool(row["significant_holm_5pct"]),
                ))
        lines.append(
            "| " + " | ".join([f"({row_number}) {METHOD_LABELS[model]}", *cells]) + " |"
        )
    return "\n".join(lines) + "\n"


def render_dm_latex(
    results: pd.DataFrame,
    method_order: tuple[str, ...],
    *,
    horizon_months: int,
) -> str:
    lookup = results.set_index(["row_number", "column_number"])
    if horizon_months == 12:
        caption = (
            "All pairwise Diebold--Mariano tests for the final 2016 Tourism "
            "forecasts using monthly panel-average training-z-score-scaled "
            "squared-error losses."
        )
        label = "tab:tourism_dm_results"
        accuracy_label = "tab:tourism_forecasting_results"
    else:
        caption = (
            "All pairwise Diebold--Mariano tests over the 24 monthly Tourism "
            "rolling origins from January 2015 through December 2016 using "
            "training-z-score-scaled squared-error losses."
        )
        label = "tab:tourism_dm_results_h24"
        accuracy_label = "tab:tourism_forecasting_results_h24"
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{caption} The \(10\times10\) lower triangle contains all 55 unique comparisons among the 11 methods.}}",
        rf"\label{{{label}}}",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}l*{10}{c}@{}}",
        r"\toprule",
        "Row method & "
        + " & ".join(f"({number})" for number in range(1, 11))
        + r" \\",
        r"\midrule",
    ]
    for row_number in range(2, 12):
        model = method_order[row_number - 1]
        cells = []
        for column_number in range(1, 11):
            if column_number >= row_number:
                cells.append("")
            else:
                row = lookup.loc[(row_number, column_number)]
                cells.append(_latex_p_value(
                    float(row["p_value_raw_two_sided"]),
                    bool(row["significant_holm_5pct"]),
                ))
        lines.append(
            rf"({row_number}) {LATEX_LABELS[model]} & "
            + " & ".join(cells)
            + r" \\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}[flushleft]",
        r"\scriptsize",
        (
            r"\item Notes: Method (1) is identified in the accuracy table; "
            r"methods (2)--(11) are identified by the row labels. Numbering "
            rf"follows ascending scaled MSE in Table~\ref{{{accuracy_label}}}."
        ),
        (
            r"\item Each cell reports the raw two-sided asymptotic-normal "
            r"\(p\)-value for the monthly panel-average scaled squared-error "
            r"loss differential (row minus column). At each origin, every "
            r"method's error for a series is divided by that series' standard "
            r"deviation calculated from observations through \(t-1\). A "
            r"significant cell favors the lower-MSE column method. "
            r"Superscripts \textsuperscript{***}, \textsuperscript{**}, and "
            r"\textsuperscript{*} denote raw \(p<0.001\), \(p<0.05\), and "
            r"\(p<0.1\). Bold entries remain significant at 5\% after Holm "
            r"correction over all 55 tests."
        ),
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def write_evaluation(
    inputs: EvaluationInputs,
    tourism_data: pd.DataFrame,
    output_dir: Path,
) -> dict[str, object]:
    panel = load_forecast_panel(inputs)
    cells = add_training_zscore_errors(panel, tourism_data)
    accuracy, origins, rank_audit = calculate_accuracy(cells)
    method_order = tuple(accuracy["model"].astype(str))
    dm_results, losses = calculate_pairwise_dm_tests(origins, method_order)
    horizon_months = int(origins["date"].nunique())
    numbered = horizon_months == 24

    output_dir.mkdir(parents=True, exist_ok=True)
    cells.to_csv(output_dir / "scaled_errors_by_cell.csv", index=False)
    origins.to_csv(output_dir / "scaled_metrics_by_origin.csv", index=False)
    accuracy.to_csv(output_dir / "accuracy_full.csv", index=False)
    rank_audit.to_csv(output_dir / "rankscore_by_origin.csv", index=False)
    losses.to_csv(output_dir / "monthly_scaled_mse_losses.csv", index_label="date")
    dm_results.to_csv(output_dir / "pairwise_dm_tests.csv", index=False)
    accuracy_md = render_accuracy_markdown(accuracy, numbered=numbered)
    dm_md = render_dm_markdown(dm_results, method_order)
    (output_dir / "accuracy.md").write_text(accuracy_md)
    (output_dir / "accuracy.tex").write_text(render_accuracy_latex(
        accuracy,
        numbered=numbered,
        horizon_months=horizon_months,
    ))
    (output_dir / "pairwise_dm.md").write_text(dm_md)
    (output_dir / "pairwise_dm.tex").write_text(render_dm_latex(
        dm_results,
        method_order,
        horizon_months=horizon_months,
    ))

    # Files with the source model ids expected by the existing Table 6 plotter.
    origins[origins["model"].isin(STANDARD_MODELS)].to_csv(
        output_dir / "origin_standard.csv", index=False
    )
    origins[origins["model"].isin(ZZA_MODELS)].to_csv(
        output_dir / "origin_zza.csv", index=False
    )
    vara_origins = origins[origins["model"].eq(CALIBRATED_MODEL)].copy()
    vara_origins["model"] = CALIBRATED_SOURCE_MODEL
    vara_origins.to_csv(output_dir / "origin_vara.csv", index=False)
    origins[origins["model"].eq(INSAMPLE_MODEL)].to_csv(
        output_dir / "origin_insample.csv", index=False
    )

    metadata = {
        "name": inputs.name,
        "evaluation_start": inputs.start_date,
        "evaluation_end": inputs.end_date,
        "n_origins": horizon_months,
        "n_series": 76,
        "n_methods": 11,
        "forecast_lead_months": 1,
        "metric": "mean squared training-z-score-scaled forecast error",
        "training_scale_ddof": 0,
        "training_data_end": "target month t-1",
        "forecast_values_modified": False,
        "model_fits_modified": False,
        "lambda_or_gate_modified": False,
        "method_order": list(method_order),
        "raw_significant_5pct": int(dm_results["significant_raw_5pct"].sum()),
        "holm_significant_5pct": int(dm_results["significant_holm_5pct"].sum()),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(f"\n{inputs.name} scaled-MSE accuracy:\n")
    print(accuracy_md, end="")
    print(
        f"Raw p<0.05: {metadata['raw_significant_5pct']}/55; "
        f"Holm p<0.05: {metadata['holm_significant_5pct']}/55",
        flush=True,
    )
    return {
        "accuracy": accuracy,
        "origins": origins,
        "dm_results": dm_results,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/tourism/tourism.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_scaled_mse_tables"),
    )
    args = parser.parse_args()
    tourism_data = pd.read_csv(args.data_path)
    evaluations = (
        EvaluationInputs(
            name="h12",
            start_date="2016-01-01",
            end_date="2016-12-01",
            standard_forecasts_path=Path(
                "outputs/tourism_benchmark_standardized_run/rolling-origin/"
                "benchmark_forecasts_long.csv"
            ),
            zza_forecasts_path=Path(
                "outputs/tourism_benchmark_zza_train_zscore_run/rolling-origin/"
                "benchmark_forecasts_long.csv"
            ),
            calibrated_predictions_path=Path(
                "outputs/tourism_76_reliability_vara_autoets_zza_train_zscore/"
                "final_predictions.csv"
            ),
            insample_predictions_path=Path(
                "outputs/tourism_76_insample_error_vara_autoets/predictions.csv"
            ),
        ),
        EvaluationInputs(
            name="h24",
            start_date="2015-01-01",
            end_date="2016-12-01",
            standard_forecasts_path=Path(
                "outputs/tourism_benchmark_standardized_h24_run/rolling-origin/"
                "benchmark_forecasts_long.csv"
            ),
            zza_forecasts_path=Path(
                "outputs/tourism_benchmark_zza_train_zscore_h24_run/rolling-origin/"
                "benchmark_forecasts_long.csv"
            ),
            calibrated_predictions_path=Path(
                "outputs/tourism_76_reliability_vara_autoets_zza_train_zscore_h24/"
                "final_predictions.csv"
            ),
            insample_predictions_path=Path(
                "outputs/tourism_76_insample_error_vara_autoets_h24/predictions.csv"
            ),
        ),
    )
    for inputs in evaluations:
        write_evaluation(inputs, tourism_data, args.output_dir / inputs.name)


if __name__ == "__main__":
    main()
