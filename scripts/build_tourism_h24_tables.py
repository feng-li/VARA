#!/usr/bin/env python3
"""Build Tourism accuracy and pairwise-DM tables over 24 monthly origins."""

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

from scripts.build_tourism_table7_pairwise_dm import (
    LATEX_LABELS,
    METHOD_LABELS,
    _latex_p_value,
    _markdown_p_value,
)
from vara.forecast_significance import diebold_mariano_test, holm_adjust


STANDARD_MODELS = {
    "seasonal_naive": "seasonal_naive",
    "auto_arima": "auto_arima",
    "deepar": "deepar",
    "nbeats": "nbeats",
    "ar1": "ar1",
    "var1": "var1",
}
ZZA_MODELS = {
    "auto_ets": "auto_ets",
    "ar1_auto_ets": "ar1_auto_ets",
    "var1_auto_ets": "var1_auto_ets",
}
CALIBRATED_VARA_MODELS = {
    "reliability_gated_vara_auto_ets_oos": "vara_auto_ets_lambda",
}
INSAMPLE_VARA_MODELS = {
    "vara_auto_ets_insample": "vara_auto_ets_insample",
}
EXPECTED_MODELS = frozenset(METHOD_LABELS)


def _select_origin_metrics(
    path: Path,
    model_mapping: dict[str, str],
    *,
    source: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"model", "date", "rmse", "mae"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    selected = frame[frame["model"].astype(str).isin(model_mapping)].copy()
    selected["model"] = selected["model"].astype(str).map(model_mapping)
    selected["date"] = pd.to_datetime(selected["date"])
    selected = selected[selected["date"].between(start_date, end_date)].copy()
    expected = set(model_mapping.values())
    if set(selected["model"]) != expected:
        missing_models = sorted(expected.difference(selected["model"]))
        raise ValueError(f"{source} is missing models: {missing_models}")
    if selected.duplicated(["model", "date"]).any():
        raise ValueError(f"{source} has duplicate model-date rows.")
    values = selected.loc[:, ["rmse", "mae"]].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{source} contains non-finite origin metrics.")
    return selected.loc[:, ["model", "date", "rmse", "mae"]]


def load_origin_metrics(
    standard_origins_path: Path,
    zza_origins_path: Path,
    calibrated_vara_origins_path: Path,
    insample_vara_origins_path: Path,
    *,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Load the matched 11-method origin metrics for the requested window."""
    specifications = (
        (standard_origins_path, STANDARD_MODELS, "standard benchmarks"),
        (zza_origins_path, ZZA_MODELS, "training-normalized ZZA benchmarks"),
        (
            calibrated_vara_origins_path,
            CALIBRATED_VARA_MODELS,
            "calibrated VARA",
        ),
        (
            insample_vara_origins_path,
            INSAMPLE_VARA_MODELS,
            "in-sample-error VARA",
        ),
    )
    origins = pd.concat(
        [
            _select_origin_metrics(
                path,
                mapping,
                source=source,
                start_date=start_date,
                end_date=end_date,
            )
            for path, mapping, source in specifications
        ],
        ignore_index=True,
    )
    if set(origins["model"]) != EXPECTED_MODELS:
        raise ValueError("Inputs do not contain the prespecified 11 methods.")
    expected_dates = tuple(pd.date_range(start_date, end_date, freq="MS"))
    dates_by_model = origins.groupby("model", observed=True)["date"].agg(
        lambda values: tuple(sorted(pd.Timestamp(value) for value in values))
    )
    if any(dates != expected_dates for dates in dates_by_model):
        raise ValueError(
            "Every method must cover each requested monthly forecast origin."
        )
    if len(expected_dates) != 24:
        raise ValueError(
            f"The h24 robustness table requires 24 origins, found "
            f"{len(expected_dates)}."
        )
    return origins.sort_values(["date", "model"]).reset_index(drop=True)


def calculate_accuracy(
    origins: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return RMSE-sorted accuracy, squared losses, and RankScore audit."""
    squared_losses = (
        origins.assign(squared_error=origins["rmse"].pow(2.0))
        .pivot(index="date", columns="model", values="squared_error")
        .sort_index()
    )
    absolute_errors = (
        origins.pivot(index="date", columns="model", values="mae").sort_index()
    )
    if squared_losses.isna().any().any() or absolute_errors.isna().any().any():
        raise ValueError("Accuracy inputs must form complete origin-method panels.")
    if set(squared_losses.columns) != EXPECTED_MODELS:
        raise ValueError("Accuracy inputs do not contain all 11 methods.")

    common_order = sorted(EXPECTED_MODELS)
    squared_losses = squared_losses.loc[:, common_order]
    absolute_errors = absolute_errors.loc[:, common_order]
    values = absolute_errors.to_numpy(dtype=float)
    n_comparators = values.shape[1] - 1
    if n_comparators != 10:
        raise ValueError("RankScore requires exactly 10 competing methods.")
    wins = (values[:, :, None] < values[:, None, :]).sum(axis=2)
    scores = 100.0 * wins / float(n_comparators)

    accuracy = pd.DataFrame({
        "model": common_order,
        "rmse": np.sqrt(squared_losses.mean(axis=0)).to_numpy(dtype=float),
        "rank_score": scores.mean(axis=0),
        "mean_monthly_absolute_error_wins": wins.mean(axis=0),
        "n_origins": len(squared_losses),
        "n_rank_score_comparators": n_comparators,
    })
    accuracy["method"] = accuracy["model"].map(METHOD_LABELS)
    accuracy = accuracy.sort_values(
        ["rmse", "model"],
        ascending=[True, True],
    ).reset_index(drop=True)
    accuracy.insert(0, "method_number", np.arange(1, len(accuracy) + 1))

    audit = (
        absolute_errors.rename_axis(columns="model")
        .stack()
        .rename("absolute_prediction_error")
        .reset_index()
    )
    audit["absolute_error_wins"] = wins.reshape(-1)
    audit["rank_score_s"] = scores.reshape(-1)
    return accuracy, squared_losses.loc[:, accuracy["model"]], audit


def calculate_pairwise_dm_tests(
    monthly_squared_losses: pd.DataFrame,
    method_order: tuple[str, ...],
) -> pd.DataFrame:
    """Calculate all 55 unique lower-triangular DM comparisons."""
    if tuple(monthly_squared_losses.columns) != method_order:
        raise ValueError("Squared-loss columns must follow ascending RMSE.")
    if len(monthly_squared_losses) != 24:
        raise ValueError("The h24 DM comparison requires 24 monthly losses.")

    rows: list[dict[str, object]] = []
    for row_index in range(1, len(method_order)):
        row_model = method_order[row_index]
        for column_index in range(row_index):
            column_model = method_order[column_index]
            test = diebold_mariano_test(
                monthly_squared_losses[row_model].to_numpy(dtype=float),
                monthly_squared_losses[column_model].to_numpy(dtype=float),
                forecast_horizon=1,
            )
            rows.append({
                "row_number": row_index + 1,
                "row_model": row_model,
                "row_method": METHOD_LABELS[row_model],
                "column_number": column_index + 1,
                "column_model": column_model,
                "column_method": METHOD_LABELS[column_model],
                "loss": "monthly_panel_average_squared_error",
                "lower_rmse_method": column_model,
                **test,
            })
    results = pd.DataFrame(rows)
    if len(results) != 55:
        raise ValueError("Expected all 55 unique method pairs.")
    results["p_value_raw_two_sided_holm_55"] = holm_adjust(
        results["p_value_raw_two_sided"].to_numpy(dtype=float)
    )
    results["significant_raw_10pct"] = (
        results["p_value_raw_two_sided"] < 0.10
    )
    results["significant_raw_5pct"] = (
        results["p_value_raw_two_sided"] < 0.05
    )
    results["significant_holm_5pct"] = (
        results["p_value_raw_two_sided_holm_55"] < 0.05
    )
    return results


def render_accuracy_markdown(accuracy: pd.DataFrame) -> str:
    lines = [
        "| No. | Method | RMSE | RankScore |",
        "|---:|---|---:|---:|",
    ]
    for row in accuracy.itertuples(index=False):
        lines.append(
            f"| {row.method_number} | {row.method} | "
            f"{row.rmse:.2f} | {row.rank_score:.2f} |"
        )
    return "\n".join(lines) + "\n"


def render_accuracy_latex(accuracy: pd.DataFrame) -> str:
    best_rmse = float(accuracy["rmse"].min())
    best_rank_score = float(accuracy["rank_score"].max())
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        (
            r"\caption{Twenty-four-month rolling-origin robustness accuracy "
            r"for the Tourism application, covering January 2015 through "
            r"December 2016. Each target is forecast one month ahead.}"
        ),
        r"\label{tab:tourism_forecasting_results_h24}",
        r"\begin{tabular}{clrr}",
        r"\toprule",
        r"No. & Method & RMSE & RankScore \\",
        r"\midrule",
    ]
    for row in accuracy.itertuples(index=False):
        rmse = f"{row.rmse:.2f}"
        rank_score = f"{row.rank_score:.2f}"
        if np.isclose(float(row.rmse), best_rmse):
            rmse = rf"\textbf{{{rmse}}}"
        if np.isclose(float(row.rank_score), best_rank_score):
            rank_score = rf"\textbf{{{rank_score}}}"
        lines.append(
            f"({row.method_number}) & {LATEX_LABELS[row.model]} & "
            f"{rmse} & {rank_score} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def render_dm_markdown(
    results: pd.DataFrame,
    method_order: tuple[str, ...],
) -> str:
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
                continue
            result = lookup.loc[(row_number, column_number)]
            cells.append(_markdown_p_value(
                float(result["p_value_raw_two_sided"]),
                bool(result["significant_holm_5pct"]),
            ))
        lines.append(
            "| "
            + " | ".join([
                f"({row_number}) {METHOD_LABELS[model]}",
                *cells,
            ])
            + " |"
        )
    return "\n".join(lines) + "\n"


def render_dm_latex(
    results: pd.DataFrame,
    method_order: tuple[str, ...],
) -> str:
    lookup = results.set_index(["row_number", "column_number"])
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{All pairwise Diebold--Mariano tests over the "
            r"24 monthly Tourism rolling origins from January 2015 through "
            r"December 2016. The \(10\times10\) lower triangle contains all "
            r"55 unique comparisons among the 11 methods.}"
        ),
        r"\label{tab:tourism_dm_results_h24}",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}l*{10}{c}@{}}",
        r"\toprule",
        "Row method & "
        + " & ".join(rf"({number})" for number in range(1, 11))
        + r" \\",
        r"\midrule",
    ]
    for row_number in range(2, 12):
        model = method_order[row_number - 1]
        cells = []
        for column_number in range(1, 11):
            if column_number >= row_number:
                cells.append("")
                continue
            result = lookup.loc[(row_number, column_number)]
            cells.append(_latex_p_value(
                float(result["p_value_raw_two_sided"]),
                bool(result["significant_holm_5pct"]),
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
            r"follows ascending RMSE in Table~"
            r"\ref{tab:tourism_forecasting_results_h24}."
        ),
        (
            r"\item Each cell gives the raw two-sided asymptotic-normal "
            r"\(p\)-value for the monthly panel-average squared-error loss "
            r"differential (row minus column). A significant cell favors the "
            r"lower-RMSE column method. The 24 months are an evaluation "
            r"window; every forecast has a one-month lead, so the DM test uses "
            r"forecast horizon one and HAC lag zero. Superscripts "
            r"\textsuperscript{***}, \textsuperscript{**}, and "
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standard-origins-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_standardized_h24_run/rolling-origin/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--zza-origins-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_zza_train_zscore_h24_run/"
            "rolling-origin/metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--calibrated-vara-origins-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_reliability_vara_autoets_zza_"
            "train_zscore_h24/metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--insample-vara-origins-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_insample_error_vara_autoets_h24/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date", default="2016-12-01")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_h24_tables"),
    )
    args = parser.parse_args()

    start_date = pd.Timestamp(args.start_date)
    end_date = pd.Timestamp(args.end_date)
    origins = load_origin_metrics(
        args.standard_origins_path,
        args.zza_origins_path,
        args.calibrated_vara_origins_path,
        args.insample_vara_origins_path,
        start_date=start_date,
        end_date=end_date,
    )
    accuracy, squared_losses, rankscore_audit = calculate_accuracy(origins)
    method_order = tuple(accuracy["model"].astype(str))
    dm_results = calculate_pairwise_dm_tests(squared_losses, method_order)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    accuracy.to_csv(args.output_dir / "table_h24_accuracy.csv", index=False)
    squared_losses.to_csv(
        args.output_dir / "monthly_squared_losses.csv",
        index_label="date",
    )
    rankscore_audit.to_csv(
        args.output_dir / "rankscore_by_origin.csv",
        index=False,
    )
    dm_results.to_csv(
        args.output_dir / "pairwise_dm_tests.csv",
        index=False,
    )
    accuracy_markdown = render_accuracy_markdown(accuracy)
    dm_markdown = render_dm_markdown(dm_results, method_order)
    (args.output_dir / "table_h24_accuracy.md").write_text(accuracy_markdown)
    (args.output_dir / "table_h24_accuracy.tex").write_text(
        render_accuracy_latex(accuracy)
    )
    (args.output_dir / "table_h24_pairwise_dm.md").write_text(dm_markdown)
    (args.output_dir / "table_h24_pairwise_dm.tex").write_text(
        render_dm_latex(dm_results, method_order)
    )
    metadata = {
        "evaluation_start": str(start_date.date()),
        "evaluation_end": str(end_date.date()),
        "n_origins": len(squared_losses),
        "forecast_lead_months": 1,
        "dm_forecast_horizon": 1,
        "dm_hac_lag": 0,
        "method_order": list(method_order),
        "pairwise_test_count": len(dm_results),
        "raw_significant_5pct": int(
            dm_results["significant_raw_5pct"].sum()
        ),
        "holm_significant_5pct": int(
            dm_results["significant_holm_5pct"].sum()
        ),
        "rank_score_comparators": 10,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print("24-month accuracy table:\n", flush=True)
    print(accuracy_markdown, end="", flush=True)
    print("\n24-month pairwise DM table:\n", flush=True)
    print(dm_markdown, end="", flush=True)
    print(
        f"\nRaw p<0.05: {metadata['raw_significant_5pct']}/55; "
        f"Holm p<0.05: {metadata['holm_significant_5pct']}/55",
        flush=True,
    )


if __name__ == "__main__":
    main()
