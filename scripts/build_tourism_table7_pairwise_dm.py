#!/usr/bin/env python3
"""Build the all-pairs Tourism Diebold--Mariano test matrix for Table 7."""

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

from vara.forecast_significance import diebold_mariano_test, holm_adjust


METHOD_ORDER = (
    "vara_auto_ets_lambda",
    "vara_auto_ets_insample",
    "auto_ets",
    "ar1_auto_ets",
    "auto_arima",
    "nbeats",
    "var1_auto_ets",
    "seasonal_naive",
    "deepar",
    "ar1",
    "var1",
)
METHOD_LABELS = {
    "vara_auto_ets_lambda": "VARA–AutoETS_λ",
    "vara_auto_ets_insample": "VARA–AutoETS",
    "auto_ets": "AutoETS",
    "ar1_auto_ets": "AR(1)–AutoETS",
    "auto_arima": "AutoARIMA",
    "nbeats": "N-BEATS",
    "var1_auto_ets": "VAR(1)–AutoETS",
    "seasonal_naive": "Seasonal naive",
    "deepar": "DeepAR",
    "ar1": "AR(1)",
    "var1": "VAR(1)",
}
LATEX_LABELS = {
    **METHOD_LABELS,
    "vara_auto_ets_lambda": r"VARA--AutoETS$_\lambda$",
    "vara_auto_ets_insample": r"VARA--AutoETS",
    "ar1_auto_ets": r"AR(1)--AutoETS",
    "var1_auto_ets": r"VAR(1)--AutoETS",
}
FINAL_DATES = tuple(pd.date_range("2016-01-01", periods=12, freq="MS"))


def _select_monthly_rmse(
    path: Path,
    model_mapping: dict[str, str],
    *,
    source: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"model", "date", "rmse"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")
    selected = frame[frame["model"].astype(str).isin(model_mapping)].copy()
    selected["model"] = selected["model"].astype(str).map(model_mapping)
    selected["date"] = pd.to_datetime(selected["date"])
    expected_models = set(model_mapping.values())
    if set(selected["model"]) != expected_models:
        missing_models = sorted(expected_models.difference(selected["model"]))
        raise ValueError(f"{source} is missing models: {missing_models}")
    if selected.duplicated(["model", "date"]).any():
        raise ValueError(f"{source} has duplicate model-date rows.")
    if not np.isfinite(selected["rmse"].to_numpy(dtype=float)).all():
        raise ValueError(f"{source} contains non-finite monthly RMSE values.")
    return selected.loc[:, ["model", "date", "rmse"]]


def load_monthly_squared_losses(
    standard_origins_path: Path,
    zza_origins_path: Path,
    calibrated_vara_origins_path: Path,
    insample_vara_origins_path: Path,
) -> pd.DataFrame:
    """Return the 12-by-11 monthly panel-average squared-error loss matrix."""
    frames = [
        _select_monthly_rmse(
            standard_origins_path,
            {
                "seasonal_naive": "seasonal_naive",
                "auto_arima": "auto_arima",
                "deepar": "deepar",
                "nbeats": "nbeats",
                "ar1": "ar1",
                "var1": "var1",
            },
            source="standard benchmark origins",
        ),
        _select_monthly_rmse(
            zza_origins_path,
            {
                "auto_ets": "auto_ets",
                "ar1_auto_ets": "ar1_auto_ets",
                "var1_auto_ets": "var1_auto_ets",
            },
            source="training-normalized ZZA benchmark origins",
        ),
        _select_monthly_rmse(
            calibrated_vara_origins_path,
            {
                "reliability_gated_vara_auto_ets_oos": (
                    "vara_auto_ets_lambda"
                ),
            },
            source="calibrated VARA origins",
        ),
        _select_monthly_rmse(
            insample_vara_origins_path,
            {"vara_auto_ets_insample": "vara_auto_ets_insample"},
            source="in-sample-error VARA origins",
        ),
    ]
    origins = pd.concat(frames, ignore_index=True)
    if set(origins["model"]) != set(METHOD_ORDER):
        raise ValueError("The pairwise DM input does not contain the 11 Table 6 methods.")
    wide = origins.pivot(index="date", columns="model", values="rmse").sort_index()
    observed_dates = tuple(pd.Timestamp(value) for value in wide.index)
    if observed_dates != FINAL_DATES:
        raise ValueError("The pairwise DM test requires January--December 2016.")
    wide = wide.loc[:, list(METHOD_ORDER)]
    return wide.pow(2.0)


def validate_table6_order(
    monthly_squared_losses: pd.DataFrame,
    table6_path: Path,
    *,
    tolerance: float = 1e-8,
) -> None:
    """Require the DM method order and losses to reproduce RMSE-sorted Table 6."""
    table = pd.read_csv(table6_path)
    required = {"model", "rmse"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"Table 6 data are missing columns: {missing}")
    if table["model"].duplicated().any():
        raise ValueError("Table 6 data contain duplicate methods.")
    observed_order = tuple(table["model"].astype(str))
    if observed_order != METHOD_ORDER:
        raise ValueError("Table 6 is not in the prespecified ascending-RMSE order.")
    rmse = table["rmse"].to_numpy(dtype=float)
    if np.any(np.diff(rmse) < 0.0):
        raise ValueError("Table 6 RMSE values are not sorted in ascending order.")
    reproduced = np.sqrt(
        monthly_squared_losses.loc[:, list(METHOD_ORDER)].mean(axis=0)
    ).to_numpy(dtype=float)
    if not np.allclose(rmse, reproduced, rtol=0.0, atol=float(tolerance)):
        raise ValueError("Monthly squared losses do not reproduce Table 6 RMSE.")


def calculate_pairwise_dm_tests(
    monthly_squared_losses: pd.DataFrame,
) -> pd.DataFrame:
    """Calculate all 55 unique row-versus-column DM tests."""
    if tuple(monthly_squared_losses.columns) != METHOD_ORDER:
        raise ValueError("Squared-loss columns do not follow the Table 6 order.")
    if tuple(pd.Timestamp(value) for value in monthly_squared_losses.index) != FINAL_DATES:
        raise ValueError("Squared-loss rows do not cover the final 12 origins.")
    if not np.isfinite(monthly_squared_losses.to_numpy(dtype=float)).all():
        raise ValueError("Squared losses must all be finite.")

    rows: list[dict[str, object]] = []
    for row_index in range(1, len(METHOD_ORDER)):
        row_model = METHOD_ORDER[row_index]
        for column_index in range(row_index):
            column_model = METHOD_ORDER[column_index]
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
    expected_pairs = len(METHOD_ORDER) * (len(METHOD_ORDER) - 1) // 2
    if len(results) != expected_pairs:
        raise ValueError(f"Expected {expected_pairs} pairwise tests.")
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


def _star(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def _latex_p_value(p_value: float, holm_significant: bool) -> str:
    value = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
    core = rf"${value}$"
    if holm_significant:
        core = rf"\textbf{{{core}}}"
    stars = _star(p_value)
    return core if not stars else core + rf"\textsuperscript{{{stars}}}"


def _markdown_p_value(p_value: float, holm_significant: bool) -> str:
    value = "<0.001" if p_value < 0.001 else f"{p_value:.3f}"
    core = f"**{value}**" if holm_significant else value
    stars = _star(p_value)
    return core if not stars else core + f"<sup>{stars}</sup>"


def render_latex(results: pd.DataFrame) -> str:
    """Render the 10-by-10 lower-triangular p-value table."""
    lookup = results.set_index(["row_number", "column_number"])
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\scriptsize",
        (
            r"\caption{All pairwise Diebold--Mariano tests for the final 2016 "
            r"Tourism forecasts. The displayed body is a \(10\times10\) lower "
            r"triangle containing all 55 unique comparisons among the 11 "
            r"methods in Table~\ref{tab:tourism_forecasting_results}.}"
        ),
        r"\label{tab:tourism_dm_results}",
        r"\begin{threeparttable}",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}l*{10}{c}@{}}",
        r"\toprule",
        "Row method & " + " & ".join(
            rf"({number})" for number in range(1, 11)
        ) + r" \\",
        r"\midrule",
    ]
    for row_number in range(2, 12):
        model = METHOD_ORDER[row_number - 1]
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
            r"\item Notes: Method (1) is VARA--AutoETS$_\lambda$; methods "
            r"(2)--(11) are identified by the row labels. The numbering follows "
            r"the ascending-RMSE order in Table~"
            r"\ref{tab:tourism_forecasting_results}."
        ),
        (
            r"\item Each cell reports the raw two-sided asymptotic-normal "
            r"\(p\)-value for the monthly panel-average squared-error loss "
            r"differential, defined as row method minus column method. Because "
            r"the methods are RMSE ordered, the column method has the lower "
            r"sample-average squared loss; thus, a significant cell favors the "
            r"column method. Displayed \(p\)-values are rounded to three "
            r"decimals, while significance markers use the unrounded values. "
            r"Superscripts \textsuperscript{***}, "
            r"\textsuperscript{**}, and \textsuperscript{*} denote raw "
            r"\(p<0.001\), \(p<0.05\), and \(p<0.1\), respectively. Bold "
            r"entries remain significant at 5\% after Holm correction over "
            r"all 55 tests."
        ),
        r"\end{tablenotes}",
        r"\end{threeparttable}",
        r"\end{table}",
        "",
    ])
    return "\n".join(lines)


def render_markdown(results: pd.DataFrame) -> str:
    lookup = results.set_index(["row_number", "column_number"])
    headers = ["Row method", *[f"({number})" for number in range(1, 11)]]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---", *(["---:"] * 10)]) + "|",
    ]
    for row_number in range(2, 12):
        model = METHOD_ORDER[row_number - 1]
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standard-origins-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_standardized_run/rolling-origin/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--zza-origins-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_zza_train_zscore_run/rolling-origin/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--calibrated-vara-origins-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_reliability_vara_autoets_zza_train_zscore/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--insample-vara-origins-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_insample_error_vara_autoets/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--table6-path",
        type=Path,
        default=Path(
            "outputs/tourism_table6_zza_train_zscore/table6_zza_full.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_table7_pairwise_dm"),
    )
    args = parser.parse_args()

    losses = load_monthly_squared_losses(
        args.standard_origins_path,
        args.zza_origins_path,
        args.calibrated_vara_origins_path,
        args.insample_vara_origins_path,
    )
    validate_table6_order(losses, args.table6_path)
    results = calculate_pairwise_dm_tests(losses)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "pairwise_dm_tests.csv", index=False)
    losses.to_csv(args.output_dir / "monthly_squared_losses.csv")
    latex = render_latex(results)
    markdown = render_markdown(results)
    (args.output_dir / "table7_pairwise_dm.tex").write_text(latex)
    (args.output_dir / "table7_pairwise_dm.md").write_text(markdown)
    metadata = {
        "method_order": list(METHOD_ORDER),
        "method_labels": METHOD_LABELS,
        "method_count": len(METHOD_ORDER),
        "pairwise_test_count": len(results),
        "matrix_body_shape": [10, 10],
        "loss": "monthly panel-average squared error",
        "n_origins": len(losses),
        "forecast_horizon": 1,
        "reported_p_value": "raw two-sided asymptotic-normal",
        "multiple_testing": "Holm correction over all 55 raw two-sided p-values",
        "raw_significant_5pct": int(results["significant_raw_5pct"].sum()),
        "holm_significant_5pct": int(results["significant_holm_5pct"].sum()),
    }
    (args.output_dir / "table7_pairwise_dm.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(markdown, end="")
    print(
        f"\nRaw p<0.05: {metadata['raw_significant_5pct']}/55; "
        f"Holm p<0.05: {metadata['holm_significant_5pct']}/55",
        flush=True,
    )


if __name__ == "__main__":
    main()
