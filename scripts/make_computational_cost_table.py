#!/usr/bin/env python3
"""Build a computational-cost table from a VARA simulation output folder."""

from __future__ import annotations

import argparse
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from pandas.errors import EmptyDataError


SUMMARY_PREFIX = "summary_full_simulation_"
RUN_TAG_RE = re.compile(r"^summary_full_simulation_(?P<tag>.+)\.csv$")


def _format_number(value: object, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    if number.is_integer() and abs(number) >= 10:
        return str(int(number))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _format_int(value: object) -> str:
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return "--"


def _latest_summary(results_dir: Path, run_tag: Optional[str]) -> Path:
    if run_tag:
        path = results_dir / f"{SUMMARY_PREFIX}{run_tag}.csv"
        if path.exists():
            return path
        raise FileNotFoundError(f"No summary file found for run_tag={run_tag!r} in {results_dir}")

    candidates = sorted(results_dir.glob(f"{SUMMARY_PREFIX}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No {SUMMARY_PREFIX}*.csv file found in {results_dir}")
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def _run_tag_from_summary(path: Path) -> str:
    match = RUN_TAG_RE.match(path.name)
    if not match:
        return path.stem
    return match.group("tag")


def _read_summary(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def build_cost_table(results_dir: Path, run_tag: Optional[str] = None) -> pd.DataFrame:
    summary_path = _latest_summary(results_dir, run_tag)
    summary = _read_summary(summary_path)
    if summary.empty:
        raise ValueError(f"Summary file is empty: {summary_path}")

    actual_run_tag = _run_tag_from_summary(summary_path)
    finish_time = datetime.fromtimestamp(summary_path.stat().st_mtime).astimezone()
    finish_time_text = finish_time.strftime("%Y-%m-%d %H:%M:%S %Z")

    columns = [
        "setting",
        "dist",
        "T",
        "p",
        "q",
        "G",
        "method",
        "n_rep",
        "valid_rate",
        "cpu_sec_mean",
        "cpu_sec_sum",
    ]
    existing = [col for col in columns if col in summary.columns]
    table = summary.loc[:, existing].copy()
    if "cpu_sec_sum" in table.columns:
        table["cpu_hour_sum"] = table["cpu_sec_sum"] / 3600.0
    if "cpu_sec_mean" in table.columns:
        table["cpu_min_mean"] = table["cpu_sec_mean"] / 60.0

    table.insert(0, "finish_time", finish_time_text)
    table.insert(0, "run_tag", actual_run_tag)
    table.insert(0, "output_dir", str(results_dir))
    table.insert(0, "summary_file", summary_path.name)

    sort_cols = [col for col in ["setting", "dist", "T", "method"] if col in table.columns]
    if sort_cols:
        table = table.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    return table


def _render_latex_table(table: pd.DataFrame) -> str:
    headers = [
        "Setting",
        "Innovation",
        "$T$",
        "$p$",
        "$q$",
        "$G$",
        "Estimator",
        "$R$",
        "Mean sec.",
        "Total hours",
    ]
    body_rows = []
    for _, row in table.iterrows():
        body_rows.append(
            " & ".join([
                str(row.get("setting", "--")),
                str(row.get("dist", "--")).capitalize(),
                _format_int(row.get("T", "")),
                _format_int(row.get("p", "")),
                _format_int(row.get("q", "")),
                _format_int(row.get("G", "")),
                str(row.get("method", "--")).replace("_", "-"),
                _format_int(row.get("n_rep", "")),
                _format_number(row.get("cpu_sec_mean", math.nan), digits=2),
                _format_number(row.get("cpu_hour_sum", math.nan), digits=2),
            ]) + r" \\"
        )

    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Computational cost of the simulation runs.}\n"
        "\\label{tab:simulation_computational_cost}\n"
        "\\resizebox{\\textwidth}{!}{%\n"
        "\\begin{tabular}{llcccclccc}\n"
        "\\toprule\n"
        + " & ".join(headers)
        + " \\\\\n"
        "\\midrule\n"
        + "\n".join(body_rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "}\n"
        "\\end{table}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a computational-cost table from a VARA simulation output folder."
    )
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional exact run tag after summary_full_simulation_. Defaults to latest summary file.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to results-dir/table_computational_cost_<run_tag>.csv.",
    )
    parser.add_argument(
        "--tex-out",
        type=Path,
        default=None,
        help="Output TeX path. Defaults to results-dir/table_computational_cost_<run_tag>.tex.",
    )
    args = parser.parse_args()

    table = build_cost_table(args.results_dir, run_tag=args.run_tag)
    output_table = table.drop(columns=["valid_rate", "finish_time"], errors="ignore")
    run_tag = str(table["run_tag"].iloc[0])
    csv_out = args.csv_out or args.results_dir / f"table_computational_cost_{run_tag}.csv"
    tex_out = args.tex_out or args.results_dir / f"table_computational_cost_{run_tag}.tex"

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    output_table.to_csv(csv_out, index=False)
    tex_out.write_text(_render_latex_table(output_table), encoding="utf-8")

    print(f"Summary file : {table['summary_file'].iloc[0]}")
    print(f"Run tag      : {run_tag}")
    print(f"Finish time  : {table['finish_time'].iloc[0]}")
    print(f"CSV table    : {csv_out}")
    print(f"TeX table    : {tex_out}")
    print(output_table.to_string(index=False))


if __name__ == "__main__":
    main()
