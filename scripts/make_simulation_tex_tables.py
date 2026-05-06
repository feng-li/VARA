#!/usr/bin/env python3
"""Generate LaTeX simulation tables from VARA simulation CSV outputs.

Example
-------
python scripts/make_simulation_tex_tables.py \
    --results-dir outputs/paper_simulation \
    --tex-dir docs/generated_tables

Then merge the generated tables into the manuscript with
``\\input{generated_tables/simulation_tables.tex}`` from ``docs/VAR_Atten.tex``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from pandas.errors import EmptyDataError


TABLE_PREFIXES = {
    "simulation_design": ["table_simulation_setup_"],
    "simulation_low_rmse": ["table_simulation_low_rmse_"],
    "simulation_low_ecp": ["table_simulation_low_ecp_"],
    "simulation_high_rmse": ["table_simulation_high_rmse_"],
    "simulation_high_ecp": ["table_simulation_high_ecp_"],
    "simulation_xhigh_diagnostics": [
        "table_simulation_publication_wide_",
        "summary_full_simulation_",
        "table_simulation_eta_diagnostics_",
    ],
}


OUTPUT_FILES = {
    "simulation_design": "simulation_design.tex",
    "simulation_low_rmse": "simulation_low_rmse.tex",
    "simulation_low_ecp": "simulation_low_ecp.tex",
    "simulation_high_rmse": "simulation_high_rmse.tex",
    "simulation_high_ecp": "simulation_high_ecp.tex",
    "simulation_xhigh_diagnostics": "simulation_xhigh_diagnostics.tex",
}


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _format_number(value: object, digits: int = 4) -> str:
    if _is_missing(value):
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _latex_escape(value)
    if not math.isfinite(number):
        return "--"
    if number.is_integer() and abs(number) >= 10:
        return str(int(number))
    return f"{number:.{digits}f}".rstrip("0").rstrip(".")


def _format_int(value: object) -> str:
    if _is_missing(value):
        return "--"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return _latex_escape(value)


def _dist_label(value: object) -> str:
    labels = {
        "normal": "Normal",
        "laplace": "Laplace",
        "t5": "Student-$t_5$",
    }
    return labels.get(str(value).lower(), _latex_escape(value))


def _method_label(value: object) -> str:
    labels = {
        "qmle": "QMLE",
        "two_stage": "Two-stage",
    }
    return labels.get(str(value).lower(), _latex_escape(value))


def _setting_label(setting: object, attention_mode: object = "") -> str:
    if str(attention_mode).lower() == "topk":
        return "Extra-high-dimensional"
    labels = {
        "low": "Low-dimensional",
        "high": "High-dimensional",
    }
    return labels.get(str(setting).lower(), _latex_escape(setting))


def _p_rule_label(value: object) -> str:
    text = str(value)
    if text == "floor(sqrt(T))":
        return r"$[\sqrt T]$"
    if text == "floor(T^1.5)":
        return r"$[T^{1.5}]$"
    return _latex_escape(value)


def _attention_label(row: pd.Series) -> str:
    mode = str(row.get("attention_mode", "dense")).lower()
    if mode == "topk":
        k = row.get("attention_top_k", "")
        if not _is_missing(k) and str(k) != "":
            return f"Sparse top-$K$ ($K={_format_int(k)}$)"
        return "Sparse top-$K$"
    return "Dense"


def _read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _latest_matching_csv(results_dir: Path, prefixes: Sequence[str], config_tag: Optional[str] = None) -> Optional[Path]:
    candidates: List[Path] = []
    for prefix in prefixes:
        if config_tag:
            path = results_dir / f"{prefix}{config_tag}.csv"
            if path.exists():
                candidates.append(path)
        else:
            candidates.extend(results_dir.glob(f"{prefix}*.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))


def _sort_table(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return df
    return df.sort_values(existing, kind="stable").reset_index(drop=True)


def _table_environment(label: str, caption: str, body: str) -> str:
    return (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}\n"
        "\\end{table}\n"
    )


def render_design_table(setup: pd.DataFrame) -> str:
    if setup.empty:
        rows = [
            r"Low-dimensional & 100, 200, 500 & $[\sqrt T]$ & 50 & Normal, Laplace & QMLE, two-stage & Dense \\",
            r"High-dimensional & 100, 200, 500 & $[T^{1.5}]$ & 50 & Normal, Laplace & Two-stage & Dense \\",
            r"Extra-high-dimensional & TBD & TBD & 50 & Normal & Two-stage & Sparse top-$K$ \\",
        ]
    else:
        group_cols = ["setting", "p_rule", "q", "attention_mode", "attention_top_k"]
        existing_group_cols = [c for c in group_cols if c in setup.columns]
        rows = []
        grouped = setup.groupby(existing_group_cols, dropna=False, sort=False) if existing_group_cols else [((), setup)]
        for _, group in grouped:
            first = group.iloc[0]
            setting = _setting_label(first.get("setting", ""), first.get("attention_mode", ""))
            t_values = ", ".join(_format_int(v) for v in sorted(group["T"].dropna().unique())) if "T" in group else "--"
            p_rule = _p_rule_label(first.get("p_rule", ""))
            q_values = ", ".join(_format_int(v) for v in sorted(group["q"].dropna().unique())) if "q" in group else "--"
            innovations = ", ".join(_dist_label(v) for v in sorted(group["dist"].dropna().unique())) if "dist" in group else "--"
            estimator = "QMLE, two-stage" if str(first.get("setting", "")).lower() == "low" else "Two-stage"
            rows.append(
                " & ".join([
                    setting,
                    t_values,
                    p_rule,
                    q_values,
                    innovations,
                    estimator,
                    _attention_label(first),
                ]) + r" \\"
            )

    body = (
        "\\begin{threeparttable}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        "\\begin{tabular}{@{}>{\\RaggedRight\\arraybackslash}p{0.18\\textwidth}\n"
        ">{\\centering\\arraybackslash}p{0.12\\textwidth}\n"
        ">{\\centering\\arraybackslash}p{0.12\\textwidth}\n"
        ">{\\centering\\arraybackslash}p{0.05\\textwidth}\n"
        ">{\\RaggedRight\\arraybackslash}p{0.15\\textwidth}\n"
        ">{\\RaggedRight\\arraybackslash}p{0.15\\textwidth}\n"
        ">{\\RaggedRight\\arraybackslash}p{0.13\\textwidth}@{}}\n"
        "\\toprule\n"
        "Setting & $T$ & $p$ & $q$ & Innovations & Estimator & Attention design \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\footnotesize\n"
        "\\item Notes: The table is generated from the simulation setup CSV. The paper uses $\\gamma$ for code ``eta\\_a'', $\\alpha$ for code ``beta'', and $\\beta$ for code ``gamma\\_a''.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}"
    )
    return _table_environment("tab:simulation_design", "Simulation designs.", body)


def _metric_rows(
    df: pd.DataFrame,
    metric_cols: Sequence[str],
    include_method: bool,
) -> List[str]:
    if df.empty:
        width = 3 + len(metric_cols) if include_method else 2 + len(metric_cols)
        return [" & ".join(["--"] * width) + r" \\"]

    rows = []
    sort_cols = ["dist", "T", "method"] if include_method else ["dist", "T"]
    df = _sort_table(df, sort_cols)
    for _, row in df.iterrows():
        cells = [_dist_label(row.get("dist", ""))]
        if include_method:
            cells.append(_method_label(row.get("method", "")))
        cells.append(_format_int(row.get("T", "")))
        cells.extend(_format_number(row.get(col, math.nan)) for col in metric_cols)
        rows.append(" & ".join(cells) + r" \\")
    return rows


def render_metric_table(
    df: pd.DataFrame,
    *,
    label: str,
    caption: str,
    metric_cols: Sequence[str],
    headers: Sequence[str],
    include_method: bool,
) -> str:
    column_spec = "ll" if include_method else "l"
    column_spec += "c" * (1 + len(metric_cols))
    header_cells = ["Innovation"]
    if include_method:
        header_cells.append("Method")
    header_cells.extend(["$T$", *headers])
    body = (
        f"\\begin{{tabular}}{{{column_spec}}}\n"
        "\\toprule\n"
        + " & ".join(header_cells)
        + " \\\\\n"
        "\\midrule\n"
        + "\n".join(_metric_rows(df, metric_cols, include_method=include_method))
        + "\n\\bottomrule\n"
        "\\end{tabular}"
    )
    return _table_environment(label, caption, body)


def _format_pair(row: pd.Series, pos_col: str, neg_col: str) -> str:
    pos = _format_number(row.get(pos_col, math.nan), digits=3)
    neg = _format_number(row.get(neg_col, math.nan), digits=3)
    if pos == "--" and neg == "--":
        return "--"
    if pos == "--":
        return neg
    if neg == "--":
        return pos
    return f"{pos}/{neg}"


def render_xhigh_diagnostics(df: pd.DataFrame) -> str:
    if not df.empty and "attention_mode" in df.columns:
        topk = df.loc[df["attention_mode"].astype(str).str.lower() == "topk"]
        if not topk.empty:
            df = topk
    if not df.empty and "setting" in df.columns:
        high = df.loc[df["setting"].astype(str).str.lower() == "high"]
        if not high.empty:
            df = high

    rows = []
    if df.empty:
        rows = [r"TBD & TBD & TBD & -- & -- & -- & -- \\"]
    else:
        df = _sort_table(df, ["T", "p", "dist", "method"])
        for _, row in df.iterrows():
            cache_recall = _format_pair(row, "TopK_cache_recall_pos_mean", "TopK_cache_recall_neg_mean")
            score_loss = _format_pair(
                row,
                "TopK_cache_score_loss_pos_abs_mean",
                "TopK_cache_score_loss_neg_abs_mean",
            )
            rows.append(
                " & ".join([
                    _format_int(row.get("T", "")),
                    _format_int(row.get("p", "")),
                    _format_int(row.get("attention_top_k", "")),
                    _format_number(row.get("RMSE_gamma", math.nan)),
                    _format_number(row.get("eta_boundary_rate", math.nan)),
                    cache_recall,
                    score_loss,
                ]) + r" \\"
            )

    body = (
        "\\begin{threeparttable}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n"
        "\\renewcommand{\\arraystretch}{1.15}\n"
        "\\begin{tabular}{@{}>{\\centering\\arraybackslash}p{0.09\\textwidth}\n"
        ">{\\centering\\arraybackslash}p{0.12\\textwidth}\n"
        ">{\\centering\\arraybackslash}p{0.07\\textwidth}\n"
        ">{\\centering\\arraybackslash}p{0.14\\textwidth}\n"
        ">{\\centering\\arraybackslash}p{0.15\\textwidth}\n"
        ">{\\centering\\arraybackslash}p{0.15\\textwidth}\n"
        ">{\\centering\\arraybackslash}p{0.13\\textwidth}@{}}\n"
        "\\toprule\n"
        "$T$ & $p$ & $K$ & $\\RMSE_{\\gamma}$ & Boundary rate & Cache recall & Score loss \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\begin{tablenotes}\n"
        "\\footnotesize\n"
        "\\item Notes: Cache recall and score loss are reported as positive/negative eta-direction diagnostics, computed by comparing the sparse top-$K$ cache with exact row-to-all top-$K$ attention on sampled rows.\n"
        "\\end{tablenotes}\n"
        "\\end{threeparttable}"
    )
    return _table_environment(
        "tab:simulation_xhigh_diagnostics",
        "Extra-high-dimensional sparse-attention diagnostics.",
        body,
    )


def render_table(table_name: str, df: pd.DataFrame) -> str:
    gamma = r"$\gamma$"
    alpha = r"$\balpha$"
    beta = r"$\bbeta$"
    sigma2 = r"$\bsigma^2$"
    if table_name == "simulation_design":
        return render_design_table(df)
    if table_name == "simulation_low_rmse":
        return render_metric_table(
            df,
            label="tab:simulation_low_rmse",
            caption="Low-dimensional setting: RMSE.",
            metric_cols=["RMSE_gamma", "RMSE_alpha", "RMSE_beta", "RMSE_sigma2"],
            headers=[gamma, alpha, beta, sigma2],
            include_method=True,
        )
    if table_name == "simulation_low_ecp":
        return render_metric_table(
            df,
            label="tab:simulation_low_ecp",
            caption="Low-dimensional setting: empirical coverage probabilities.",
            metric_cols=["ECP_gamma", "ECP_alpha", "ECP_beta", "ECP_sigma2"],
            headers=[gamma, alpha, beta, sigma2],
            include_method=True,
        )
    if table_name == "simulation_high_rmse":
        return render_metric_table(
            df,
            label="tab:simulation_high_rmse",
            caption="High-dimensional setting: RMSE.",
            metric_cols=["RMSE_gamma", "RMSE_alpha", "RMSE_beta", "RMSE_sigma2"],
            headers=[gamma, alpha, beta, sigma2],
            include_method=False,
        )
    if table_name == "simulation_high_ecp":
        return render_metric_table(
            df,
            label="tab:simulation_high_ecp",
            caption="High-dimensional setting: empirical coverage probabilities.",
            metric_cols=["ECP_alpha", "ECP_beta", "ECP_sigma2"],
            headers=[alpha, beta, sigma2],
            include_method=False,
        )
    if table_name == "simulation_xhigh_diagnostics":
        return render_xhigh_diagnostics(df)
    raise ValueError(f"Unknown table name: {table_name}")


def generate_tex_tables(
    *,
    results_dir: Path,
    tex_dir: Path,
    config_tag: Optional[str] = None,
    table_names: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
    tex_dir.mkdir(parents=True, exist_ok=True)
    selected_tables = list(table_names) if table_names is not None else list(OUTPUT_FILES)
    written: Dict[str, Path] = {}
    for table_name in selected_tables:
        if table_name not in OUTPUT_FILES:
            raise ValueError(f"Unknown table name: {table_name}")
        csv_path = _latest_matching_csv(results_dir, TABLE_PREFIXES[table_name], config_tag=config_tag)
        df = _read_csv(csv_path)
        tex = "% Auto-generated by scripts/make_simulation_tex_tables.py\n"
        if csv_path is not None:
            tex += f"% Source: {csv_path}\n"
        else:
            tex += "% Source: no matching CSV found; placeholder emitted\n"
        tex += render_table(table_name, df)
        out_path = tex_dir / OUTPUT_FILES[table_name]
        out_path.write_text(tex, encoding="utf-8")
        written[table_name] = out_path

    combined = tex_dir / "simulation_tables.tex"
    combined_lines = [
        "% Auto-generated by scripts/make_simulation_tex_tables.py",
        "% Include from docs/VAR_Atten.tex with:",
        "% \\input{generated_tables/simulation_tables.tex}",
        "",
    ]
    for table_name in selected_tables:
        combined_lines.append((tex_dir / OUTPUT_FILES[table_name]).read_text(encoding="utf-8"))
        combined_lines.append("")
    combined.write_text("\n".join(combined_lines), encoding="utf-8")
    written["combined"] = combined
    return written


def _parse_table_list(text: str) -> List[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    unknown = sorted(set(values) - set(OUTPUT_FILES))
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown table name(s): {', '.join(unknown)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX VARA simulation tables from CSV outputs.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/results"))
    parser.add_argument("--tex-dir", type=Path, default=Path("docs/generated_tables"))
    parser.add_argument(
        "--config-tag",
        type=str,
        default=None,
        help="Optional exact config tag after table_simulation_*_. If omitted, latest matching CSVs are used.",
    )
    parser.add_argument(
        "--tables",
        type=_parse_table_list,
        default=list(OUTPUT_FILES),
        help=f"Comma-separated table names. Choices: {','.join(OUTPUT_FILES)}",
    )
    args = parser.parse_args()

    written = generate_tex_tables(
        results_dir=args.results_dir,
        tex_dir=args.tex_dir,
        config_tag=args.config_tag,
        table_names=args.tables,
    )
    print("Generated LaTeX simulation tables:")
    for name, path in written.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
