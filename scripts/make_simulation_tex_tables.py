#!/usr/bin/env python3
"""Generate LaTeX simulation tables from VARA simulation CSV outputs.

The default output matches the simulation table layout used in
``docs/VAR_Atten.tex``: one design table, two low-dimensional tables split by
innovation distribution, and one combined high-dimensional table.

Example
-------
python scripts/make_simulation_tex_tables.py \
    --results-dir outputs/paper_final_low_screened_z_p8_12_16_G2_3_4_actual \
                  outputs/paper_final_high_screened_z_gamma_a_design \
    --tex-dir docs/generated_tables
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from pandas.errors import EmptyDataError


TABLE_PREFIXES = {
    "simulation_low_uniform": ["summary_full_simulation_", "table_simulation_publication_wide_"],
    "simulation_design": ["table_simulation_setup_", "summary_full_simulation_"],
    "simulation_low_normal": ["summary_full_simulation_", "table_simulation_publication_wide_"],
    "simulation_low_laplace": ["summary_full_simulation_", "table_simulation_publication_wide_"],
    "simulation_high_combined": ["summary_full_simulation_", "table_simulation_publication_wide_"],
    "simulation_xhigh_diagnostics": [
        "table_simulation_publication_wide_",
        "summary_full_simulation_",
        "table_simulation_eta_diagnostics_",
    ],
}


OUTPUT_FILES = {
    "simulation_low_uniform": "simulation_low_uniform.tex",
    "simulation_design": "simulation_design.tex",
    "simulation_low_normal": "simulation_low_normal.tex",
    "simulation_low_laplace": "simulation_low_laplace.tex",
    "simulation_high_combined": "simulation_high_combined.tex",
    "simulation_xhigh_diagnostics": "simulation_xhigh_diagnostics.tex",
}


DEFAULT_TABLES = [
    "simulation_low_uniform",
    "simulation_design",
    "simulation_low_normal",
    "simulation_low_laplace",
    "simulation_high_combined",
]


LEGACY_TABLE_ALIASES = {
    "simulation_low_rmse": "simulation_low_normal",
    "simulation_low_ecp": "simulation_low_laplace",
    "simulation_high_rmse": "simulation_high_combined",
    "simulation_high_ecp": "simulation_high_combined",
}


METRIC_COLS = ["RMSE_gamma", "RMSE_alpha", "RMSE_beta", "RMSE_sigma2"]
ECP_COLS = ["ECP_gamma", "ECP_alpha", "ECP_beta", "ECP_sigma2"]
HIGH_ECP_COLS = ["ECP_alpha", "ECP_beta", "ECP_sigma2"]


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


def _format_number(value: object, digits: int = 3) -> str:
    if _is_missing(value):
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _latex_escape(value)
    if not math.isfinite(number):
        return "--"
    return f"{number:.{digits}f}"


def _format_time(value: object) -> str:
    if _is_missing(value):
        return "--"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _latex_escape(value)
    if not math.isfinite(number):
        return "--"
    return f"{number:.2f}"


def _format_int(value: object) -> str:
    if _is_missing(value):
        return "--"
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return _latex_escape(value)


def _format_set(values: Sequence[object]) -> str:
    clean = []
    for value in values:
        if not _is_missing(value):
            clean.append(_format_int(value))
    return ", ".join(dict.fromkeys(clean)) if clean else "--"


def _dist_label(value: object) -> str:
    labels = {
        "normal": "Normal",
        "uniform": "Uniform",
        "laplace": "Laplace",
        "t5": "Student-$t_5$",
    }
    return labels.get(str(value).lower(), _latex_escape(value))


def _ordered_dists(values: Sequence[object]) -> List[object]:
    order = {"normal": 0, "uniform": 1, "laplace": 2}
    return sorted(values, key=lambda x: order.get(str(x).lower(), 99))


def _method_label(value: object) -> str:
    labels = {
        "qmle": "QMLE",
        "two_stage": "Full pilot",
        "two_stage_subsampled": "Subsampled pilot",
    }
    return labels.get(str(value).lower(), _latex_escape(value))


def _read_csv(path: Optional[Path]) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()


def _latest_matching_csv(results_dir: Path, prefixes: Sequence[str], config_tag: Optional[str] = None) -> Optional[Path]:
    # Prefix order is meaningful. For example, metric tables prefer
    # summary_full_simulation_* because it contains cpu_sec_mean;
    # publication_wide is only a fallback for older outputs.
    for prefix in prefixes:
        candidates: List[Path] = []
        if config_tag:
            path = results_dir / f"{prefix}{config_tag}.csv"
            if path.exists():
                candidates.append(path)
        else:
            candidates.extend(results_dir.glob(f"{prefix}*.csv"))
        if candidates:
            return max(candidates, key=lambda p: (p.stat().st_mtime, p.name))
    return None


def _latest_matching_csvs(
    results_dirs: Sequence[Path],
    prefixes: Sequence[str],
    config_tag: Optional[str] = None,
) -> List[Path]:
    paths: List[Path] = []
    for results_dir in results_dirs:
        path = _latest_matching_csv(results_dir, prefixes, config_tag=config_tag)
        if path is not None:
            paths.append(path)
    return paths


def _read_and_concat(paths: Sequence[Path]) -> pd.DataFrame:
    frames = [_read_csv(path) for path in paths]
    frames = [df for df in frames if not df.empty]
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    if "setting" in df.columns and "dist" in df.columns and "T" in df.columns:
        subset = [col for col in ["setting", "dist", "T", "p", "q", "G", "method"] if col in df.columns]
        df = df.drop_duplicates(subset=subset, keep="last")
    return df.reset_index(drop=True)


def _sort_table(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if df.empty:
        return df
    existing = [col for col in columns if col in df.columns]
    if not existing:
        return df
    return df.sort_values(existing, kind="stable").reset_index(drop=True)


def _table_environment(label: str, caption: str, body: str, *, color_red: bool = False) -> str:
    color = "\\color{red}\n" if color_red else ""
    return (
        "\\begin{table}[!htbp]\n"
        "\\centering\n"
        f"{color}"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"{body}\n"
        "\\end{table}\n"
    )


def render_design_table(setup: pd.DataFrame) -> str:
    if setup.empty:
        rows = [
            r"Low & 100, 200, 500 & $8, 12, 16$ & 50 & $2, 3, 4$ & Normal, Laplace & QMLE, two-stage  \\",
            r"High & 100, 200, 500 & $100, 300, 1000$ & 50 & $5, 10, 15$ & Normal, Laplace & Two-stage  \\",
        ]
    else:
        rows = []
        for setting in ["low", "high"]:
            group = setup.loc[setup["setting"].astype(str).str.lower() == setting] if "setting" in setup else pd.DataFrame()
            if group.empty:
                continue
            t_values = _format_set(sorted(group["T"].dropna().unique())) if "T" in group else "--"
            p_values = _format_set(sorted(group["p"].dropna().unique())) if "p" in group else "--"
            q_values = _format_set(sorted(group["q"].dropna().unique())) if "q" in group else "--"
            g_values = _format_set(sorted(group["G"].dropna().unique())) if "G" in group else "--"
            dist_values = ", ".join(_dist_label(v) for v in _ordered_dists(group["dist"].dropna().unique())) if "dist" in group else "--"
            method = "QMLE" if setting == "low" else "Full/subsampled two-stage"
            rows.append(
                " & ".join([
                    "Low" if setting == "low" else "High",
                    t_values,
                    f"${p_values}$" if "," in p_values else p_values,
                    q_values,
                    f"${g_values}$" if "," in g_values else g_values,
                    dist_values,
                    method,
                ]) + r"  \\" 
            )
    body = (
        "\\resizebox{\\textwidth}{!}{%\n"
        "  \\begin{tabular}{ccccccc}\n"
        "    \\toprule\n"
        "    Dimension & $T$ & $p$ & $q$ & $G$ & Innovations & Method \\\\\n"
        "    \\midrule\n"
        + "\n".join(f"    {row}" for row in rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "}"
    )
    return _table_environment("tab:simulation_design", "Simulation designs.", body)


def _filter_summary(df: pd.DataFrame, *, setting: str, dist: Optional[str] = None) -> pd.DataFrame:
    if df.empty or "setting" not in df.columns:
        return pd.DataFrame()
    out = df.loc[df["setting"].astype(str).str.lower() == setting].copy()
    if dist is not None and "dist" in out.columns:
        out = out.loc[out["dist"].astype(str).str.lower() == dist].copy()
    return out


def _method_order_key(method: object) -> int:
    order = {"qmle": 0, "two_stage": 1}
    return order.get(str(method).lower(), 99)


def _low_group_rows(group: pd.DataFrame, *, include_cmidrule: bool) -> List[str]:
    rows: List[str] = []
    group = group.sort_values("method", key=lambda s: s.map(_method_order_key), kind="stable")
    n_methods = len(group)
    multirow = 2 * n_methods
    first_data_row = True
    tpg = f"$({_format_int(group.iloc[0].get('T'))},{_format_int(group.iloc[0].get('p'))},{_format_int(group.iloc[0].get('G'))})$"
    for _, row in group.iterrows():
        prefix = f"\\multirow{{{multirow}}}{{*}}{{{tpg}}}" if first_data_row else ""
        rows.append(
            " & ".join([
                prefix,
                _method_label(row.get("method", "")),
                *[_format_number(row.get(col)) for col in METRIC_COLS],
                _format_time(row.get("cpu_sec_mean")),
            ]) + r" \\"
        )
        rows.append(
            " & ".join([
                "",
                "",
                *[f"({_format_number(row.get(col))})" for col in ECP_COLS],
                "",
            ]) + r" \\"
        )
        first_data_row = False
    if include_cmidrule:
        rows.append(r"\cmidrule(l){2-7}")
    return rows


def render_low_innovation_table(df: pd.DataFrame, *, dist: str) -> str:
    filtered = _filter_summary(df, setting="low", dist=dist)
    filtered = _sort_table(filtered, ["T", "p", "G", "method"])
    rows: List[str] = []
    if filtered.empty:
        rows = [r"-- & -- & -- & -- & -- & -- & -- \\"]
    else:
        groups = list(filtered.groupby(["T", "p", "G"], sort=True, dropna=False))
        for idx, (_, group) in enumerate(groups):
            rows.extend(_low_group_rows(group, include_cmidrule=idx < len(groups) - 1))
    dist_key = str(dist).lower()
    dist_label = _dist_label(dist_key)
    label = f"tab:simulation_low_{dist_key}"
    caption = f"Low-dimensional setting under {dist_label} innovations: RMSEs, ECPs reported below in parentheses, and average computation time."
    body = (
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}llccccc}\n"
        "\\toprule\n"
        "$(T,p,G)$ & Method & $\\gamma$ & $\\balpha$ & $\\bbeta$ & $\\bsigma^2$ & Time (sec.) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular*}"
    )
    return _table_environment(label, caption, body)


def _high_dist_rows(group: pd.DataFrame, *, include_midrule: bool) -> List[str]:
    rows: List[str] = []
    group = _sort_table(group, ["T", "p", "G"])
    n_settings = len(group)
    first_data_row = True
    dist = _dist_label(group.iloc[0].get("dist", ""))
    for _, row in group.iterrows():
        prefix = f"\\multirow{{{2 * n_settings}}}{{*}}{{{dist}}}" if first_data_row else ""
        tpg = f"$({_format_int(row.get('T'))},{_format_int(row.get('p'))},{_format_int(row.get('G'))})$"
        rows.append(
            " & ".join([
                prefix,
                tpg,
                _method_label(row.get("method", "")),
                *[_format_number(row.get(col)) for col in METRIC_COLS],
                _format_time(row.get("cpu_sec_mean")),
            ]) + r" \\"
        )
        rows.append(
            " & ".join([
                "",
                "",
                "",
                "",
                *[f"({_format_number(row.get(col))})" for col in HIGH_ECP_COLS],
                "",
            ]) + r" \\"
        )
        first_data_row = False
    if include_midrule:
        rows.append(r"\midrule")
    return rows


def render_high_combined_table(df: pd.DataFrame) -> str:
    filtered = _filter_summary(df, setting="high")
    filtered = _sort_table(filtered, ["dist", "T", "p", "G"])
    rows: List[str] = []
    if filtered.empty:
        rows = [r"-- & -- & -- & -- & -- & -- & -- & -- \\"]
    else:
        # Match the manuscript order: normal first, then Laplace.
        order = {"normal": 0, "laplace": 1}
        dists = _ordered_dists(filtered["dist"].dropna().unique())
        for idx, dist in enumerate(dists):
            group = filtered.loc[filtered["dist"].astype(str).str.lower() == str(dist).lower()]
            rows.extend(_high_dist_rows(group, include_midrule=idx < len(dists) - 1))
    body = (
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lllccccc}\n"
        "\\toprule\n"
        "Innovation & $(T,p,G)$ & Pilot & $\\gamma$ & $\\balpha$ & $\\bbeta$ & $\\bsigma^2$ & Time (sec.) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular*}"
    )
    return _table_environment(
        "tab:simulation_high_combined",
        "High-dimensional setting: RMSEs, ECPs reported below in parentheses, and average computation time.",
        body,
    )


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
            score_loss = _format_pair(row, "TopK_cache_score_loss_pos_abs_mean", "TopK_cache_score_loss_neg_abs_mean")
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
        "\\begin{tabular}{ccccccc}\n"
        "\\toprule\n"
        "$T$ & $p$ & $K$ & $\\RMSE_{\\gamma}$ & Boundary rate & Cache recall & Score loss \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}"
    )
    return _table_environment(
        "tab:simulation_xhigh_diagnostics",
        "Extra-high-dimensional sparse-attention diagnostics.",
        body,
    )


def render_table(table_name: str, df: pd.DataFrame) -> str:
    table_name = LEGACY_TABLE_ALIASES.get(table_name, table_name)
    if table_name == "simulation_low_uniform":
        return render_low_innovation_table(df, dist="uniform")
    if table_name == "simulation_design":
        return render_design_table(df)
    if table_name == "simulation_low_normal":
        return render_low_innovation_table(df, dist="normal")
    if table_name == "simulation_low_laplace":
        return render_low_innovation_table(df, dist="laplace")
    if table_name == "simulation_high_combined":
        return render_high_combined_table(df)
    if table_name == "simulation_xhigh_diagnostics":
        return render_xhigh_diagnostics(df)
    raise ValueError(f"Unknown table name: {table_name}")


def generate_tex_tables(
    *,
    results_dirs: Sequence[Path],
    tex_dir: Path,
    config_tag: Optional[str] = None,
    table_names: Optional[Iterable[str]] = None,
) -> Dict[str, Path]:
    tex_dir.mkdir(parents=True, exist_ok=True)
    selected_tables = list(table_names) if table_names is not None else list(DEFAULT_TABLES)
    written: Dict[str, Path] = {}
    for requested_name in selected_tables:
        table_name = LEGACY_TABLE_ALIASES.get(requested_name, requested_name)
        if table_name not in OUTPUT_FILES:
            raise ValueError(f"Unknown table name: {requested_name}")
        csv_paths = _latest_matching_csvs(results_dirs, TABLE_PREFIXES[table_name], config_tag=config_tag)
        df = _read_and_concat(csv_paths)
        tex = "% Auto-generated by scripts/make_simulation_tex_tables.py\n"
        if csv_paths:
            for csv_path in csv_paths:
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
    for requested_name in selected_tables:
        table_name = LEGACY_TABLE_ALIASES.get(requested_name, requested_name)
        out_file = tex_dir / OUTPUT_FILES[table_name]
        if out_file.exists():
            combined_lines.append(out_file.read_text(encoding="utf-8"))
            combined_lines.append("")
    combined.write_text("\n".join(combined_lines), encoding="utf-8")
    written["combined"] = combined
    return written


def _parse_table_list(text: str) -> List[str]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    valid = set(OUTPUT_FILES) | set(LEGACY_TABLE_ALIASES)
    unknown = sorted(set(values) - valid)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown table name(s): {', '.join(unknown)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate LaTeX VARA simulation tables from CSV outputs.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        nargs="+",
        default=[Path("outputs/results")],
        help="One or more result folders. Latest matching CSVs are combined across folders.",
    )
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
        default=list(DEFAULT_TABLES),
        help=f"Comma-separated table names. Choices: {','.join(OUTPUT_FILES)}",
    )
    args = parser.parse_args()

    written = generate_tex_tables(
        results_dirs=args.results_dir,
        tex_dir=args.tex_dir,
        config_tag=args.config_tag,
        table_names=args.tables,
    )
    print("Generated LaTeX simulation tables:")
    for name, path in written.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
