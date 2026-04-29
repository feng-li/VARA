#!/usr/bin/env python3

import sys
from pathlib import Path

import pandas as pd

def summarize_empirical_coverage(file_path: str, latex: bool):
    """
    Summarize empirical coverage results across all parameters, sample sizes (T),
    and nominal confidence levels.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing simulation results.
        Expected columns: ["T", "nominal", "param", "coverage", "rmse", "avg_len"]

    Returns
    -------
    dict
        Dictionary with summary tables:
        - summary_group_nominal
        - param_summary
        - grand_summary
    """
    # --- Load results ---
    csv_path = Path(file_path)
    df = pd.read_csv(csv_path)


    # --- Convert numeric columns explicitly ---
    numeric_cols = ["R", "T", "nominal", "coverage", "rmse", "avg_len"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing numeric values if any
    df = df.dropna(subset=numeric_cols)

    # --- Summary by group and nominal ---
    summary_group_nominal = (
        df.groupby(["R", "T", "nominal", "dist"], as_index=False)
        .agg(
            mean_coverage=("coverage", "mean"),
            std_coverage=("coverage", "std"),
            mean_rmse=("rmse", "mean"),
            mean_len=("avg_len", "mean")
        )
        .sort_values(["T", "nominal"])
    )

    print("\n"+"="*70+"\n=== ECP, RMSE, and CI Length by R, T\n"+"="*70)
    print(summary_group_nominal.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    # --- Parameter-level coverage table ---
    print("\n"+"="*70+"\n=== ECP: Parameter-level \n"+"="*70)
    param_summary = (
        df.pivot_table(
            index=["param_group","param"],
            columns=["dist", "R", "T", "p", "q", "nominal"],
            values="coverage",
            aggfunc="mean"
        )
        .round(3)
    )
    print(param_summary.T)
    # param_summary.to_csv("param_summary.csv")
    # order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    # for group in order:
    #     if group not in param_summary.index.get_level_values(0):
    #         continue
    #     sub = param_summary.loc[group]
    #     for param_name, row in sub.iterrows():
    #         print(f"\n[{group}] {param_name}")
    #         print(row.to_string())
    print("\n"+"="*70+"\n=== ECP: Parameter group level \n"+"="*70)
    param_ECP_summary = (
        df.pivot_table(
            index=["param_group"],
            columns=["dist", "R", "T", "p", "q", "nominal"],
            values="coverage",
            aggfunc="mean"
        )
        .round(3)
    )
    print(param_ECP_summary.T.to_string())

    # --- Parameter group ---
    print("\n"+"="*70+"\n=== RMSE: Parameter group level \n"+"="*70)
    param_RMSE_summary = (
        df.pivot_table(
            index="param_group",
            columns=["dist", "R", "T", "p", "q"],
            values="rmse",
            aggfunc="mean"
        )
        .round(3)
    )
    print(param_RMSE_summary.T.to_string())

    if latex:
        param_ECP_summary.T.to_latex(
            csv_path.parent / f"{csv_path.name}_param_ECP_summary.tex",
            index=True,
            escape=False,
            float_format=lambda x: f"{x:.3f}"
        )

        param_RMSE_summary.T.to_latex(
            csv_path.parent / f"{csv_path.name}_param_RMSE_summary.tex",
            index=True,
            escape=False,
            float_format=lambda x: f"{x:.3f}"
        )


    # --- Overall coverage by nominal level ---
    print("\n"+"="*70+"\n=== ECP: Overall \n"+"="*70)
    grand_summary = (
        df.groupby(["nominal"], as_index=False)
        .agg(overall_mean_coverage=("coverage", "mean"))
    )
    print(grand_summary.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\n"+"="*70+"\n=== Computing time: Overall \n"+"="*70)
    elapsed_time = df.groupby(["R", "T", "p", "q", "dist"])["elapsed_min"].mean().reset_index()
    print(elapsed_time.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    print("\n"+"="*70+"\n=== Summary ends here \n"+"="*70)

    return {
        "summary_group_nominal": summary_group_nominal,
        "param_summary": param_summary,
        "grand_summary": grand_summary
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/summarize_ecp.py <path_to_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    print(f"Reading results from: {csv_path}")

    results = summarize_empirical_coverage(csv_path, latex=True)
    # print(results)

if __name__ == "__main__":
    main()
