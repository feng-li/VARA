#!/usr/bin/env python3
"""Plot per-series MAE boxplots for the Tourism and M5 applications."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


TOURISM_METHODS = [
    ("VARA reduced", Path("outputs/tourism_vara_auto_ets_core8_features/vara/rolling-origin/metrics_by_series.csv"), "vara"),
    ("VARA full tsfeatures", Path("outputs/tourism_vara_auto_ets_dense_workflow/vara/rolling-origin/metrics_by_series.csv"), "vara"),
    ("AR(1) AutoETS resid.", Path("outputs/tourism_benchmark_filtered_arvar/rolling-origin/metrics_by_series.csv"), "ar1_auto_ets"),
    ("AutoETS", Path("outputs/tourism_benchmark_auto_ets_compare_all_modes/rolling-origin/metrics_by_series.csv"), "auto_ets"),
    ("AutoARIMA", Path("outputs/tourism_benchmark_auto_ets_compare_all_modes/rolling-origin/metrics_by_series.csv"), "auto_arima"),
    ("N-BEATS", Path("outputs/tourism_benchmark_auto_ets_compare_all_modes/rolling-origin/metrics_by_series.csv"), "nbeats"),
    ("VAR(1) AutoETS resid.", Path("outputs/tourism_benchmark_filtered_arvar/rolling-origin/metrics_by_series.csv"), "var1_auto_ets"),
    ("Seasonal naive", Path("outputs/tourism_benchmark_auto_ets_compare_all_modes/rolling-origin/metrics_by_series.csv"), "seasonal_naive"),
    ("DeepAR", Path("outputs/tourism_benchmark_auto_ets_compare_all_modes/rolling-origin/metrics_by_series.csv"), "deepar"),
    ("AR(1)", Path("outputs/tourism_benchmark_auto_ets_compare_all_modes/rolling-origin/metrics_by_series.csv"), "ar1"),
    ("VAR(1)", Path("outputs/tourism_benchmark_auto_ets_compare_all_modes/rolling-origin/metrics_by_series.csv"), "var1"),
]

M5_METHODS = [
    ("VARA lambda=0.4", Path("outputs/m5_vara_auto_ets_dense_full_learned_lambda/vara_lambda04_aux/metrics_by_series.csv"), "vara"),
    ("VARA lambda=0.632", Path("outputs/m5_vara_auto_ets_dense_full_learned_lambda/vara/metrics_by_series.csv"), "vara"),
    ("AutoETS", Path("outputs/m5_benchmark_all_modes_with_variance/rolling-origin/metrics_by_series.csv"), "auto_ets"),
    ("AutoARIMA", Path("outputs/m5_benchmark_all_modes_with_variance/rolling-origin/metrics_by_series.csv"), "auto_arima"),
    ("DeepAR", Path("outputs/m5_benchmark_all_modes_with_variance/rolling-origin/metrics_by_series.csv"), "deepar"),
    ("AR(1)", Path("outputs/m5_benchmark_all_modes_with_variance/rolling-origin/metrics_by_series.csv"), "ar1"),
    ("N-BEATS", Path("outputs/m5_benchmark_all_modes_with_variance/rolling-origin/metrics_by_series.csv"), "nbeats"),
    ("Naive", Path("outputs/m5_benchmark_all_modes_with_variance/rolling-origin/metrics_by_series.csv"), "naive"),
    ("AR(1) SNaive resid.", Path("outputs/m5_benchmark_all_modes_with_variance/rolling-origin/metrics_by_series.csv"), "ar1_snaive"),
    ("Seasonal naive", Path("outputs/m5_benchmark_all_modes_with_variance/rolling-origin/metrics_by_series.csv"), "seasonal_naive"),
]


def _load_method_mae(methods: list[tuple[str, Path, str]]) -> dict[str, np.ndarray]:
    cache: dict[Path, pd.DataFrame] = {}
    out: dict[str, np.ndarray] = {}
    for label, path, model in methods:
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics_by_series.csv for {label}: {path}")
        if path not in cache:
            cache[path] = pd.read_csv(path, usecols=["model", "mae"])
        df = cache[path]
        values = df.loc[df["model"].astype(str).eq(model), "mae"].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            raise ValueError(f"No finite MAE values found for model {model!r} in {path}")
        out[label] = values
    return out


def _summary_rows(dataset: str, data: dict[str, np.ndarray]) -> list[dict[str, float | int | str]]:
    rows = []
    for method, values in data.items():
        rows.append({
            "dataset": dataset,
            "method": method,
            "n_series": int(values.size),
            "mean_mae": float(np.mean(values)),
            "median_mae": float(np.median(values)),
            "q25_mae": float(np.quantile(values, 0.25)),
            "q75_mae": float(np.quantile(values, 0.75)),
            "iqr_mae": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
        })
    return rows


def _draw_boxplot(ax, data: dict[str, np.ndarray], title: str) -> None:
    labels = list(data)
    values = [data[label] for label in labels]
    boxes = ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        showfliers=False,
        widths=0.62,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#333333",
            "markersize": 4.0,
        },
        medianprops={"color": "#111111", "linewidth": 1.3},
        whiskerprops={"color": "#666666", "linewidth": 1.0},
        capprops={"color": "#666666", "linewidth": 1.0},
    )
    for label, patch in zip(labels, boxes["boxes"]):
        if label.startswith("VARA"):
            patch.set_facecolor("#6baed6")
            patch.set_edgecolor("#1f5d8a")
        else:
            patch.set_facecolor("#d9d9d9")
            patch.set_edgecolor("#6b6b6b")
        patch.set_alpha(0.9)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Per-series MAE")
    ax.grid(axis="y", alpha=0.25)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    ax.tick_params(axis="x", labelrotation=35, labelsize=8)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")


def plot_mae_boxplots(output_png: Path, output_pdf: Path, summary_csv: Path) -> None:
    tourism = _load_method_mae(TOURISM_METHODS)
    m5 = _load_method_mae(M5_METHODS)

    summary = pd.DataFrame([
        *_summary_rows("Tourism", tourism),
        *_summary_rows("M5", m5),
    ])
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_csv, index=False)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.2), constrained_layout=True)
    _draw_boxplot(axes[0], tourism, "Tourism: per-series MAE by method")
    _draw_boxplot(axes[1], m5, "M5: per-series MAE by method")
    fig.suptitle("MAE Distributions Across Forecasted Series (outlier points hidden)", fontsize=14)
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAE boxplots for Tourism and M5 forecast methods.")
    parser.add_argument("--output-png", type=Path, default=Path("outputs/mae_boxplots_tourism_m5.png"))
    parser.add_argument("--output-pdf", type=Path, default=Path("outputs/mae_boxplots_tourism_m5.pdf"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/mae_boxplots_tourism_m5_summary.csv"))
    args = parser.parse_args()

    plot_mae_boxplots(args.output_png, args.output_pdf, args.summary_csv)
    print(f"Saved MAE boxplot PNG to: {args.output_png}")
    print(f"Saved MAE boxplot PDF to: {args.output_pdf}")
    print(f"Saved MAE boxplot summary to: {args.summary_csv}")


if __name__ == "__main__":
    main()
