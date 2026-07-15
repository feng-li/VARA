#!/usr/bin/env python3
"""Plot per-series MAE violin plots for the Tourism and M5 applications."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
import zlib

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from scripts.plot_mae_boxplots import M5_METHODS, TOURISM_METHODS, _load_method_mae, _summary_rows


def _violin_sample(values: np.ndarray, *, label: str, max_points: int) -> np.ndarray:
    if values.size <= max_points:
        return values
    seed = zlib.adler32(label.encode("utf-8")) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    idx = rng.choice(values.size, size=max_points, replace=False)
    return values[idx]


def _draw_violinplot(
    ax,
    data: dict[str, np.ndarray],
    title: str,
    *,
    ylim_quantile: float = 0.99,
    max_points: int = 10000,
) -> None:
    labels = list(data)
    full_values = [data[label] for label in labels]
    plot_values = [_violin_sample(values, label=label, max_points=max_points) for label, values in data.items()]
    positions = np.arange(1, len(labels) + 1)

    violins = ax.violinplot(
        plot_values,
        positions=positions,
        widths=0.82,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for label, body in zip(labels, violins["bodies"]):
        if label.startswith("VARA"):
            body.set_facecolor("#756bb1")
            body.set_edgecolor("#3f2d78")
        else:
            body.set_facecolor("#c7c7c7")
            body.set_edgecolor("#606060")
        body.set_alpha(0.82)
        body.set_linewidth(0.9)

    for pos, values in zip(positions, full_values):
        q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
        mean = float(np.mean(values))
        ax.vlines(pos, q1, q3, color="#202020", linewidth=3.2, alpha=0.92)
        ax.hlines(median, pos - 0.24, pos + 0.24, color="#101010", linewidth=1.6)
        ax.scatter([pos], [mean], marker="D", s=18, facecolor="white", edgecolor="#303030", zorder=4)

    all_values = np.concatenate(full_values)
    ymax = float(np.quantile(all_values, ylim_quantile))
    ax.set_ylim(bottom=0.0, top=ymax * 1.08)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Per-series MAE")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))


def plot_mae_violinplots(output_png: Path, output_pdf: Path, summary_csv: Path) -> None:
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
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9.4), constrained_layout=True)
    _draw_violinplot(axes[0], tourism, "Tourism: per-series MAE by method")
    _draw_violinplot(axes[1], m5, "M5: per-series MAE by method")
    fig.suptitle("MAE Violin Plots Across Forecasted Series (y-axis capped at 99th percentile)", fontsize=14)
    fig.savefig(output_png, dpi=180)
    fig.savefig(output_pdf)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MAE violin plots for Tourism and M5 forecast methods.")
    parser.add_argument("--output-png", type=Path, default=Path("outputs/mae_violinplots_tourism_m5.png"))
    parser.add_argument("--output-pdf", type=Path, default=Path("outputs/mae_violinplots_tourism_m5.pdf"))
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/mae_violinplots_tourism_m5_summary.csv"))
    args = parser.parse_args()

    plot_mae_violinplots(args.output_png, args.output_pdf, args.summary_csv)
    print(f"Saved MAE violin plot PNG to: {args.output_png}")
    print(f"Saved MAE violin plot PDF to: {args.output_pdf}")
    print(f"Saved MAE violin plot summary to: {args.summary_csv}")


if __name__ == "__main__":
    main()
