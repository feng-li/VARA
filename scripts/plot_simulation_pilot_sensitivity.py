"""Plot pilot-size sensitivity from run_full_simulation.py summary CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import FIGURES_DIR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_pilot_sensitivity(summary: Path, output: Path) -> None:
    data = pd.read_csv(summary)
    data = data.loc[data["method"].str.startswith("two_stage_m")].copy()
    if data.empty or len(data[["T", "p", "dist"]].drop_duplicates()) != 1:
        raise ValueError("Select a pilot summary containing exactly one design and innovation distribution.")
    data = data.sort_values("pilot_size")
    if data["pilot_size"].duplicated().any():
        raise ValueError("The summary contains duplicate pilot sizes.")
    if not data["valid_rate"].eq(1).all():
        raise ValueError("Resolve failed replications before drawing the pilot comparison.")
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.8, 4.1), layout="constrained")
    ax.plot(data["cpu_sec_mean"], data["RMSE_pool"], "o-", color="#25649B")
    for row in data.itertuples():
        ax.annotate(rf"$m_p={int(row.pilot_size)}$", (row.cpu_sec_mean, row.RMSE_pool),
                    xytext=(5, 7), textcoords="offset points")
    ax.set(xlabel="Mean CPU time (seconds)", ylabel="Pooled RMSE")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#DEDEDE", linewidth=0.6)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "high_dim_pilot_sensitivity.pdf")
    args = parser.parse_args()
    plot_pilot_sensitivity(args.summary, args.output)
