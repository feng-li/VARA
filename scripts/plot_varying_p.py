import argparse

try:
    from ._bootstrap import DATA_DIR, FIGURES_DIR
except ImportError:
    from _bootstrap import DATA_DIR, FIGURES_DIR

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot RMSE against dimension p.")
    parser.add_argument(
        "--input-csv",
        default=str(DATA_DIR / "varying_p.csv"),
        help="Input CSV path.",
    )
    parser.add_argument(
        "--output",
        default=str(FIGURES_DIR / "p_comparison.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, header=0)

    p = df["p"].values
    rmse_eta_a = df["eta_a"].values
    rmse_eta_b = df["eta_b"].values

    plt.figure(figsize=(8, 5))
    plt.plot(p, rmse_eta_b, "-s", linewidth=6, markersize=8, label="RMSE (eta_b)")
    plt.plot(p, rmse_eta_a, "-o", linewidth=2, markersize=8, label="RMSE (eta_a)")
    plt.yscale("log")
    plt.xlabel("P (Dimension)")
    plt.ylabel("RMSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.output, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
