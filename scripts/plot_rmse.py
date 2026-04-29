import argparse

try:
    from ._bootstrap import FIGURES_DIR
except ImportError:
    from _bootstrap import FIGURES_DIR

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Plot RMSE curves from a CSV file.")
    parser.add_argument("input_csv", help="CSV with columns T, RMSE_FULL, RMSE_Row1, RMSE_Rows5.")
    parser.add_argument(
        "--output",
        default=str(FIGURES_DIR / "rmse_comparison.pdf"),
        help="Output PDF path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, header=0)

    T = df["T"].values
    rmse_full = df["RMSE_FULL"].values
    rmse_row1 = df["RMSE_Row1"].values
    rmse_rows5 = df["RMSE_Rows5"].values

    plt.figure(figsize=(8, 5))
    plt.plot(T, rmse_full, "-o", linewidth=2, markersize=8, label="RMSE (Full Frobenius)")
    plt.plot(T, rmse_row1, "-s", linewidth=2, markersize=8, label="RMSE (Row 1 Approx)")
    plt.plot(T, rmse_rows5, "-p", linewidth=2, markersize=8, label="RMSE (Rows 1:5 Approx)")
    plt.yscale("log")
    plt.xlabel("T (Sample Size)")
    plt.ylabel("RMSE")
    plt.title("RMSE vs T")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(args.output, bbox_inches="tight")
    plt.show()
    print(f"Saved plot to {args.output}")


if __name__ == "__main__":
    main()
