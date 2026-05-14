#!/usr/bin/env python3
"""Plot tourism forecasts from a forecast output folder."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ._bootstrap import DATA_DIR
except ImportError:
    from _bootstrap import DATA_DIR

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


def _read_metrics_source(results_dir: Path) -> Path | None:
    metrics_path = results_dir / "tourism_metrics.csv"
    if not metrics_path.exists():
        return None
    metrics = pd.read_csv(metrics_path)
    if metrics.empty or "source" not in metrics.columns:
        return None
    source = Path(str(metrics.loc[0, "source"]))
    return source if source.exists() else None


def _load_actual_history(results_dir: Path, source_path: Path | None) -> pd.DataFrame:
    test = pd.read_csv(results_dir / "tourism_test.csv")
    test["date"] = pd.to_datetime(test["date"])
    if source_path is None:
        return test

    actual = pd.read_csv(source_path)
    actual["date"] = pd.to_datetime(actual["date"])
    series_cols = [col for col in test.columns if col != "date" and col in actual.columns]
    return actual.loc[:, ["date", *series_cols]]


def _series_metrics(forecast: pd.DataFrame, test: pd.DataFrame, series_cols: list[str]) -> pd.DataFrame:
    errors = forecast[series_cols].to_numpy(dtype=float) - test[series_cols].to_numpy(dtype=float)
    return pd.DataFrame({
        "series": series_cols,
        "rmse": np.sqrt(np.mean(errors ** 2, axis=0)),
        "mae": np.mean(np.abs(errors), axis=0),
    }).set_index("series")


def plot_tourism_forecasts_pdf(
    results_dir: Path,
    output: Path,
    *,
    source_path: Path | None = None,
    history_months: int = 36,
    rows: int = 4,
    cols: int = 4,
    max_series: int = 0,
    series: list[str] | None = None,
) -> Path:
    forecast_path = results_dir / "tourism_forecast.csv"
    test_path = results_dir / "tourism_test.csv"
    if not forecast_path.exists() or not test_path.exists():
        raise FileNotFoundError("results_dir must contain tourism_forecast.csv and tourism_test.csv.")

    forecast = pd.read_csv(forecast_path)
    test = pd.read_csv(test_path)
    forecast["date"] = pd.to_datetime(forecast["date"])
    test["date"] = pd.to_datetime(test["date"])

    source_path = source_path or _read_metrics_source(results_dir) or (DATA_DIR / "tourism" / "tourism.csv")
    actual = _load_actual_history(results_dir, source_path if source_path.exists() else None)

    available = [col for col in forecast.columns if col != "date" and col in test.columns and col in actual.columns]
    if series:
        missing = [name for name in series if name not in available]
        if missing:
            raise ValueError(f"Requested series not available in forecast/test/history data: {missing[:5]}")
        series_cols = series
    else:
        series_cols = available
    if max_series > 0:
        series_cols = series_cols[:max_series]
    if not series_cols:
        raise ValueError("No series selected for plotting.")

    per_series = _series_metrics(forecast, test, series_cols)
    test_start = test["date"].min()
    history = actual[actual["date"] < test_start].copy()
    if history_months > 0:
        history = history.tail(history_months)

    output.parent.mkdir(parents=True, exist_ok=True)
    per_page = rows * cols
    date_locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    date_formatter = mdates.ConciseDateFormatter(date_locator)

    with PdfPages(output) as pdf:
        for page_start in range(0, len(series_cols), per_page):
            page_series = series_cols[page_start:page_start + per_page]
            fig, axes = plt.subplots(rows, cols, figsize=(4.8 * cols, 3.0 * rows), sharex=False)
            axes = np.asarray(axes).reshape(-1)

            handles = None
            labels = None
            for ax, series_id in zip(axes, page_series):
                if not history.empty:
                    ax.plot(history["date"], history[series_id], color="0.35", linewidth=1.0, label="Observed history")
                ax.plot(test["date"], test[series_id], color="black", linewidth=1.4, marker="o", markersize=2.5, label="Observed test")
                ax.plot(
                    forecast["date"],
                    forecast[series_id],
                    color="tab:red",
                    linestyle="--",
                    linewidth=1.5,
                    marker="o",
                    markersize=2.5,
                    label="Forecast",
                )
                ax.axvline(test_start, color="0.55", linestyle=":", linewidth=1.0)
                stats = per_series.loc[series_id]
                ax.set_title(f"{series_id}  RMSE={stats['rmse']:.1f}, MAE={stats['mae']:.1f}", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis="both", which="major", labelsize=6)
                ax.xaxis.set_major_locator(date_locator)
                ax.xaxis.set_major_formatter(date_formatter)
                handles, labels = ax.get_legend_handles_labels()

            for ax in axes[len(page_series):]:
                ax.axis("off")

            page_num = page_start // per_page + 1
            total_pages = int(np.ceil(len(series_cols) / per_page))
            fig.suptitle(f"Tourism VARA Forecasts ({page_num}/{total_pages})", fontsize=14)
            if handles and labels:
                fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
            fig.tight_layout(rect=[0, 0.04, 1, 0.96])
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot tourism forecast outputs into a multi-page PDF.")
    parser.add_argument("--results-dir", type=Path, default=Path("outputs/tourism_forecast_detail76_rolling"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--source", type=Path, default=None, help="Optional original tourism CSV for history.")
    parser.add_argument("--history-months", type=int, default=36, help="Observed months before the test window; 0 uses all history.")
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--max-series", type=int, default=0)
    parser.add_argument("--series", type=str, default="", help="Optional comma-separated series IDs.")
    args = parser.parse_args()

    selected_series = [item.strip() for item in args.series.split(",") if item.strip()] or None
    output = args.output or (args.results_dir / "tourism_forecast_plots.pdf")
    out = plot_tourism_forecasts_pdf(
        args.results_dir,
        output,
        source_path=args.source,
        history_months=args.history_months,
        rows=args.rows,
        cols=args.cols,
        max_series=args.max_series,
        series=selected_series,
    )
    print(f"Saved tourism forecast plots to: {out}")


if __name__ == "__main__":
    main()
