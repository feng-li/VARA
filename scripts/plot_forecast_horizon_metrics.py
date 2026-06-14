#!/usr/bin/env python3
"""Plot forecast metrics over the forecast horizon."""

from __future__ import annotations

import argparse
import math
from itertools import cycle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import pandas as pd


DEFAULT_METRICS = "rmse=RMSE,crps=CRPS,bias=Bias"


def _parse_mapping(value: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            key, label = item.split("=", 1)
            mapping[key.strip()] = label.strip()
        else:
            mapping[item] = item
    return mapping


def _parse_results_dirs(value: str, search_root: Path) -> list[Path]:
    if value.strip():
        return [Path(item.strip()) for item in value.split(",") if item.strip()]
    return sorted(path.parent for path in search_root.glob("*/metrics_by_origin.csv"))


def _load_metrics_by_horizon(
    results_dirs: list[Path],
    *,
    model_labels: dict[str, str],
) -> pd.DataFrame:
    frames = []
    for results_dir in results_dirs:
        path = results_dir / "metrics_by_origin.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing metrics_by_origin.csv in {results_dir}")
        df = pd.read_csv(path)
        if "model" not in df.columns or "date" not in df.columns:
            raise ValueError(f"{path} must contain model and date columns.")
        df["date"] = pd.to_datetime(df["date"])
        df["source_dir"] = results_dir.as_posix()
        df["method"] = df["model"].map(model_labels).fillna(df["model"])
        frames.append(df)

    if not frames:
        raise ValueError("No metrics_by_origin.csv files were found.")

    metrics = pd.concat(frames, ignore_index=True)
    metrics = metrics.sort_values(["source_dir", "method", "date"]).copy()
    metrics["horizon"] = metrics.groupby(["source_dir", "method"]).cumcount() + 1
    keep = [
        "source_dir",
        "method",
        "model",
        "horizon",
        "date",
        "rmse",
        "mae",
        "mape_pct",
        "smape_pct",
        "mase",
        "rmsse",
        "bias",
        "crps",
        "forecast_variance_mean",
    ]
    return metrics.loc[:, [col for col in keep if col in metrics.columns]]


def _style_maps(methods: list[str]) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    palette = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    colors = {method: next(palette) for method in methods}
    marker_cycle = cycle(["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "8"])
    markers = {method: next(marker_cycle) for method in methods}
    linestyle_cycle = cycle(["-", "--", "-.", ":", (0, (5, 2)), (0, (3, 1, 1, 1))])
    linestyles = {method: next(linestyle_cycle) for method in methods}
    return colors, markers, linestyles


def _is_vara_method(method: str) -> bool:
    return str(method).strip().startswith("VARA")


def _plot_method_line(ax, sub: pd.DataFrame, col: str, method: str, colors: dict, markers: dict, linestyles: dict) -> None:
    is_vara = _is_vara_method(method)
    ax.plot(
        sub["horizon"],
        sub[col],
        label=method,
        color=colors.get(method),
        linestyle="-" if is_vara else "None",
        marker=markers.get(method, "o"),
        markersize=5.2,
        markerfacecolor="white",
        markeredgewidth=1.1,
        linewidth=1.9 if is_vara else 0.0,
    )


def _add_panel_legends(axes, plot_metrics: list[tuple[str, str]], *, legend_ncol: int = 2) -> None:
    for ax, (metric, _) in zip(axes, plot_metrics):
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(
                handles,
                labels,
                loc="lower right" if metric == "bias" else "upper right",
                ncol=legend_ncol,
                frameon=True,
                framealpha=0.88,
                facecolor="white",
                edgecolor="0.75",
                fontsize=7.2,
                columnspacing=0.65,
                handlelength=1.8,
                handletextpad=0.4,
                borderpad=0.35,
            )


def _dense_log_ticks_from_limits(ymin: float, ymax: float) -> list[float]:
    if ymin <= 0.0 or ymax <= 0.0 or not math.isfinite(ymin) or not math.isfinite(ymax):
        return []
    mantissas = (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0)
    lo_exp = math.floor(math.log10(ymin)) - 1
    hi_exp = math.ceil(math.log10(ymax)) + 1
    ticks: list[float] = []
    for exp in range(lo_exp, hi_exp + 1):
        scale = 10.0 ** exp
        ticks.extend(value for value in (m * scale for m in mantissas) if ymin * 0.98 <= value <= ymax * 1.02)
    return ticks


def _format_log_axis_ticks(ax, *, dense: bool = False) -> None:
    if dense:
        ticks = _dense_log_ticks_from_limits(*ax.get_ylim())
        if len(ticks) >= 4:
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
            ax.yaxis.set_minor_formatter(NullFormatter())
            return
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(1.5, 2.5, 4.0, 6.0, 7.0, 8.0, 9.0)))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_horizon_metrics(
    metrics: pd.DataFrame,
    output_pdf: Path,
    *,
    output_png: Path | None = None,
    metric_labels: dict[str, str],
    method_order: list[str] | None = None,
    title: str = "",
    note: str = "",
    log_positive_metrics: bool = False,
    legend_ncol: int = 2,
) -> None:
    plot_metrics = [(col, label) for col, label in metric_labels.items() if col in metrics.columns]
    if not plot_metrics:
        raise ValueError("None of the requested plot metrics are present in the metrics table.")

    methods = method_order or sorted(metrics["method"].dropna().astype(str).unique())
    colors, markers, linestyles = _style_maps(methods)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)

    max_horizon = int(metrics["horizon"].max())
    x_padding = 0.45 if max_horizon > 1 else 0.25
    n_panels = len(plot_metrics)
    n_cols = 1
    n_rows = n_panels
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.4, 3.0 * n_rows), sharex=False)
    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for ax, (col, label) in zip(axes, plot_metrics):
        panel_title = label
        for method in methods:
            sub = metrics[metrics["method"].eq(method)]
            if sub.empty:
                continue
            _plot_method_line(ax, sub, col, method, colors, markers, linestyles)
        ax.set_xlim(1 - x_padding, max_horizon + x_padding)
        ax.set_xticks(range(1, max_horizon + 1))
        ax.set_xlabel("Horizon")
        ax.grid(True, alpha=0.25)
        if log_positive_metrics and col != "bias" and (metrics[col].dropna() > 0.0).all():
            ax.set_yscale("log")
            _format_log_axis_ticks(ax, dense=col == "rmse")
            panel_title = f"{label} (log scale)"
        if col == "bias":
            ax.axhline(0.0, color="0.4", linewidth=0.9, alpha=0.8)
        if not (log_positive_metrics and col != "bias" and (metrics[col].dropna() > 0.0).all()):
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
        ax.set_title(panel_title, fontsize=12)

    _add_panel_legends(axes, plot_metrics, legend_ncol=legend_ncol)
    if note:
        fig.text(0.01, 0.015, note, fontsize=10, color="0.25")

    for ax in axes[n_panels:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=15, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
    else:
        fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_relative_metrics(
    metrics: pd.DataFrame,
    *,
    baseline_method: str,
    metric_labels: dict[str, str],
) -> pd.DataFrame:
    available = [col for col in metric_labels if col in metrics.columns]
    if not available:
        raise ValueError("None of the requested relative metrics are present in the metrics table.")

    baseline = metrics[metrics["method"].eq(baseline_method)]
    if baseline.empty:
        raise ValueError(f"Baseline method not found in metrics table: {baseline_method}")

    baseline_by_horizon = baseline.groupby("horizon", as_index=False)[available].mean()
    rows = []
    for _, row in metrics.iterrows():
        horizon = row["horizon"]
        base_row = baseline_by_horizon[baseline_by_horizon["horizon"].eq(horizon)]
        if base_row.empty:
            continue
        base_row = base_row.iloc[0]
        for metric in available:
            base_value = float(base_row[metric])
            if base_value == 0.0 or pd.isna(base_value) or pd.isna(row[metric]):
                continue
            rows.append({
                "source_dir": row["source_dir"],
                "method": row["method"],
                "model": row["model"],
                "horizon": int(horizon),
                "date": row["date"],
                "metric": metric,
                "metric_label": metric_labels[metric],
                "baseline_method": baseline_method,
                "baseline_value": base_value,
                "value": float(row[metric]),
                "relative_pct": 100.0 * (float(row[metric]) / base_value - 1.0),
            })
    return pd.DataFrame(rows)


def plot_relative_metrics(
    relative: pd.DataFrame,
    output_pdf: Path,
    *,
    output_png: Path | None = None,
    method_order: list[str] | None = None,
    title: str = "",
    note: str = "",
    legend_ncol: int = 2,
) -> None:
    if relative.empty:
        raise ValueError("Relative metrics table is empty.")

    methods = method_order or sorted(relative["method"].dropna().astype(str).unique())
    metric_labels = relative[["metric", "metric_label"]].drop_duplicates().sort_values("metric")
    plot_metrics = list(metric_labels.itertuples(index=False, name=None))

    colors, markers, linestyles = _style_maps(methods)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    if output_png is not None:
        output_png.parent.mkdir(parents=True, exist_ok=True)

    max_horizon = int(relative["horizon"].max())
    x_padding = 0.45 if max_horizon > 1 else 0.25
    n_panels = len(plot_metrics)
    n_cols = 1
    n_rows = n_panels
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.4, 3.0 * n_rows), sharex=False)
    if n_panels == 1:
        axes = [axes]
    else:
        axes = axes.ravel()

    for ax, (metric, label) in zip(axes, plot_metrics):
        panel = relative[relative["metric"].eq(metric)]
        baseline_method = str(panel["baseline_method"].iloc[0])
        for method in methods:
            sub = panel[panel["method"].eq(method)]
            if sub.empty:
                continue
            _plot_method_line(
                ax,
                sub.rename(columns={"relative_pct": "_relative_pct"}),
                "_relative_pct",
                method,
                colors,
                markers,
                linestyles,
            )
        ax.axhline(0.0, color="0.4", linewidth=0.9, alpha=0.8)
        ax.set_title(f"{label} vs {baseline_method} (%)", fontsize=12)
        ax.set_xlim(1 - x_padding, max_horizon + x_padding)
        ax.set_xticks(range(1, max_horizon + 1))
        ax.set_xlabel("Horizon")
        ax.set_ylabel("Relative change (%)")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}"))
        ax.grid(True, alpha=0.25)

    _add_panel_legends(axes, plot_metrics, legend_ncol=legend_ncol)
    if note:
        fig.text(0.01, 0.015, note, fontsize=10, color="0.25")

    for ax in axes[n_panels:]:
        ax.axis("off")

    if title:
        fig.suptitle(title, fontsize=15, y=0.995)
        fig.tight_layout(rect=[0, 0, 1, 0.975])
    else:
        fig.tight_layout()
    fig.savefig(output_pdf, bbox_inches="tight")
    if output_png is not None:
        fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot forecast metrics over forecast horizons.")
    parser.add_argument(
        "--results-dirs",
        type=str,
        default="",
        help="Comma-separated output folders containing metrics_by_origin.csv. If empty, discover under --search-root.",
    )
    parser.add_argument("--search-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--model-labels",
        type=str,
        default="",
        help="Comma-separated model=label mappings used in the plot legend.",
    )
    parser.add_argument(
        "--method-order",
        type=str,
        default="",
        help="Comma-separated display labels controlling legend and line order.",
    )
    parser.add_argument(
        "--include-methods",
        type=str,
        default="",
        help="Optional comma-separated display labels to keep before plotting.",
    )
    parser.add_argument(
        "--exclude-methods",
        type=str,
        default="",
        help="Optional comma-separated display labels to drop before plotting.",
    )
    parser.add_argument(
        "--plot-metrics",
        type=str,
        default=DEFAULT_METRICS,
        help="Comma-separated metric=label mappings to plot.",
    )
    parser.add_argument("--output-csv", type=Path, default=Path("outputs/forecast_metrics_by_horizon.csv"))
    parser.add_argument("--output-pdf", type=Path, default=Path("outputs/forecast_metrics_by_horizon_plot.pdf"))
    parser.add_argument("--output-png", type=Path, default=Path("outputs/forecast_metrics_by_horizon_plot.png"))
    parser.add_argument("--title", type=str, default="")
    parser.add_argument("--note", type=str, default="")
    parser.add_argument(
        "--log-positive-metrics",
        action="store_true",
        help="Use a log y-axis for strictly positive absolute metric panels except bias.",
    )
    parser.add_argument("--legend-ncol", type=int, default=2, help="Number of columns in each panel legend.")
    parser.add_argument(
        "--relative-baseline",
        type=str,
        default="",
        help="Method label used as the denominator for relative metric plots, for example SNaive.",
    )
    parser.add_argument(
        "--relative-metrics",
        type=str,
        default="rmse=RMSE,mae=MAE",
        help="Comma-separated metric=label mappings for the relative-to-baseline plot.",
    )
    parser.add_argument("--relative-output-csv", type=Path, default=None)
    parser.add_argument("--relative-output-pdf", type=Path, default=None)
    parser.add_argument("--relative-output-png", type=Path, default=None)
    args = parser.parse_args()

    results_dirs = _parse_results_dirs(args.results_dirs, args.search_root)
    model_labels = _parse_mapping(args.model_labels)
    metric_labels = _parse_mapping(args.plot_metrics)
    method_order = [item.strip() for item in args.method_order.split(",") if item.strip()] or None
    include_methods = [item.strip() for item in args.include_methods.split(",") if item.strip()]
    exclude_methods = [item.strip() for item in args.exclude_methods.split(",") if item.strip()]

    metrics = _load_metrics_by_horizon(results_dirs, model_labels=model_labels)
    if include_methods:
        metrics = metrics[metrics["method"].isin(include_methods)].copy()
    if exclude_methods:
        metrics = metrics[~metrics["method"].isin(exclude_methods)].copy()
    if metrics.empty:
        raise ValueError("No rows remain after applying method filters.")
    if method_order is not None:
        method_order = [method for method in method_order if method in set(metrics["method"])]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_csv, index=False)
    plot_horizon_metrics(
        metrics,
        args.output_pdf,
        output_png=args.output_png,
        metric_labels=metric_labels,
        method_order=method_order,
        title=args.title,
        note=args.note,
        log_positive_metrics=args.log_positive_metrics,
        legend_ncol=max(1, args.legend_ncol),
    )

    if args.relative_baseline:
        relative_metric_labels = _parse_mapping(args.relative_metrics)
        relative = make_relative_metrics(
            metrics,
            baseline_method=args.relative_baseline,
            metric_labels=relative_metric_labels,
        )
        relative_output_csv = args.relative_output_csv or args.output_csv.with_name(f"{args.output_csv.stem}_relative.csv")
        relative_output_pdf = args.relative_output_pdf or args.output_pdf.with_name(f"{args.output_pdf.stem}_relative.pdf")
        relative_output_png = args.relative_output_png or args.output_png.with_name(f"{args.output_png.stem}_relative.png")
        relative_output_csv.parent.mkdir(parents=True, exist_ok=True)
        relative.to_csv(relative_output_csv, index=False)
        plot_relative_metrics(
            relative,
            relative_output_pdf,
            output_png=relative_output_png,
            method_order=method_order,
            title=args.title,
            note=args.note,
            legend_ncol=max(1, args.legend_ncol),
        )
        print(f"Saved relative horizon metrics to: {relative_output_csv}")
        print(f"Saved relative horizon metrics plot to: {relative_output_pdf}")
        print(f"Saved relative horizon metrics plot to: {relative_output_png}")

    print(f"Saved horizon metrics to: {args.output_csv}")
    print(f"Saved horizon metrics plot to: {args.output_pdf}")
    print(f"Saved horizon metrics plot to: {args.output_png}")


if __name__ == "__main__":
    main()
