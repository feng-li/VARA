#!/usr/bin/env python3
"""Plot the MSE distribution for Tourism Table 6."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from _bootstrap import MPLCONFIG_DIR
except ModuleNotFoundError:
    from scripts._bootstrap import MPLCONFIG_DIR

os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIG_DIR))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from scripts.build_tourism_table6_zza import (
    DISPLAY_NAMES,
    INSAMPLE_VARA_MODEL,
    NON_AUTOETS_MODELS,
    RANK_SCORE_END,
    RANK_SCORE_START,
    ZZA_BENCHMARK_MODELS,
    ZZA_VARA_MODEL_NAMES,
)


AUTOETS_RELATED_MODELS = {
    *ZZA_BENCHMARK_MODELS,
    *ZZA_VARA_MODEL_NAMES.values(),
    INSAMPLE_VARA_MODEL,
}
PLOT_LABELS = {
    "vara_auto_ets_lambda": r"VARA–AutoETS$_{\lambda}$",
    "auto_ets": "AutoETS",
    "ar1_auto_ets": "AR(1)–AutoETS",
    "nbeats": "N-BEATS",
    INSAMPLE_VARA_MODEL: "VARA–AutoETS",
    "auto_arima": "AutoARIMA",
    "deepar": "DeepAR",
    "var1_auto_ets": "VAR(1)–AutoETS",
    "ar1": "AR(1)",
    "var1": "VAR(1)",
    "seasonal_naive": "Seasonal naive",
}
CALIBRATED_COLOR = "#2166ac"
INSAMPLE_COLOR = "#67a9cf"
AUTOETS_COLOR = "#d1e5f0"
BENCHMARK_COLOR = "#d9d9d9"


def _model_color(model: str) -> str:
    if model == "vara_auto_ets_lambda":
        return CALIBRATED_COLOR
    if model == INSAMPLE_VARA_MODEL:
        return INSAMPLE_COLOR
    if model in AUTOETS_RELATED_MODELS:
        return AUTOETS_COLOR
    return BENCHMARK_COLOR


def _select_origin_mse(
    frame: pd.DataFrame,
    model_names: tuple[str, ...],
    *,
    source: str,
) -> pd.DataFrame:
    required = {"model", "date"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} origin metrics are missing columns: {missing}")
    if "mse" not in frame and "rmse" not in frame:
        raise ValueError(f"{source} origin metrics require MSE or RMSE.")
    frame = frame.copy()
    if "mse" not in frame:
        frame["mse"] = np.square(frame["rmse"].to_numpy(dtype=float))
    selected = frame[frame["model"].astype(str).isin(model_names)].copy()
    missing_models = sorted(
        set(model_names).difference(selected["model"].astype(str).unique())
    )
    if missing_models:
        raise ValueError(f"{source} origin metrics are missing models: {missing_models}")
    selected["model"] = selected["model"].astype(str)
    selected["date"] = pd.to_datetime(selected["date"])
    selected = selected[selected["date"].between(
        RANK_SCORE_START,
        RANK_SCORE_END,
    )].copy()
    if selected.duplicated(["model", "date"]).any():
        raise ValueError(f"{source} origin metrics contain duplicate rows.")
    values = selected["mse"].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0.0).any():
        raise ValueError(f"{source} origin MSE values must be finite and positive.")
    selected["source"] = source
    return selected.loc[:, ["model", "date", "mse", "source"]]


def load_plot_data(
    standard_origin_metrics_path: Path,
    zza_origin_metrics_path: Path,
    vara_origin_metrics_path: Path,
    insample_origin_metrics_path: Path,
    table6_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the exact forecast inputs used by normalized Table 6."""
    standard = _select_origin_mse(
        pd.read_csv(standard_origin_metrics_path),
        NON_AUTOETS_MODELS,
        source="unchanged_non_autoets_benchmark",
    )
    zza = _select_origin_mse(
        pd.read_csv(zza_origin_metrics_path),
        ZZA_BENCHMARK_MODELS,
        source="zza_training_zscore_benchmark",
    )
    vara = _select_origin_mse(
        pd.read_csv(vara_origin_metrics_path),
        tuple(ZZA_VARA_MODEL_NAMES),
        source="zza_training_zscore_vara",
    )
    vara["model"] = vara["model"].map(ZZA_VARA_MODEL_NAMES)
    insample = _select_origin_mse(
        pd.read_csv(insample_origin_metrics_path),
        (INSAMPLE_VARA_MODEL,),
        source="insample_error_vara",
    )
    origins = pd.concat(
        [standard, zza, vara, insample],
        ignore_index=True,
    )

    table = pd.read_csv(table6_path)
    required_table = {
        "model",
        "method",
        "response_normalization",
    }
    missing_table = sorted(required_table.difference(table.columns))
    if missing_table:
        raise ValueError(f"Table 6 data are missing columns: {missing_table}")
    if "mse" not in table and "rmse" not in table:
        raise ValueError("Table 6 data require MSE or RMSE.")
    if "mse" not in table:
        table["mse"] = np.square(table["rmse"].to_numpy(dtype=float))
    if table["model"].duplicated().any():
        raise ValueError("Table 6 contains duplicate model rows.")
    expected_models = set(DISPLAY_NAMES)
    if set(table["model"].astype(str)) != expected_models:
        raise ValueError("Table 6 does not contain the prespecified 11 methods.")
    if set(origins["model"]) != expected_models:
        raise ValueError("Origin MSE data do not contain the Table 6 method set.")

    expected_dates = tuple(pd.date_range(
        RANK_SCORE_START,
        RANK_SCORE_END,
        freq="MS",
    ))
    dates_by_model = origins.groupby("model", observed=True)["date"].agg(
        lambda values: tuple(sorted(pd.Timestamp(value) for value in values))
    )
    if any(dates != expected_dates for dates in dates_by_model):
        raise ValueError(
            "Every plotted method must have exactly the 12 monthly origins in 2016."
        )

    normalized = table[table["model"].isin(AUTOETS_RELATED_MODELS)]
    if not normalized["response_normalization"].eq("training_zscore").all():
        raise ValueError(
            "Every AutoETS-related plotted method must use training z-scores."
        )
    other = table[~table["model"].isin(AUTOETS_RELATED_MODELS)]
    if not other["response_normalization"].eq("not_applicable").all():
        raise ValueError(
            "Non-AutoETS Table 6 methods unexpectedly declare normalization."
        )

    table = table.sort_values(
        ["mse", "model"],
        ascending=[True, True],
    ).reset_index(drop=True)
    table["plot_label"] = table["model"].map(PLOT_LABELS)
    if table["plot_label"].isna().any():
        raise ValueError("A Table 6 model has no plotting label.")
    origins = origins.merge(
        table.loc[:, [
            "model",
            "method",
            "mse",
            "response_normalization",
            "plot_label",
        ]].rename(columns={"mse": "pooled_mse"}),
        on="model",
        how="left",
        validate="many_to_one",
    )
    return origins.sort_values(["pooled_mse", "date"]), table


def plot_table6_comparison(
    origin_metrics: pd.DataFrame,
    table: pd.DataFrame,
    *,
    output_pdf: Path,
    output_png: Path,
    plot_data_csv: Path,
    evaluation_scale: str = "original",
) -> None:
    """Draw monthly MSE boxplots with pooled Table 6 MSE markers."""
    if evaluation_scale not in {"original", "training_zscore"}:
        raise ValueError(
            "evaluation_scale must be 'original' or 'training_zscore'."
        )
    order = table["model"].astype(str).tolist()
    labels = table["plot_label"].astype(str).tolist()
    positions = np.arange(len(order), 0, -1, dtype=float)
    mse_values = [
        origin_metrics.loc[
            origin_metrics["model"].eq(model),
            "mse",
        ].to_numpy(dtype=float)
        for model in order
    ]
    pooled_mse = table.set_index("model").loc[order, "mse"].to_numpy(dtype=float)
    colors = [_model_color(model) for model in order]

    plot_data_csv.parent.mkdir(parents=True, exist_ok=True)
    origin_metrics.to_csv(plot_data_csv, index=False)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context({
        "font.family": "serif",
        "font.size": 9.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }):
        fig, ax_mse = plt.subplots(
            figsize=(7.2, 6.0),
            constrained_layout=True,
        )
        boxes = ax_mse.boxplot(
            mse_values,
            positions=positions,
            orientation="horizontal",
            widths=0.56,
            patch_artist=True,
            showmeans=False,
            showfliers=False,
            manage_ticks=False,
            medianprops={"color": "#202020", "linewidth": 1.35},
            whiskerprops={"color": "#666666", "linewidth": 0.9},
            capprops={"color": "#666666", "linewidth": 0.9},
            flierprops={
                "marker": "o",
                "markerfacecolor": "#777777",
                "markeredgecolor": "none",
                "markersize": 3.0,
                "alpha": 0.55,
            },
        )
        for patch, color in zip(boxes["boxes"], colors, strict=True):
            patch.set_facecolor(color)
            patch.set_edgecolor("#4d4d4d")
            patch.set_alpha(0.95)

        jitter = np.linspace(-0.17, 0.17, 12)
        for position, values, color in zip(
            positions,
            mse_values,
            colors,
            strict=True,
        ):
            ax_mse.scatter(
                values,
                position + jitter,
                s=10,
                color=color,
                edgecolor="#4d4d4d",
                linewidth=0.25,
                alpha=0.55,
                zorder=3,
            )
        ax_mse.scatter(
            pooled_mse,
            positions,
            marker="D",
            s=31,
            facecolor="white",
            edgecolor="#111111",
            linewidth=0.9,
            zorder=5,
        )
        all_mse = np.concatenate([*mse_values, pooled_mse])
        ax_mse.set_xlim(0.0, float(np.max(all_mse)) * 1.08)
        ax_mse.set_yticks(positions)
        ax_mse.set_yticklabels(labels, fontsize=8.4)
        if evaluation_scale == "training_zscore":
            mse_xlabel = "Monthly cross-series MSE (training z-score scale)"
            mse_title = "Scaled MSE across the 12 rolling origins"
            pooled_label = "Pooled MSE in Table 6"
        else:
            mse_xlabel = "Monthly cross-series MSE"
            mse_title = "MSE across the 12 rolling origins"
            pooled_label = "Pooled MSE in Table 6"
        ax_mse.set_xlabel(mse_xlabel)
        ax_mse.set_title(mse_title, loc="left")
        ax_mse.grid(axis="x", alpha=0.24)

        legend_handles = [
            Patch(
                facecolor=CALIBRATED_COLOR,
                edgecolor="#4d4d4d",
                label=r"VARA–AutoETS$_{\lambda}$",
            ),
            Patch(
                facecolor=INSAMPLE_COLOR,
                edgecolor="#4d4d4d",
                label="VARA–AutoETS",
            ),
            Patch(
                facecolor=AUTOETS_COLOR,
                edgecolor="#4d4d4d",
                label="Other AutoETS-related",
            ),
            Patch(
                facecolor=BENCHMARK_COLOR,
                edgecolor="#4d4d4d",
                label="Other benchmark",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor="white",
                markeredgecolor="#111111",
                markersize=5.5,
                label=pooled_label,
            ),
        ]
        fig.legend(
            handles=legend_handles,
            loc="outside lower center",
            ncol=3,
            frameon=False,
            fontsize=8.0,
        )
        fig.savefig(output_pdf, bbox_inches="tight")
        fig.savefig(output_png, dpi=220, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--standard-origin-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_standardized_run/rolling-origin/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--zza-origin-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_zza_train_zscore_run/rolling-origin/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--vara-origin-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_reliability_vara_autoets_zza_train_zscore/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--insample-origin-metrics-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_insample_error_vara_autoets/"
            "metrics_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--table6-path",
        type=Path,
        default=Path(
            "outputs/tourism_table6_zza_train_zscore/table6_zza_full.csv"
        ),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("docs/figures/tourism_table6_rmse_rankscore.pdf"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("docs/figures/tourism_table6_rmse_rankscore.png"),
    )
    parser.add_argument(
        "--plot-data-csv",
        type=Path,
        default=Path(
            "outputs/tourism_table6_zza_train_zscore/"
            "tourism_table6_plot_by_origin.csv"
        ),
    )
    parser.add_argument(
        "--evaluation-scale",
        choices=("original", "training_zscore"),
        default="original",
    )
    args = parser.parse_args()

    origins, table = load_plot_data(
        args.standard_origin_metrics_path,
        args.zza_origin_metrics_path,
        args.vara_origin_metrics_path,
        args.insample_origin_metrics_path,
        args.table6_path,
    )
    plot_table6_comparison(
        origins,
        table,
        output_pdf=args.output_pdf,
        output_png=args.output_png,
        plot_data_csv=args.plot_data_csv,
        evaluation_scale=args.evaluation_scale,
    )
    print(f"Saved figure PDF to: {args.output_pdf}")
    print(f"Saved figure PNG to: {args.output_png}")
    print(f"Saved plot data to: {args.plot_data_csv}")


if __name__ == "__main__":
    main()
