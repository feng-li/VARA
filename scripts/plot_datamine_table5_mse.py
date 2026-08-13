#!/usr/bin/env python3
"""Plot the five paired metric comparisons reported in risk-application Table 5."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/vara-matplotlib-cache")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METRICS = (
    REPOSITORY_ROOT
    / "outputs"
    / "datamine_vara"
    / "ew_log_risk_dense_attention_table10"
    / "paper"
    / "paper_table_metrics.csv"
)
DEFAULT_CONTRACT = DEFAULT_METRICS.with_name("paper_contract.json")
DEFAULT_OUTPUT = (
    REPOSITORY_ROOT / "docs" / "figures" / "datamine_table5_mse.pdf"
)
DEFAULT_R2_OUTPUT = (
    REPOSITORY_ROOT / "docs" / "figures" / "datamine_table5_r2_oos.pdf"
)
DEFAULT_RANK_OUTPUT = (
    REPOSITORY_ROOT / "docs" / "figures" / "datamine_table5_rankscore.pdf"
)

MODEL_PAIRS = (
    ("HAR ridge", "har_ridge", "har_vara_lambda"),
    ("Pooled\nLightGBM", "pooled_lightgbm", "pooled_lightgbm_vara_lambda"),
    (
        "RiskMetrics\nEWMA",
        "riskmetrics_ewma_validation_offset",
        "riskmetrics_ewma_vara_lambda",
    ),
    ("GARCH(1,1)", "garch11_offset", "garch_vara_lambda"),
    (
        "GARCH(1,1)–$t$",
        "garch11_student_t_offset",
        "garch11_student_t_vara_lambda",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="MSPE figure output (kept for backward compatibility).",
    )
    parser.add_argument("--r2-output", type=Path, default=DEFAULT_R2_OUTPUT)
    parser.add_argument("--rank-output", type=Path, default=DEFAULT_RANK_OUTPUT)
    return parser.parse_args()


def load_table5_metrics(
    metrics_path: Path,
) -> tuple[list[str], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Return exact paired Table 5 metrics in manuscript order."""
    metrics = pd.read_csv(metrics_path)
    if "log_risk_mspe" not in metrics.columns and "log_risk_rmse" in metrics.columns:
        rmse = pd.to_numeric(metrics["log_risk_rmse"], errors="coerce")
        metrics["log_risk_mspe"] = np.square(rmse)
    required = {"model", "log_risk_mspe", "absolute_r2", "rank_score"}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError("Metrics file is missing columns: " + ", ".join(missing))
    if metrics["model"].duplicated().any():
        raise ValueError("Metrics file must contain one row per model.")

    indexed = metrics.set_index("model")
    required_models = {
        model
        for _, base_model, vara_model in MODEL_PAIRS
        for model in (base_model, vara_model)
    }
    missing_models = sorted(required_models.difference(indexed.index))
    if missing_models:
        raise ValueError("Metrics file is missing models: " + ", ".join(missing_models))

    def paired_column(column: str, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        base_values = scale * np.asarray(
            [float(indexed.loc[base, column]) for _, base, _ in MODEL_PAIRS]
        )
        vara_values = scale * np.asarray(
            [float(indexed.loc[vara, column]) for _, _, vara in MODEL_PAIRS]
        )
        if not np.isfinite(base_values).all() or not np.isfinite(vara_values).all():
            raise ValueError(f"Table 5 {column} values must be finite.")
        return base_values, vara_values

    base_mspe, vara_mspe = paired_column("log_risk_mspe")
    if np.any(base_mspe < 0.0) or np.any(vara_mspe < 0.0):
        raise ValueError("Table 5 MSPE values must be nonnegative.")
    base_r2, vara_r2 = paired_column("absolute_r2", scale=100.0)
    base_rank, vara_rank = paired_column("rank_score")

    rank_values = np.concatenate((base_rank, vara_rank))
    if np.any(rank_values < 0.0) or np.any(rank_values > 100.0):
        raise ValueError("Table 5 RankScore values must lie between 0 and 100.")

    labels = [label for label, _, _ in MODEL_PAIRS]
    return labels, {
        "mse": (base_mspe, vara_mspe),
        "r2": (base_r2, vara_r2),
        "rank": (base_rank, vara_rank),
    }


def _table5_forecast_paths(contract: dict[str, object]) -> dict[str, Path]:
    """Resolve the ten forecast artifacts used to construct Table 5."""
    required_locations = {
        "source_output",
        "har_vara_output",
        "pooled_lightgbm_vara_output",
        "riskmetrics_ewma_vara_output",
        "garch_vara_output",
        "student_t_garch_output",
        "student_t_garch_vara_output",
    }
    missing = sorted(required_locations.difference(contract))
    if missing:
        raise ValueError("Paper contract is missing paths: " + ", ".join(missing))

    source = Path(str(contract["source_output"]))
    har = Path(str(contract["har_vara_output"])) / "forecasts"
    lightgbm = Path(str(contract["pooled_lightgbm_vara_output"])) / "forecasts"
    ewma = Path(str(contract["riskmetrics_ewma_vara_output"])) / "forecasts"
    garch = Path(str(contract["garch_vara_output"])) / "forecasts"
    garch_t = Path(str(contract["student_t_garch_output"])) / "forecasts"
    garch_t_vara = (
        Path(str(contract["student_t_garch_vara_output"])) / "forecasts"
    )
    paths = {
        "har_ridge": source / "baseline" / "baseline_forecast.npy",
        "har_vara_lambda": har / "vara_final_forecast.npy",
        "pooled_lightgbm": lightgbm / "pooled_lightgbm_forecast.npy",
        "pooled_lightgbm_vara_lambda": (
            lightgbm / "pooled_lightgbm_vara_lambda_forecast.npy"
        ),
        "riskmetrics_ewma_validation_offset": (
            ewma / "riskmetrics_ewma_validation_offset_forecast.npy"
        ),
        "riskmetrics_ewma_vara_lambda": (
            ewma / "riskmetrics_ewma_vara_lambda_forecast.npy"
        ),
        "garch11_offset": garch / "garch_validation_offset_forecast.npy",
        "garch_vara_lambda": garch / "garch_vara_lambda_forecast.npy",
        "garch11_student_t_offset": (
            garch_t / "garch11_t_validation_offset_forecast.npy"
        ),
        "garch11_student_t_vara_lambda": (
            garch_t_vara / "garch11_t_vara_lambda_forecast.npy"
        ),
    }
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError("Missing Table 5 forecasts: " + ", ".join(missing_files))
    return paths


def load_table5_monthly_r2(
    contract_path: Path,
    pooled_r2: dict[str, float],
) -> dict[str, np.ndarray]:
    """Decompose pooled Table 5 R-squared into its 120 monthly values.

    Every month uses the same pooled test-cell response mean as Table 5. The
    Table 5 value is therefore the monthly R-squared average weighted by each
    month's total-sum-of-squares contribution.
    """
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("Paper contract must contain a JSON object.")
    forecast_paths = _table5_forecast_paths(contract)
    if set(forecast_paths) != set(pooled_r2):
        raise ValueError("R-squared inputs do not match the ten Table 5 methods.")

    source = Path(str(contract["source_output"]))
    target = np.load(source / "baseline" / "target.npy", mmap_mode="r")
    dates = pd.DatetimeIndex(np.load(source / "processed" / "dates.npy"))
    split = contract.get("split")
    if not isinstance(split, dict):
        raise ValueError("Paper contract has no split definition.")
    test_start = pd.Timestamp(str(split["test_start"]))
    test_end = pd.Timestamp(str(split["test_end"]))
    test_indices = np.flatnonzero((dates >= test_start) & (dates <= test_end))
    if len(test_indices) != 120:
        raise ValueError(f"Expected 120 Table 5 test months; found {len(test_indices)}.")

    expected_shape = target.shape
    actual = np.asarray(target[test_indices], dtype=float)
    matched = np.isfinite(actual)
    for path in forecast_paths.values():
        forecast = np.load(path, mmap_mode="r")
        if forecast.shape != expected_shape:
            raise ValueError(f"Forecast shape mismatch for {path}.")
        matched &= np.isfinite(np.asarray(forecast[test_indices], dtype=float))

    observed_cells = matched.sum(axis=1)
    if np.any(observed_cells == 0):
        raise ValueError("At least one Table 5 test month has no matched cells.")
    rank_contract = contract.get("rank_score")
    if isinstance(rank_contract, dict):
        expected_cells = rank_contract.get("common_matched_cells")
        if expected_cells is not None and int(observed_cells.sum()) != int(expected_cells):
            raise ValueError("Reconstructed Table 5 mask has the wrong cell count.")

    pooled_mean = float(np.mean(actual[matched]))
    monthly_tss = np.where(matched, np.square(actual - pooled_mean), 0.0).sum(axis=1)
    if not np.isfinite(monthly_tss).all() or np.any(monthly_tss <= 0.0):
        raise ValueError("Monthly Table 5 R-squared denominators must be positive.")

    monthly_r2: dict[str, np.ndarray] = {}
    total_tss = float(np.sum(monthly_tss))
    for model, path in forecast_paths.items():
        forecast = np.load(path, mmap_mode="r")
        forecast_test = np.asarray(forecast[test_indices], dtype=float)
        squared_error = np.square(forecast_test - actual)
        monthly_sse = np.where(matched, squared_error, 0.0).sum(axis=1)
        values = 100.0 * (1.0 - monthly_sse / monthly_tss)
        reconstructed_pooled = 100.0 * (1.0 - float(monthly_sse.sum()) / total_tss)
        if not np.isclose(reconstructed_pooled, pooled_r2[model], rtol=0.0, atol=5e-10):
            raise ValueError(
                f"Monthly R-squared values do not reproduce Table 5 for {model}."
            )
        monthly_r2[model] = values
    return monthly_r2


def plot_table5_metric(
    labels: list[str],
    base_values: np.ndarray,
    vara_values: np.ndarray,
    output_path: Path,
    *,
    ylabel: str,
    value_decimals: int,
    annotations: list[str],
    ylim_padding: float,
    bracket_offset: float,
    bracket_tick: float,
    text_offset: float,
    metadata_title: str,
) -> None:
    """Write a vector-PDF grouped bar chart with a truthful zero baseline."""
    positions = np.arange(len(labels), dtype=float)
    width = 0.34

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(8.4, 4.5), constrained_layout=True)
    base_bars = axis.bar(
        positions - width / 2,
        base_values,
        width,
        label="Base model",
        color="#4C78A8",
        edgecolor="#263238",
        linewidth=0.6,
    )
    vara_bars = axis.bar(
        positions + width / 2,
        vara_values,
        width,
        label=r"Base + $\lambda^\ast\mathrm{VARA}$",
        color="#F58518",
        edgecolor="#263238",
        linewidth=0.6,
        hatch="//",
    )

    maximum = float(max(np.max(base_values), np.max(vara_values)))
    axis.set_ylim(0.0, maximum + ylim_padding)
    axis.set_ylabel(ylabel)
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=2, frameon=False)

    value_format = f"{{:.{value_decimals}f}}"
    axis.bar_label(
        base_bars,
        labels=[value_format.format(value) for value in base_values],
        padding=2,
        fontsize=7.5,
    )
    axis.bar_label(
        vara_bars,
        labels=[value_format.format(value) for value in vara_values],
        padding=2,
        fontsize=7.5,
    )
    for position, base_value, vara_value, annotation in zip(
        positions,
        base_values,
        vara_values,
        annotations,
        strict=True,
    ):
        bracket_y = float(max(base_value, vara_value) + bracket_offset)
        axis.plot(
            [position - width / 2, position - width / 2, position + width / 2, position + width / 2],
            [bracket_y - bracket_tick, bracket_y, bracket_y, bracket_y - bracket_tick],
            color="#455A64",
            linewidth=0.7,
        )
        axis.text(
            position,
            bracket_y + text_offset,
            annotation,
            ha="center",
            va="bottom",
            color="#1B5E20",
            fontsize=8,
            fontweight="semibold",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".pdf":
        raise ValueError("The manuscript figure output must use the .pdf suffix.")
    figure.savefig(
        output_path,
        bbox_inches="tight",
        metadata={
            "Title": metadata_title,
            "Creator": "scripts/plot_datamine_table5_mse.py",
        },
    )
    plt.close(figure)


def plot_table5_mse(
    labels: list[str],
    base_mse: np.ndarray,
    vara_mse: np.ndarray,
    output_path: Path,
) -> None:
    """Write a vector-PDF grouped bar chart with a zero MSPE baseline."""
    reductions = 100.0 * (base_mse - vara_mse) / base_mse
    plot_table5_metric(
        labels,
        base_mse,
        vara_mse,
        output_path,
        ylabel="Log-risk MSPE",
        value_decimals=4,
        annotations=[f"-{reduction:.2f}%" for reduction in reductions],
        ylim_padding=0.34,
        bracket_offset=0.15,
        bracket_tick=0.025,
        text_offset=0.018,
        metadata_title="Accounting-Signal Risk Table 5 MSPE comparison",
    )


def plot_table5_r2(
    labels: list[str],
    base_monthly_r2: list[np.ndarray],
    vara_monthly_r2: list[np.ndarray],
    base_pooled_r2: np.ndarray,
    vara_pooled_r2: np.ndarray,
    output_path: Path,
) -> None:
    """Write paired monthly R-squared boxplots with pooled Table 5 markers."""
    positions = np.arange(len(labels), dtype=float)
    offset = 0.19
    width = 0.31
    base_positions = positions - offset
    vara_positions = positions + offset
    gains = vara_pooled_r2 - base_pooled_r2

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(8.4, 4.5), constrained_layout=True)

    common_boxplot = {
        "orientation": "vertical",
        "widths": width,
        "patch_artist": True,
        "showmeans": False,
        "showfliers": False,
        "manage_ticks": False,
        "medianprops": {"color": "#202020", "linewidth": 1.35},
        "whiskerprops": {"color": "#666666", "linewidth": 0.9},
        "capprops": {"color": "#666666", "linewidth": 0.9},
    }
    base_boxes = axis.boxplot(
        base_monthly_r2,
        positions=base_positions,
        **common_boxplot,
    )
    vara_boxes = axis.boxplot(
        vara_monthly_r2,
        positions=vara_positions,
        **common_boxplot,
    )
    for patch in base_boxes["boxes"]:
        patch.set_facecolor("#4C78A8")
        patch.set_edgecolor("#263238")
        patch.set_linewidth(0.7)
        patch.set_alpha(0.92)
    for patch in vara_boxes["boxes"]:
        patch.set_facecolor("#F58518")
        patch.set_edgecolor("#263238")
        patch.set_linewidth(0.7)
        patch.set_hatch("//")
        patch.set_alpha(0.92)

    axis.scatter(
        base_positions,
        base_pooled_r2,
        marker="D",
        s=34,
        facecolor="white",
        edgecolor="#111111",
        linewidth=0.9,
        zorder=5,
    )
    axis.scatter(
        vara_positions,
        vara_pooled_r2,
        marker="D",
        s=34,
        facecolor="white",
        edgecolor="#111111",
        linewidth=0.9,
        zorder=5,
    )

    def whisker_limits(values: np.ndarray) -> tuple[float, float]:
        lower_quartile, upper_quartile = np.percentile(values, [25.0, 75.0])
        interquartile_range = upper_quartile - lower_quartile
        inliers = values[
            (values >= lower_quartile - 1.5 * interquartile_range)
            & (values <= upper_quartile + 1.5 * interquartile_range)
        ]
        return float(np.min(inliers)), float(np.max(inliers))

    base_whiskers = [whisker_limits(values) for values in base_monthly_r2]
    vara_whiskers = [whisker_limits(values) for values in vara_monthly_r2]
    visible_min = min(
        *(lower for lower, _ in base_whiskers),
        *(lower for lower, _ in vara_whiskers),
        float(np.min(base_pooled_r2)),
        float(np.min(vara_pooled_r2)),
    )
    visible_max = max(
        *(upper for _, upper in base_whiskers),
        *(upper for _, upper in vara_whiskers),
        float(np.max(base_pooled_r2)),
        float(np.max(vara_pooled_r2)),
    )
    data_span = max(visible_max - visible_min, 1.0)
    bracket_offset = 0.045 * data_span
    bracket_tick = 0.012 * data_span
    text_offset = 0.008 * data_span
    bracket_heights: list[float] = []
    for family_index, gain in enumerate(gains):
        pair_maximum = max(
            base_whiskers[family_index][1],
            vara_whiskers[family_index][1],
            float(base_pooled_r2[family_index]),
            float(vara_pooled_r2[family_index]),
        )
        bracket_y = pair_maximum + bracket_offset
        bracket_heights.append(bracket_y)
        axis.plot(
            [base_positions[family_index], base_positions[family_index],
             vara_positions[family_index], vara_positions[family_index]],
            [bracket_y - bracket_tick, bracket_y, bracket_y, bracket_y - bracket_tick],
            color="#455A64",
            linewidth=0.7,
        )
        axis.text(
            positions[family_index],
            bracket_y + text_offset,
            f"+{gain:.2f}%",
            ha="center",
            va="bottom",
            color="#1B5E20",
            fontsize=8,
            fontweight="semibold",
        )

    axis.set_ylim(
        visible_min - 0.07 * data_span,
        max(bracket_heights) + 0.07 * data_span,
    )
    axis.axhline(0.0, color="#555555", linewidth=0.8, alpha=0.75, zorder=0)
    axis.set_ylabel(r"Monthly $R^2_{\mathrm{oos}}$ (%)")
    axis.set_xticks(positions, labels)
    axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        handles=[
            Patch(
                facecolor="#4C78A8",
                edgecolor="#263238",
                label="Base model",
            ),
            Patch(
                facecolor="#F58518",
                edgecolor="#263238",
                hatch="//",
                label=r"Base + $\lambda^\ast\mathrm{VARA}$",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=2,
        frameon=False,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".pdf":
        raise ValueError("The manuscript figure output must use the .pdf suffix.")
    figure.savefig(
        output_path,
        bbox_inches="tight",
        metadata={
            "Title": "Accounting-Signal Risk Table 5 monthly R-squared boxplots",
            "Creator": "scripts/plot_datamine_table5_mse.py",
        },
    )
    plt.close(figure)


def plot_table5_rankscore(
    labels: list[str],
    base_rank: np.ndarray,
    vara_rank: np.ndarray,
    output_path: Path,
) -> None:
    """Write the Table 5 RankScore comparison."""
    gains = vara_rank - base_rank
    plot_table5_metric(
        labels,
        base_rank,
        vara_rank,
        output_path,
        ylabel="RankScore (%)",
        value_decimals=2,
        annotations=[f"+{gain:.2f}%" for gain in gains],
        ylim_padding=7.0,
        bracket_offset=2.8,
        bracket_tick=0.48,
        text_offset=0.32,
        metadata_title="Accounting-Signal Risk Table 5 RankScore comparison",
    )


def main() -> None:
    args = parse_args()
    labels, table_metrics = load_table5_metrics(args.metrics)
    base_mse, vara_mse = table_metrics["mse"]
    base_r2, vara_r2 = table_metrics["r2"]
    base_rank, vara_rank = table_metrics["rank"]
    pooled_r2 = {
        model: float(value)
        for pair_index, (_, base_model, vara_model) in enumerate(MODEL_PAIRS)
        for model, value in (
            (base_model, base_r2[pair_index]),
            (vara_model, vara_r2[pair_index]),
        )
    }
    monthly_r2 = load_table5_monthly_r2(args.contract, pooled_r2)
    base_monthly_r2 = [monthly_r2[base] for _, base, _ in MODEL_PAIRS]
    vara_monthly_r2 = [monthly_r2[vara] for _, _, vara in MODEL_PAIRS]
    plot_table5_mse(labels, base_mse, vara_mse, args.output)
    plot_table5_r2(
        labels,
        base_monthly_r2,
        vara_monthly_r2,
        base_r2,
        vara_r2,
        args.r2_output,
    )
    plot_table5_rankscore(labels, base_rank, vara_rank, args.rank_output)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.r2_output}")
    print(f"Wrote {args.rank_output}")


if __name__ == "__main__":
    main()
