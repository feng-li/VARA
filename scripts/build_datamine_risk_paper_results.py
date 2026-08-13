#!/usr/bin/env python3
"""Rebuild the Accounting-Signal Risk paper comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .datamine_paper_eta import (
        APPLICATION_CONTRACT_SCHEMA,
        ETA_PARAMETERIZATION,
        file_fingerprints,
    )
else:
    from datamine_paper_eta import (  # type: ignore[no-redef]
        APPLICATION_CONTRACT_SCHEMA,
        ETA_PARAMETERIZATION,
        file_fingerprints,
    )

from vara.datamine import evaluate_risk_forecasts, load_panel, relative_panel_split, write_json
from vara.forecast_metrics import add_origin_win_rank_score
from vara.forecast_significance import diebold_mariano_test


METHOD_FILES = {
    "garch_vara_lambda": ("garch", "garch_vara_lambda_forecast.npy"),
    "har_vara_lambda": ("har", "vara_final_forecast.npy"),
    "garch11_offset": ("garch", "garch_validation_offset_forecast.npy"),
    "har_ridge": ("source", "baseline_forecast.npy"),
    "garch_vara_unit": ("garch", "garch_vara_unit_forecast.npy"),
    "har_vara_unit": ("har", "vara_unit_forecast.npy"),
    "ar1_ridge": ("source", "ar1_forecast.npy"),
}
STUDENT_T_GARCH_METHOD = "garch11_student_t_offset"
STUDENT_T_GARCH_VARA_METHOD = "garch11_student_t_vara_lambda"
RISKMETRICS_EWMA_METHOD = "riskmetrics_ewma_validation_offset"
RISKMETRICS_EWMA_VARA_METHOD = "riskmetrics_ewma_vara_lambda"
POOLED_LIGHTGBM_METHOD = "pooled_lightgbm"
POOLED_LIGHTGBM_VARA_METHOD = "pooled_lightgbm_vara_lambda"
TRAINING_ZSCORE_SCALE_FLOOR = 1e-12
RESPONSE_STANDARDIZATION_CHOICES = ("median_mad", "training_zscore")
LEGACY_RESPONSE_STANDARDIZATION = "median_mad"
LIGHTGBM_RESPONSE_NORMALIZATION_CHOICES = ("none", "training_zscore")
LEGACY_LIGHTGBM_RESPONSE_NORMALIZATION = "none"
TABLE_METHODS = (
    "har_ridge",
    "har_vara_lambda",
    POOLED_LIGHTGBM_METHOD,
    POOLED_LIGHTGBM_VARA_METHOD,
    RISKMETRICS_EWMA_METHOD,
    RISKMETRICS_EWMA_VARA_METHOD,
    "garch11_offset",
    "garch_vara_lambda",
    STUDENT_T_GARCH_METHOD,
    STUDENT_T_GARCH_VARA_METHOD,
)
TABLE_METHOD_LABELS = {
    "har_ridge": "HAR ridge",
    "har_vara_lambda": "HAR ridge + lambda*VARA",
    POOLED_LIGHTGBM_METHOD: "Pooled LightGBM",
    POOLED_LIGHTGBM_VARA_METHOD: "Pooled LightGBM + lambda*VARA",
    RISKMETRICS_EWMA_METHOD: "RiskMetrics-style EWMA",
    RISKMETRICS_EWMA_VARA_METHOD: "RiskMetrics-style EWMA + lambda*VARA",
    "garch11_offset": "GARCH(1,1)",
    "garch_vara_lambda": "GARCH(1,1) + lambda*VARA",
    STUDENT_T_GARCH_METHOD: "GARCH(1,1)-t",
    STUDENT_T_GARCH_VARA_METHOD: "GARCH(1,1)-t + lambda*VARA",
}
TABLE_METHOD_LATEX_LABELS = {
    **TABLE_METHOD_LABELS,
    "har_vara_lambda": r"HAR ridge + $\lambda^\ast\mathrm{VARA}$",
    POOLED_LIGHTGBM_VARA_METHOD: (
        r"Pooled LightGBM + $\lambda^\ast\mathrm{VARA}$"
    ),
    RISKMETRICS_EWMA_VARA_METHOD: (
        r"RiskMetrics-style EWMA + $\lambda^\ast\mathrm{VARA}$"
    ),
    "garch_vara_lambda": r"GARCH(1,1) + $\lambda^\ast\mathrm{VARA}$",
    STUDENT_T_GARCH_METHOD: r"GARCH(1,1)--$t$",
    STUDENT_T_GARCH_VARA_METHOD: (
        r"GARCH(1,1)--$t$ + $\lambda^\ast\mathrm{VARA}$"
    ),
}
FULL_DENSE_ATTENTION_SUPPORT = "full_dense_off_diagonal_scores"


def render_risk_table(
    metrics: pd.DataFrame,
    *,
    attention_support: str,
) -> tuple[str, str]:
    """Render the original-scale ten-method comparison as Markdown and LaTeX.

    The displayed order and within-family emphasis match Table 5: lower MSE
    and larger absolute R2 and RankScore are preferred. Exact ties are bolded
    on both rows.
    """
    source_columns = (
        "log_risk_rmse",
        "absolute_r2",
        "rank_score",
    )
    metric_columns = ("log_risk_mse", "absolute_r2", "rank_score")
    required = {"model", *source_columns}
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(
            "Risk table rendering is missing columns: " + ", ".join(missing)
        )
    if metrics["model"].duplicated().any():
        raise ValueError("Risk table rendering requires one row per model.")
    indexed = metrics.set_index("model")
    if set(indexed.index) != set(TABLE_METHODS):
        raise ValueError("Risk table rendering has the wrong method set.")
    source_values = indexed.loc[list(TABLE_METHODS), list(source_columns)].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = source_values.loc[:, ["absolute_r2", "rank_score"]].copy()
    values.insert(0, "log_risk_mse", np.square(source_values["log_risk_rmse"]))
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("Risk table rendering requires finite metric values.")

    dense_attention = str(attention_support) == FULL_DENSE_ATTENTION_SUPPORT
    if dense_attention:
        markdown_title = "Parallel full-dense-attention Table 5 comparison"
        caption = (
            "Parallel full-dense-attention comparison for the Accounting-Signal "
            "Risk application. Each family reports its base forecast followed by "
            "the forecast with the refit-specific pooled LSE-weighted VARA "
            "correction. Every VARA correction uses all off-diagonal component "
            "scores. Within each pair, bold entries identify lower MSE or "
            "higher absolute R2 and RankScore."
        )
        label = "tab:datamine_forecasting_results_full_dense_attention"
    else:
        markdown_title = "Standalone Table 5 comparison"
        caption = (
            "Standalone Table 5 comparison for the Accounting-Signal Risk "
            "application. Each family reports its base forecast followed by the "
            "forecast with the refit-specific pooled LSE-weighted VARA correction. "
            "Within each pair, bold entries identify lower MSE or higher "
            "absolute R2 and RankScore."
        )
        label = "tab:datamine_forecasting_results_table10_comparison"

    minimize = {"log_risk_mse"}
    maximize = {"absolute_r2", "rank_score"}

    def preferred(model_position: int, column: str) -> bool:
        pair_start = 2 * (model_position // 2)
        pair = values.iloc[pair_start : pair_start + 2][column].to_numpy(dtype=float)
        current = float(values.iloc[model_position][column])
        if column in minimize:
            return current == float(np.min(pair))
        if column in maximize:
            return current == float(np.max(pair))
        raise ValueError(f"Unknown display preference for column {column!r}.")

    formats = {
        "log_risk_mse": ".4f",
        "absolute_r2": ".4f",
        "rank_score": ".2f",
    }

    def display_value(column: str, value: float) -> float:
        return 100.0 * value if column == "absolute_r2" else value

    markdown = [
        f"### {markdown_title}",
        "",
        caption,
        "",
        "| Method | Log-risk MSE | Absolute R2 (%) | RankScore (%) |",
        "|---|---:|---:|---:|",
    ]
    latex = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & Log-risk MSE & $R^2_{\mathrm{abs}}$ (\%) & RankScore (\%) \\",
        r"\midrule",
    ]
    for position, model in enumerate(TABLE_METHODS):
        row = values.loc[model]
        markdown_cells: list[str] = []
        latex_cells: list[str] = []
        for column in metric_columns:
            value = display_value(column, float(row[column]))
            rendered = format(value, formats[column])
            if preferred(position, column):
                markdown_cells.append(f"**{rendered}**")
                latex_cells.append(rf"\textbf{{{rendered}}}")
            else:
                markdown_cells.append(rendered)
                latex_cells.append(rendered)
        markdown.append(
            f"| {TABLE_METHOD_LABELS[model]} | "
            + " | ".join(markdown_cells)
            + " |"
        )
        latex.append(
            f"{TABLE_METHOD_LATEX_LABELS[model]} & "
            + " & ".join(latex_cells)
            + r" \\"
        )
        if position % 2 == 1 and position != len(TABLE_METHODS) - 1:
            latex.append(r"\addlinespace")
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])
    return "\n".join(markdown) + "\n", "\n".join(latex) + "\n"


def write_risk_table(
    metrics: pd.DataFrame,
    *,
    output_dir: Path,
    attention_support: str,
) -> tuple[Path, Path]:
    """Write the standalone original-scale comparison beside its CSV."""
    markdown, latex = render_risk_table(
        metrics,
        attention_support=attention_support,
    )
    markdown_path = Path(output_dir) / "paper_table.md"
    latex_path = Path(output_dir) / "paper_table.tex"
    markdown_path.write_text(markdown, encoding="utf-8")
    latex_path.write_text(latex, encoding="utf-8")
    return markdown_path, latex_path


def _validate_selectable_contract_field(
    *,
    contract: dict[str, object],
    label: str,
    field: str,
    expected: str,
    choices: tuple[str, ...],
    legacy_default: str,
) -> None:
    """Validate a selectable field while accepting a missing legacy default."""
    if expected not in choices:
        raise ValueError(
            f"Expected {field}={expected!r} is invalid; choose from {choices!r}."
        )
    actual = contract[field] if field in contract else legacy_default
    if actual != expected:
        raise ValueError(
            f"{label} contract has {field}={actual!r}; expected {expected!r}."
        )


def evaluate_training_zscore_risk_table(
    *,
    target: np.ndarray,
    dates: pd.DatetimeIndex | np.ndarray,
    component_ids: np.ndarray,
    forecasts: dict[str, np.ndarray],
    training_indices: np.ndarray,
    test_indices: np.ndarray,
    scale_floor: float = TRAINING_ZSCORE_SCALE_FLOOR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate a common forecast panel using training-only component z-scores.

    The forecasts are not refit or transformed in storage.  For component
    ``k``, every test error is divided by the population standard deviation of
    the observed log-risk target in ``training_indices``.  The corresponding
    training mean is used only to define the transformed-level absolute R2;
    it cancels from all forecast errors.
    """
    target_array = np.asarray(target, dtype=float)
    if target_array.ndim != 2:
        raise ValueError("Training-z-score evaluation requires a two-dimensional target.")
    dates_index = pd.DatetimeIndex(pd.to_datetime(dates))
    component_array = np.asarray(component_ids, dtype=object)
    if len(dates_index) != target_array.shape[0]:
        raise ValueError("Training-z-score dates do not match the target rows.")
    if len(component_array) != target_array.shape[1]:
        raise ValueError("Training-z-score component IDs do not match the target columns.")
    if pd.Index(component_array).duplicated().any():
        raise ValueError("Training-z-score component IDs must be unique.")
    if not np.isfinite(scale_floor) or scale_floor <= 0.0:
        raise ValueError("Training-z-score scale_floor must be positive and finite.")
    if not forecasts:
        raise ValueError("Training-z-score evaluation requires at least two forecasts.")

    training_indices = np.asarray(training_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    for label, indices in (("training", training_indices), ("test", test_indices)):
        if indices.ndim != 1 or len(indices) == 0:
            raise ValueError(f"Training-z-score {label} indices must be nonempty and one-dimensional.")
        if np.any(np.diff(indices) != 1):
            raise ValueError(f"Training-z-score {label} indices must be contiguous and increasing.")
        if indices[0] < 0 or indices[-1] >= target_array.shape[0]:
            raise ValueError(f"Training-z-score {label} indices are outside the target panel.")
    if training_indices[-1] >= test_indices[0]:
        raise ValueError("Training-z-score statistics must end before the test period.")

    for name, forecast in forecasts.items():
        if np.asarray(forecast).shape != target_array.shape:
            raise ValueError(
                f"Training-z-score forecast {name!r} does not match the target shape."
            )

    training = target_array[training_indices]
    finite_training = np.isfinite(training)
    training_counts = finite_training.sum(axis=0)
    if np.any(training_counts < 2):
        invalid = int(np.count_nonzero(training_counts < 2))
        raise ValueError(
            "Training-z-score evaluation requires at least two finite training "
            f"observations per component; {invalid} components fail."
        )
    training_sums = np.where(finite_training, training, 0.0).sum(axis=0)
    training_means = training_sums / training_counts
    centered_training = np.where(
        finite_training,
        training - training_means[None, :],
        0.0,
    )
    training_variances = np.square(centered_training).sum(axis=0) / training_counts
    training_scales = np.sqrt(training_variances)
    invalid_scale = ~np.isfinite(training_scales) | (training_scales <= scale_floor)
    if invalid_scale.any():
        raise ValueError(
            "Training-z-score evaluation found "
            f"{int(invalid_scale.sum())} nonfinite or near-zero component scales."
        )

    actual = target_array[test_indices]
    matched_cells = np.isfinite(actual)
    for forecast in forecasts.values():
        matched_cells &= np.isfinite(np.asarray(forecast, dtype=float)[test_indices])
    monthly_cells = matched_cells.sum(axis=1)
    if not matched_cells.any() or np.any(monthly_cells == 0):
        raise ValueError(
            "Training-z-score evaluation requires matched cells for every test month."
        )

    actual_z = (actual - training_means[None, :]) / training_scales[None, :]
    observed_mean = float(np.mean(actual_z[matched_cells]))
    total_sum_of_squares = float(
        np.sum(np.square(actual_z[matched_cells] - observed_mean))
    )
    if not np.isfinite(total_sum_of_squares) or total_sum_of_squares <= 0.0:
        raise ValueError("Training-z-score absolute R2 has a nonpositive denominator.")

    summary_rows: list[dict[str, float | int | str]] = []
    monthly_rows: list[pd.DataFrame] = []
    evaluation_dates = dates_index[test_indices]
    for name, forecast in forecasts.items():
        forecast_array = np.asarray(forecast, dtype=float)[test_indices]
        error_z = (forecast_array - actual) / training_scales[None, :]
        squared_error_z = np.square(error_z)
        absolute_error_z = np.abs(error_z)
        sse_z = float(np.sum(squared_error_z[matched_cells]))
        summary_rows.append({
            "model": name,
            "target_type": "training_zscore_log_squared_return",
            "evaluation_scale": "fixed_pre_validation_training_zscore",
            "test_cells": int(matched_cells.sum()),
            "z_log_risk_mse": float(np.mean(squared_error_z[matched_cells])),
            "z_log_risk_rmse": float(np.sqrt(np.mean(squared_error_z[matched_cells]))),
            "z_log_risk_mae": float(np.mean(absolute_error_z[matched_cells])),
            "z_log_risk_bias": float(np.mean(error_z[matched_cells])),
            "z_absolute_r2": 1.0 - sse_z / total_sum_of_squares,
        })
        monthly_rows.append(pd.DataFrame({
            "date": evaluation_dates,
            "model": name,
            "observed_cells": monthly_cells,
            "z_squared_error": np.divide(
                np.where(matched_cells, squared_error_z, 0.0).sum(axis=1),
                monthly_cells,
            ),
            "z_absolute_error": np.divide(
                np.where(matched_cells, absolute_error_z, 0.0).sum(axis=1),
                monthly_cells,
            ),
            "z_log_risk_bias": np.divide(
                np.where(matched_cells, error_z, 0.0).sum(axis=1),
                monthly_cells,
            ),
        }))

    monthly = pd.concat(monthly_rows, ignore_index=True)
    metrics, rankscore_audit = add_origin_win_rank_score(
        pd.DataFrame(summary_rows),
        monthly,
        error_column="z_absolute_error",
    )
    monthly = monthly.merge(
        rankscore_audit.loc[:, [
            "date",
            "model",
            "absolute_error_wins",
            "rank_score_s",
        ]],
        on=["date", "model"],
        how="left",
        validate="one_to_one",
    )
    scalers = pd.DataFrame({
        "component_id": component_array,
        "training_observations": training_counts,
        "training_mean": training_means,
        "training_variance": training_variances,
        "training_scale": training_scales,
        "training_start": dates_index[training_indices[0]],
        "training_end": dates_index[training_indices[-1]],
        "training_window_months": len(training_indices),
        "scale_ddof": 0,
    })
    return metrics, monthly, rankscore_audit, scalers


def render_training_zscore_table(metrics: pd.DataFrame) -> tuple[str, str]:
    """Render the optional z-score table as Markdown and standalone LaTeX."""
    required = {
        "model",
        "z_log_risk_rmse",
        "z_absolute_r2",
        "rank_score",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(
            "Training-z-score table rendering is missing columns: "
            + ", ".join(missing)
        )
    indexed = metrics.set_index("model")
    if set(indexed.index) != set(TABLE_METHODS):
        raise ValueError("Training-z-score table rendering has the wrong method set.")

    markdown = [
        "| Method | z-RMSE | z-absolute R2 (%) | RankScore (%) |",
        "|---|---:|---:|---:|",
    ]
    latex = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\small",
        (
            r"\caption{Training-z-score-scaled accuracy for the Accounting-Signal Risk "
            r"application. Forecasts are identical to Table~\ref{tab:datamine_forecasting_results}; "
            r"only their evaluation scale changes. For each series, the mean and population "
            r"standard deviation are estimated from the fixed January 1988--December 2007 "
            r"training window. RMSE uses the resulting standardized log-risk errors; "
            r"$R^2_{\mathrm{abs}}$ is calculated after applying the same transformation "
            r"to the observations, and RankScore uses monthly strict wins based on "
            r"cross-sectional mean standardized absolute error.}"
        ),
        r"\label{tab:datamine_forecasting_results_training_zscore}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Method & z-RMSE & $z$-$R^2_{\mathrm{abs}}$ (\%) & RankScore (\%) \\",
        r"\midrule",
    ]
    for model in TABLE_METHODS:
        row = indexed.loc[model]
        values = (
            float(row["z_log_risk_rmse"]),
            100.0 * float(row["z_absolute_r2"]),
            float(row["rank_score"]),
        )
        markdown.append(
            f"| {TABLE_METHOD_LABELS[model]} | {values[0]:.6f} | "
            f"{values[1]:.4f} | {values[2]:.2f} |"
        )
        latex.append(
            f"{TABLE_METHOD_LATEX_LABELS[model]} & {values[0]:.6f} & "
            f"{values[1]:.4f} & {values[2]:.2f} \\\\"
        )
    latex.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])
    return "\n".join(markdown) + "\n", "\n".join(latex) + "\n"


DM_COMPARISONS = (
    (
        "A: primary versus alternative bases",
        "har_vara_lambda",
        "ar1_ridge",
    ),
    (
        "A: primary versus alternative bases",
        "har_vara_lambda",
        POOLED_LIGHTGBM_METHOD,
    ),
    (
        "A: primary versus alternative bases",
        "har_vara_lambda",
        RISKMETRICS_EWMA_METHOD,
    ),
    (
        "A: primary versus alternative bases",
        "har_vara_lambda",
        "garch11_offset",
    ),
    (
        "A: primary versus alternative bases",
        "har_vara_lambda",
        STUDENT_T_GARCH_METHOD,
    ),
    (
        "B: incremental VARA within family",
        "har_vara_lambda",
        "har_ridge",
    ),
    (
        "B: incremental VARA within family",
        POOLED_LIGHTGBM_VARA_METHOD,
        POOLED_LIGHTGBM_METHOD,
    ),
    (
        "B: incremental VARA within family",
        RISKMETRICS_EWMA_VARA_METHOD,
        RISKMETRICS_EWMA_METHOD,
    ),
    (
        "B: incremental VARA within family",
        "garch_vara_lambda",
        "garch11_offset",
    ),
    (
        "B: incremental VARA within family",
        STUDENT_T_GARCH_VARA_METHOD,
        STUDENT_T_GARCH_METHOD,
    ),
)


def validate_paper_input_contracts(
    *,
    har_contract: dict[str, object],
    garch_contract: dict[str, object],
    source_output: str,
    har_vara_output: str,
    component_count: int,
    split: dict[str, str],
    response_standardization: str = LEGACY_RESPONSE_STANDARDIZATION,
) -> None:
    """Prevent legacy or mismatched application artifacts from entering the table."""
    common = {
        "application_contract_schema": APPLICATION_CONTRACT_SCHEMA,
        "source_output": source_output,
        "component_count": int(component_count),
        "forecast_scope": "validation_and_test",
        "split": split,
    }
    for label, contract in (("HAR", har_contract), ("GARCH", garch_contract)):
        for field, expected in common.items():
            if contract.get(field) != expected:
                raise ValueError(
                    f"{label} application contract has {field}={contract.get(field)!r}; "
                    f"expected {expected!r}."
                )
        eta = contract.get("eta_estimation")
        if not isinstance(eta, dict) or eta.get("parameterization") != ETA_PARAMETERIZATION:
            raise ValueError(f"{label} application contract is not paper-eta scaled.")
        _validate_selectable_contract_field(
            contract=contract,
            label=f"{label} application",
            field="response_standardization",
            expected=response_standardization,
            choices=RESPONSE_STANDARDIZATION_CHOICES,
            legacy_default=LEGACY_RESPONSE_STANDARDIZATION,
        )

    if har_contract.get("eta_estimation") != garch_contract.get("eta_estimation"):
        raise ValueError("HAR and GARCH application contracts use different eta specifications.")
    for field in ("lambda_window_months", "lambda_ridge"):
        if har_contract.get(field) != garch_contract.get(field):
            raise ValueError(f"HAR and GARCH application contracts use different {field}.")
    if garch_contract.get("har_vara_output") != har_vara_output:
        raise ValueError("GARCH metrics were not generated with the requested HAR comparator.")


def validate_student_t_vara_contract(
    *,
    contract: dict[str, object],
    source_output: str,
    har_vara_output: str,
    student_t_garch_output: str,
    component_count: int,
    split: dict[str, str],
    eta_estimation: object,
    lambda_window_months: object,
    lambda_ridge: object,
    response_standardization: str = LEGACY_RESPONSE_STANDARDIZATION,
) -> None:
    """Validate that Student-t GARCH residuals received their own VARA refit."""
    expected_fields = {
        "application_contract_schema": APPLICATION_CONTRACT_SCHEMA,
        "source_output": source_output,
        "har_vara_output": har_vara_output,
        "garch_variant": "student_t",
        "innovation_distribution": "variance-standardized Student-t",
        "upstream_garch_output": student_t_garch_output,
        "component_count": int(component_count),
        "forecast_scope": "validation_and_test",
        "split": split,
        "eta_estimation": eta_estimation,
        "lambda_window_months": lambda_window_months,
        "lambda_ridge": lambda_ridge,
    }
    for field, expected in expected_fields.items():
        if contract.get(field) != expected:
            raise ValueError(
                f"Student-t GARCH--VARA contract has "
                f"{field}={contract.get(field)!r}; expected {expected!r}."
            )
    _validate_selectable_contract_field(
        contract=contract,
        label="Student-t GARCH--VARA",
        field="response_standardization",
        expected=response_standardization,
        choices=RESPONSE_STANDARDIZATION_CHOICES,
        legacy_default=LEGACY_RESPONSE_STANDARDIZATION,
    )
    model_ids = contract.get("model_ids")
    artifacts = contract.get("forecast_artifacts")
    if not isinstance(model_ids, dict) or model_ids.get("lambda_vara") != (
        STUDENT_T_GARCH_VARA_METHOD
    ):
        raise ValueError("Student-t GARCH--VARA contract has the wrong lambda model ID.")
    if not isinstance(artifacts, dict) or artifacts.get("lambda_vara") != (
        "garch11_t_vara_lambda_forecast.npy"
    ):
        raise ValueError("Student-t GARCH--VARA contract has the wrong forecast artifact.")


def validate_riskmetrics_ewma_vara_contract(
    *,
    contract: dict[str, object],
    source_output: str,
    har_vara_output: str,
    component_count: int,
    split: dict[str, str],
    eta_estimation: object,
    lambda_window_months: object,
    lambda_ridge: object,
    variance_floor: float,
    decay: float = 0.97,
    response_standardization: str = LEGACY_RESPONSE_STANDARDIZATION,
) -> None:
    """Validate the dedicated fixed-decay EWMA residual VARA refit."""
    expected_fields = {
        "application_contract_schema": APPLICATION_CONTRACT_SCHEMA,
        "source_output": source_output,
        "har_vara_output": har_vara_output,
        "base_variant": "riskmetrics_ewma",
        # The shared runner retains this legacy field for reuse compatibility.
        "garch_variant": "riskmetrics_ewma",
        "innovation_distribution": "not parametrically specified",
        "upstream_garch_output": None,
        "component_count": int(component_count),
        "forecast_scope": "validation_and_test",
        "split": split,
        "eta_estimation": eta_estimation,
        "lambda_window_months": lambda_window_months,
        "lambda_ridge": lambda_ridge,
    }
    for field, expected in expected_fields.items():
        if contract.get(field) != expected:
            raise ValueError(
                "RiskMetrics EWMA--VARA contract has "
                f"{field}={contract.get(field)!r}; expected {expected!r}."
            )
    _validate_selectable_contract_field(
        contract=contract,
        label="RiskMetrics EWMA--VARA",
        field="response_standardization",
        expected=response_standardization,
        choices=RESPONSE_STANDARDIZATION_CHOICES,
        legacy_default=LEGACY_RESPONSE_STANDARDIZATION,
    )

    expected_ewma = {
        "decay": float(decay),
        "observation_frequency": "monthly",
        "conditional_mean": 0.0,
        "initialization": "first_finite_squared_return_per_series",
        "missing_return_update": "carry_previous_variance_state",
        "one_step_ahead": True,
        "decay_tuned": False,
        "variance_floor": float(variance_floor),
    }
    if contract.get("riskmetrics_ewma") != expected_ewma:
        raise ValueError(
            "RiskMetrics EWMA--VARA contract has the wrong fixed-decay EWMA specification."
        )

    expected_model_ids = {
        "raw": "riskmetrics_ewma_raw",
        "validation_offset": RISKMETRICS_EWMA_METHOD,
        "unit_vara": "riskmetrics_ewma_vara_unit",
        "lambda_vara": RISKMETRICS_EWMA_VARA_METHOD,
        "raw_unit_vara": "riskmetrics_ewma_raw_vara_unit",
    }
    model_ids = contract.get("model_ids")
    if not isinstance(model_ids, dict) or any(
        model_ids.get(key) != expected
        for key, expected in expected_model_ids.items()
    ):
        raise ValueError("RiskMetrics EWMA--VARA contract has the wrong model IDs.")

    expected_artifacts = {
        "raw": "riskmetrics_ewma_raw_forecast.npy",
        "conditional_variance": "riskmetrics_ewma_variance_forecast.npy",
        "validation_offset": "riskmetrics_ewma_validation_offset_forecast.npy",
        "unit_vara": "riskmetrics_ewma_vara_unit_forecast.npy",
        "lambda_vara": "riskmetrics_ewma_vara_lambda_forecast.npy",
        "raw_unit_vara_correction": (
            "riskmetrics_ewma_raw_vara_unit_correction.npy"
        ),
        "raw_unit_vara": "riskmetrics_ewma_raw_vara_unit_forecast.npy",
    }
    artifacts = contract.get("forecast_artifacts")
    if not isinstance(artifacts, dict) or any(
        artifacts.get(key) != expected
        for key, expected in expected_artifacts.items()
    ):
        raise ValueError(
            "RiskMetrics EWMA--VARA contract has the wrong forecast artifacts."
        )


def validate_pooled_lightgbm_vara_contract(
    *,
    contract: dict[str, object],
    source_output: str,
    har_vara_output: str,
    component_count: int,
    split: dict[str, str],
    eta_estimation: object,
    lambda_window_months: object,
    lambda_ridge: object,
    error_window_months: int = 240,
    refit_frequency_months: int = 12,
    training_window_months: int = 240,
    response_standardization: str = LEGACY_RESPONSE_STANDARDIZATION,
    lightgbm_response_normalization: str = LEGACY_LIGHTGBM_RESPONSE_NORMALIZATION,
) -> None:
    """Validate the pooled LightGBM base and its dedicated residual VARA fit."""
    expected_fields = {
        "application_contract_schema": APPLICATION_CONTRACT_SCHEMA,
        "source_output": source_output,
        "har_vara_output": har_vara_output,
        "base_variant": "pooled_lightgbm",
        "component_count": int(component_count),
        "forecast_scope": "validation_and_test",
        "split": split,
        "eta_estimation": eta_estimation,
        "error_window_months": int(error_window_months),
        "refit_frequency_months": int(refit_frequency_months),
        "lambda_window_months": lambda_window_months,
        "lambda_ridge": lambda_ridge,
        "target_month_excluded_from_base_and_vara_refits": True,
        "current_or_future_test_target_used_for_base_fit": False,
    }
    for field, expected in expected_fields.items():
        if contract.get(field) != expected:
            raise ValueError(
                "Pooled LightGBM--VARA contract has "
                f"{field}={contract.get(field)!r}; expected {expected!r}."
            )
    _validate_selectable_contract_field(
        contract=contract,
        label="Pooled LightGBM--VARA",
        field="response_standardization",
        expected=response_standardization,
        choices=RESPONSE_STANDARDIZATION_CHOICES,
        legacy_default=LEGACY_RESPONSE_STANDARDIZATION,
    )

    specification = contract.get("lightgbm")
    if not isinstance(specification, dict):
        raise ValueError("Pooled LightGBM--VARA contract has no LightGBM specification.")
    expected_lightgbm = {
        "target": "log(epsilon + squared return)",
        "formula_features_used": False,
        "component_identifiers_used": False,
        "features": ["lag_1", "mean_3", "mean_12"],
        "feature_windows": [1, 3, 12],
        "training_window_months": int(training_window_months),
        "refit_frequency_months": int(refit_frequency_months),
        "hyperparameter_selection": (
            "prespecified single configuration; no validation or test search"
        ),
        "one_step_ahead": True,
    }
    _validate_selectable_contract_field(
        contract=specification,
        label="Pooled LightGBM--VARA LightGBM specification",
        field="normalization",
        expected=lightgbm_response_normalization,
        choices=LIGHTGBM_RESPONSE_NORMALIZATION_CHOICES,
        legacy_default=LEGACY_LIGHTGBM_RESPONSE_NORMALIZATION,
    )
    for field, expected in expected_lightgbm.items():
        if specification.get(field) != expected:
            raise ValueError(
                "Pooled LightGBM--VARA contract has LightGBM "
                f"{field}={specification.get(field)!r}; expected {expected!r}."
            )
    if not isinstance(specification.get("version"), str):
        raise ValueError("Pooled LightGBM--VARA contract has no package version.")
    if not isinstance(specification.get("parameters"), dict):
        raise ValueError("Pooled LightGBM--VARA contract has no parameter record.")

    expected_model_ids = {
        "base": POOLED_LIGHTGBM_METHOD,
        "unit_vara": "pooled_lightgbm_vara_unit",
        "lambda_vara": POOLED_LIGHTGBM_VARA_METHOD,
    }
    model_ids = contract.get("model_ids")
    if not isinstance(model_ids, dict) or any(
        model_ids.get(key) != expected
        for key, expected in expected_model_ids.items()
    ):
        raise ValueError("Pooled LightGBM--VARA contract has the wrong model IDs.")
    expected_artifacts = {
        "base": "pooled_lightgbm_forecast.npy",
        "base_error": "pooled_lightgbm_error.npy",
        "unit_vara": "pooled_lightgbm_vara_unit_forecast.npy",
        "lambda_vara": "pooled_lightgbm_vara_lambda_forecast.npy",
        "vara_correction": "vara_correction.npy",
    }
    artifacts = contract.get("forecast_artifacts")
    if not isinstance(artifacts, dict) or any(
        artifacts.get(key) != expected
        for key, expected in expected_artifacts.items()
    ):
        raise ValueError(
            "Pooled LightGBM--VARA contract has the wrong forecast artifacts."
        )


def build_paper_dm_results(
    *,
    monthly: pd.DataFrame,
    metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Build the prespecified two-panel, ten-pair paper DM artifact."""
    required_losses = ("squared_error", "gaussian_return_crps")
    required_monthly_columns = {"date", "model", *required_losses}
    missing_columns = required_monthly_columns.difference(monthly.columns)
    if missing_columns:
        raise ValueError(
            "Monthly DM input is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    if not {"model", "log_risk_rmse"}.issubset(metrics.columns):
        raise ValueError("DM metric input must contain model and log_risk_rmse.")
    if metrics["model"].duplicated().any():
        raise ValueError("DM metric input must contain one row per model.")

    available_models = set(monthly["model"].astype(str))
    required_models = {
        model
        for _, focal, comparator in DM_COMPARISONS
        for model in (focal, comparator)
    }
    missing_models = sorted(required_models.difference(available_models))
    if missing_models:
        raise ValueError(
            "Monthly DM input is missing required models: "
            + ", ".join(missing_models)
        )
    metric_index = metrics.set_index("model")
    missing_metric_models = sorted(required_models.difference(metric_index.index))
    if missing_metric_models:
        raise ValueError(
            "DM metric input is missing required models: "
            + ", ".join(missing_metric_models)
        )

    rows: list[dict[str, float | int | str]] = []
    for panel, focal_name, comparator_name in DM_COMPARISONS:
        focal_rmse = float(metric_index.loc[focal_name, "log_risk_rmse"])
        comparator_rmse = float(
            metric_index.loc[comparator_name, "log_risk_rmse"]
        )
        row: dict[str, float | int | str] = {
            "panel": panel,
            "focal_model": focal_name,
            "comparator_model": comparator_name,
            "rmse_gain_percent": 100.0 * (1.0 - focal_rmse / comparator_rmse),
        }
        n_origins: int | None = None
        for loss in required_losses:
            focal = monthly.loc[
                monthly["model"].eq(focal_name), ["date", loss]
            ]
            comparator = monthly.loc[
                monthly["model"].eq(comparator_name), ["date", loss]
            ]
            if focal["date"].duplicated().any() or comparator["date"].duplicated().any():
                raise ValueError("DM input must contain one loss per model and month.")
            matched = focal.merge(
                comparator,
                on="date",
                suffixes=("_focal", "_comparator"),
                validate="one_to_one",
            ).dropna()
            test = diebold_mariano_test(
                matched[f"{loss}_focal"],
                matched[f"{loss}_comparator"],
                forecast_horizon=1,
            )
            current_n_origins = int(test["n_origins"])
            if n_origins is None:
                n_origins = current_n_origins
            elif current_n_origins != n_origins:
                raise ValueError(
                    "Squared-error and CRPS DM tests use different month counts."
                )
            prefix = "crps" if loss == "gaussian_return_crps" else loss
            # The paper defines the reported statistic and p-value as the raw
            # normal-reference, two-sided DM result. Retain the corrected result
            # as explicit audit columns rather than silently discarding it.
            row[f"{prefix}_mean_loss_difference"] = float(
                test["mean_loss_difference"]
            )
            row[f"{prefix}_standard_error"] = float(test["standard_error"])
            row[f"{prefix}_dm_statistic"] = float(test["dm_statistic_raw"])
            row[f"{prefix}_p_value_two_sided"] = float(
                test["p_value_raw_two_sided"]
            )
            row[f"{prefix}_dm_statistic_hln"] = float(test["dm_statistic"])
            row[f"{prefix}_p_value_two_sided_hln"] = float(
                test["p_value_two_sided"]
            )
        assert n_origins is not None
        row["n_origins"] = n_origins
        rows.append(row)
    return pd.DataFrame(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the paper-result CLI, including expected input-contract settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_output", type=Path)
    parser.add_argument("har_vara_output", type=Path)
    parser.add_argument("garch_vara_output", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--student-t-garch-output", type=Path, required=True)
    parser.add_argument("--student-t-garch-vara-output", type=Path, required=True)
    parser.add_argument(
        "--riskmetrics-ewma-vara-output",
        type=Path,
        required=True,
        help=(
            "Completed fixed-decay RiskMetrics-style EWMA output containing "
            "both the validation-offset base and its dedicated VARA refit."
        ),
    )
    parser.add_argument(
        "--pooled-lightgbm-vara-output",
        type=Path,
        required=True,
        help=(
            "Completed pooled LightGBM output containing the genuine rolling "
            "base forecast and its dedicated rolling-LSE VARA correction."
        ),
    )
    parser.add_argument(
        "--response-standardization",
        choices=RESPONSE_STANDARDIZATION_CHOICES,
        default=LEGACY_RESPONSE_STANDARDIZATION,
        help=(
            "Expected rolling response standardization shared by the HAR, "
            "GARCH, Student-t GARCH, EWMA, and LightGBM VARA inputs."
        ),
    )
    parser.add_argument(
        "--lightgbm-response-normalization",
        choices=LIGHTGBM_RESPONSE_NORMALIZATION_CHOICES,
        default=LEGACY_LIGHTGBM_RESPONSE_NORMALIZATION,
        help="Expected response normalization used to fit the pooled LightGBM base.",
    )
    parser.add_argument(
        "--write-training-zscore-table",
        action="store_true",
        help=(
            "Also write a separate evaluation-only table whose log-risk errors "
            "are divided by each component's population standard deviation in "
            "the fixed pre-validation training window."
        ),
    )
    return parser


def main() -> None:
    args = build_argument_parser().parse_args()

    source_contract = json.loads(
        (args.source_output / "forecasts" / "locked_test_config.json").read_text(
            encoding="utf-8"
        )
    )
    panel = load_panel(args.source_output / "processed")
    split = relative_panel_split(
        panel.dates,
        validation_months=int(source_contract["validation_months"]),
        test_months=int(source_contract["test_months"]),
    )
    split_contract = {
        "universe_selection_end": panel.dates[
            int(split.validation_indices[0]) - 1
        ].date().isoformat(),
        "validation_start": panel.dates[int(split.validation_indices[0])].date().isoformat(),
        "validation_end": panel.dates[int(split.validation_indices[-1])].date().isoformat(),
        "test_start": panel.dates[int(split.test_indices[0])].date().isoformat(),
        "test_end": panel.dates[int(split.test_indices[-1])].date().isoformat(),
    }
    har_contract_path = args.har_vara_output / "forecasts" / "contract.json"
    garch_contract_path = args.garch_vara_output / "forecasts" / "contract.json"
    har_contract = json.loads(har_contract_path.read_text(encoding="utf-8"))
    garch_contract = json.loads(garch_contract_path.read_text(encoding="utf-8"))
    validate_paper_input_contracts(
        har_contract=har_contract,
        garch_contract=garch_contract,
        source_output=args.source_output.resolve().as_posix(),
        har_vara_output=args.har_vara_output.resolve().as_posix(),
        component_count=len(panel.component_ids),
        split=split_contract,
        response_standardization=args.response_standardization,
    )
    riskmetrics_ewma_contract_path = (
        args.riskmetrics_ewma_vara_output / "forecasts" / "contract.json"
    )
    riskmetrics_ewma_contract = json.loads(
        riskmetrics_ewma_contract_path.read_text(encoding="utf-8")
    )
    validate_riskmetrics_ewma_vara_contract(
        contract=riskmetrics_ewma_contract,
        source_output=args.source_output.resolve().as_posix(),
        har_vara_output=args.har_vara_output.resolve().as_posix(),
        component_count=len(panel.component_ids),
        split=split_contract,
        eta_estimation=har_contract["eta_estimation"],
        lambda_window_months=har_contract["lambda_window_months"],
        lambda_ridge=har_contract["lambda_ridge"],
        variance_floor=float(source_contract["variance_floor"]),
        response_standardization=args.response_standardization,
    )
    pooled_lightgbm_contract_path = (
        args.pooled_lightgbm_vara_output / "forecasts" / "contract.json"
    )
    pooled_lightgbm_contract = json.loads(
        pooled_lightgbm_contract_path.read_text(encoding="utf-8")
    )
    validate_pooled_lightgbm_vara_contract(
        contract=pooled_lightgbm_contract,
        source_output=args.source_output.resolve().as_posix(),
        har_vara_output=args.har_vara_output.resolve().as_posix(),
        component_count=len(panel.component_ids),
        split=split_contract,
        eta_estimation=har_contract["eta_estimation"],
        lambda_window_months=har_contract["lambda_window_months"],
        lambda_ridge=har_contract["lambda_ridge"],
        error_window_months=int(har_contract["error_window_months"]),
        refit_frequency_months=int(har_contract["refit_frequency_months"]),
        response_standardization=args.response_standardization,
        lightgbm_response_normalization=args.lightgbm_response_normalization,
    )
    student_t_contract_path: Path | None = None
    student_t_contract: dict[str, object] | None = None
    if args.student_t_garch_output is not None:
        student_t_contract_path = (
            args.student_t_garch_output / "forecasts" / "contract.json"
        )
        student_t_contract = json.loads(
            student_t_contract_path.read_text(encoding="utf-8")
        )
        expected_student_t_fields = {
            "source_output": args.source_output.resolve().as_posix(),
            "component_count": len(panel.component_ids),
            "innovation_distribution": "variance-standardized Student-t",
            "parameters_frozen_during_validation_and_test": True,
            "split": split_contract,
        }
        for field, expected in expected_student_t_fields.items():
            if student_t_contract.get(field) != expected:
                raise ValueError(
                    f"Student-t GARCH contract has "
                    f"{field}={student_t_contract.get(field)!r}; expected {expected!r}."
                )
    student_t_vara_contract_path: Path | None = None
    student_t_vara_contract: dict[str, object] | None = None
    if args.student_t_garch_vara_output is not None:
        if args.student_t_garch_output is None:
            raise ValueError(
                "--student-t-garch-vara-output requires --student-t-garch-output."
            )
        student_t_vara_contract_path = (
            args.student_t_garch_vara_output / "forecasts" / "contract.json"
        )
        student_t_vara_contract = json.loads(
            student_t_vara_contract_path.read_text(encoding="utf-8")
        )
        validate_student_t_vara_contract(
            contract=student_t_vara_contract,
            source_output=args.source_output.resolve().as_posix(),
            har_vara_output=args.har_vara_output.resolve().as_posix(),
            student_t_garch_output=args.student_t_garch_output.resolve().as_posix(),
            component_count=len(panel.component_ids),
            split=split_contract,
            eta_estimation=har_contract["eta_estimation"],
            lambda_window_months=har_contract["lambda_window_months"],
            lambda_ridge=har_contract["lambda_ridge"],
            response_standardization=args.response_standardization,
        )
    locations = {
        "source": args.source_output / "baseline",
        "har": args.har_vara_output / "forecasts",
        "garch": args.garch_vara_output / "forecasts",
        "riskmetrics_ewma_vara": (
            args.riskmetrics_ewma_vara_output / "forecasts"
        ),
        "pooled_lightgbm_vara": (
            args.pooled_lightgbm_vara_output / "forecasts"
        ),
    }
    method_files = dict(METHOD_FILES)
    method_files[RISKMETRICS_EWMA_METHOD] = (
        "riskmetrics_ewma_vara",
        "riskmetrics_ewma_validation_offset_forecast.npy",
    )
    method_files[RISKMETRICS_EWMA_VARA_METHOD] = (
        "riskmetrics_ewma_vara",
        "riskmetrics_ewma_vara_lambda_forecast.npy",
    )
    method_files[POOLED_LIGHTGBM_METHOD] = (
        "pooled_lightgbm_vara",
        "pooled_lightgbm_forecast.npy",
    )
    method_files[POOLED_LIGHTGBM_VARA_METHOD] = (
        "pooled_lightgbm_vara",
        "pooled_lightgbm_vara_lambda_forecast.npy",
    )
    if args.student_t_garch_output is not None:
        locations["student_t_garch"] = args.student_t_garch_output / "forecasts"
        method_files[STUDENT_T_GARCH_METHOD] = (
            "student_t_garch",
            "garch11_t_validation_offset_forecast.npy",
        )
    if args.student_t_garch_vara_output is not None:
        locations["student_t_garch_vara"] = (
            args.student_t_garch_vara_output / "forecasts"
        )
        method_files[STUDENT_T_GARCH_VARA_METHOD] = (
            "student_t_garch_vara",
            "garch11_t_vara_lambda_forecast.npy",
        )
    missing_table_methods = [
        method for method in TABLE_METHODS if method not in method_files
    ]
    if missing_table_methods:
        raise ValueError(
            "Complete Table 10 requires forecast inputs for: "
            + ", ".join(missing_table_methods)
        )
    forecasts = {
        method: np.load(locations[location] / filename, mmap_mode="r")
        for method, (location, filename) in method_files.items()
    }
    target = np.load(args.source_output / "baseline" / "target.npy", mmap_mode="r")
    expected_shape = (len(panel.dates), len(panel.component_ids))
    if target.shape != expected_shape or any(
        forecast.shape != expected_shape for forecast in forecasts.values()
    ):
        raise ValueError("Paper forecast inputs do not match the processed panel shape.")
    metrics, monthly, calibration = evaluate_risk_forecasts(
        target=target,
        returns=panel.returns,
        dates=panel.dates,
        forecasts=forecasts,
        validation_indices=split.validation_indices,
        test_indices=split.test_indices,
        benchmark_name="har_ridge",
        epsilon=float(source_contract["epsilon"]),
        variance_floor=float(source_contract["variance_floor"]),
    )
    displayed_methods = TABLE_METHODS
    table_metrics, table_monthly, _ = evaluate_risk_forecasts(
        target=target,
        returns=panel.returns,
        dates=panel.dates,
        forecasts={method: forecasts[method] for method in displayed_methods},
        validation_indices=split.validation_indices,
        test_indices=split.test_indices,
        benchmark_name="har_ridge",
        epsilon=float(source_contract["epsilon"]),
        variance_floor=float(source_contract["variance_floor"]),
    )
    zscore_metrics: pd.DataFrame | None = None
    zscore_monthly: pd.DataFrame | None = None
    zscore_rankscore_audit: pd.DataFrame | None = None
    zscore_scalers: pd.DataFrame | None = None
    zscore_contract: dict[str, object] | None = None
    if args.write_training_zscore_table:
        training_window_months = int(source_contract["universe_window_months"])
        training_end = int(split.validation_indices[0])
        training_start = training_end - training_window_months
        if training_start < 0:
            raise ValueError(
                "The requested training-z-score window starts before the panel."
            )
        training_indices = np.arange(
            training_start,
            training_end,
            dtype=np.int64,
        )
        zscore_metrics, zscore_monthly, zscore_rankscore_audit, zscore_scalers = (
            evaluate_training_zscore_risk_table(
                target=target,
                dates=panel.dates,
                component_ids=np.asarray(panel.component_ids, dtype=object),
                forecasts={method: forecasts[method] for method in displayed_methods},
                training_indices=training_indices,
                test_indices=np.asarray(split.test_indices, dtype=np.int64),
            )
        )
        if set(zscore_metrics["test_cells"]) != set(table_metrics["test_cells"]):
            raise ValueError(
                "Training-z-score and original-scale tables use different test cells."
            )
        zscore_origin_count = int(zscore_monthly["date"].nunique())
        expected_zscore_rows = zscore_origin_count * len(displayed_methods)
        if zscore_origin_count != len(split.test_indices):
            raise ValueError(
                "Training-z-score table does not cover every test month."
            )
        if len(zscore_rankscore_audit) != expected_zscore_rows:
            raise ValueError(
                "Training-z-score RankScore audit is not a complete method-month grid."
            )
        zscore_contract = {
            "enabled": True,
            "evaluation_only": True,
            "forecast_refit": False,
            "target": "log(epsilon + squared return)",
            "normalization": "componentwise_training_zscore",
            "training_window": "fixed_pre_validation",
            "training_window_months": training_window_months,
            "training_start": panel.dates[training_indices[0]].date().isoformat(),
            "training_end": panel.dates[training_indices[-1]].date().isoformat(),
            "statistics": "finite_only_componentwise_mean_and_population_variance",
            "scale_ddof": 0,
            "scale_floor": TRAINING_ZSCORE_SCALE_FLOOR,
            "minimum_finite_training_observations": int(
                zscore_scalers["training_observations"].min()
            ),
            "maximum_finite_training_observations": int(
                zscore_scalers["training_observations"].max()
            ),
            "minimum_training_scale": float(zscore_scalers["training_scale"].min()),
            "maximum_training_scale": float(zscore_scalers["training_scale"].max()),
            "method_count": len(displayed_methods),
            "methods": list(displayed_methods),
            "test_origins": zscore_origin_count,
            "common_matched_cells": int(zscore_metrics["test_cells"].iloc[0]),
            "rank_score": "monthly_strict_win_on_cross_sectional_mean_z_absolute_error",
            "comparators_per_method": len(displayed_methods) - 1,
            "crps_included": False,
            "artifacts": {
                "metrics": "paper_table_metrics_training_zscore.csv",
                "monthly_losses": "paper_table_monthly_losses_training_zscore.csv",
                "rankscore_audit": "paper_table_rankscore_by_month_training_zscore.csv",
                "scalers": "paper_table_training_zscore_scalers.csv",
                "contract": "paper_table_training_zscore_contract.json",
                "markdown_table": "paper_table_training_zscore.md",
                "latex_table": "paper_table_training_zscore.tex",
            },
        }
    rankscore_audit = table_monthly.loc[:, [
        "date",
        "model",
        "observed_cells",
        "log_risk_mae",
        "absolute_error_wins",
        "rank_score_s",
    ]].rename(columns={"log_risk_mae": "absolute_prediction_error"})
    rankscore_origin_count = int(rankscore_audit["date"].nunique())
    rankscore_method_count = len(displayed_methods)
    if rankscore_origin_count != len(split.test_indices):
        raise ValueError("Table RankScore audit does not cover every test month.")
    if rankscore_method_count != 10:
        raise ValueError("Complete Table 10 must contain exactly ten methods.")
    expected_rankscore_rows = rankscore_origin_count * rankscore_method_count
    if len(rankscore_audit) != expected_rankscore_rows:
        raise ValueError("Table RankScore audit is not a complete method-month grid.")
    observed_cells_per_month = rankscore_audit.pivot(
        index="date",
        columns="model",
        values="observed_cells",
    )
    if not observed_cells_per_month.nunique(axis=1).eq(1).all():
        raise ValueError("Table RankScore methods do not use a common monthly cell mask.")
    rankscore_pair_months = (
        rankscore_origin_count * rankscore_method_count * (rankscore_method_count - 1) // 2
    )
    rankscore_strict_wins = int(rankscore_audit["absolute_error_wins"].sum())
    rankscore_exact_ties = rankscore_pair_months - rankscore_strict_wins

    dm = build_paper_dm_results(monthly=monthly, metrics=metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output_dir / "paper_metrics.csv", index=False)
    table_metrics.to_csv(args.output_dir / "paper_table_metrics.csv", index=False)
    eta_contract = har_contract["eta_estimation"]
    assert isinstance(eta_contract, dict)
    table_attention_support = str(
        eta_contract.get("attention_support", "unspecified")
    )
    write_risk_table(
        table_metrics,
        output_dir=args.output_dir,
        attention_support=table_attention_support,
    )
    monthly.to_csv(args.output_dir / "paper_monthly_losses.csv", index=False)
    rankscore_audit.to_csv(
        args.output_dir / "paper_table_rankscore_by_month.csv",
        index=False,
    )
    if zscore_metrics is not None:
        assert zscore_monthly is not None
        assert zscore_rankscore_audit is not None
        assert zscore_scalers is not None
        assert zscore_contract is not None
        zscore_metrics.to_csv(
            args.output_dir / "paper_table_metrics_training_zscore.csv",
            index=False,
        )
        zscore_monthly.to_csv(
            args.output_dir / "paper_table_monthly_losses_training_zscore.csv",
            index=False,
        )
        zscore_rankscore_audit.to_csv(
            args.output_dir / "paper_table_rankscore_by_month_training_zscore.csv",
            index=False,
        )
        zscore_scalers.to_csv(
            args.output_dir / "paper_table_training_zscore_scalers.csv",
            index=False,
        )
        write_json(
            args.output_dir / "paper_table_training_zscore_contract.json",
            zscore_contract,
        )
        zscore_markdown, zscore_latex = render_training_zscore_table(zscore_metrics)
        (args.output_dir / "paper_table_training_zscore.md").write_text(
            zscore_markdown,
            encoding="utf-8",
        )
        (args.output_dir / "paper_table_training_zscore.tex").write_text(
            zscore_latex,
            encoding="utf-8",
        )
    dm.to_csv(args.output_dir / "paper_dm_results.csv", index=False)
    dm.loc[dm["panel"].str.startswith("A:")].to_csv(
        args.output_dir / "paper_dm_har_vara.csv",
        index=False,
    )
    write_json(args.output_dir / "paper_variance_calibration.json", calibration)
    contract_fingerprint_paths = {
        "har_contract": har_contract_path,
        "garch_contract": garch_contract_path,
        "riskmetrics_ewma_vara_contract": riskmetrics_ewma_contract_path,
        "pooled_lightgbm_vara_contract": pooled_lightgbm_contract_path,
    }
    if student_t_contract_path is not None:
        contract_fingerprint_paths["student_t_garch_contract"] = (
            student_t_contract_path
        )
    if student_t_vara_contract_path is not None:
        contract_fingerprint_paths["student_t_garch_vara_contract"] = (
            student_t_vara_contract_path
        )
    write_json(
        args.output_dir / "paper_contract.json",
        {
            "application_contract_schema": APPLICATION_CONTRACT_SCHEMA,
            "source_output": args.source_output.resolve().as_posix(),
            "har_vara_output": args.har_vara_output.resolve().as_posix(),
            "garch_vara_output": args.garch_vara_output.resolve().as_posix(),
            "riskmetrics_ewma_vara_output": (
                args.riskmetrics_ewma_vara_output.resolve().as_posix()
            ),
            "pooled_lightgbm_vara_output": (
                args.pooled_lightgbm_vara_output.resolve().as_posix()
            ),
            "student_t_garch_output": (
                None
                if args.student_t_garch_output is None
                else args.student_t_garch_output.resolve().as_posix()
            ),
            "student_t_garch_vara_output": (
                None
                if args.student_t_garch_vara_output is None
                else args.student_t_garch_vara_output.resolve().as_posix()
            ),
            "component_count": len(panel.component_ids),
            "split": split_contract,
            "eta_estimation": har_contract["eta_estimation"],
            "response_standardization": args.response_standardization,
            "lightgbm_response_normalization": (
                args.lightgbm_response_normalization
            ),
            "table_methods": list(displayed_methods),
            "standalone_table": {
                "attention_support": table_attention_support,
                "full_dense_attention_comparison": (
                    table_attention_support == FULL_DENSE_ATTENTION_SUPPORT
                ),
                "metrics": "paper_table_metrics.csv",
                "markdown": "paper_table.md",
                "latex": "paper_table.tex",
            },
            "training_zscore_table": (
                {"enabled": False}
                if zscore_contract is None
                else zscore_contract
            ),
            "rank_score": {
                "definition": "monthly_strict_absolute_error_win_rate",
                "absolute_prediction_error": (
                    "cross_sectional_mean_absolute_log_risk_error"
                ),
                "origin_count": rankscore_origin_count,
                "method_count": rankscore_method_count,
                "comparators_per_method": rankscore_method_count - 1,
                "comparison": "strict_unrounded_less_than",
                "ties": "no_win_for_either_method",
                "common_matched_cell_mask": True,
                "common_matched_cells": int(table_metrics["test_cells"].iloc[0]),
                "minimum_matched_cells_per_month": int(
                    observed_cells_per_month.iloc[:, 0].min()
                ),
                "maximum_matched_cells_per_month": int(
                    observed_cells_per_month.iloc[:, 0].max()
                ),
                "exact_pair_month_ties": rankscore_exact_ties,
                "pair_month_comparisons": rankscore_pair_months,
                "expected_audit_rows": expected_rankscore_rows,
                "audit_file": "paper_table_rankscore_by_month.csv",
            },
            "diebold_mariano": {
                "comparison_count": len(DM_COMPARISONS),
                "panels": [
                    "A: primary versus alternative bases",
                    "B: incremental VARA within family",
                ],
                "losses": ["squared_error", "gaussian_return_crps"],
                "sampling_unit": "monthly_panel_average_loss",
                "forecast_horizon": 1,
                "reported_statistic": "raw_dm_statistic",
                "reported_p_value": "raw_two_sided_standard_normal",
                "multiplicity_adjustment": None,
                "results_file": "paper_dm_results.csv",
            },
            "input_contract_fingerprints": file_fingerprints(
                contract_fingerprint_paths
            ),
        },
    )

    display_columns = [
        "model",
        "log_risk_rmse",
        "log_risk_mae",
        "gaussian_return_crps",
        "log_risk_bias",
        "absolute_r2",
        "rank_score",
        "overall_rank",
    ]
    print(
        table_metrics.loc[:, display_columns]
        .sort_values("log_risk_rmse")
        .to_string(index=False)
    )
    if zscore_metrics is not None:
        zscore_display_columns = [
            "model",
            "z_log_risk_mse",
            "z_log_risk_rmse",
            "z_log_risk_mae",
            "z_log_risk_bias",
            "z_absolute_r2",
            "rank_score",
            "overall_rank",
        ]
        print("\nTraining-z-score-scaled Table 10 comparison:")
        print(
            zscore_metrics.loc[:, zscore_display_columns]
            .sort_values("z_log_risk_rmse")
            .to_string(index=False)
        )
    print(dm.to_string(index=False))


if __name__ == "__main__":
    main()
