#!/usr/bin/env python3
"""Run the Tourism AutoETS residual correction with the Risk VARA config.

The Risk application and the Tourism application have different base models
and different static metadata.  This runner keeps the established Tourism
ZZA AutoETS rolling forecasts, supplies a response-free Tourism hierarchy
feature matrix, and otherwise carries over the saved Risk HAR--VARA settings.
Final accuracy is reported for January--December 2016.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

try:
    from _bootstrap import REPO_ROOT
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT

from scripts.datamine_paper_eta import (
    attention_support_name,
    file_fingerprints,
)
from vara.datamine_vara_runner import run_vara_family
from vara.forecast_contracts import (
    BaseForecastResult,
    ExecutionConfig,
    ForecastPanel,
    PaperEtaConfig,
    RollingEvaluationPlan,
    RollingVARAConfig,
    StaticFeatureSet,
    VARARunRequest,
)


DEFAULT_AUTOETS_PANEL = (
    REPO_ROOT
    / "outputs/tourism_76_autoets_zza_train_zscore_extended/rolling_forecasts.csv"
)
DEFAULT_RISK_CONTRACT = (
    REPO_ROOT
    / "outputs/datamine_vara/"
    "ew_log_risk_har_vara_papereta_logp_topk500_w240_refit12_"
    "lambda24_training_zscore_ray15/forecasts/contract.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "outputs/tourism_76_autoets_zza_risk_config_har_vara_training_zscore"
)
MODEL_FORECAST_COLUMNS = {
    "AutoETS": "autoets_forecast",
    "AutoETS + unit VARA (Risk config)": "unit_vara_forecast",
    "AutoETS + lambda*VARA (Risk config)": "lambda_vara_forecast",
}


@dataclass(frozen=True)
class TourismAutoETSPanel:
    """Validated complete rolling-origin Tourism panel."""

    dates: pd.DatetimeIndex
    series_ids: tuple[str, ...]
    metadata: pd.DataFrame
    actual: np.ndarray
    base_forecast: np.ndarray
    base_error: np.ndarray
    training_scale: np.ndarray
    source_metadata: dict[str, Any]


@dataclass(frozen=True)
class EvaluationSplit:
    """Trailing validation and final-test indices."""

    validation_indices: np.ndarray
    test_indices: np.ndarray
    evaluation_indices: np.ndarray


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return value.resolve().as_posix()
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__} to JSON.")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(dict(payload), indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _parse_bool_column(values: pd.Series, *, name: str) -> np.ndarray:
    parsed: list[bool] = []
    for value in values.to_numpy():
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
        elif isinstance(value, (int, np.integer, float, np.floating)) and value in (0, 1):
            parsed.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            parsed.append(value.strip().lower() == "true")
        else:
            raise ValueError(f"Column {name} contains a non-Boolean value: {value!r}.")
    return np.asarray(parsed, dtype=bool)


def load_tourism_autoets_panel(
    forecast_path: Path,
    *,
    metadata_path: Path | None = None,
    expected_series: int = 76,
) -> TourismAutoETSPanel:
    """Load and validate the authoritative ZZA training-z AutoETS panel."""
    source = Path(forecast_path).resolve()
    sidecar = (
        Path(metadata_path).resolve()
        if metadata_path is not None
        else Path(f"{source}.metadata.json")
    )
    source_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    autoets_config = source_metadata.get("autoets_config", {})
    required_contract = {
        "response_transform": "none",
        "response_normalization": "training_zscore",
    }
    for field, expected in required_contract.items():
        if source_metadata.get(field) != expected:
            raise ValueError(
                f"AutoETS metadata has {field}={source_metadata.get(field)!r}; "
                f"expected {expected!r}."
            )
    if autoets_config.get("model_specification") != "ZZA":
        raise ValueError("The Tourism base forecast must use AutoETS specification ZZA.")
    if int(autoets_config.get("forecast_horizon", -1)) != 1:
        raise ValueError("The Tourism base forecast must be one-step rolling AutoETS.")
    if not bool(autoets_config.get("refit_every_origin", False)):
        raise ValueError("The Tourism AutoETS source must refit at every origin.")

    frame = pd.read_csv(source)
    required = {
        "date",
        "series_id",
        "group",
        "actual",
        "direct_auto_ets",
        "autoets_training_scale",
        "autoets_history_end",
        "response_transform",
        "response_normalization",
        "autoets_clip_nonnegative",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Tourism AutoETS panel is missing columns: {missing}.")
    frame["date"] = pd.to_datetime(frame["date"])
    frame["autoets_history_end"] = pd.to_datetime(frame["autoets_history_end"])
    if frame.duplicated(["date", "series_id"]).any():
        raise ValueError("Tourism AutoETS panel contains duplicate date-series cells.")
    if not frame["response_transform"].eq("none").all():
        raise ValueError("Tourism AutoETS cells must use no response transformation.")
    if not frame["response_normalization"].eq("training_zscore").all():
        raise ValueError("Tourism AutoETS cells must use training-only z-normalization.")
    if not _parse_bool_column(
        frame["autoets_clip_nonnegative"], name="autoets_clip_nonnegative"
    ).all():
        raise ValueError("Tourism AutoETS forecasts must use nonnegative clipping.")
    if not (frame["autoets_history_end"] < frame["date"]).all():
        raise ValueError("Some AutoETS fits use the forecast target or later data.")

    dates = pd.DatetimeIndex(sorted(frame["date"].unique()))
    expected_dates = pd.date_range(dates[0], dates[-1], freq="MS")
    if not dates.equals(expected_dates):
        raise ValueError("Tourism AutoETS dates must be a complete monthly sequence.")
    series_ids = tuple(sorted(frame["series_id"].astype(str).unique()))
    if len(series_ids) != int(expected_series):
        raise ValueError(
            f"Tourism AutoETS panel contains {len(series_ids)} series; "
            f"expected {int(expected_series)}."
        )
    if len(frame) != len(dates) * len(series_ids):
        raise ValueError("Tourism AutoETS panel is not a complete rectangular panel.")

    group_counts = frame.groupby("series_id", observed=True)["group"].nunique()
    if not group_counts.eq(1).all():
        raise ValueError("Each Tourism series must have one fixed group label.")
    group_lookup = (
        frame.loc[:, ["series_id", "group"]]
        .drop_duplicates()
        .set_index("series_id")["group"]
    )
    metadata = pd.DataFrame({
        "series_id": series_ids,
        "group": [str(group_lookup.loc[series_id]) for series_id in series_ids],
    })

    def pivot(column: str) -> np.ndarray:
        values = frame.pivot(index="date", columns="series_id", values=column)
        values = values.reindex(index=dates, columns=series_ids)
        array = values.to_numpy(dtype=float)
        if array.shape != (len(dates), len(series_ids)) or not np.isfinite(array).all():
            raise ValueError(f"Tourism AutoETS column {column} is incomplete or non-finite.")
        return array

    actual_column = "actual_original" if "actual_original" in frame else "actual"
    actual = pivot(actual_column)
    if actual_column != "actual" and not np.allclose(
        actual, pivot("actual"), rtol=0.0, atol=1e-10
    ):
        raise ValueError("actual and actual_original disagree in the AutoETS panel.")
    base_forecast = pivot("direct_auto_ets")
    training_scale = pivot("autoets_training_scale")
    if np.any(training_scale <= 1e-12):
        raise ValueError("Tourism AutoETS training scales must be positive.")
    return TourismAutoETSPanel(
        dates=dates,
        series_ids=series_ids,
        metadata=metadata,
        actual=actual,
        base_forecast=base_forecast,
        base_error=actual - base_forecast,
        training_scale=training_scale,
        source_metadata=source_metadata,
    )


def build_tourism_hierarchy_features(
    series_ids: tuple[str, ...] | list[str],
) -> tuple[np.ndarray, list[str]]:
    """Build a fixed unit-row-norm level-1/level-2 hierarchy incidence matrix."""
    identifiers = [str(value) for value in series_ids]
    codes: list[str] = []
    for series_id in identifiers:
        code = series_id[:-3] if series_id.endswith("All") else series_id
        if len(code) != 3 or not code.isalpha():
            raise ValueError(
                f"Tourism leaf series id {series_id!r} does not contain a 3-letter code."
            )
        codes.append(code.upper())
    level_1 = sorted({code[:1] for code in codes})
    level_2 = sorted({code[:2] for code in codes})
    feature_names = [f"L1_{value}" for value in level_1]
    feature_names.extend(f"L2_{value}" for value in level_2)
    lookup_1 = {value: index for index, value in enumerate(level_1)}
    lookup_2 = {
        value: len(level_1) + index for index, value in enumerate(level_2)
    }
    matrix = np.zeros((len(codes), len(feature_names)), dtype=float)
    weight = 1.0 / np.sqrt(2.0)
    for row, code in enumerate(codes):
        matrix[row, lookup_1[code[:1]]] = weight
        matrix[row, lookup_2[code[:2]]] = weight
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Tourism hierarchy feature rows were not normalized.")
    return matrix, feature_names


def trailing_evaluation_split(
    dates: pd.DatetimeIndex,
    *,
    validation_months: int,
    test_months: int,
) -> EvaluationSplit:
    """Return a trailing validation/test split with increasing integer indices."""
    validation = int(validation_months)
    test = int(test_months)
    if validation < 1 or test < 1:
        raise ValueError("validation_months and test_months must be positive.")
    total = validation + test
    if len(dates) <= total:
        raise ValueError("The rolling panel has no pre-evaluation residual history.")
    validation_indices = np.arange(len(dates) - total, len(dates) - test, dtype=np.int64)
    test_indices = np.arange(len(dates) - test, len(dates), dtype=np.int64)
    evaluation_indices = np.concatenate((validation_indices, test_indices))
    return EvaluationSplit(
        validation_indices=validation_indices,
        test_indices=test_indices,
        evaluation_indices=evaluation_indices,
    )


def calculate_tourism_metrics(
    *,
    actual: np.ndarray,
    training_scale: np.ndarray,
    forecasts: Mapping[str, np.ndarray],
    evaluation_indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate pooled/raw and training-z-scaled accuracy for selected origins."""
    target = np.asarray(actual, dtype=float)
    scale = np.asarray(training_scale, dtype=float)
    indices = np.asarray(evaluation_indices, dtype=np.int64)
    if target.ndim != 2 or target.shape != scale.shape:
        raise ValueError("actual and training_scale must be equal-shape panels.")
    if indices.size == 0 or np.any(np.diff(indices) <= 0):
        raise ValueError("evaluation_indices must be nonempty and increasing.")
    if np.any(scale[indices] <= 1e-12) or not np.isfinite(scale[indices]).all():
        raise ValueError("Evaluation training scales must be finite and positive.")

    rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    base_scaled_sse: float | None = None
    for method, forecast in forecasts.items():
        values = np.asarray(forecast, dtype=float)
        if values.shape != target.shape or not np.isfinite(values[indices]).all():
            raise ValueError(f"Forecast panel for {method} is incomplete or non-finite.")
        error = values[indices] - target[indices]
        scaled_error = error / scale[indices]
        raw_sse = float(np.sum(np.square(error)))
        scaled_sse = float(np.sum(np.square(scaled_error)))
        if base_scaled_sse is None:
            base_scaled_sse = scaled_sse
        rows.append({
            "method": str(method),
            "scaled_mse": float(np.mean(np.square(scaled_error))),
            "scaled_rmse": float(np.sqrt(np.mean(np.square(scaled_error)))),
            "scaled_mae": float(np.mean(np.abs(scaled_error))),
            "scaled_bias": float(np.mean(scaled_error)),
            "raw_mse": float(np.mean(np.square(error))),
            "raw_rmse": float(np.sqrt(np.mean(np.square(error)))),
            "raw_mae": float(np.mean(np.abs(error))),
            "raw_bias": float(np.mean(error)),
            "scaled_r2_vs_autoets": float(1.0 - scaled_sse / base_scaled_sse),
            "cells": int(error.size),
            "origins": int(len(indices)),
            "series": int(error.shape[1]),
        })
        for position, target_index in enumerate(indices):
            month_error = error[position]
            month_scaled_error = scaled_error[position]
            monthly_rows.append({
                "method": str(method),
                "target_index": int(target_index),
                "scaled_mse": float(np.mean(np.square(month_scaled_error))),
                "scaled_mae": float(np.mean(np.abs(month_scaled_error))),
                "raw_mse": float(np.mean(np.square(month_error))),
                "raw_mae": float(np.mean(np.abs(month_error))),
                "series": int(error.shape[1]),
            })
    metrics = pd.DataFrame(rows).sort_values(["scaled_mse", "method"]).reset_index(drop=True)
    metrics.insert(0, "mse_order", np.arange(1, len(metrics) + 1))
    monthly = pd.DataFrame(monthly_rows)
    return metrics, monthly


def _risk_validation_months(contract: Mapping[str, Any]) -> int:
    split = contract.get("split", {})
    start = pd.Timestamp(split["validation_start"])
    end = pd.Timestamp(split["validation_end"])
    months = (end.year - start.year) * 12 + end.month - start.month + 1
    if months < 1:
        raise ValueError("Risk contract has an invalid validation interval.")
    return int(months)


def _risk_settings(contract: Mapping[str, Any]) -> dict[str, Any]:
    eta = contract.get("eta_estimation", {})
    execution = contract.get("rolling_refit_execution", {})
    bounds = eta.get("bounds", [])
    required_top_level = (
        "error_window_months",
        "response_standardization",
        "response_clip",
        "refit_frequency_months",
        "ridge",
        "attention_top_k",
        "attention_cache_mode",
        "attention_batch_size",
        "lambda_window_months",
        "lambda_ridge",
        "zero_own_lag_coefficients",
    )
    missing = [field for field in required_top_level if field not in contract]
    if missing:
        raise ValueError(f"Risk contract is missing settings: {missing}.")
    if len(bounds) != 2:
        raise ValueError("Risk contract must provide two paper-eta bounds.")
    return {
        **{field: contract[field] for field in required_top_level},
        "eta_lower": float(bounds[0]),
        "eta_upper": float(bounds[1]),
        "eta_grid_size": int(eta["grid_size"]),
        "eta_local_tolerance": float(eta["local_tolerance"]),
        "validation_months": _risk_validation_months(contract),
        "risk_test_months": (
            (pd.Timestamp(contract["split"]["test_end"]).year - pd.Timestamp(contract["split"]["test_start"]).year) * 12
            + pd.Timestamp(contract["split"]["test_end"]).month
            - pd.Timestamp(contract["split"]["test_start"]).month
            + 1
        ),
        "ray_num_cpus": int(execution.get("ray_num_cpus", 1)),
        "max_cpus_per_task": int(execution.get("max_cpus_per_task", 1)),
    }


def _refit_date(index: int, dates: pd.DatetimeIndex) -> str | None:
    return dates[int(index)].date().isoformat() if int(index) >= 0 else None


def _typed_vara_inputs(
    *,
    panel: TourismAutoETSPanel,
    split: EvaluationSplit,
    features: np.ndarray,
    feature_names: list[str],
    settings: Mapping[str, Any],
    ray_num_cpus: int,
    max_cpus_per_task: int,
    ray_debug: bool,
) -> tuple[VARARunRequest, RollingVARAConfig, ExecutionConfig]:
    """Map the existing Tourism/Risk inputs to the shared typed interface."""
    request = VARARunRequest(
        panel=ForecastPanel(
            dates=panel.dates,
            series_ids=panel.series_ids,
            target=panel.actual,
            metadata=panel.metadata,
        ),
        evaluation=RollingEvaluationPlan(
            validation_indices=split.validation_indices,
            test_indices=split.test_indices,
        ),
        features=StaticFeatureSet(
            values=features,
            names=feature_names,
            provenance={
                "source": "response-free Tourism level-1/level-2 hierarchy incidence",
            },
        ),
        base=BaseForecastResult(
            model_name="autoets_zza",
            forecast=panel.base_forecast,
            error=panel.base_error,
        ),
    )
    config = RollingVARAConfig(
        error_window_months=int(settings["error_window_months"]),
        response_standardization=str(settings["response_standardization"]),
        response_clip=float(settings["response_clip"]),
        refit_frequency_months=int(settings["refit_frequency_months"]),
        ridge=float(settings["ridge"]),
        attention_top_k=int(settings["attention_top_k"]),
        attention_cache_mode=str(settings["attention_cache_mode"]),
        attention_batch_size=int(settings["attention_batch_size"]),
        eta=PaperEtaConfig(
            lower=float(settings["eta_lower"]),
            upper=float(settings["eta_upper"]),
            grid_size=int(settings["eta_grid_size"]),
            local_tolerance=float(settings["eta_local_tolerance"]),
        ),
        lambda_window_months=int(settings["lambda_window_months"]),
        lambda_ridge=float(settings["lambda_ridge"]),
        zero_own_lag_coefficients=bool(settings["zero_own_lag_coefficients"]),
    )
    execution = ExecutionConfig(
        ray_num_cpus=ray_num_cpus,
        max_cpus_per_task=max_cpus_per_task,
        ray_debug=ray_debug,
    )
    return request, config, execution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--autoets-panel", type=Path, default=DEFAULT_AUTOETS_PANEL)
    parser.add_argument("--autoets-metadata", type=Path)
    parser.add_argument("--risk-contract", type=Path, default=DEFAULT_RISK_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--test-months",
        type=int,
        default=12,
        help="Tourism final holdout length; Risk's 120 months cannot fit this panel.",
    )
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        help="Override the Risk contract's Ray CPU budget; use 1 for sequential fitting.",
    )
    parser.add_argument("--ray-debug", action="store_true")
    args = parser.parse_args()

    risk_contract_path = args.risk_contract.resolve()
    risk_contract = json.loads(risk_contract_path.read_text(encoding="utf-8"))
    settings = _risk_settings(risk_contract)
    ray_num_cpus = (
        int(args.ray_num_cpus)
        if args.ray_num_cpus is not None
        else int(settings["ray_num_cpus"])
    )
    if ray_num_cpus < 1:
        raise ValueError("ray_num_cpus must be positive.")
    max_cpus_per_task = int(settings["max_cpus_per_task"])
    if max_cpus_per_task > ray_num_cpus:
        raise ValueError("max_cpus_per_task cannot exceed ray_num_cpus.")

    autoets_path = args.autoets_panel.resolve()
    autoets_metadata_path = (
        args.autoets_metadata.resolve()
        if args.autoets_metadata is not None
        else Path(f"{autoets_path}.metadata.json")
    )
    panel = load_tourism_autoets_panel(
        autoets_path,
        metadata_path=autoets_metadata_path,
    )
    split = trailing_evaluation_split(
        panel.dates,
        validation_months=int(settings["validation_months"]),
        test_months=int(args.test_months),
    )
    if len(panel.dates[: int(split.evaluation_indices[0])]) < 24:
        raise ValueError("The first Tourism VARA refit has fewer than 24 residual months.")
    features, feature_names = build_tourism_hierarchy_features(panel.series_ids)
    vara_request, vara_config, execution_config = _typed_vara_inputs(
        panel=panel,
        split=split,
        features=features,
        feature_names=feature_names,
        settings=settings,
        ray_num_cpus=ray_num_cpus,
        max_cpus_per_task=max_cpus_per_task,
        ray_debug=bool(args.ray_debug),
    )
    eta_specification = vara_config.eta.resolve(len(panel.series_ids))

    print(
        "Tourism Risk-config HAR-VARA: "
        f"series={len(panel.series_ids)} "
        f"validation={panel.dates[split.validation_indices[0]].date()}.."
        f"{panel.dates[split.validation_indices[-1]].date()} "
        f"test={panel.dates[split.test_indices[0]].date()}.."
        f"{panel.dates[split.test_indices[-1]].date()} "
        f"error_window_cap={int(settings['error_window_months'])} "
        f"top_k_requested={int(settings['attention_top_k'])} "
        f"top_k_effective={min(int(settings['attention_top_k']), len(panel.series_ids) - 1)}",
        flush=True,
    )

    def print_refit(row: dict[str, Any]) -> None:
        print(
            "eta refit "
            f"history_end={panel.dates[int(row['refit_index'])].date()} "
            f"error_rows={int(row['error_rows'])} "
            f"paper_eta={eta_specification.paper_eta(float(row['eta_hat'])):.6f} "
            f"boundary={bool(row['eta_on_boundary'])}",
            flush=True,
        )

    family = run_vara_family(
        vara_request,
        vara_config,
        execution=execution_config,
        progress_callback=print_refit,
    )
    fit_summary = eta_specification.convert_core_fit_summary(
        family.core_fit_summary
    )
    eta_specification.validate_refit_coverage(
        fit_summary,
        family.refit_index,
        split.evaluation_indices,
    )

    unit_unclipped = np.asarray(family.unit_forecast, dtype=float)
    lambda_unclipped = np.asarray(family.lambda_forecast, dtype=float)
    unit_forecast = np.maximum(unit_unclipped, 0.0)
    lambda_forecast = np.maximum(lambda_unclipped, 0.0)
    forecasts = {
        "AutoETS": panel.base_forecast,
        "AutoETS + unit VARA (Risk config)": unit_forecast,
        "AutoETS + lambda*VARA (Risk config)": lambda_forecast,
    }
    metrics, monthly = calculate_tourism_metrics(
        actual=panel.actual,
        training_scale=panel.training_scale,
        forecasts=forecasts,
        evaluation_indices=split.test_indices,
    )
    monthly["date"] = [
        panel.dates[int(index)].date().isoformat()
        for index in monthly["target_index"]
    ]

    output_dir = args.output_dir.resolve()
    forecast_dir = output_dir / "forecasts"
    metric_dir = output_dir / "metrics"
    feature_dir = output_dir / "features"
    for directory in (forecast_dir, metric_dir, feature_dir):
        directory.mkdir(parents=True, exist_ok=True)

    evaluation = split.evaluation_indices
    row_dates = np.repeat(panel.dates[evaluation].to_numpy(), len(panel.series_ids))
    row_series = np.tile(np.asarray(panel.series_ids, dtype=object), len(evaluation))
    row_refits = np.repeat(family.refit_index[evaluation], len(panel.series_ids))
    long = pd.DataFrame({
        "date": row_dates,
        "series_id": row_series,
        "is_final_test": np.repeat(np.isin(evaluation, split.test_indices), len(panel.series_ids)),
        "actual": panel.actual[evaluation].reshape(-1),
        "autoets_training_scale": panel.training_scale[evaluation].reshape(-1),
        "autoets_forecast": panel.base_forecast[evaluation].reshape(-1),
        "autoets_error_actual_minus_forecast": panel.base_error[evaluation].reshape(-1),
        "vara_correction": family.correction[evaluation].reshape(-1),
        "lambda": family.lambda_weights[evaluation].reshape(-1),
        "unit_vara_forecast_unclipped": unit_unclipped[evaluation].reshape(-1),
        "lambda_vara_forecast_unclipped": lambda_unclipped[evaluation].reshape(-1),
        "unit_vara_forecast": unit_forecast[evaluation].reshape(-1),
        "lambda_vara_forecast": lambda_forecast[evaluation].reshape(-1),
        "vara_refit_history_end_index": row_refits,
        "vara_refit_history_end_date": [
            _refit_date(index, panel.dates) for index in row_refits
        ],
    })
    long.to_csv(forecast_dir / "rolling_forecasts.csv", index=False)

    fit_output = fit_summary.copy()
    fit_output.insert(
        1,
        "history_end_date",
        [_refit_date(index, panel.dates) for index in fit_output["refit_index"]],
    )
    fit_output.insert(
        2,
        "first_forecast_date",
        [
            _refit_date(int(index) + 1, panel.dates)
            for index in fit_output["refit_index"]
        ],
    )
    fit_output.to_csv(forecast_dir / "fit_summary.csv", index=False)

    lambda_table = family.lambda_refit_fits.copy()
    for column, output_column in (
        ("lambda_refit_target_index", "lambda_refit_target_date"),
        ("lambda_training_start_index", "lambda_training_start_date"),
        ("lambda_training_end_index", "lambda_training_end_date"),
    ):
        if column in lambda_table:
            lambda_table[output_column] = [
                _refit_date(index, panel.dates) for index in lambda_table[column]
            ]
    lambda_table.to_csv(forecast_dir / "lambda_refit_fits.csv", index=False)

    lambda_rows: list[dict[str, Any]] = []
    for target_index in evaluation:
        values = family.lambda_weights[int(target_index)]
        if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
            raise RuntimeError("Expected a common pooled lambda across Tourism series.")
        lambda_rows.append({
            "target_index": int(target_index),
            "date": panel.dates[int(target_index)].date().isoformat(),
            "lambda": float(values[0]),
            "refit_index": int(family.refit_index[int(target_index)]),
            "refit_history_end_date": _refit_date(
                int(family.refit_index[int(target_index)]), panel.dates
            ),
            "is_final_test": bool(np.isin(target_index, split.test_indices)),
        })
    lambda_by_origin = pd.DataFrame(lambda_rows)
    lambda_by_origin.to_csv(forecast_dir / "lambda_by_origin.csv", index=False)

    metrics.to_csv(metric_dir / "metrics.csv", index=False)
    monthly.to_csv(metric_dir / "monthly_losses.csv", index=False)
    np.save(feature_dir / "Z.npy", features)
    pd.DataFrame({"feature": feature_names}).to_csv(
        feature_dir / "feature_columns.csv", index=False
    )
    panel.metadata.to_csv(feature_dir / "component_order.csv", index=False)

    first_test = int(split.test_indices[0])
    test_lambda = family.lambda_weights[split.test_indices]
    first_test_lambda_row = lambda_table.loc[
        lambda_table["lambda_refit_target_index"].eq(first_test)
    ]
    if len(first_test_lambda_row) != 1:
        raise RuntimeError("Expected exactly one pooled lambda fit for the 2016 block.")
    lambda_training_end = int(first_test_lambda_row.iloc[0]["lambda_training_end_index"])
    if lambda_training_end >= first_test:
        raise RuntimeError("The 2016 lambda fit includes a 2016 target outcome.")

    risk_test_months = int(settings["risk_test_months"])
    effective_top_k = min(int(settings["attention_top_k"]), len(panel.series_ids) - 1)
    contract = {
        "application_contract_schema": "tourism_risk_config_har_vara_v1",
        "purpose": "Port the locked Risk HAR-VARA correction config to Tourism AutoETS residuals.",
        "source_inputs": file_fingerprints({
            "autoets_forecast_panel": autoets_path,
            "autoets_metadata": autoets_metadata_path,
            "risk_contract": risk_contract_path,
        }),
        "base_model": panel.source_metadata.get("autoets_config", {}),
        "response": {
            "target": "monthly visitor nights",
            "transform": "none",
            "base_fit_normalization": "per-origin per-series training_zscore",
            "evaluation": "error divided by the same origin-specific AutoETS training scale",
            "corrected_forecast_clip": "max(forecast, 0)",
        },
        "split": {
            "validation_months": int(len(split.validation_indices)),
            "validation_start": panel.dates[split.validation_indices[0]].date().isoformat(),
            "validation_end": panel.dates[split.validation_indices[-1]].date().isoformat(),
            "test_months": int(len(split.test_indices)),
            "test_start": panel.dates[split.test_indices[0]].date().isoformat(),
            "test_end": panel.dates[split.test_indices[-1]].date().isoformat(),
            "risk_source_validation_months": int(settings["validation_months"]),
            "risk_source_test_months": risk_test_months,
        },
        "risk_settings_carried_over": {
            "error_window_months": int(settings["error_window_months"]),
            "response_standardization": str(settings["response_standardization"]),
            "response_clip": float(settings["response_clip"]),
            "refit_frequency_months": int(settings["refit_frequency_months"]),
            "har_windows": [1, 3, 12],
            "ridge": float(settings["ridge"]),
            "attention_top_k_requested": int(settings["attention_top_k"]),
            "attention_cache_mode": str(settings["attention_cache_mode"]),
            "attention_batch_size": int(settings["attention_batch_size"]),
            "lambda_window_months": int(settings["lambda_window_months"]),
            "lambda_ridge": float(settings["lambda_ridge"]),
            "lambda_bounds": [0.0, 1.0],
            "zero_own_lag_coefficients": bool(settings["zero_own_lag_coefficients"]),
        },
        "tourism_adaptations": {
            "static_feature_source": "response-free level-1/level-2 hierarchy incidence",
            "feature_shape": list(features.shape),
            "feature_row_normalization": "two active entries, each 1/sqrt(2)",
            "attention_top_k_effective": effective_top_k,
            "attention_top_k_note": "requested 500 is capped at p-1=75 by the shared core",
            "error_window_interpretation": "maximum trailing cap",
            "minimum_error_rows_used": int(fit_summary["error_rows"].min()),
            "maximum_error_rows_used": int(fit_summary["error_rows"].max()),
            "error_window_note": "240 prior OOS residual months are unavailable in this Tourism panel",
            "test_length_note": (
                f"Risk has {risk_test_months} test months; Tourism uses its final "
                f"{len(split.test_indices)} months."
            ),
        },
        "eta_estimation": eta_specification.contract(
            attention_support=attention_support_name(
                str(settings["attention_cache_mode"])
            )
        ),
        "eta_refit_summary": eta_specification.refit_summary(fit_summary),
        "lambda_estimation": {
            "method": "rolling_refit_block_pooled_bounded_lse",
            "window_months": int(settings["lambda_window_months"]),
            "minimum_history_months": int(settings["lambda_window_months"]),
            "ridge": float(settings["lambda_ridge"]),
            "target_month_excluded": True,
            "held_fixed_within_refit_block": True,
            "test_lambda": float(test_lambda[0, 0]),
            "test_lambda_training_start": _refit_date(
                int(first_test_lambda_row.iloc[0]["lambda_training_start_index"]),
                panel.dates,
            ),
            "test_lambda_training_end": _refit_date(
                lambda_training_end, panel.dates
            ),
        },
        "rolling_refit_execution": family.correction_execution,
        "leakage_audit": {
            "autoets_history_end_strictly_before_target": True,
            "vara_refit_history_end_strictly_before_first_block_target": True,
            "each_correction_uses_residuals_only_through_target_minus_one": True,
            "lambda_training_end_strictly_before_2016_test": True,
        },
        "metrics_path": (metric_dir / "metrics.csv").as_posix(),
    }
    _write_json(forecast_dir / "contract.json", contract)

    print("\nFinal 2016 Tourism accuracy (MSE uses training-z-scaled errors):")
    print(
        metrics.loc[:, [
            "mse_order",
            "method",
            "scaled_mse",
            "scaled_rmse",
            "raw_rmse",
            "scaled_r2_vs_autoets",
        ]].to_string(index=False, float_format=lambda value: f"{value:.6f}"),
        flush=True,
    )
    print(
        "\n2016 lambda: "
        f"{float(test_lambda[0, 0]):.6f}, trained on "
        f"{_refit_date(int(first_test_lambda_row.iloc[0]['lambda_training_start_index']), panel.dates)}"
        " through "
        f"{_refit_date(lambda_training_end, panel.dates)}.",
        flush=True,
    )
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
