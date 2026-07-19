#!/usr/bin/env python3
"""Combine AutoETS and VARA using only the 76 Tourism all-purpose series.

The selected model first predicts the direct AutoETS error from the current
VARA correction and the same series' preceding ten same-month AutoETS errors.
It then calibrates that predicted error with rolling geographic-group/month
weights and a robust offset for historical AutoETS-error reliability groups.
Purpose-component series are never loaded.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from search_tourism_76_state_blend import (
        StateBlendConfig,
        build_state_panel,
        forecast_config,
    )
except ModuleNotFoundError:
    from scripts.search_tourism_76_state_blend import (
        StateBlendConfig,
        build_state_panel,
        forecast_config,
    )
from vara.forecast_metrics import forecast_metric_tables
from vara.forecast_significance import panel_diebold_mariano_tests


CORE_MODEL = "reliability_calibrated_vara_auto_ets_oos"
MODEL = "reliability_gated_vara_auto_ets_oos"
PAPER_MODEL = "global_shared_lambda_purpose_vara"
DIRECT_AUTO_ETS = "direct_auto_ets"
DIRECT_AUTO_ETS_VARIANCE = f"{DIRECT_AUTO_ETS}_variance"
STATE_CORRECTION_VARIANCE = "state_correction_variance"
CORE_MODEL_VARIANCE = f"{CORE_MODEL}_variance"
MODEL_VARIANCE = f"{MODEL}_variance"


@dataclass(frozen=True)
class ReliabilityCombinationConfig:
    state_window_months: int = 36
    state_feature_set: str = "vara_seasonal10"
    state_ridge_alpha: float = 10.0
    calibration_window_months: int = 36
    lambda_prior_strength: float = 3.0
    lambda_cap: float = 1.25
    reliability_bins: int = 7
    residual_quantile: float = 0.5
    offset_prior_observations: float = 0.0
    variance_prior_observations: float = 36.0
    state_history_start: str = "2010-01-01"
    reliability_history_end: str = "2009-12-01"


def build_reliability_mapping(
    autoets_forecasts: pd.DataFrame,
    *,
    history_end: str,
    n_bins: int,
    expected_n_series: int = 76,
) -> pd.DataFrame:
    """Assign series to fixed error-MAE quantile groups using early OOS errors."""
    required = {"date", "series_id", "actual", "direct_auto_ets"}
    missing = sorted(required.difference(autoets_forecasts.columns))
    if missing:
        raise ValueError(f"AutoETS reliability input is missing columns: {missing}")
    frame = autoets_forecasts.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame["series_id"] = frame["series_id"].astype(str)
    frame = frame[frame["date"] <= pd.Timestamp(history_end)].copy()
    if frame.empty:
        raise ValueError("No AutoETS errors precede the reliability-history cutoff.")
    if frame["series_id"].nunique() != int(expected_n_series):
        raise ValueError(
            f"Expected {expected_n_series} reliability series, found "
            f"{frame['series_id'].nunique()}."
        )
    frame["absolute_error"] = np.abs(
        frame["actual"].to_numpy(dtype=float)
        - frame["direct_auto_ets"].to_numpy(dtype=float)
    )
    scores = (
        frame.groupby("series_id", observed=True)["absolute_error"]
        .mean()
        .sort_index()
    )
    if int(n_bins) < 2 or int(n_bins) > len(scores):
        raise ValueError("reliability_bins must be between two and n_series.")
    # Ranking first makes the quantile assignment deterministic even under ties.
    groups = pd.qcut(
        scores.rank(method="first"),
        q=int(n_bins),
        labels=False,
    ).astype(int)
    return pd.DataFrame({
        "series_id": scores.index.astype(str),
        "historical_autoets_mae": scores.to_numpy(dtype=float),
        "reliability_group": groups.to_numpy(dtype=int),
        "reliability_history_end": pd.Timestamp(history_end),
    })


def learn_group_season_lambdas(
    train: pd.DataFrame,
    *,
    prior_strength: float,
    cap: float,
) -> tuple[pd.Series, float, float]:
    """Estimate pooled group/month slopes from previously realized errors."""
    required = {"group", "month", "state_correction", "error_target"}
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"Lambda training input is missing columns: {missing}")
    if train.empty:
        raise ValueError("Lambda training input is empty.")
    if float(prior_strength) < 0.0 or float(cap) <= 0.0:
        raise ValueError("Lambda prior strength must be nonnegative and cap positive.")

    work = train.loc[:, [
        "group",
        "month",
        "state_correction",
        "error_target",
    ]].copy()
    work["cross_product"] = (
        work["state_correction"] * work["error_target"]
    )
    work["correction_square"] = work["state_correction"] ** 2
    cells = work.groupby(["group", "month"], observed=True)[
        ["cross_product", "correction_square"]
    ].sum()
    total_square = float(work["correction_square"].sum())
    if total_square <= 1e-12:
        raise ValueError("State corrections have zero calibration variation.")
    global_lambda = float(work["cross_product"].sum() / total_square)
    median_information = float(cells["correction_square"].median())
    prior_information = float(prior_strength) * median_information
    lambdas = (
        (
            cells["cross_product"]
            + prior_information * global_lambda
        )
        / (cells["correction_square"] + prior_information)
    ).clip(lower=0.0, upper=float(cap))
    lambdas.name = "lambda"
    return lambdas, global_lambda, prior_information


def _apply_group_season_lambdas(
    frame: pd.DataFrame,
    lambdas: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    keys = pd.MultiIndex.from_frame(frame.loc[:, ["group", "month"]])
    row_lambdas = keys.map(lambdas)
    if pd.isna(row_lambdas).any():
        raise ValueError("A target group/month cell has no learned lambda.")
    prediction = (
        frame["direct_auto_ets"].to_numpy(dtype=float)
        + np.asarray(row_lambdas, dtype=float)
        * frame["state_correction"].to_numpy(dtype=float)
    )
    return prediction, np.asarray(row_lambdas, dtype=float)


def learn_reliability_offsets(
    train: pd.DataFrame,
    calibrated_prediction: np.ndarray,
    *,
    quantile: float,
    prior_observations: float,
) -> tuple[pd.Series, float]:
    """Estimate robust residual locations for fixed reliability groups."""
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("residual_quantile must lie between zero and one.")
    if float(prior_observations) < 0.0:
        raise ValueError("offset_prior_observations must be nonnegative.")
    if len(train) != len(calibrated_prediction):
        raise ValueError("Training rows and calibrated predictions must align.")
    work = train.loc[:, ["reliability_group", "actual"]].copy()
    work["residual"] = (
        work["actual"].to_numpy(dtype=float)
        - np.asarray(calibrated_prediction, dtype=float)
    )
    grouped = work.groupby("reliability_group", observed=True)["residual"]
    counts = grouped.count().astype(float)
    locations = grouped.quantile(float(quantile))
    global_location = float(work["residual"].quantile(float(quantile)))
    prior = float(prior_observations)
    offsets = (
        counts * locations + prior * global_location
    ) / (counts + prior)
    offsets.name = "residual_offset"
    return offsets, global_location


def learn_state_residual_variances(
    train: pd.DataFrame,
    *,
    prior_observations: float,
) -> tuple[pd.Series, pd.Series, float]:
    """Estimate series variances from prior OOS state-forecast residuals.

    Series estimates are shrunk toward their fixed reliability-group estimate,
    and group estimates are shrunk toward the panel estimate. All residuals in
    ``train`` must precede the forecast origin supplied by the caller.
    """
    required = {
        "series_id",
        "reliability_group",
        "error_target",
        "state_correction",
    }
    missing = sorted(required.difference(train.columns))
    if missing:
        raise ValueError(f"State-variance input is missing columns: {missing}")
    prior = float(prior_observations)
    if prior < 0.0:
        raise ValueError("variance_prior_observations must be nonnegative.")
    if train.empty:
        raise ValueError("State-variance input is empty.")

    work = train.loc[:, [
        "series_id",
        "reliability_group",
        "error_target",
        "state_correction",
    ]].copy()
    work["state_residual"] = (
        work["error_target"].to_numpy(dtype=float)
        - work["state_correction"].to_numpy(dtype=float)
    )
    if not np.isfinite(work["state_residual"].to_numpy(dtype=float)).all():
        raise ValueError("State-variance residuals must all be finite.")
    series_groups = work.groupby("series_id", observed=True)[
        "reliability_group"
    ].nunique()
    if (series_groups != 1).any():
        raise ValueError("A series changes reliability group within the variance window.")

    global_variance = float(work["state_residual"].var(ddof=1))
    if not np.isfinite(global_variance) or global_variance < 0.0:
        raise ValueError("The panel state-residual variance is unavailable.")

    group_stats = work.groupby("reliability_group", observed=True)[
        "state_residual"
    ].agg(["count", "var"])
    group_information = np.maximum(
        group_stats["count"].to_numpy(dtype=float) - 1.0,
        0.0,
    )
    group_raw = group_stats["var"].to_numpy(dtype=float)
    group_raw = np.where(np.isfinite(group_raw), group_raw, global_variance)
    group_denominator = group_information + prior
    group_values = np.divide(
        group_information * group_raw + prior * global_variance,
        group_denominator,
        out=np.full_like(group_information, global_variance, dtype=float),
        where=group_denominator > 0.0,
    )
    group_variances = pd.Series(
        np.maximum(group_values, 0.0),
        index=group_stats.index,
        name="reliability_group_state_variance",
    )

    series_stats = work.groupby("series_id", observed=True).agg(
        count=("state_residual", "count"),
        variance=("state_residual", "var"),
        reliability_group=("reliability_group", "first"),
    )
    series_information = np.maximum(
        series_stats["count"].to_numpy(dtype=float) - 1.0,
        0.0,
    )
    series_targets = series_stats["reliability_group"].map(
        group_variances
    ).to_numpy(dtype=float)
    series_raw = series_stats["variance"].to_numpy(dtype=float)
    series_raw = np.where(np.isfinite(series_raw), series_raw, series_targets)
    series_denominator = series_information + prior
    series_values = np.divide(
        series_information * series_raw + prior * series_targets,
        series_denominator,
        out=series_targets.copy(),
        where=series_denominator > 0.0,
    )
    series_variances = pd.Series(
        np.maximum(series_values, 0.0),
        index=series_stats.index,
        name=STATE_CORRECTION_VARIANCE,
    )
    return series_variances, group_variances, global_variance


def rolling_recombine(
    state_predictions: pd.DataFrame,
    target_dates: list[pd.Timestamp],
    config: ReliabilityCombinationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply leakage-safe rolling calibration at each requested target month."""
    pieces = []
    lambda_rows = []
    offset_rows = []
    for target_date in sorted(pd.Timestamp(value) for value in target_dates):
        prior_dates = sorted(
            state_predictions.loc[
                state_predictions["date"] < target_date,
                "date",
            ].unique()
        )[-int(config.calibration_window_months):]
        if len(prior_dates) != int(config.calibration_window_months):
            raise ValueError(
                f"Target {target_date.date()} has {len(prior_dates)} prior state "
                f"months; expected {config.calibration_window_months}."
            )
        train = state_predictions[state_predictions["date"].isin(prior_dates)].copy()
        test = state_predictions[state_predictions["date"] == target_date].copy()
        if test.empty:
            raise ValueError(f"No state predictions exist for {target_date.date()}.")
        lambdas, global_lambda, prior_information = learn_group_season_lambdas(
            train,
            prior_strength=config.lambda_prior_strength,
            cap=config.lambda_cap,
        )
        train_prediction, _ = _apply_group_season_lambdas(train, lambdas)
        test_prediction, row_lambdas = _apply_group_season_lambdas(test, lambdas)
        offsets, global_offset = learn_reliability_offsets(
            train,
            train_prediction,
            quantile=config.residual_quantile,
            prior_observations=config.offset_prior_observations,
        )
        state_variances, group_state_variances, global_state_variance = (
            learn_state_residual_variances(
                train,
                prior_observations=config.variance_prior_observations,
            )
        )
        row_offsets = test["reliability_group"].map(offsets)
        if row_offsets.isna().any():
            raise ValueError("A target reliability group has no learned offset.")
        row_state_variances = test["series_id"].map(state_variances)
        if row_state_variances.isna().any():
            raise ValueError("A target series has no learned state variance.")
        row_group_state_variances = test["reliability_group"].map(
            group_state_variances
        )
        if row_group_state_variances.isna().any():
            raise ValueError("A target reliability group has no learned state variance.")

        result = test.copy()
        result["lambda"] = row_lambdas
        result["residual_offset"] = row_offsets.to_numpy(dtype=float)
        result[STATE_CORRECTION_VARIANCE] = row_state_variances.to_numpy(
            dtype=float
        )
        result["reliability_group_state_variance"] = (
            row_group_state_variances.to_numpy(dtype=float)
        )
        result["global_state_variance"] = global_state_variance
        result[CORE_MODEL] = np.maximum(
            test_prediction + result["residual_offset"].to_numpy(dtype=float),
            0.0,
        )
        result["calibration_start"] = pd.Timestamp(prior_dates[0])
        result["calibration_end"] = pd.Timestamp(prior_dates[-1])
        result["n_calibration_months"] = int(len(prior_dates))
        result["global_lambda"] = global_lambda
        result["global_residual_offset"] = global_offset
        pieces.append(result)

        for (group, month), value in lambdas.items():
            lambda_rows.append({
                "target_date": target_date,
                "group": group,
                "month": int(month),
                "lambda": float(value),
                "global_lambda": global_lambda,
                "prior_information": prior_information,
                "calibration_start": pd.Timestamp(prior_dates[0]),
                "calibration_end": pd.Timestamp(prior_dates[-1]),
            })
        for group, value in offsets.items():
            offset_rows.append({
                "target_date": target_date,
                "reliability_group": int(group),
                "residual_offset": float(value),
                "global_residual_offset": global_offset,
                "calibration_start": pd.Timestamp(prior_dates[0]),
                "calibration_end": pd.Timestamp(prior_dates[-1]),
            })
    return (
        pd.concat(pieces, ignore_index=True),
        pd.DataFrame(lambda_rows),
        pd.DataFrame(offset_rows),
    )


def learn_reliability_gates(validation: pd.DataFrame) -> pd.DataFrame:
    """Select core VARA or direct AutoETS by validation MAE in each group."""
    required = {
        "reliability_group",
        "actual",
        "direct_auto_ets",
        CORE_MODEL,
    }
    missing = sorted(required.difference(validation.columns))
    if missing:
        raise ValueError(f"Reliability-gate input is missing columns: {missing}")
    if validation.empty:
        raise ValueError("Reliability-gate validation input is empty.")
    rows = []
    for group, frame in validation.groupby("reliability_group", observed=True):
        direct_mae = float(np.mean(np.abs(
            frame["direct_auto_ets"].to_numpy(dtype=float)
            - frame["actual"].to_numpy(dtype=float)
        )))
        core_mae = float(np.mean(np.abs(
            frame[CORE_MODEL].to_numpy(dtype=float)
            - frame["actual"].to_numpy(dtype=float)
        )))
        rows.append({
            "reliability_group": int(group),
            "gate": float(core_mae < direct_mae),
            "selected_forecast": CORE_MODEL if core_mae < direct_mae else DIRECT_AUTO_ETS,
            "validation_direct_mae": direct_mae,
            "validation_core_mae": core_mae,
            "validation_mae_improvement": direct_mae - core_mae,
        })
    return pd.DataFrame(rows).sort_values("reliability_group").reset_index(drop=True)


def apply_reliability_gates(
    predictions: pd.DataFrame,
    gates: pd.DataFrame,
) -> pd.DataFrame:
    """Apply validation-locked baseline/VARA gates to every forecast row."""
    if gates["reliability_group"].duplicated().any():
        raise ValueError("Reliability gates contain duplicate groups.")
    result = predictions.merge(
        gates.loc[:, ["reliability_group", "gate"]],
        on="reliability_group",
        how="left",
        validate="many_to_one",
    )
    if result["gate"].isna().any():
        raise ValueError("Some forecast rows have no validation-locked gate.")
    result[MODEL] = (
        result["direct_auto_ets"].to_numpy(dtype=float)
        + result["gate"].to_numpy(dtype=float)
        * (
            result[CORE_MODEL].to_numpy(dtype=float)
            - result["direct_auto_ets"].to_numpy(dtype=float)
        )
    )
    return result


def load_autoets_forecast_variances(path: Path) -> pd.DataFrame:
    """Read an original-scale wide AutoETS variance file as a long table."""
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"AutoETS variance file {path} has no date column.")
    frame["date"] = pd.to_datetime(frame["date"])
    if frame["date"].duplicated().any():
        raise ValueError(f"AutoETS variance file {path} has duplicate dates.")
    series_columns = [column for column in frame.columns if column != "date"]
    if not series_columns:
        raise ValueError(f"AutoETS variance file {path} has no series columns.")
    long = frame.melt(
        id_vars="date",
        value_vars=series_columns,
        var_name="series_id",
        value_name=DIRECT_AUTO_ETS_VARIANCE,
    )
    values = long[DIRECT_AUTO_ETS_VARIANCE].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("AutoETS forecast variances must be finite and nonnegative.")
    return long


def attach_plugin_forecast_variances(
    predictions: pd.DataFrame,
    autoets_variances: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach fixed-parameter plug-in variances to gated forecasts."""
    required = {
        "date",
        "series_id",
        "gate",
        "lambda",
        STATE_CORRECTION_VARIANCE,
    }
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Variance propagation input is missing columns: {missing}")
    result = predictions.copy()
    if autoets_variances is None:
        result[DIRECT_AUTO_ETS_VARIANCE] = np.nan
    else:
        required_variance = {"date", "series_id", DIRECT_AUTO_ETS_VARIANCE}
        missing_variance = sorted(
            required_variance.difference(autoets_variances.columns)
        )
        if missing_variance:
            raise ValueError(
                f"AutoETS variance input is missing columns: {missing_variance}"
            )
        if autoets_variances.duplicated(["date", "series_id"]).any():
            raise ValueError("AutoETS variances contain duplicate date/series rows.")
        result = result.merge(
            autoets_variances.loc[:, [
                "date",
                "series_id",
                DIRECT_AUTO_ETS_VARIANCE,
            ]],
            on=["date", "series_id"],
            how="left",
            validate="one_to_one",
        )

    baseline_variance = result[DIRECT_AUTO_ETS_VARIANCE].to_numpy(dtype=float)
    state_variance = result[STATE_CORRECTION_VARIANCE].to_numpy(dtype=float)
    lambdas = result["lambda"].to_numpy(dtype=float)
    gates = result["gate"].to_numpy(dtype=float)
    if not np.isfinite(state_variance).all() or (state_variance < 0.0).any():
        raise ValueError("State-correction variances must be finite and nonnegative.")
    result[CORE_MODEL_VARIANCE] = baseline_variance + lambdas ** 2 * state_variance
    result[MODEL_VARIANCE] = (
        baseline_variance + (gates * lambdas) ** 2 * state_variance
    )
    return result


def _wide(
    frame: pd.DataFrame,
    column: str,
    dates: list[pd.Timestamp],
    series_ids: list[str],
) -> np.ndarray:
    return (
        frame.pivot(index="date", columns="series_id", values=column)
        .reindex(index=dates, columns=series_ids)
        .to_numpy(dtype=float)
    )


def _metric_tables(
    predictions: pd.DataFrame,
    raw: pd.DataFrame,
    *,
    model_columns: dict[str, str],
    variance_columns: dict[str, str] | None,
    train_end: str,
    context: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    dates = sorted(pd.to_datetime(predictions["date"].unique()))
    series_ids = sorted(predictions["series_id"].astype(str).unique())
    forecasts = {
        model: _wide(predictions, column, dates, series_ids)
        for model, column in model_columns.items()
    }
    forecast_variances = None
    if variance_columns is not None:
        forecast_variances = {
            model: _wide(predictions, column, dates, series_ids)
            for model, column in variance_columns.items()
        }
    actual = _wide(predictions, "actual", dates, series_ids)
    train = (
        raw[raw["date"] <= pd.Timestamp(train_end)]
        .sort_values("date")
        .loc[:, series_ids]
        .to_numpy(dtype=float)
    )
    by_series, by_origin, summary = forecast_metric_tables(
        forecasts,
        actual,
        train,
        dates,
        series_ids,
        season_length=12,
        forecast_mode="rolling-origin",
        horizon=len(dates),
        summary_context=context,
        forecast_variances=forecast_variances,
    )
    return by_series, by_origin, summary, {**forecasts, "actual": actual}


def _load_paper_comparator(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "series_id", "rolling_purpose_oos_vara"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Paper comparator is missing columns: {missing}")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.loc[:, list(required)].rename(columns={
        "rolling_purpose_oos_vara": PAPER_MODEL,
    })


def run_recombination(
    *,
    data_path: Path,
    autoets_forecasts_path: Path,
    autoets_variance_path: Path | None,
    vara_results_path: Path,
    strategy_id: str,
    output_dir: Path,
    validation_start: str,
    validation_end: str,
    final_start: str,
    final_end: str,
    paper_final_predictions_path: Path | None,
    config: ReliabilityCombinationConfig,
) -> pd.DataFrame:
    autoets = pd.read_csv(autoets_forecasts_path)
    autoets["date"] = pd.to_datetime(autoets["date"])
    reliability = build_reliability_mapping(
        autoets,
        history_end=config.reliability_history_end,
        n_bins=config.reliability_bins,
    )
    panel, feature_sets = build_state_panel(
        autoets_forecasts_path,
        vara_results_path,
        strategy_id,
        drop_incomplete_state_origins=True,
    )
    state_dates = sorted(
        panel.loc[
            panel["date"].between(
                pd.Timestamp(config.state_history_start),
                pd.Timestamp(final_end),
            ),
            "date",
        ].unique()
    )
    state = forecast_config(
        panel,
        feature_sets,
        StateBlendConfig(
            config.state_window_months,
            config.state_feature_set,
            config.state_ridge_alpha,
        ),
        state_dates,
        blends=(1.0,),
    )
    state["error_target"] = (
        state["actual"].to_numpy(dtype=float)
        - state["direct_auto_ets"].to_numpy(dtype=float)
    )
    state["month"] = state["date"].dt.month.astype(int)
    state = state.merge(
        reliability,
        on="series_id",
        how="left",
        validate="many_to_one",
    )
    if state["reliability_group"].isna().any():
        raise ValueError("Some state predictions lack a reliability group.")
    state["reliability_group"] = state["reliability_group"].astype(int)

    target_dates = sorted(
        state.loc[
            state["date"].between(
                pd.Timestamp(validation_start),
                pd.Timestamp(final_end),
            ),
            "date",
        ].unique()
    )
    predictions, lambdas, offsets = rolling_recombine(state, target_dates, config)
    selection_validation = predictions[predictions["date"].between(
        pd.Timestamp(validation_start),
        pd.Timestamp(validation_end),
    )].copy()
    gates = learn_reliability_gates(selection_validation)
    predictions = apply_reliability_gates(predictions, gates)
    autoets_variances = (
        load_autoets_forecast_variances(autoets_variance_path)
        if autoets_variance_path is not None
        else None
    )
    predictions = attach_plugin_forecast_variances(
        predictions,
        autoets_variances,
    )
    validation = predictions[predictions["date"].between(
        pd.Timestamp(validation_start),
        pd.Timestamp(validation_end),
    )].copy()
    final = predictions[predictions["date"].between(
        pd.Timestamp(final_start),
        pd.Timestamp(final_end),
    )].copy()
    if validation.empty or final.empty:
        raise ValueError("Validation and final windows must both contain forecasts.")
    if (
        autoets_variance_path is not None
        and final[DIRECT_AUTO_ETS_VARIANCE].isna().any()
    ):
        missing_rows = final.loc[
            final[DIRECT_AUTO_ETS_VARIANCE].isna(),
            ["date", "series_id"],
        ]
        raise ValueError(
            "AutoETS variance input does not cover the final forecast panel; "
            f"first missing row is {missing_rows.iloc[0].to_dict()}."
        )
    if not (predictions["calibration_end"] < predictions["date"]).all():
        raise RuntimeError("Rolling calibration includes a target or future month.")
    if not (predictions["training_end"] < predictions["date"]).all():
        raise RuntimeError("State-ridge training includes a target or future month.")
    if not (predictions["autoets_history_end"] < predictions["date"]).all():
        raise RuntimeError("AutoETS training includes a target or future month.")
    if not (predictions["history_end"] < predictions["date"]).all():
        raise RuntimeError("VARA training includes a target or future month.")

    raw = pd.read_csv(data_path)
    raw["date"] = pd.to_datetime(raw["date"])
    validation_models = {
        MODEL: MODEL,
        DIRECT_AUTO_ETS: "direct_auto_ets",
    }
    validation_variances = {
        MODEL: MODEL_VARIANCE,
        DIRECT_AUTO_ETS: DIRECT_AUTO_ETS_VARIANCE,
    }
    validation_tables = _metric_tables(
        validation,
        raw,
        model_columns=validation_models,
        variance_columns=validation_variances,
        train_end=str(pd.Timestamp(validation_start) - pd.offsets.MonthBegin(1)),
        context={
            "period": "validation",
            "strategy_id": strategy_id,
            "purpose_component_series_used": False,
        },
    )

    final_models = dict(validation_models)
    if paper_final_predictions_path is not None:
        comparator = _load_paper_comparator(paper_final_predictions_path)
        final = final.merge(
            comparator,
            on=["date", "series_id"],
            how="left",
            validate="one_to_one",
        )
        if final[PAPER_MODEL].isna().any():
            raise ValueError("Paper comparator does not cover the final panel.")
        final_models[PAPER_MODEL] = PAPER_MODEL
    final_tables = _metric_tables(
        final,
        raw,
        model_columns=final_models,
        variance_columns=validation_variances,
        train_end=str(pd.Timestamp(final_start) - pd.offsets.MonthBegin(1)),
        context={
            "period": "final",
            "strategy_id": strategy_id,
            "purpose_component_series_used": False,
        },
    )
    final_summary = final_tables[2].copy()
    comparator_rmse = (
        float(final_summary.loc[final_summary["model"] == PAPER_MODEL, "rmse"].iloc[0])
        if PAPER_MODEL in final_models
        else np.nan
    )
    final_summary["rmse_improvement_vs_paper_pct"] = (
        np.nan
        if not np.isfinite(comparator_rmse)
        else 100.0 * (comparator_rmse - final_summary["rmse"]) / comparator_rmse
    )

    dm = None
    if PAPER_MODEL in final_models:
        arrays = final_tables[3]
        dm = panel_diebold_mariano_tests(
            {model: arrays[model] for model in final_models},
            arrays["actual"],
            sorted(final["date"].unique()),
            focal_model=MODEL,
            forecast_horizon=1,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output_dir / "rolling_predictions.csv", index=False)
    validation.to_csv(output_dir / "validation_predictions.csv", index=False)
    final.to_csv(output_dir / "final_predictions.csv", index=False)
    reliability.to_csv(output_dir / "reliability_groups.csv", index=False)
    lambdas.to_csv(output_dir / "rolling_lambdas.csv", index=False)
    offsets.to_csv(output_dir / "rolling_offsets.csv", index=False)
    gates.to_csv(output_dir / "reliability_gates.csv", index=False)
    validation_tables[0].to_csv(output_dir / "validation_metrics_by_series.csv", index=False)
    validation_tables[1].to_csv(output_dir / "validation_metrics_by_origin.csv", index=False)
    validation_tables[2].to_csv(output_dir / "validation_metrics_summary.csv", index=False)
    final_tables[0].to_csv(output_dir / "metrics_by_series.csv", index=False)
    final_tables[1].to_csv(output_dir / "metrics_by_origin.csv", index=False)
    final_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    for column, filename in (
        (MODEL_VARIANCE, "forecast_variance.csv"),
        (DIRECT_AUTO_ETS_VARIANCE, "baseline_forecast_variance.csv"),
        (STATE_CORRECTION_VARIANCE, "state_correction_variance.csv"),
    ):
        variance_wide = final.pivot(
            index="date",
            columns="series_id",
            values=column,
        ).sort_index()
        variance_wide.columns.name = None
        variance_wide.reset_index().to_csv(output_dir / filename, index=False)
    if dm is not None:
        dm.to_csv(output_dir / "diebold_mariano_tests.csv", index=False)

    leakage_audit = {
        "all_checks_passed": True,
        "n_series": int(final["series_id"].nunique()),
        "n_final_origins": int(final["date"].nunique()),
        "purpose_component_series_used": False,
        "reliability_history_end": config.reliability_history_end,
        "reliability_precedes_state_history": bool(
            pd.Timestamp(config.reliability_history_end)
            < pd.Timestamp(config.state_history_start)
        ),
        "autoets_history_precedes_target": True,
        "vara_history_precedes_target": True,
        "state_training_precedes_target": True,
        "calibration_training_precedes_target": True,
        "state_variance_training_precedes_target": True,
        "final_autoets_variances_complete": bool(
            final[DIRECT_AUTO_ETS_VARIANCE].notna().all()
        ),
    }
    (output_dir / "leakage_audit.json").write_text(
        json.dumps(leakage_audit, indent=2) + "\n"
    )
    metadata = {
        "model": MODEL,
        "selection_status": "selected from 2013-2015; final year excluded from rule",
        "selection_rule": (
            "lowest validation MAE among core candidates within 1 percent of "
            "the minimum validation RMSE, followed by a reliability-group "
            "core-versus-direct gate selected by validation MAE"
        ),
        "strategy_id": strategy_id,
        "config": asdict(config),
        "data_path": str(data_path),
        "autoets_forecasts_path": str(autoets_forecasts_path),
        "autoets_variance_path": (
            str(autoets_variance_path)
            if autoets_variance_path is not None
            else None
        ),
        "vara_results_path": str(vara_results_path),
        "validation_start": validation_start,
        "validation_end": validation_end,
        "final_start": final_start,
        "final_end": final_end,
        "purpose_component_series_used": False,
        "variance_formula": (
            "autoets_variance + (reliability_gate * lambda)^2 * "
            "state_correction_variance"
        ),
        "variance_covariance_assumption": "zero",
        "variance_parameter_uncertainty_included": False,
    }
    (output_dir / "selection.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(final_summary.to_string(index=False), flush=True)
    if dm is not None:
        print("\nDiebold-Mariano tests:", flush=True)
        print(dm.to_string(index=False), flush=True)
    return final_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/tourism/tourism.csv"))
    parser.add_argument(
        "--autoets-forecasts-path",
        type=Path,
        default=Path("outputs/tourism_76_autoets_extended/rolling_forecasts.csv"),
    )
    parser.add_argument(
        "--autoets-variance-path",
        type=Path,
        default=Path(
            "outputs/tourism_benchmark_standardized_run/rolling-origin/"
            "forecast_variance_auto_ets.csv"
        ),
        help=(
            "Original-scale wide AutoETS predictive variances for the final "
            "forecast period."
        ),
    )
    parser.add_argument(
        "--vara-results-path",
        type=Path,
        default=Path("outputs/tourism_76_vara_extended/origin_results.csv"),
    )
    parser.add_argument(
        "--strategy-id",
        default="predictive_avg_topk10_w24_stable05",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_76_reliability_vara_autoets"),
    )
    parser.add_argument("--validation-start", default="2013-01-01")
    parser.add_argument("--validation-end", default="2015-12-01")
    parser.add_argument("--final-start", default="2016-01-01")
    parser.add_argument("--final-end", default="2016-12-01")
    parser.add_argument(
        "--paper-final-predictions-path",
        type=Path,
        default=None,
        help=(
            "Optional aggregated paper-model forecasts used only as an evaluation "
            "comparator. Omit this argument for a strictly 76-series run."
        ),
    )
    parser.add_argument("--state-window-months", type=int, default=36)
    parser.add_argument("--state-feature-set", default="vara_seasonal10")
    parser.add_argument("--state-ridge-alpha", type=float, default=10.0)
    parser.add_argument("--calibration-window-months", type=int, default=36)
    parser.add_argument("--lambda-prior-strength", type=float, default=3.0)
    parser.add_argument("--lambda-cap", type=float, default=1.25)
    parser.add_argument("--reliability-bins", type=int, default=7)
    parser.add_argument("--residual-quantile", type=float, default=0.5)
    parser.add_argument("--offset-prior-observations", type=float, default=0.0)
    parser.add_argument("--variance-prior-observations", type=float, default=36.0)
    parser.add_argument("--state-history-start", default="2010-01-01")
    parser.add_argument("--reliability-history-end", default="2009-12-01")
    args = parser.parse_args()
    config = ReliabilityCombinationConfig(
        state_window_months=args.state_window_months,
        state_feature_set=args.state_feature_set,
        state_ridge_alpha=args.state_ridge_alpha,
        calibration_window_months=args.calibration_window_months,
        lambda_prior_strength=args.lambda_prior_strength,
        lambda_cap=args.lambda_cap,
        reliability_bins=args.reliability_bins,
        residual_quantile=args.residual_quantile,
        offset_prior_observations=args.offset_prior_observations,
        variance_prior_observations=args.variance_prior_observations,
        state_history_start=args.state_history_start,
        reliability_history_end=args.reliability_history_end,
    )
    run_recombination(
        data_path=args.data_path,
        autoets_forecasts_path=args.autoets_forecasts_path,
        autoets_variance_path=args.autoets_variance_path,
        vara_results_path=args.vara_results_path,
        strategy_id=args.strategy_id,
        output_dir=args.output_dir,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        final_start=args.final_start,
        final_end=args.final_end,
        paper_final_predictions_path=args.paper_final_predictions_path,
        config=config,
    )


if __name__ == "__main__":
    main()
