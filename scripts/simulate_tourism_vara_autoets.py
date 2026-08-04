#!/usr/bin/env python3
"""Tourism-sized simulation comparing AutoETS with VARA--AutoETS.

This is an end-to-end forecasting experiment, not a simulation of abstract
forecast-error pairs.  It generates a VARA panel using the numerical-experiment
DGP in ``docs/VAR_Atten.tex`` and then applies the Tourism forecasting workflow:

1. Simulate 228 observations of 76 series after a burn-in.
2. Produce genuine expanding-origin AutoETS forecasts.
3. Fit VARA to the preceding 120 AutoETS-error vectors at every correction
   origin, using the simulated clustered feature matrix.
4. Estimate a pooled correction weight on 36 completed pre-test origins.
5. Compare AutoETS, unit-weight VARA--AutoETS_O, and tuned
   VARA--AutoETS_lambda on the final 12 origins.

The default DGP uses q=50, five feature clusters, attention scale log(p),
eta=1, component-specific own-lag coefficients from U(-0.2, 0.2),
attention-lag coefficients from U(0.55, 0.75), and innovation variances from
U(0.5, 1), matching the manuscript simulation section.  Both normal and
Laplace innovations are supported.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from vara.forecast_baselines import (
    _statsforecast_long_from_panel,
    auto_ets_config,
    make_auto_ets_model,
)


AUTOETS_ALIAS = "auto_ets"
SUPPORTED_DISTRIBUTIONS = ("normal", "laplace")


@dataclass(frozen=True)
class OriginSchedule:
    """Locations of the error-history, calibration, and final test origins."""

    total_time: int
    history_window: int
    calibration_origins: int
    test_origins: int
    n_autoets_windows: int
    first_autoets_target: int
    correction_start_position: int
    calibration_positions: tuple[int, ...]
    test_positions: tuple[int, ...]


def make_origin_schedule(
    *,
    total_time: int,
    history_window: int,
    calibration_origins: int,
    test_origins: int,
) -> OriginSchedule:
    """Construct a leakage-safe schedule ending at the final observation."""
    for name, value in (
        ("total_time", total_time),
        ("history_window", history_window),
        ("calibration_origins", calibration_origins),
        ("test_origins", test_origins),
    ):
        if int(value) <= 0:
            raise ValueError(f"{name} must be positive.")

    n_windows = (
        int(history_window) + int(calibration_origins) + int(test_origins)
    )
    first_target = int(total_time) - n_windows
    if first_target < 2:
        raise ValueError(
            "The requested history, calibration, and test origins leave fewer "
            "than two observations before the first AutoETS target."
        )
    correction_start = int(history_window)
    calibration_positions = tuple(
        range(correction_start, correction_start + int(calibration_origins))
    )
    test_positions = tuple(
        range(
            correction_start + int(calibration_origins),
            n_windows,
        )
    )
    return OriginSchedule(
        total_time=int(total_time),
        history_window=int(history_window),
        calibration_origins=int(calibration_origins),
        test_origins=int(test_origins),
        n_autoets_windows=n_windows,
        first_autoets_target=first_target,
        correction_start_position=correction_start,
        calibration_positions=calibration_positions,
        test_positions=test_positions,
    )


def parse_distributions(value: str | Iterable[str]) -> list[str]:
    """Parse and validate a comma-separated innovation-distribution list."""
    if isinstance(value, str):
        values = [piece.strip().lower() for piece in value.split(",") if piece.strip()]
    else:
        values = [str(piece).strip().lower() for piece in value]
    if not values:
        raise ValueError("At least one innovation distribution is required.")
    unknown = sorted(set(values).difference(SUPPORTED_DISTRIBUTIONS))
    if unknown:
        raise ValueError(f"Unsupported innovation distributions: {unknown}")
    return list(dict.fromkeys(values))


def draw_paper_dgp(
    *,
    seed: int,
    p: int,
    q: int,
    clusters: int,
    screen_features: bool,
    feature_max_tries: int,
) -> tuple[dict[str, np.ndarray | float], np.ndarray, np.ndarray, dict[str, object]]:
    """Draw the fixed parameters and clustered feature design used in a run."""
    if int(p) < 2 or int(q) < 1:
        raise ValueError("p must be at least two and q must be positive.")
    if int(clusters) < 1 or int(clusters) > int(p):
        raise ValueError("clusters must lie between one and p.")
    if int(feature_max_tries) <= 0:
        raise ValueError("feature_max_tries must be positive.")

    rng = np.random.default_rng(int(seed))
    own_lag = rng.uniform(-0.20, 0.20, size=int(p))
    attention_lag = rng.uniform(0.55, 0.75, size=int(p))
    stationarity_lhs_before = float(
        np.max(np.abs(own_lag)) + np.max(np.abs(attention_lag))
    )
    stationarity_rescaled = stationarity_lhs_before >= 0.98
    if stationarity_rescaled:
        scale = 0.98 / (stationarity_lhs_before + 1e-6)
        own_lag = own_lag * scale
        attention_lag = attention_lag * scale
    stationarity_lhs = float(
        np.max(np.abs(own_lag)) + np.max(np.abs(attention_lag))
    )
    params = {
        "eta_a": 1.0,
        "beta": own_lag,
        "gamma_a": attention_lag,
        "sigma2": rng.uniform(0.50, 1.00, size=int(p)),
    }
    attention_scale = math.log(int(p))

    best: tuple[float, np.ndarray, np.ndarray, dict[str, object]] | None = None
    attempts = int(feature_max_tries) if screen_features else 1
    for attempt in range(1, attempts + 1):
        centers = rng.normal(size=(int(clusters), int(q)))
        labels = rng.integers(0, int(clusters), size=int(p))
        noise = rng.normal(scale=math.sqrt(0.1), size=(int(p), int(q)))
        raw_features = centers[labels] + noise
        features = raw_features / np.maximum(
            np.linalg.norm(raw_features, axis=1, keepdims=True),
            1e-12,
        )
        similarities = features @ features.T
        logits = attention_scale * similarities
        np.fill_diagonal(logits, -np.inf)
        row_max = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - row_max)
        np.fill_diagonal(exp_logits, 0.0)
        attention = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        off_diagonal = similarities[~np.eye(int(p), dtype=bool)]
        weighted_similarity = np.sum(attention * similarities, axis=1)
        attention_derivative = (
            attention_scale
            * attention
            * (similarities - weighted_similarity[:, None])
        )
        eta_signal = float(
            np.linalg.norm(attention_lag[:, None] * attention_derivative)
        )
        effective_neighbors = 1.0 / np.maximum(
            np.sum(attention * attention, axis=1),
            1e-12,
        )
        variance_proxy = params["sigma2"] / np.maximum(
            1.0 - np.minimum(
                (np.abs(own_lag) + np.abs(attention_lag)) ** 2,
                0.99,
            ),
            1e-6,
        )
        attention_x2_std = np.sqrt(
            np.maximum(attention * attention @ variance_proxy, 0.0)
        )
        offdiag_std = float(np.std(off_diagonal))
        score_tail = float(np.quantile(np.abs(off_diagonal), 0.999))
        x2_std_mean = float(np.mean(attention_x2_std))
        effective_neighbors_mean = float(np.mean(effective_neighbors))
        accepted = bool(
            offdiag_std > 0.15
            and score_tail < 5.0
            and eta_signal > 0.05
            and x2_std_mean > 0.02
        )
        selection_score = (
            eta_signal
            * max(x2_std_mean, 1e-12)
            / math.sqrt(max(effective_neighbors_mean, 1.0))
        )
        feature_info = {
            "accepted": accepted,
            "attempt": attempt,
            "offdiag_std": offdiag_std,
            "abs_score_quantile_0.999": score_tail,
            "eta_signal": eta_signal,
            "attention_x2_std_mean": x2_std_mean,
            "attention_effective_neighbors_mean": effective_neighbors_mean,
            "selection_score": selection_score,
        }
        candidate = (selection_score, features, attention, feature_info)
        if best is None or (
            accepted and not bool(best[3]["accepted"])
        ) or (
            accepted == bool(best[3]["accepted"]) and selection_score > best[0]
        ):
            best = candidate
    if best is None:
        raise RuntimeError("No clustered feature candidate was generated.")
    _, features, attention, feature_info = best

    diagnostics: dict[str, object] = {
        "screened": bool(screen_features),
        "feature_info": feature_info,
        "stationarity_rescaled": stationarity_rescaled,
        "stationarity_lhs": stationarity_lhs,
        "attention_scale": attention_scale,
        "attention_effective_neighbors_mean": float(
            np.mean(
                1.0
                / np.maximum(
                    np.sum(attention ** 2, axis=1),
                    1e-12,
                )
            )
        ),
    }
    return params, features, attention, diagnostics


def simulate_panel(
    *,
    seed: int,
    params: dict[str, np.ndarray | float],
    features: np.ndarray,
    attention: np.ndarray,
    total_time: int,
    burn_in: int,
    distribution: str,
) -> np.ndarray:
    """Simulate and burn in one Tourism-sized VARA panel."""
    if distribution not in SUPPORTED_DISTRIBUTIONS:
        raise ValueError(f"Unsupported innovation distribution: {distribution}")
    if int(total_time) <= 0 or int(burn_in) < 0:
        raise ValueError("total_time must be positive and burn_in nonnegative.")
    p = int(features.shape[0])
    n_total = int(total_time) + int(burn_in)
    rng = np.random.default_rng(int(seed))
    if distribution == "normal":
        standardized_innovations = rng.normal(size=(n_total, p))
    else:
        standardized_innovations = rng.laplace(
            scale=1.0 / math.sqrt(2.0),
            size=(n_total, p),
        )
    innovations = (
        standardized_innovations
        * np.sqrt(np.asarray(params["sigma2"], dtype=float))[None, :]
    )
    own_lag = np.asarray(params["beta"], dtype=float)
    attention_lag = np.asarray(params["gamma_a"], dtype=float)
    attention_values = np.asarray(attention, dtype=float)
    values = np.zeros((n_total, p), dtype=float)
    for index in range(1, n_total):
        previous = values[index - 1]
        values[index] = (
            own_lag * previous
            + attention_lag * (attention_values @ previous)
            + innovations[index]
        )
    panel = values[int(burn_in):]
    if panel.shape != (int(total_time), p):
        raise RuntimeError(
            f"Simulated panel has shape {panel.shape}; expected {(int(total_time), p)}."
        )
    if not np.isfinite(panel).all():
        raise RuntimeError("Simulated panel contains non-finite observations.")
    return panel


def rolling_autoets_forecasts(
    panel: np.ndarray,
    *,
    n_windows: int,
    season_length: int,
    n_jobs: int,
    refit_every: int,
) -> dict[str, np.ndarray]:
    """Return genuine expanding-origin AutoETS actuals, forecasts, and errors."""
    from statsforecast import StatsForecast

    values = np.asarray(panel, dtype=float)
    if values.ndim != 2:
        raise ValueError("panel must have shape (T, p).")
    if int(n_windows) <= 0 or int(n_windows) >= values.shape[0] - 1:
        raise ValueError("n_windows must be positive and leave two training points.")
    if int(season_length) <= 0 or int(n_jobs) <= 0 or int(refit_every) < 0:
        raise ValueError("season_length/n_jobs must be positive; refit_every nonnegative.")

    series_ids = [f"series_{index + 1:03d}" for index in range(values.shape[1])]
    long = _statsforecast_long_from_panel(values, series_ids)
    refit: bool | int
    if int(refit_every) == 0:
        refit = False
    elif int(refit_every) == 1:
        refit = True
    else:
        refit = int(refit_every)
    sf = StatsForecast(
        models=[
            make_auto_ets_model(
                int(season_length),
                alias=AUTOETS_ALIAS,
            )
        ],
        freq=1,
        n_jobs=min(int(n_jobs), values.shape[1]),
    )
    cross_validation = sf.cross_validation(
        h=1,
        df=long,
        n_windows=int(n_windows),
        step_size=1,
        refit=refit,
    )
    expected_targets = np.arange(
        values.shape[0] - int(n_windows),
        values.shape[0],
    )
    actual = (
        cross_validation.pivot(
            index="ds",
            columns="unique_id",
            values="y",
        )
        .sort_index()
        .loc[:, series_ids]
    )
    forecast = (
        cross_validation.pivot(
            index="ds",
            columns="unique_id",
            values=AUTOETS_ALIAS,
        )
        .sort_index()
        .loc[:, series_ids]
    )
    if not np.array_equal(actual.index.to_numpy(dtype=int), expected_targets):
        raise RuntimeError("AutoETS cross-validation returned unexpected targets.")
    cutoff_check = cross_validation.loc[:, ["ds", "cutoff"]].drop_duplicates()
    if not (
        cutoff_check["cutoff"].to_numpy(dtype=int)
        < cutoff_check["ds"].to_numpy(dtype=int)
    ).all():
        raise RuntimeError("An AutoETS cutoff does not precede its forecast target.")

    actual_values = actual.to_numpy(dtype=float)
    forecast_values = forecast.to_numpy(dtype=float)
    if not (
        np.isfinite(actual_values).all()
        and np.isfinite(forecast_values).all()
    ):
        raise RuntimeError("AutoETS cross-validation returned non-finite values.")
    return {
        "target_indices": expected_targets,
        "actual": actual_values,
        "forecast": forecast_values,
        "errors": actual_values - forecast_values,
    }


def fit_vara_error_correction(
    error_history: np.ndarray,
    features: np.ndarray,
    *,
    grid_size: int,
    ridge: float,
    eta_lower: float,
    eta_upper: float,
    stability_cap: float | None,
) -> tuple[np.ndarray, dict[str, float | bool]]:
    """Fit one VARA model to prior AutoETS errors and forecast the next error."""
    history = np.asarray(error_history, dtype=float)
    feature_values = np.asarray(features, dtype=float)
    if history.ndim != 2 or feature_values.ndim != 2:
        raise ValueError("error_history and features must both be matrices.")
    if history.shape[1] != feature_values.shape[0]:
        raise ValueError("Error-history columns must match feature rows.")
    if history.shape[0] < 2:
        raise ValueError("At least two error vectors are required.")
    mean = np.mean(history, axis=0, keepdims=True)
    scale = np.std(history, axis=0, keepdims=True)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (history - mean) / scale

    import jax.numpy as jnp

    from vara.forecast import VARAForecaster

    p = history.shape[1]
    forecaster = VARAForecaster(
        params_scale={"eta_a": math.log(p)},
        eta_bounds=(float(eta_lower), float(eta_upper)),
        grid_size=int(grid_size),
        ridge=float(ridge),
    )
    forecaster.fit(jnp.asarray(standardized), jnp.asarray(feature_values))
    if stability_cap is not None:
        cap = float(stability_cap)
        if not np.isfinite(cap) or cap <= 0.0:
            raise ValueError("stability_cap must be positive when supplied.")
        alpha = np.asarray(forecaster.alpha_hat, dtype=float)
        beta = np.asarray(forecaster.beta_hat, dtype=float)
        coefficient_size = np.abs(alpha) + np.abs(beta)
        projection = np.minimum(
            1.0,
            cap / np.maximum(coefficient_size, 1e-12),
        )
        forecaster.alpha_hat = jnp.asarray(alpha * projection)
        forecaster.beta_hat = jnp.asarray(beta * projection)

    correction = (
        np.asarray(forecaster.predict(h=1), dtype=float)[0] * scale[0]
        + mean[0]
    )
    alpha = np.asarray(forecaster.alpha_hat, dtype=float)
    beta = np.asarray(forecaster.beta_hat, dtype=float)
    diagnostics = forecaster.diagnostics()
    return correction, {
        "eta_hat": float(diagnostics["eta_hat"]),
        "eta_on_boundary": bool(diagnostics["eta_on_boundary"]),
        "stationarity_proxy": float(np.max(np.abs(alpha) + np.abs(beta))),
    }


def rolling_vara_corrections(
    autoets_errors: np.ndarray,
    features: np.ndarray,
    *,
    history_window: int,
    grid_size: int,
    ridge: float,
    eta_lower: float,
    eta_upper: float,
    stability_cap: float | None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit VARA recursively and forecast every error after the history window."""
    errors = np.asarray(autoets_errors, dtype=float)
    if errors.ndim != 2:
        raise ValueError("autoets_errors must have shape (origins, p).")
    if int(history_window) < 2 or int(history_window) >= errors.shape[0]:
        raise ValueError("history_window must be at least two and below origins.")

    corrections = np.full_like(errors, np.nan, dtype=float)
    diagnostic_rows = []
    for position in range(int(history_window), errors.shape[0]):
        correction, diagnostics = fit_vara_error_correction(
            errors[position - int(history_window):position],
            features,
            grid_size=int(grid_size),
            ridge=float(ridge),
            eta_lower=float(eta_lower),
            eta_upper=float(eta_upper),
            stability_cap=stability_cap,
        )
        corrections[position] = correction
        diagnostic_rows.append({
            "position": int(position),
            **diagnostics,
        })
    return corrections, pd.DataFrame(diagnostic_rows)


def learn_pooled_lambda(
    realized_errors: np.ndarray,
    corrections: np.ndarray,
    *,
    lambda_cap: float,
) -> tuple[float, float]:
    """Learn and clip the no-intercept pooled correction weight."""
    error = np.asarray(realized_errors, dtype=float).ravel()
    correction = np.asarray(corrections, dtype=float).ravel()
    if error.shape != correction.shape or error.size == 0:
        raise ValueError("realized_errors and corrections must be nonempty and aligned.")
    if not (np.isfinite(error).all() and np.isfinite(correction).all()):
        raise ValueError("Lambda inputs must be finite.")
    if not np.isfinite(lambda_cap) or float(lambda_cap) <= 0.0:
        raise ValueError("lambda_cap must be finite and positive.")
    denominator = float(np.dot(correction, correction))
    raw_lambda = (
        float(np.dot(error, correction) / denominator)
        if denominator > 1e-12
        else 0.0
    )
    return float(np.clip(raw_lambda, 0.0, float(lambda_cap))), raw_lambda


def _rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(error, dtype=float) ** 2)))


def _mae(error: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(error, dtype=float))))


def evaluate_replication(
    *,
    distribution: str,
    replication: int,
    autoets: dict[str, np.ndarray],
    corrections: np.ndarray,
    correction_diagnostics: pd.DataFrame,
    schedule: OriginSchedule,
    lambda_cap: float,
) -> tuple[dict[str, float | int | str | bool], pd.DataFrame]:
    """Estimate lambda on calibration origins and evaluate held-out forecasts."""
    calibration = np.asarray(schedule.calibration_positions, dtype=int)
    test = np.asarray(schedule.test_positions, dtype=int)
    errors = np.asarray(autoets["errors"], dtype=float)
    actual = np.asarray(autoets["actual"], dtype=float)
    base = np.asarray(autoets["forecast"], dtype=float)
    correction = np.asarray(corrections, dtype=float)

    lambda_hat, raw_lambda = learn_pooled_lambda(
        errors[calibration],
        correction[calibration],
        lambda_cap=float(lambda_cap),
    )
    test_error_autoets = errors[test]
    test_correction = correction[test]
    test_error_unit = test_error_autoets - test_correction
    test_error_tuned = test_error_autoets - lambda_hat * test_correction
    oracle_lambda, _ = learn_pooled_lambda(
        test_error_autoets,
        test_correction,
        lambda_cap=float(lambda_cap),
    )

    rmse_autoets = _rmse(test_error_autoets)
    rmse_unit = _rmse(test_error_unit)
    rmse_tuned = _rmse(test_error_tuned)
    row: dict[str, float | int | str | bool] = {
        "distribution": distribution,
        "replication": int(replication),
        "lambda_hat": lambda_hat,
        "raw_lambda_hat": raw_lambda,
        "oracle_test_lambda": oracle_lambda,
        "rmse_autoets": rmse_autoets,
        "rmse_unit": rmse_unit,
        "rmse_tuned": rmse_tuned,
        "mae_autoets": _mae(test_error_autoets),
        "mae_unit": _mae(test_error_unit),
        "mae_tuned": _mae(test_error_tuned),
        "unit_improvement_pct": 100.0 * (rmse_autoets - rmse_unit) / rmse_autoets,
        "tuned_improvement_pct": 100.0 * (rmse_autoets - rmse_tuned) / rmse_autoets,
        "unit_better": bool(rmse_unit < rmse_autoets),
        "tuned_better": bool(rmse_tuned < rmse_autoets),
        "correction_error_correlation": float(
            np.corrcoef(test_error_autoets.ravel(), test_correction.ravel())[0, 1]
        ),
        "eta_hat_mean": float(correction_diagnostics["eta_hat"].mean()),
        "eta_boundary_rate": float(
            correction_diagnostics["eta_on_boundary"].astype(float).mean()
        ),
        "stationarity_proxy_max": float(
            correction_diagnostics["stationarity_proxy"].max()
        ),
    }

    target_indices = np.asarray(autoets["target_indices"], dtype=int)[test]
    p = errors.shape[1]
    predictions = pd.DataFrame({
        "distribution": distribution,
        "replication": int(replication),
        "target_index": np.repeat(target_indices, p),
        "series_index": np.tile(np.arange(1, p + 1), len(test)),
        "actual": actual[test].ravel(),
        "autoets": base[test].ravel(),
        "vara_correction": test_correction.ravel(),
        "lambda_hat": lambda_hat,
        "vara_autoets_unit": (base[test] + test_correction).ravel(),
        "vara_autoets_tuned": (
            base[test] + lambda_hat * test_correction
        ).ravel(),
    })
    return row, predictions


def summarize_replications(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate equal-sized replication metrics by distribution and overall."""
    required = {
        "distribution",
        "rmse_autoets",
        "rmse_unit",
        "rmse_tuned",
        "mae_autoets",
        "mae_unit",
        "mae_tuned",
        "lambda_hat",
        "oracle_test_lambda",
        "unit_better",
        "tuned_better",
        "eta_hat_mean",
    }
    missing = sorted(required.difference(metrics.columns))
    if missing:
        raise ValueError(f"Replication metrics are missing columns: {missing}")
    if metrics.empty:
        raise ValueError("No replication metrics are available.")

    groups: list[tuple[str, pd.DataFrame]] = [
        (str(name), frame.copy())
        for name, frame in metrics.groupby("distribution", sort=False)
    ]
    if len(groups) > 1:
        groups.append(("pooled", metrics.copy()))

    rows = []
    for distribution, frame in groups:
        rmse_autoets = math.sqrt(float(np.mean(frame["rmse_autoets"] ** 2)))
        rmse_unit = math.sqrt(float(np.mean(frame["rmse_unit"] ** 2)))
        rmse_tuned = math.sqrt(float(np.mean(frame["rmse_tuned"] ** 2)))
        rows.append({
            "distribution": distribution,
            "replications": int(len(frame)),
            "rmse_autoets": rmse_autoets,
            "rmse_unit": rmse_unit,
            "rmse_tuned": rmse_tuned,
            "unit_improvement_pct": (
                100.0 * (rmse_autoets - rmse_unit) / rmse_autoets
            ),
            "tuned_improvement_pct": (
                100.0 * (rmse_autoets - rmse_tuned) / rmse_autoets
            ),
            "mae_autoets": float(frame["mae_autoets"].mean()),
            "mae_unit": float(frame["mae_unit"].mean()),
            "mae_tuned": float(frame["mae_tuned"].mean()),
            "unit_better_rate": float(frame["unit_better"].mean()),
            "tuned_better_rate": float(frame["tuned_better"].mean()),
            "mean_lambda_hat": float(frame["lambda_hat"].mean()),
            "sd_lambda_hat": float(frame["lambda_hat"].std(ddof=1))
            if len(frame) > 1
            else 0.0,
            "mean_oracle_test_lambda": float(frame["oracle_test_lambda"].mean()),
            "mean_eta_hat": float(frame["eta_hat_mean"].mean()),
        })
    return pd.DataFrame(rows)


def format_summary(summary: pd.DataFrame) -> str:
    view = summary.loc[:, [
        "distribution",
        "replications",
        "rmse_autoets",
        "rmse_unit",
        "unit_improvement_pct",
        "rmse_tuned",
        "tuned_improvement_pct",
        "mean_lambda_hat",
        "unit_better_rate",
        "tuned_better_rate",
    ]].copy()
    view.columns = [
        "innovations",
        "R",
        "AutoETS RMSE",
        "unit RMSE",
        "unit gain (%)",
        "tuned RMSE",
        "tuned gain (%)",
        "mean lambda",
        "unit win rate",
        "tuned win rate",
    ]
    return view.to_string(
        index=False,
        formatters={
            "AutoETS RMSE": "{:.4f}".format,
            "unit RMSE": "{:.4f}".format,
            "unit gain (%)": "{:+.2f}".format,
            "tuned RMSE": "{:.4f}".format,
            "tuned gain (%)": "{:+.2f}".format,
            "mean lambda": "{:.3f}".format,
            "unit win rate": "{:.3f}".format,
            "tuned win rate": "{:.3f}".format,
        },
    )


def run_experiment(
    *,
    seed: int,
    replications: int,
    distributions: Iterable[str],
    total_time: int,
    p: int,
    q: int,
    clusters: int,
    burn_in: int,
    history_window: int,
    calibration_origins: int,
    test_origins: int,
    season_length: int,
    autoets_n_jobs: int,
    autoets_refit_every: int,
    grid_size: int,
    ridge: float,
    eta_lower: float,
    eta_upper: float,
    stability_cap: float | None,
    lambda_cap: float,
    screen_features: bool,
    feature_max_tries: int,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray | float],
    np.ndarray,
    dict[str, object],
    OriginSchedule,
]:
    """Run the complete DGP, AutoETS, VARA, calibration, and test workflow."""
    if int(replications) <= 0:
        raise ValueError("replications must be positive.")
    distribution_values = parse_distributions(distributions)
    schedule = make_origin_schedule(
        total_time=int(total_time),
        history_window=int(history_window),
        calibration_origins=int(calibration_origins),
        test_origins=int(test_origins),
    )
    params, features, attention, dgp_diagnostics = draw_paper_dgp(
        seed=int(seed),
        p=int(p),
        q=int(q),
        clusters=int(clusters),
        screen_features=bool(screen_features),
        feature_max_tries=int(feature_max_tries),
    )

    simulation_seeds = np.random.SeedSequence(int(seed) + 1).spawn(
        len(distribution_values) * int(replications)
    )
    autoets_runs = []
    seed_position = 0
    for distribution in distribution_values:
        for replication in range(1, int(replications) + 1):
            started = time.time()
            simulation_seed = int(
                simulation_seeds[seed_position].generate_state(
                    1,
                    dtype=np.uint64,
                )[0]
            )
            seed_position += 1
            panel = simulate_panel(
                seed=simulation_seed,
                params=params,
                features=features,
                attention=attention,
                total_time=int(total_time),
                burn_in=int(burn_in),
                distribution=distribution,
            )
            autoets = rolling_autoets_forecasts(
                panel,
                n_windows=schedule.n_autoets_windows,
                season_length=int(season_length),
                n_jobs=int(autoets_n_jobs),
                refit_every=int(autoets_refit_every),
            )
            autoets_runs.append({
                "distribution": distribution,
                "replication": replication,
                "autoets": autoets,
                "autoets_seconds": float(time.time() - started),
            })
            print(
                f"[{distribution} {replication}/{replications}] "
                "recursive AutoETS forecasts ready",
                flush=True,
            )

    # Keep all process-parallel AutoETS work ahead of the first JAX operation.
    # This avoids forking a process after JAX has initialized worker threads.
    metric_rows = []
    prediction_frames = []
    for run in autoets_runs:
        distribution = str(run["distribution"])
        replication = int(run["replication"])
        autoets = run["autoets"]
        vara_started = time.time()
        corrections, correction_diagnostics = rolling_vara_corrections(
            autoets["errors"],
            np.asarray(features, dtype=float),
            history_window=int(history_window),
            grid_size=int(grid_size),
            ridge=float(ridge),
            eta_lower=float(eta_lower),
            eta_upper=float(eta_upper),
            stability_cap=stability_cap,
        )
        metric, predictions = evaluate_replication(
            distribution=distribution,
            replication=replication,
            autoets=autoets,
            corrections=corrections,
            correction_diagnostics=correction_diagnostics,
            schedule=schedule,
            lambda_cap=float(lambda_cap),
        )
        metric["autoets_seconds"] = float(run["autoets_seconds"])
        metric["vara_seconds"] = float(time.time() - vara_started)
        metric["elapsed_seconds"] = (
            float(metric["autoets_seconds"]) + float(metric["vara_seconds"])
        )
        metric_rows.append(metric)
        prediction_frames.append(predictions)
        print(
            f"[{distribution} {replication}/{replications}] "
            f"AutoETS={metric['rmse_autoets']:.4f}, "
            f"unit={metric['rmse_unit']:.4f}, "
            f"tuned={metric['rmse_tuned']:.4f}, "
            f"lambda={metric['lambda_hat']:.3f}, "
            f"elapsed={metric['elapsed_seconds']:.1f}s",
            flush=True,
        )

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    summary = summarize_replications(metrics)
    return (
        metrics,
        summary,
        predictions,
        params,
        features,
        dgp_diagnostics,
        schedule,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--replications", type=int, default=10)
    parser.add_argument("--distributions", default="normal,laplace")
    parser.add_argument("--total-time", type=int, default=228)
    parser.add_argument("--p", type=int, default=76)
    parser.add_argument("--q", type=int, default=50)
    parser.add_argument("--clusters", type=int, default=5)
    parser.add_argument("--burn-in", type=int, default=100)
    parser.add_argument(
        "--history-window",
        type=int,
        default=120,
        help=(
            "Prior recursive AutoETS-error vectors used by each VARA fit. "
            "The default uses the longest fixed window compatible with a "
            "60-month initial AutoETS history, 36 calibration origins, and "
            "12 final test origins."
        ),
    )
    parser.add_argument("--calibration-origins", type=int, default=36)
    parser.add_argument("--test-origins", type=int, default=12)
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--autoets-n-jobs", type=int, default=76)
    parser.add_argument(
        "--autoets-refit-every",
        type=int,
        default=1,
        help="Refit AutoETS every N origins; 1 means every origin and 0 never refits.",
    )
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--eta-lower", type=float, default=-2.0)
    parser.add_argument("--eta-upper", type=float, default=2.0)
    parser.add_argument(
        "--stability-cap",
        type=float,
        default=None,
        help="Optional cap on |own slope| + |attention slope| after fitting.",
    )
    parser.add_argument("--lambda-cap", type=float, default=1.25)
    parser.add_argument(
        "--screen-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--feature-max-tries", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_size_vara_autoets_simulation"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    distributions = parse_distributions(args.distributions)
    (
        metrics,
        summary,
        predictions,
        params,
        features,
        dgp_diagnostics,
        schedule,
    ) = run_experiment(
        seed=args.seed,
        replications=args.replications,
        distributions=distributions,
        total_time=args.total_time,
        p=args.p,
        q=args.q,
        clusters=args.clusters,
        burn_in=args.burn_in,
        history_window=args.history_window,
        calibration_origins=args.calibration_origins,
        test_origins=args.test_origins,
        season_length=args.season_length,
        autoets_n_jobs=args.autoets_n_jobs,
        autoets_refit_every=args.autoets_refit_every,
        grid_size=args.grid_size,
        ridge=args.ridge,
        eta_lower=args.eta_lower,
        eta_upper=args.eta_upper,
        stability_cap=args.stability_cap,
        lambda_cap=args.lambda_cap,
        screen_features=args.screen_features,
        feature_max_tries=args.feature_max_tries,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "replication_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    predictions.to_csv(output_dir / "test_predictions.csv", index=False)
    pd.DataFrame({
        "series_index": np.arange(1, int(args.p) + 1),
        "own_lag_alpha": np.asarray(params["beta"], dtype=float),
        "attention_lag_beta": np.asarray(params["gamma_a"], dtype=float),
        "innovation_variance": np.asarray(params["sigma2"], dtype=float),
    }).to_csv(output_dir / "dgp_parameters.csv", index=False)
    feature_frame = pd.DataFrame(
        np.asarray(features, dtype=float),
        columns=[f"feature_{index + 1:02d}" for index in range(int(args.q))],
    )
    feature_frame.insert(0, "series_index", np.arange(1, int(args.p) + 1))
    feature_frame.to_csv(output_dir / "dgp_features.csv", index=False)

    config = {
        "command": " ".join(os.sys.argv),
        "seed": args.seed,
        "replications_per_distribution": args.replications,
        "distributions": distributions,
        "total_time": args.total_time,
        "p": args.p,
        "q": args.q,
        "clusters": args.clusters,
        "burn_in": args.burn_in,
        "history_window": args.history_window,
        "calibration_origins": args.calibration_origins,
        "test_origins": args.test_origins,
        "season_length": args.season_length,
        "autoets_config": auto_ets_config(args.season_length),
        "autoets_n_jobs": args.autoets_n_jobs,
        "autoets_refit_every": args.autoets_refit_every,
        "grid_size": args.grid_size,
        "ridge": args.ridge,
        "eta_bounds": [args.eta_lower, args.eta_upper],
        "stability_cap": args.stability_cap,
        "lambda_cap": args.lambda_cap,
        "screen_features": args.screen_features,
        "feature_max_tries": args.feature_max_tries,
        "schedule": asdict(schedule),
        "dgp": {
            "eta": 1.0,
            "attention_scale": "log(p)",
            "own_lag_alpha": "U(-0.2, 0.2)",
            "attention_lag_beta": "U(0.55, 0.75)",
            "innovation_variance": "U(0.5, 1.0)",
            "feature_design": "clustered normalized Gaussian",
            "feature_noise_variance": 0.1,
        },
        "dgp_diagnostics": dgp_diagnostics,
        "leakage_control": (
            "every AutoETS cutoff and every VARA error-history endpoint "
            "precedes its target; lambda uses only pre-test origins"
        ),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print("\nTourism-sized VARA--AutoETS simulation:", flush=True)
    print(format_summary(summary), flush=True)
    print(f"\nSaved outputs to {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
