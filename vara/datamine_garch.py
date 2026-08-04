"""GARCH benchmarks for the Accounting-Signal Risk application.

The functions in this module deliberately keep parameter estimation separate
from the Ray orchestration in :mod:`scripts.run_datamine_t_garch`.  This makes
the statistical core usable in serial tests while the application runner can
parallelize independent component blocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StudentTGARCH11Fit:
    """One fitted Student-t GARCH(1,1) and its second-moment path."""

    second_moment: np.ndarray
    parameters: dict[str, Any]


def _finite_location_variance(
    values: np.ndarray,
    *,
    variance_floor: float,
) -> tuple[float, float, int]:
    """Return a finite sample mean and variance with a deterministic floor."""
    observed = np.asarray(values, dtype=float)
    observed = observed[np.isfinite(observed)]
    count = int(observed.size)
    if count == 0:
        return 0.0, float(variance_floor), 0
    mean = float(np.mean(observed))
    variance = float(np.mean(np.square(observed - mean)))
    if not np.isfinite(variance) or variance <= float(variance_floor):
        variance = float(variance_floor)
    return mean, variance, count


def _fallback_student_t_parameters(
    estimation: np.ndarray,
    *,
    variance_floor: float,
) -> tuple[dict[str, float], np.ndarray]:
    """Construct a stable variance-targeted t-GARCH fallback.

    The fallback is used only when a series is too short or the optimizer does
    not return an admissible solution.  It remains a Student-t GARCH(1,1), but
    its parameters are fixed and the status is exposed in the output table.
    """
    mean, variance, _ = _finite_location_variance(
        estimation,
        variance_floor=variance_floor,
    )
    alpha = 0.05
    beta = 0.90
    omega = max(
        (1.0 - alpha - beta) * variance,
        float(np.finfo(float).tiny),
    )
    degrees_of_freedom = 8.0
    finite_indices = np.flatnonzero(np.isfinite(estimation))
    conditional = np.empty(len(finite_indices), dtype=float)
    if len(finite_indices):
        conditional[0] = variance
        for position in range(1, len(finite_indices)):
            previous = float(estimation[finite_indices[position - 1]])
            innovation_squared = (previous - mean) ** 2
            conditional[position] = max(
                omega + alpha * innovation_squared + beta * conditional[position - 1],
                float(variance_floor),
            )
    return (
        {
            "mean_return": mean,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "degrees_of_freedom": degrees_of_freedom,
        },
        conditional,
    )


def _admissible_student_t_parameters(parameters: dict[str, float]) -> bool:
    """Check the constraints required for stable forward recursion."""
    mean = parameters["mean_return"]
    omega = parameters["omega"]
    alpha = parameters["alpha"]
    beta = parameters["beta"]
    degrees_of_freedom = parameters["degrees_of_freedom"]
    return bool(
        np.isfinite(mean)
        and np.isfinite(omega)
        and omega > 0.0
        and np.isfinite(alpha)
        and alpha >= 0.0
        and np.isfinite(beta)
        and beta >= 0.0
        and alpha + beta <= 1.000001
        and np.isfinite(degrees_of_freedom)
        and degrees_of_freedom > 2.0
    )


def fit_student_t_garch11_series(
    returns: np.ndarray,
    *,
    estimation_end_index: int,
    minimum_observations: int = 120,
    return_scale: float = 100.0,
    variance_floor: float = 1e-10,
    max_iterations: int = 300,
    tolerance: float = 1e-8,
) -> StudentTGARCH11Fit:
    """Fit one constant-mean GARCH(1,1) with standardized t innovations.

    Only finite returns through ``estimation_end_index`` enter the likelihood.
    Parameters are then frozen.  Later observations update the conditional
    variance recursion, and a missing lagged return is integrated out by using
    its conditional expected squared innovation.

    Returns are multiplied by ``return_scale`` during numerical optimization
    and all location/variance parameters are converted back to the original
    return scale before forecasting.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1:
        raise ValueError("returns must be one-dimensional for a component fit.")
    end = int(estimation_end_index)
    if end < 1 or end >= len(values):
        raise ValueError("estimation_end_index must precede at least one forecast row.")
    if int(minimum_observations) < 24:
        raise ValueError("minimum_observations must be at least 24.")
    if not np.isfinite(return_scale) or float(return_scale) <= 0.0:
        raise ValueError("return_scale must be finite and positive.")
    if not np.isfinite(variance_floor) or float(variance_floor) <= 0.0:
        raise ValueError("variance_floor must be finite and positive.")
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be positive.")
    if not np.isfinite(tolerance) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be finite and positive.")

    estimation = values[: end + 1]
    finite_indices = np.flatnonzero(np.isfinite(estimation))
    observed = estimation[finite_indices]
    fit_status = "success"
    convergence_flag = -1
    optimizer_iterations = 0
    loglikelihood = np.nan
    aic = np.nan
    bic = np.nan
    used_fallback = False
    fit_error = ""
    parameters: dict[str, float]
    fitted_conditional: np.ndarray

    if len(observed) >= int(minimum_observations):
        try:
            # Import lazily so the core package does not import the optimizer in
            # processes that never request this benchmark.
            from arch import arch_model

            scaled = np.ascontiguousarray(observed * float(return_scale))
            model = arch_model(
                scaled,
                mean="Constant",
                vol="GARCH",
                p=1,
                o=0,
                q=1,
                dist="StudentsT",
                rescale=False,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(
                    disp="off",
                    show_warning=False,
                    tol=float(tolerance),
                    options={"maxiter": int(max_iterations)},
                )
            scale = float(return_scale)
            fitted = result.params
            parameters = {
                "mean_return": float(fitted["mu"]) / scale,
                "omega": float(fitted["omega"]) / (scale * scale),
                "alpha": float(fitted["alpha[1]"]),
                "beta": float(fitted["beta[1]"]),
                "degrees_of_freedom": float(fitted["nu"]),
            }
            fitted_conditional = (
                np.square(np.asarray(result.conditional_volatility, dtype=float))
                / (scale * scale)
            )
            convergence_flag = int(result.convergence_flag)
            optimizer_iterations = int(
                getattr(result.optimization_result, "nit", 0) or 0
            )
            loglikelihood = float(result.loglikelihood)
            aic = float(result.aic)
            bic = float(result.bic)
            admissible = (
                convergence_flag == 0
                and _admissible_student_t_parameters(parameters)
                and fitted_conditional.shape == (len(observed),)
                and np.isfinite(fitted_conditional).all()
                and (fitted_conditional > 0.0).all()
            )
            if not admissible:
                if convergence_flag != 0:
                    raise RuntimeError(
                        f"optimizer_convergence_flag_{convergence_flag}"
                    )
                raise RuntimeError("non_admissible_optimizer_result")
        except Exception as exc:
            parameters, fitted_conditional = _fallback_student_t_parameters(
                estimation,
                variance_floor=float(variance_floor),
            )
            fit_error = str(exc)
            if isinstance(exc, RuntimeError) and fit_error:
                fit_status = f"fallback:{fit_error}"
            else:
                fit_status = f"fallback:{type(exc).__name__}"
            used_fallback = True
            loglikelihood = np.nan
            aic = np.nan
            bic = np.nan
    else:
        parameters, fitted_conditional = _fallback_student_t_parameters(
            estimation,
            variance_floor=float(variance_floor),
        )
        fit_status = "fallback:insufficient_observations"
        fit_error = (
            f"only {len(observed)} finite pre-validation returns; "
            f"requires {int(minimum_observations)}"
        )
        used_fallback = True

    mean = float(parameters["mean_return"])
    omega = max(float(parameters["omega"]), float(np.finfo(float).tiny))
    alpha = float(parameters["alpha"])
    beta = float(parameters["beta"])
    degrees_of_freedom = float(parameters["degrees_of_freedom"])
    second_moment = np.full(len(values), np.nan, dtype=float)

    if len(finite_indices):
        conditional = np.maximum(
            np.asarray(fitted_conditional, dtype=float),
            float(variance_floor),
        )
        second_moment[finite_indices] = mean * mean + conditional
        last_observed_index = int(finite_indices[-1])
        current_variance = float(conditional[-1])
        for time_index in range(last_observed_index + 1, len(values)):
            previous_return = values[time_index - 1]
            if np.isfinite(previous_return):
                innovation_squared = (float(previous_return) - mean) ** 2
            else:
                # E[epsilon^2 | F] = h for a missing lagged observation.
                innovation_squared = current_variance
            current_variance = (
                omega + alpha * innovation_squared + beta * current_variance
            )
            if not np.isfinite(current_variance):
                _, current_variance, _ = _finite_location_variance(
                    estimation,
                    variance_floor=float(variance_floor),
                )
            current_variance = max(current_variance, float(variance_floor))
            second_moment[time_index] = mean * mean + current_variance

    parameter_row: dict[str, Any] = {
        "mean_return": mean,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "degrees_of_freedom": degrees_of_freedom,
        "student_t_loglikelihood": loglikelihood,
        "aic": aic,
        "bic": bic,
        "estimation_observations": int(len(observed)),
        "estimation_first_finite_index": (
            int(finite_indices[0]) if len(finite_indices) else -1
        ),
        "estimation_last_finite_index": (
            int(finite_indices[-1]) if len(finite_indices) else -1
        ),
        "convergence_flag": int(convergence_flag),
        "optimizer_iterations": int(optimizer_iterations),
        "fit_status": fit_status,
        "fit_error": fit_error,
        "used_fallback": bool(used_fallback),
    }
    return StudentTGARCH11Fit(
        second_moment=second_moment,
        parameters=parameter_row,
    )


def fit_student_t_garch11_block(
    returns: np.ndarray,
    *,
    estimation_end_index: int,
    minimum_observations: int = 120,
    return_scale: float = 100.0,
    variance_floor: float = 1e-10,
    max_iterations: int = 300,
    tolerance: float = 1e-8,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Fit independent Student-t GARCH models for a component block."""
    values = np.asarray(returns)
    if values.ndim != 2:
        raise ValueError("returns must have shape (time, components).")
    second_moment = np.full(values.shape, np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    for component_position in range(values.shape[1]):
        fit = fit_student_t_garch11_series(
            values[:, component_position],
            estimation_end_index=int(estimation_end_index),
            minimum_observations=int(minimum_observations),
            return_scale=float(return_scale),
            variance_floor=float(variance_floor),
            max_iterations=int(max_iterations),
            tolerance=float(tolerance),
        )
        second_moment[:, component_position] = fit.second_moment
        row = dict(fit.parameters)
        row["block_component_position"] = int(component_position)
        rows.append(row)
    return second_moment, pd.DataFrame(rows)
