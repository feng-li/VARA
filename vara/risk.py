"""Risk-response construction and risk-specific forecast evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .data import PanelData, RollingSplit
from .datamine import evaluate_risk_forecasts
from .evaluation import EvaluationResult


def log_risk_target(returns: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    """Return ``log(epsilon + return**2)`` while preserving missing cells."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("returns must have shape (T, p).")
    if not np.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive.")
    return np.log(float(epsilon) + np.square(values))


@dataclass(frozen=True, slots=True)
class RiskForecastEvaluator:
    """Evaluate log-risk forecasts and validation-calibrated return variance."""

    benchmark_name: str | None = None
    returns_field: str = "returns"
    epsilon: float = 1e-6
    variance_floor: float = 1e-8

    def evaluate(
        self,
        data: PanelData,
        split: RollingSplit,
        forecasts: Mapping[str, np.ndarray],
    ) -> EvaluationResult:
        if self.returns_field not in data.fields:
            raise ValueError(
                f"risk evaluation requires panel field {self.returns_field!r}."
            )
        if len(forecasts) < 2:
            raise ValueError("risk evaluation requires at least two forecasts.")
        benchmark = self.benchmark_name or next(iter(forecasts))
        if benchmark not in forecasts:
            raise ValueError(f"risk benchmark {benchmark!r} is not a forecast method.")
        for name, forecast in forecasts.items():
            if np.asarray(forecast).shape != data.shape:
                raise ValueError(
                    f"forecast {name!r} has shape {np.asarray(forecast).shape}; "
                    f"expected {data.shape}."
                )
        plan = split.plan(data)
        metrics, monthly, calibration = evaluate_risk_forecasts(
            target=data.target,
            returns=data.fields[self.returns_field],
            dates=data.dates,
            forecasts=forecasts,
            validation_indices=plan.validation_indices,
            test_indices=plan.test_indices,
            benchmark_name=benchmark,
            epsilon=float(self.epsilon),
            variance_floor=float(self.variance_floor),
        )
        return EvaluationResult(
            metrics=metrics,
            by_origin=monthly,
            details={
                "benchmark": benchmark,
                "variance_calibration": calibration,
                "returns_field": self.returns_field,
            },
        )
