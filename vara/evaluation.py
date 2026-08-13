"""Model-independent point-forecast evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .data import PanelData, RollingSplit
from .forecast_metrics import add_origin_win_rank_score


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: pd.DataFrame
    by_origin: pd.DataFrame
    details: Mapping[str, Any] = field(default_factory=dict)


class ForecastEvaluator(Protocol):
    def evaluate(
        self,
        data: PanelData,
        split: RollingSplit,
        forecasts: Mapping[str, np.ndarray],
    ) -> EvaluationResult:
        ...


@dataclass(frozen=True, slots=True)
class PointForecastEvaluator:
    """Evaluate all methods on one common observed-cell mask."""

    def evaluate(
        self,
        data: PanelData,
        split: RollingSplit,
        forecasts: Mapping[str, np.ndarray],
    ) -> EvaluationResult:
        if len(forecasts) < 2:
            raise ValueError("point evaluation requires at least two forecasts.")
        plan = split.plan(data)
        indices = plan.test_indices
        actual = np.asarray(data.target, dtype=float)[indices]
        methods = list(forecasts)
        common = np.isfinite(actual)
        selected: dict[str, np.ndarray] = {}
        for name, full_forecast in forecasts.items():
            values = np.asarray(full_forecast, dtype=float)
            if values.shape != data.shape:
                raise ValueError(
                    f"forecast {name!r} has shape {values.shape}; expected {data.shape}."
                )
            selected[name] = values[indices]
            common &= np.isfinite(selected[name])
        monthly_cells = common.sum(axis=1)
        if np.any(monthly_cells == 0):
            raise ValueError("every test origin must have at least one common observed cell.")

        metric_rows: list[dict[str, Any]] = []
        monthly_rows: list[pd.DataFrame] = []
        dates = pd.to_datetime(np.asarray(data.dates)[indices])
        for name in methods:
            error = selected[name] - actual
            square = np.square(error)
            absolute = np.abs(error)
            metric_rows.append({
                "model": name,
                "test_cells": int(common.sum()),
                "mse": float(np.mean(square[common])),
                "rmse": float(np.sqrt(np.mean(square[common]))),
                "mae": float(np.mean(absolute[common])),
                "bias": float(np.mean(error[common])),
            })
            monthly_rows.append(pd.DataFrame({
                "date": dates,
                "model": name,
                "observed_cells": monthly_cells,
                "squared_error": np.divide(
                    np.where(common, square, 0.0).sum(axis=1),
                    monthly_cells,
                ),
                "absolute_error": np.divide(
                    np.where(common, absolute, 0.0).sum(axis=1),
                    monthly_cells,
                ),
                "bias": np.divide(
                    np.where(common, error, 0.0).sum(axis=1),
                    monthly_cells,
                ),
            }))

        by_origin = pd.concat(monthly_rows, ignore_index=True)
        metrics, rank_audit = add_origin_win_rank_score(
            pd.DataFrame(metric_rows),
            by_origin,
            error_column="absolute_error",
        )
        by_origin = by_origin.merge(
            rank_audit.loc[:, ["date", "model", "absolute_error_wins", "rank_score_s"]],
            on=["date", "model"],
            validate="one_to_one",
        )
        return EvaluationResult(
            metrics=metrics,
            by_origin=by_origin,
            details={
                "common_matched_cells": int(common.sum()),
                "minimum_cells_per_origin": int(monthly_cells.min()),
                "maximum_cells_per_origin": int(monthly_cells.max()),
                "rank_score": "monthly strict wins on mean absolute error",
            },
        )

