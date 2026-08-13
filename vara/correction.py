"""Public residual-correction layer over the rolling VARA engine."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable

import numpy as np
import pandas as pd

from .base import BaseForecast
from .data import PanelData, RollingSplit
from .datamine_vara_runner import RollingVARAFamilyResult, run_vara_family
from .forecast_contracts import (
    ExecutionConfig,
    RollingVARAConfig,
    StaticFeatureSet,
    VARARunRequest,
)


@dataclass(frozen=True, slots=True)
class VARAForecast:
    """Base forecast plus unit- and LSE-weighted VARA corrections."""

    name: str
    base: BaseForecast
    correction: np.ndarray
    lambda_weights: np.ndarray
    unit_forecast: np.ndarray
    combined_forecast: np.ndarray
    eta_refits: pd.DataFrame
    lambda_refits: pd.DataFrame
    execution: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise ValueError("VARA forecast name must be nonempty.")
        shape = self.base.point.shape
        arrays = {
            "correction": self.correction,
            "lambda_weights": self.lambda_weights,
            "unit_forecast": self.unit_forecast,
            "combined_forecast": self.combined_forecast,
        }
        for array_name, raw_values in arrays.items():
            values = np.asarray(raw_values)
            if values.shape != shape:
                raise ValueError(f"{array_name} must match the base forecast shape.")
            if (
                not np.issubdtype(values.dtype, np.number)
                or np.issubdtype(values.dtype, np.bool_)
                or np.iscomplexobj(values)
            ):
                raise TypeError(f"{array_name} must contain real numeric values.")
            if np.isinf(values).any():
                raise ValueError(f"{array_name} cannot contain infinite values.")
            object.__setattr__(self, array_name, values)
        if not isinstance(self.eta_refits, pd.DataFrame):
            raise TypeError("eta_refits must be a pandas DataFrame.")
        if not isinstance(self.lambda_refits, pd.DataFrame):
            raise TypeError("lambda_refits must be a pandas DataFrame.")
        if not isinstance(self.execution, Mapping):
            raise TypeError("execution must be a mapping.")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "execution", MappingProxyType(dict(self.execution)))


@dataclass(frozen=True, slots=True)
class VARAResidualCorrector:
    """Apply rolling VARA to genuine errors from any base forecaster."""

    config: RollingVARAConfig = field(default_factory=RollingVARAConfig)
    suffix: str = "+lambda*VARA"

    def forecast(
        self,
        data: PanelData,
        split: RollingSplit,
        features: StaticFeatureSet,
        base: BaseForecast,
        execution: ExecutionConfig,
        *,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> VARAForecast:
        request = VARARunRequest(
            panel=data.as_forecast_panel(),
            evaluation=split.plan(data),
            features=features,
            base=base.as_contract(),
        )
        result: RollingVARAFamilyResult = run_vara_family(
            request,
            self.config,
            execution=execution,
            progress_callback=progress_callback,
        )
        return VARAForecast(
            name=f"{base.name}{self.suffix}",
            base=base,
            correction=np.asarray(result.correction),
            lambda_weights=np.asarray(result.lambda_weights),
            unit_forecast=np.asarray(result.unit_forecast),
            combined_forecast=np.asarray(result.lambda_forecast),
            eta_refits=result.core_fit_summary.copy(),
            lambda_refits=result.lambda_refit_fits.copy(),
            execution=dict(result.correction_execution),
        )
