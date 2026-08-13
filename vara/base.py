"""Standard base-forecast interfaces and minimal adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol

import numpy as np

from .data import PanelData, RollingSplit
from .forecast_contracts import BaseForecastResult, ExecutionConfig


@dataclass(frozen=True, slots=True)
class BaseForecast:
    """Standard output of a rolling base forecaster."""

    name: str
    point: np.ndarray
    error: np.ndarray
    variance: np.ndarray | None = None
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = BaseForecastResult(self.name, self.point, self.error)
        variance = None if self.variance is None else np.asarray(self.variance)
        if variance is not None:
            if variance.shape != contract.shape:
                raise ValueError("base forecast variance must match the point forecast shape.")
            if (
                not np.issubdtype(variance.dtype, np.number)
                or np.issubdtype(variance.dtype, np.bool_)
                or np.iscomplexobj(variance)
            ):
                raise TypeError("base forecast variance must contain real numeric values.")
            if np.isinf(variance).any() or np.any(variance[np.isfinite(variance)] < 0.0):
                raise ValueError("base forecast variance must be nonnegative or NaN.")
        if not isinstance(self.diagnostics, Mapping):
            raise TypeError("base forecast diagnostics must be a mapping.")
        object.__setattr__(self, "name", contract.model_name)
        object.__setattr__(self, "point", contract.forecast)
        object.__setattr__(self, "error", contract.error)
        object.__setattr__(self, "variance", variance)
        object.__setattr__(
            self,
            "diagnostics",
            MappingProxyType(dict(self.diagnostics)),
        )

    def as_contract(self) -> BaseForecastResult:
        return BaseForecastResult(self.name, self.point, self.error)


class BaseForecaster(Protocol):
    """Interface implemented by model-specific rolling forecast runners."""

    name: str

    def rolling_forecast(
        self,
        data: PanelData,
        split: RollingSplit,
        execution: ExecutionConfig,
    ) -> BaseForecast:
        ...


@dataclass(frozen=True, slots=True)
class PrecomputedForecaster:
    """Adapt already-generated causal forecast and error arrays to the API."""

    result: BaseForecast

    @property
    def name(self) -> str:
        return self.result.name

    def rolling_forecast(
        self,
        data: PanelData,
        split: RollingSplit,
        execution: ExecutionConfig,
    ) -> BaseForecast:
        del split, execution
        if self.result.point.shape != data.shape:
            raise ValueError(
                f"precomputed forecast has shape {self.result.point.shape}; "
                f"expected {data.shape}."
            )
        return self.result


@dataclass(frozen=True, slots=True)
class HARRidgeForecaster:
    """Expanding series-wise HAR ridge forecast for a generic panel target."""

    ridge_candidates: Sequence[float] = (1.0,)
    minimum_observations: int = 120
    response_normalization: str = "none"
    name: str = "har_ridge"

    def rolling_forecast(
        self,
        data: PanelData,
        split: RollingSplit,
        execution: ExecutionConfig,
    ) -> BaseForecast:
        del execution
        from .datamine import select_ridge_forecast

        plan = split.plan(data)
        selected, point, selection = select_ridge_forecast(
            data.target,
            plan.validation_indices,
            baseline_type="har_ridge",
            ridge_candidates=self.ridge_candidates,
            minimum_observations=int(self.minimum_observations),
            response_normalization=self.response_normalization,
        )
        return BaseForecast(
            name=self.name,
            point=point,
            error=np.asarray(data.target) - np.asarray(point),
            diagnostics={
                "selected_ridge": float(selected),
                "selection": selection.to_dict(orient="records"),
            },
        )
