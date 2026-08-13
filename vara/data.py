"""Generic panel data and chronological split contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .forecast_contracts import ForecastPanel, RollingEvaluationPlan


@dataclass(frozen=True, slots=True)
class PanelData:
    """Array-first input for a panel forecasting study.

    ``target`` has shape ``(T, p)``. Optional named fields, such as raw returns,
    must use the same shape. Missing observations are represented by NaN.
    """

    target: np.ndarray
    dates: Sequence[Any]
    series_ids: Sequence[Any]
    metadata: pd.DataFrame
    fields: Mapping[str, np.ndarray] = field(default_factory=dict)

    def __post_init__(self) -> None:
        contract = ForecastPanel(
            dates=self.dates,
            series_ids=self.series_ids,
            target=self.target,
            metadata=self.metadata,
        )
        normalized_fields: dict[str, np.ndarray] = {}
        for raw_name, raw_values in self.fields.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("field names must be nonempty.")
            if name in normalized_fields:
                raise ValueError(f"duplicate panel field: {name!r}.")
            values = np.asarray(raw_values)
            if values.shape != contract.shape:
                raise ValueError(
                    f"panel field {name!r} has shape {values.shape}; "
                    f"expected {contract.shape}."
                )
            if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
                raise TypeError(f"panel field {name!r} must contain real numeric values.")
            if np.isinf(values).any():
                raise ValueError(
                    f"panel field {name!r} cannot contain infinite values; use NaN."
                )
            normalized_fields[name] = values

        object.__setattr__(self, "target", contract.target)
        object.__setattr__(self, "dates", contract.dates)
        object.__setattr__(self, "series_ids", contract.series_ids)
        object.__setattr__(self, "fields", MappingProxyType(normalized_fields))

    @property
    def shape(self) -> tuple[int, int]:
        return self.target.shape

    @property
    def time_count(self) -> int:
        return self.target.shape[0]

    @property
    def component_count(self) -> int:
        return self.target.shape[1]

    def as_forecast_panel(self) -> ForecastPanel:
        """Return the validated low-level VARA panel contract."""
        return ForecastPanel(
            dates=self.dates,
            series_ids=self.series_ids,
            target=self.target,
            metadata=self.metadata,
        )


@dataclass(frozen=True, slots=True)
class RollingSplit:
    """Trailing validation/test split expressed as numbers of target rows."""

    validation: int
    test: int

    def __post_init__(self) -> None:
        for name, value in (("validation", self.validation), ("test", self.test)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{name} must be an integer.")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive.")
            object.__setattr__(self, name, int(value))

    def plan(self, data: PanelData) -> RollingEvaluationPlan:
        """Resolve the trailing lengths to leakage-ordered target indices."""
        if data.time_count <= self.validation + self.test:
            raise ValueError(
                "panel is too short for the requested validation and test periods."
            )
        test_start = data.time_count - self.test
        validation_start = test_start - self.validation
        return RollingEvaluationPlan(
            validation_indices=np.arange(validation_start, test_start, dtype=np.int64),
            test_indices=np.arange(test_start, data.time_count, dtype=np.int64),
        )

