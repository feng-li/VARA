"""Validated, model-agnostic contracts for rolling panel forecasts.

The contracts in this module describe data passed between a base forecaster and
the shared VARA orchestration layer.  They deliberately keep the saved base
error panel authoritative: callers must supply it, and :class:`VARARunRequest`
checks it against the target and base forecast without recomputing it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .paper_eta import PaperEtaSpecification


TARGET_CONSISTENCY_ATOL = 1e-6
TARGET_CONSISTENCY_RTOL = 1e-7
RESPONSE_STANDARDIZATION_CHOICES = frozenset({"median_mad", "training_zscore"})
ATTENTION_MODE_CHOICES = frozenset({"topk", "dense"})
ATTENTION_CACHE_MODE_CHOICES = frozenset({"exact", "cluster"})


def _numeric_matrix(value: Any, *, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or min(array.shape, default=0) < 1:
        raise ValueError(f"{name} must be a nonempty two-dimensional array.")
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.issubdtype(array.dtype, np.bool_)
        or np.iscomplexobj(array)
    ):
        raise TypeError(f"{name} must contain real numeric values.")
    if np.isinf(array).any():
        raise ValueError(f"{name} cannot contain infinite values; use NaN for missing cells.")
    return array


def _string_tuple(values: Sequence[Any], *, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence, not one string.")
    result = tuple(str(value) for value in values)
    if not result or any(not value.strip() for value in result):
        raise ValueError(f"{name} must contain nonempty values.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique.")
    return result


def _index_array(values: Sequence[int] | np.ndarray, *, name: str, allow_empty: bool) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if raw.size == 0:
        if not allow_empty:
            raise ValueError(f"{name} must be nonempty.")
        return np.asarray(raw, dtype=np.int64)
    if np.issubdtype(raw.dtype, np.bool_) or not np.issubdtype(raw.dtype, np.integer):
        raise TypeError(f"{name} must contain integer indices.")
    indices = np.asarray(raw, dtype=np.int64)
    if np.any(indices < 1):
        raise ValueError(f"{name} must contain positive target indices.")
    if np.any(np.diff(indices) <= 0):
        raise ValueError(f"{name} must be strictly increasing and unique.")
    return indices


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive.")
    return result


def _nonnegative_float(value: float, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number.")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return result


@dataclass(frozen=True, slots=True)
class ForecastPanel:
    """Observed target panel and its immutable row/column identities."""

    dates: Sequence[Any]
    series_ids: Sequence[Any]
    target: np.ndarray
    metadata: pd.DataFrame

    def __post_init__(self) -> None:
        target = _numeric_matrix(self.target, name="target")
        try:
            dates = np.asarray(self.dates, dtype="datetime64[ns]")
        except (TypeError, ValueError) as exc:
            raise TypeError("dates must contain valid date-like values.") from exc
        if dates.ndim != 1 or len(dates) != target.shape[0]:
            raise ValueError("dates must be one-dimensional with one value per target row.")
        if np.isnat(dates).any():
            raise ValueError("dates cannot contain missing values.")
        if len(dates) > 1 and np.any(np.diff(dates) <= np.timedelta64(0, "ns")):
            raise ValueError("dates must be strictly increasing and unique.")

        series_ids = _string_tuple(self.series_ids, name="series_ids")
        if len(series_ids) != target.shape[1]:
            raise ValueError("series_ids must contain one value per target column.")
        if not isinstance(self.metadata, pd.DataFrame):
            raise TypeError("metadata must be a pandas DataFrame.")
        if len(self.metadata) != target.shape[1]:
            raise ValueError("metadata must contain one row per target column.")

        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "series_ids", series_ids)
        object.__setattr__(self, "target", target)

    @property
    def shape(self) -> tuple[int, int]:
        return self.target.shape

    @property
    def time_count(self) -> int:
        return self.target.shape[0]

    @property
    def component_count(self) -> int:
        return self.target.shape[1]


@dataclass(frozen=True, slots=True)
class RollingEvaluationPlan:
    """Leakage-ordered validation targets followed by optional test targets."""

    validation_indices: Sequence[int] | np.ndarray
    test_indices: Sequence[int] | np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )

    def __post_init__(self) -> None:
        validation = _index_array(
            self.validation_indices,
            name="validation_indices",
            allow_empty=False,
        )
        test = _index_array(self.test_indices, name="test_indices", allow_empty=True)
        if test.size and validation[-1] >= test[0]:
            raise ValueError("test_indices must begin after all validation_indices.")
        object.__setattr__(self, "validation_indices", validation)
        object.__setattr__(self, "test_indices", test)

    @property
    def evaluation_indices(self) -> np.ndarray:
        if self.test_indices.size == 0:
            return self.validation_indices
        return np.concatenate((self.validation_indices, self.test_indices))

    @property
    def origin_count(self) -> int:
        return len(self.validation_indices) + len(self.test_indices)


@dataclass(frozen=True, slots=True)
class StaticFeatureSet:
    """One feature row per forecast series, frozen across rolling origins."""

    values: np.ndarray
    names: Sequence[Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        values = _numeric_matrix(self.values, name="feature values")
        if not np.isfinite(values).all():
            raise ValueError("feature values must all be finite.")
        names = _string_tuple(self.names, name="feature names")
        if len(names) != values.shape[1]:
            raise ValueError("feature names must contain one value per feature column.")
        if not isinstance(self.provenance, Mapping):
            raise TypeError("feature provenance must be a mapping.")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def component_count(self) -> int:
        return self.values.shape[0]

    @property
    def feature_count(self) -> int:
        return self.values.shape[1]


@dataclass(frozen=True, slots=True)
class BaseForecastResult:
    """A base forecast and its explicitly supplied genuine one-step errors."""

    model_name: str
    forecast: np.ndarray
    error: np.ndarray

    def __post_init__(self) -> None:
        model_name = str(self.model_name).strip()
        if not model_name:
            raise ValueError("model_name must be nonempty.")
        forecast = _numeric_matrix(self.forecast, name="base forecast")
        error = _numeric_matrix(self.error, name="base error")
        if forecast.shape != error.shape:
            raise ValueError("base forecast and base error must have the same shape.")
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "forecast", forecast)
        object.__setattr__(self, "error", error)

    @property
    def shape(self) -> tuple[int, int]:
        return self.forecast.shape


@dataclass(frozen=True, slots=True)
class PaperEtaConfig:
    """Paper-scale eta settings for ``exp(eta * log(p) * score)``."""

    lower: float = -2.0
    upper: float = 2.0
    grid_size: int = 7
    local_tolerance: float = 1e-4

    def __post_init__(self) -> None:
        for name, value in (("eta lower", self.lower), ("eta upper", self.upper)):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise TypeError(f"{name} must be a real number.")
        lower = float(self.lower)
        upper = float(self.upper)
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError("eta bounds must be finite and satisfy lower < upper.")
        grid_size = _positive_integer(self.grid_size, name="eta grid_size")
        if grid_size < 2:
            raise ValueError("eta grid_size must be at least two.")
        if isinstance(self.local_tolerance, (bool, np.bool_)) or not isinstance(
            self.local_tolerance,
            Real,
        ):
            raise TypeError("eta local_tolerance must be a real number.")
        tolerance = float(self.local_tolerance)
        if not math.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("eta local_tolerance must be finite and positive.")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        object.__setattr__(self, "grid_size", grid_size)
        object.__setattr__(self, "local_tolerance", tolerance)

    def resolve(self, component_count: int) -> PaperEtaSpecification:
        """Resolve paper eta to the application adapter's effective-slope spec."""
        count = _positive_integer(component_count, name="component_count")
        if count < 2:
            raise ValueError("component_count must be at least two for paper eta.")
        # Keep the reusable contract module independent of application scripts at
        # import time.  The concrete adapter is needed only when execution starts.
        from .paper_eta import PaperEtaSpecification

        return PaperEtaSpecification(
            component_count=count,
            lower=self.lower,
            upper=self.upper,
            grid_size=self.grid_size,
            local_tolerance=self.local_tolerance,
        )


@dataclass(frozen=True, slots=True)
class RollingVARAConfig:
    """Shared rolling VARA and bounded-LSE combination settings."""

    error_window_months: int = 240
    response_standardization: str = "median_mad"
    response_clip: float | None = 8.0
    refit_frequency_months: int = 12
    ridge: float = 0.001
    attention_top_k: int = 500
    attention_cache_mode: str = "exact"
    attention_batch_size: int = 512
    attention_mode: str = "topk"
    lambda_window_months: int = 24
    lambda_ridge: float = 0.001
    zero_own_lag_coefficients: bool = False
    eta: PaperEtaConfig = field(default_factory=PaperEtaConfig)

    def __post_init__(self) -> None:
        error_window = _positive_integer(
            self.error_window_months,
            name="error_window_months",
        )
        if error_window < 24:
            raise ValueError("error_window_months must be at least 24.")
        standardization = str(self.response_standardization)
        if standardization not in RESPONSE_STANDARDIZATION_CHOICES:
            raise ValueError(
                "response_standardization must be median_mad or training_zscore."
            )
        clip = self.response_clip
        if clip is not None:
            if isinstance(clip, (bool, np.bool_)) or not isinstance(clip, Real):
                raise TypeError("response_clip must be a real number or None.")
            clip = float(clip)
            if not math.isfinite(clip) or clip <= 0.0:
                raise ValueError("response_clip must be finite and positive or None.")
        refit_frequency = _positive_integer(
            self.refit_frequency_months,
            name="refit_frequency_months",
        )
        ridge = _nonnegative_float(self.ridge, name="ridge")
        attention_top_k = _positive_integer(self.attention_top_k, name="attention_top_k")
        attention_mode = str(self.attention_mode)
        if attention_mode not in ATTENTION_MODE_CHOICES:
            raise ValueError("attention_mode must be topk or dense.")
        cache_mode = str(self.attention_cache_mode)
        if cache_mode not in ATTENTION_CACHE_MODE_CHOICES:
            raise ValueError("attention_cache_mode must be exact or cluster.")
        attention_batch_size = _positive_integer(
            self.attention_batch_size,
            name="attention_batch_size",
        )
        lambda_window = _positive_integer(
            self.lambda_window_months,
            name="lambda_window_months",
        )
        lambda_ridge = _nonnegative_float(self.lambda_ridge, name="lambda_ridge")
        if not isinstance(self.zero_own_lag_coefficients, (bool, np.bool_)):
            raise TypeError("zero_own_lag_coefficients must be Boolean.")
        if not isinstance(self.eta, PaperEtaConfig):
            raise TypeError("eta must be a PaperEtaConfig.")

        object.__setattr__(self, "error_window_months", error_window)
        object.__setattr__(self, "response_standardization", standardization)
        object.__setattr__(self, "response_clip", clip)
        object.__setattr__(self, "refit_frequency_months", refit_frequency)
        object.__setattr__(self, "ridge", ridge)
        object.__setattr__(self, "attention_top_k", attention_top_k)
        object.__setattr__(self, "attention_mode", attention_mode)
        object.__setattr__(self, "attention_cache_mode", cache_mode)
        object.__setattr__(self, "attention_batch_size", attention_batch_size)
        object.__setattr__(self, "lambda_window_months", lambda_window)
        object.__setattr__(self, "lambda_ridge", lambda_ridge)
        object.__setattr__(
            self,
            "zero_own_lag_coefficients",
            bool(self.zero_own_lag_coefficients),
        )


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    """Resource settings for sequential or Ray rolling-refit execution."""

    ray_num_cpus: int = 1
    max_cpus_per_task: int = 1
    ray_debug: bool = False

    def __post_init__(self) -> None:
        ray_num_cpus = _positive_integer(self.ray_num_cpus, name="ray_num_cpus")
        max_cpus_per_task = _positive_integer(
            self.max_cpus_per_task,
            name="max_cpus_per_task",
        )
        if max_cpus_per_task > ray_num_cpus:
            raise ValueError("max_cpus_per_task cannot exceed ray_num_cpus.")
        if not isinstance(self.ray_debug, (bool, np.bool_)):
            raise TypeError("ray_debug must be Boolean.")
        object.__setattr__(self, "ray_num_cpus", ray_num_cpus)
        object.__setattr__(self, "max_cpus_per_task", max_cpus_per_task)
        object.__setattr__(self, "ray_debug", bool(self.ray_debug))


@dataclass(frozen=True, slots=True)
class VARARunRequest:
    """Data-only request passed to one shared base-plus-VARA family run."""

    panel: ForecastPanel
    evaluation: RollingEvaluationPlan
    features: StaticFeatureSet
    base: BaseForecastResult

    def __post_init__(self) -> None:
        if not isinstance(self.panel, ForecastPanel):
            raise TypeError("panel must be a ForecastPanel.")
        if not isinstance(self.evaluation, RollingEvaluationPlan):
            raise TypeError("evaluation must be a RollingEvaluationPlan.")
        if not isinstance(self.features, StaticFeatureSet):
            raise TypeError("features must be a StaticFeatureSet.")
        if not isinstance(self.base, BaseForecastResult):
            raise TypeError("base must be a BaseForecastResult.")
        if self.base.shape != self.panel.shape:
            raise ValueError("base forecast and error must match the target panel shape.")
        if self.features.component_count != self.panel.component_count:
            raise ValueError("feature rows must match the target panel series count.")

        indices = self.evaluation.evaluation_indices
        if indices[-1] >= self.panel.time_count:
            raise ValueError("evaluation indices are outside the target panel.")
        target = np.asarray(self.panel.target)[indices]
        forecast = np.asarray(self.base.forecast)[indices]
        error = np.asarray(self.base.error)[indices]
        expected_error_cells = np.isfinite(target) & np.isfinite(forecast)
        finite_error_cells = np.isfinite(error)
        if not np.array_equal(finite_error_cells, expected_error_cells):
            raise ValueError(
                "base error finite cells must exactly match finite target/forecast cells "
                "at evaluation origins."
            )
        if np.any(expected_error_cells.sum(axis=1) == 0):
            raise ValueError("every evaluation origin must contain a finite forecast error.")
        consistent = np.isclose(
            target[expected_error_cells],
            forecast[expected_error_cells] + error[expected_error_cells],
            rtol=TARGET_CONSISTENCY_RTOL,
            atol=TARGET_CONSISTENCY_ATOL,
        )
        if not consistent.all():
            maximum_error = float(
                np.max(
                    np.abs(
                        target[expected_error_cells]
                        - forecast[expected_error_cells]
                        - error[expected_error_cells]
                    )[~consistent]
                )
            )
            raise ValueError(
                "authoritative base error is inconsistent with target - forecast "
                f"at evaluation origins (maximum absolute discrepancy={maximum_error:.3g})."
            )


__all__ = [
    "BaseForecastResult",
    "ExecutionConfig",
    "ForecastPanel",
    "PaperEtaConfig",
    "RollingEvaluationPlan",
    "RollingVARAConfig",
    "StaticFeatureSet",
    "VARARunRequest",
]
