"""Standard base-forecast interfaces and minimal adapters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
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


def _cache_json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _cache_json_value(asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _cache_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_cache_json_value(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        digest = sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "sha256": digest,
        }
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _panel_fingerprint(data: PanelData) -> str:
    digest = sha256()
    target = np.ascontiguousarray(data.target)
    digest.update(str(target.dtype).encode("utf-8"))
    digest.update(np.asarray(target.shape, dtype=np.int64).tobytes())
    digest.update(target.view(np.uint8))
    digest.update(np.asarray(data.dates, dtype="datetime64[ns]").view(np.int64).tobytes())
    digest.update(
        json.dumps(list(data.series_ids), ensure_ascii=False).encode("utf-8")
    )
    return digest.hexdigest()


def _cache_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return stem or "model"


def _atomic_save_array(path: Path, values: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, np.asarray(values), allow_pickle=False)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class CachedForecaster:
    """Cache one complete base forecast behind the standard model interface.

    A cache hit requires the same model configuration, panel values and order,
    and validation/test split. The manifest is written last, so an interrupted
    write is treated as a cache miss on the next run.
    """

    model: BaseForecaster
    cache_dir: str | Path
    refresh: bool = False

    @property
    def name(self) -> str:
        return self.model.name

    def _request(self, data: PanelData, split: RollingSplit) -> dict[str, Any]:
        plan = split.plan(data)
        model_config = _cache_json_value(self.model)
        if isinstance(model_config, dict):
            # Checkpoint locations and refresh controls affect execution only,
            # not the fitted forecast represented by the completed cache.
            model_config.pop("origin_cache_dir", None)
            model_config.pop("refresh_origin_cache", None)
        return {
            "model_name": self.name,
            "model_type": (
                f"{type(self.model).__module__}.{type(self.model).__qualname__}"
            ),
            "model_config": model_config,
            "panel_fingerprint": _panel_fingerprint(data),
            "shape": list(data.shape),
            "validation_indices": plan.validation_indices.tolist(),
            "test_indices": plan.test_indices.tolist(),
        }

    def rolling_forecast(
        self,
        data: PanelData,
        split: RollingSplit,
        execution: ExecutionConfig,
    ) -> BaseForecast:
        output = Path(self.cache_dir) / _cache_stem(self.name)
        manifest_path = output / "manifest.json"
        request = self._request(data, split)
        if not self.refresh and manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("request") == request:
                    point = np.load(output / "point.npy", mmap_mode="r", allow_pickle=False)
                    error = np.load(output / "error.npy", mmap_mode="r", allow_pickle=False)
                    variance = (
                        np.load(output / "variance.npy", mmap_mode="r", allow_pickle=False)
                        if manifest.get("has_variance", False)
                        else None
                    )
                    diagnostics = dict(manifest.get("diagnostics", {}))
                    diagnostics.update({
                        "cache_status": "hit",
                        "cache_dir": output.as_posix(),
                    })
                    return BaseForecast(
                        name=self.name,
                        point=point,
                        error=error,
                        variance=variance,
                        diagnostics=diagnostics,
                    )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

        result = self.model.rolling_forecast(data, split, execution)
        if result.name != self.name:
            raise ValueError(
                "the wrapped forecaster returned a different model name: "
                f"{result.name!r} != {self.name!r}."
            )
        output.mkdir(parents=True, exist_ok=True)
        _atomic_save_array(output / "point.npy", result.point)
        _atomic_save_array(output / "error.npy", result.error)
        if result.variance is not None:
            _atomic_save_array(output / "variance.npy", result.variance)
        _atomic_write_json(
            manifest_path,
            {
                "schema": "vara_base_forecast_cache_v1",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "request": request,
                "has_variance": result.variance is not None,
                "diagnostics": _cache_json_value(result.diagnostics),
            },
        )
        diagnostics = dict(result.diagnostics)
        diagnostics.update({
            "cache_status": "miss",
            "cache_dir": output.as_posix(),
        })
        return BaseForecast(
            name=result.name,
            point=result.point,
            error=result.error,
            variance=result.variance,
            diagnostics=diagnostics,
        )


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
