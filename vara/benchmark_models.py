"""Rolling-origin adapters for reusable statistical and neural benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import os
from pathlib import Path
import time
from typing import Any, Literal

import numpy as np

from .base import BaseForecast, _atomic_write_json, _cache_json_value, _panel_fingerprint
from .data import PanelData, RollingSplit
from .forecast_baselines import (
    AUTO_ETS_DAMPED,
    AUTO_ETS_MODEL_SPECIFICATION,
    AUTO_ETS_PHI,
    NeuralModelConfig,
    auto_arima_baseline_panel,
    auto_ets_baseline_panel,
    nbeats_baseline_panel,
)
from .forecast_contracts import ExecutionConfig


BenchmarkKind = Literal["auto_ets", "auto_arima", "nbeats"]


def _origin_cache(
    cache_dir: str | Path | None,
    *,
    data: PanelData,
    split: RollingSplit,
    kind: BenchmarkKind,
    season_length: int,
    error_history_months: int,
    minimum_history: int,
    model_options: dict[str, Any],
) -> Path | None:
    if cache_dir is None:
        return None
    plan = split.plan(data)
    request = {
        "kind": kind,
        "season_length": int(season_length),
        "error_history_months": int(error_history_months),
        "minimum_history": int(minimum_history),
        "model_options": _cache_json_value(model_options),
        "panel_fingerprint": _panel_fingerprint(data),
        "validation_indices": plan.validation_indices.tolist(),
        "test_indices": plan.test_indices.tolist(),
    }
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":")).encode("utf-8")
    output = Path(cache_dir) / kind / sha256(encoded).hexdigest()[:20]
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(
        output / "request.json",
        {"schema": "vara_rolling_origin_cache_v1", "request": request},
    )
    return output


def _origin_path(cache_dir: Path, target_index: int) -> Path:
    return cache_dir / f"target_{int(target_index):06d}.npz"


def _load_origin(path: Path, component_count: int) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as saved:
            forecast = np.asarray(saved["forecast"], dtype=float)
            variance = np.asarray(saved["forecast_variance"], dtype=float)
            config = json.loads(str(saved["config_json"].item()))
        if forecast.shape != (1, component_count) or variance.shape != forecast.shape:
            return None
        if not isinstance(config, dict):
            return None
        return {
            "forecast": forecast,
            "forecast_variance": variance,
            "config": config,
        }
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _save_origin(path: Path, output: dict[str, Any]) -> None:
    forecast = np.asarray(output["forecast"], dtype=float)
    variance = np.asarray(
        output.get("forecast_variance", np.full(forecast.shape, np.nan)),
        dtype=float,
    )
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(
            handle,
            forecast=forecast,
            forecast_variance=variance,
            config_json=np.asarray(
                json.dumps(_cache_json_value(output.get("config", {})), sort_keys=True)
            ),
        )
    os.replace(temporary, path)


def _fit_one_step(
    target: np.ndarray,
    series_ids: tuple[str, ...],
    target_index: int,
    *,
    kind: BenchmarkKind,
    season_length: int,
    model_options: dict[str, Any],
) -> dict[str, Any]:
    history = np.asarray(target[: int(target_index)], dtype=float)
    try:
        if kind == "auto_ets":
            output = auto_ets_baseline_panel(
                history,
                horizon=1,
                seasonal_lag=int(season_length),
                series_ids=series_ids,
                **model_options,
            )
        elif kind == "auto_arima":
            output = auto_arima_baseline_panel(
                history,
                horizon=1,
                seasonal_lag=int(season_length),
                series_ids=series_ids,
            )
        elif kind == "nbeats":
            output = nbeats_baseline_panel(
                history,
                horizon=1,
                seasonal_lag=int(season_length),
                series_ids=series_ids,
                neural_config=NeuralModelConfig(**model_options),
            )
        else:
            raise ValueError(f"unknown benchmark kind: {kind!r}.")
        forecast = np.asarray(output["forecast"], dtype=float)
        return {
            "forecast": forecast,
            "forecast_variance": np.asarray(
                output.get("forecast_variance", np.full(forecast.shape, np.nan)),
                dtype=float,
            ),
            "config": dict(output.get("config", {})),
        }
    except Exception as exc:
        raise RuntimeError(
            f"{kind} rolling fit failed for target index {int(target_index)}."
        ) from exc


def _rolling_target_indices(
    data: PanelData,
    split: RollingSplit,
    *,
    error_history_months: int,
    minimum_history: int,
) -> np.ndarray:
    evaluation = split.plan(data).evaluation_indices
    start = max(
        int(minimum_history),
        int(evaluation[0]) - int(error_history_months),
    )
    if start > int(evaluation[0]):
        raise ValueError("the panel has insufficient history for the base model.")
    return np.arange(start, int(evaluation[-1]) + 1, dtype=np.int64)


def _assemble_origin_results(
    results: list[tuple[int, dict[str, Any]]],
    *,
    shape: tuple[int, int],
    expected_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    point = np.full(shape, np.nan, dtype=float)
    variance = np.full(shape, np.nan, dtype=float)
    observed: list[int] = []
    model_config: dict[str, Any] = {}
    for target_index, output in sorted(results, key=lambda item: item[0]):
        forecast = np.asarray(output["forecast"], dtype=float)
        forecast_variance = np.asarray(
            output.get("forecast_variance", np.full(forecast.shape, np.nan)),
            dtype=float,
        )
        expected_shape = (1, shape[1])
        if forecast.shape != expected_shape or forecast_variance.shape != expected_shape:
            raise ValueError("one-step base forecast has the wrong panel shape.")
        point[target_index] = forecast[0]
        variance[target_index] = forecast_variance[0]
        observed.append(int(target_index))
        model_config = dict(output.get("config", model_config))
    if not np.array_equal(np.asarray(observed, dtype=np.int64), expected_indices):
        raise ValueError("rolling base forecasts do not exactly cover target indices.")
    return point, variance, model_config


def _rolling_forecast(
    data: PanelData,
    split: RollingSplit,
    execution: ExecutionConfig,
    *,
    kind: BenchmarkKind,
    season_length: int,
    error_history_months: int,
    minimum_history: int,
    model_options: dict[str, Any],
    origin_cache_dir: str | Path | None = None,
    refresh_origin_cache: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    indices = _rolling_target_indices(
        data,
        split,
        error_history_months=error_history_months,
        minimum_history=minimum_history,
    )
    started_at = time.monotonic()
    task_args = (
        np.asarray(data.target),
        tuple(data.series_ids),
    )
    task_kwargs = {
        "kind": kind,
        "season_length": int(season_length),
        "model_options": dict(model_options),
    }
    cache = _origin_cache(
        origin_cache_dir,
        data=data,
        split=split,
        kind=kind,
        season_length=season_length,
        error_history_months=error_history_months,
        minimum_history=minimum_history,
        model_options=model_options,
    )
    results: list[tuple[int, dict[str, Any]]] = []
    missing: list[int] = []
    for target_index in indices:
        cached = (
            None
            if cache is None or refresh_origin_cache
            else _load_origin(
                _origin_path(cache, int(target_index)),
                data.component_count,
            )
        )
        if cached is None:
            missing.append(int(target_index))
        else:
            results.append((int(target_index), cached))
    cache_hits = len(results)

    if not missing:
        backend = "cache"
    elif execution.ray_num_cpus == 1:
        for target_index in missing:
            output = _fit_one_step(
                *task_args,
                target_index,
                **task_kwargs,
            )
            if cache is not None:
                _save_origin(_origin_path(cache, target_index), output)
            results.append((target_index, output))
        backend = "sequential"
    else:
        import ray

        runtime_env = {
            "env_vars": {
                "JAX_PLATFORMS": "cpu",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "OMP_NUM_THREADS": str(execution.max_cpus_per_task),
                "OPENBLAS_NUM_THREADS": str(execution.max_cpus_per_task),
                "MKL_NUM_THREADS": str(execution.max_cpus_per_task),
                "NUMEXPR_NUM_THREADS": str(execution.max_cpus_per_task),
            }
        }
        started_here = False
        pending: dict[Any, int] = {}
        if not ray.is_initialized():
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                num_cpus=execution.ray_num_cpus,
                _node_ip_address="127.0.0.1",
                runtime_env=runtime_env,
            )
            started_here = True
        try:
            remote_fit = ray.remote(
                num_cpus=execution.max_cpus_per_task,
                runtime_env=runtime_env,
            )(_fit_one_step)
            target_ref = ray.put(task_args[0])
            pending = {
                remote_fit.remote(
                    target_ref,
                    task_args[1],
                    target_index,
                    **task_kwargs,
                ): target_index
                for target_index in missing
            }
            while pending:
                ready, _ = ray.wait(list(pending), num_returns=1)
                future = ready[0]
                target_index = pending.pop(future)
                output = ray.get(future)
                if cache is not None:
                    _save_origin(_origin_path(cache, target_index), output)
                results.append((target_index, output))
            backend = "ray"
        except Exception:
            for future in pending:
                try:
                    ray.cancel(future, force=True)
                except Exception:
                    pass
            raise
        finally:
            if started_here:
                ray.shutdown()

    point, variance, model_config = _assemble_origin_results(
        results,
        shape=data.shape,
        expected_indices=indices,
    )
    diagnostics = {
        "forecast_protocol": "expanding_rolling_origin_one_step",
        "forecast_start_index": int(indices[0]),
        "forecast_end_index": int(indices[-1]),
        "forecast_origin_count": int(len(indices)),
        "error_history_months": int(error_history_months),
        "season_length": int(season_length),
        "backend": backend,
        "ray_num_cpus": int(execution.ray_num_cpus),
        "max_cpus_per_task": int(execution.max_cpus_per_task),
        "origin_cache_dir": None if cache is None else cache.as_posix(),
        "origin_cache_hits": int(cache_hits),
        "origin_cache_misses": int(len(missing)),
        "elapsed_seconds": float(time.monotonic() - started_at),
        "model_config": model_config,
    }
    return point, variance, diagnostics


@dataclass(frozen=True, slots=True)
class AutoETSForecaster:
    """Expanding one-step AutoETS forecasts using the shared specification."""

    season_length: int = 12
    error_history_months: int = 240
    model_specification: str = AUTO_ETS_MODEL_SPECIFICATION
    damped: bool | None = AUTO_ETS_DAMPED
    phi: float | None = AUTO_ETS_PHI
    response_normalization: str = "none"
    origin_cache_dir: str | Path | None = None
    refresh_origin_cache: bool = False
    name: str = "auto_ets"

    def __post_init__(self) -> None:
        from .forecast_baselines import auto_ets_config

        auto_ets_config(
            self.season_length,
            model_specification=self.model_specification,
            damped=self.damped,
            phi=self.phi,
        )
        if self.response_normalization not in {"none", "training_zscore"}:
            raise ValueError(
                "response_normalization must be none or training_zscore."
            )
        if int(self.error_history_months) < 1:
            raise ValueError("error_history_months must be positive.")

    def rolling_forecast(
        self,
        data: PanelData,
        split: RollingSplit,
        execution: ExecutionConfig,
    ) -> BaseForecast:
        point, variance, diagnostics = _rolling_forecast(
            data,
            split,
            execution,
            kind="auto_ets",
            season_length=self.season_length,
            error_history_months=self.error_history_months,
            minimum_history=2,
            model_options={
                "model_specification": self.model_specification,
                "damped": self.damped,
                "phi": self.phi,
                "response_normalization": self.response_normalization,
            },
            origin_cache_dir=self.origin_cache_dir,
            refresh_origin_cache=self.refresh_origin_cache,
        )
        return BaseForecast(
            name=self.name,
            point=point,
            error=np.asarray(data.target) - point,
            variance=variance,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True, slots=True)
class AutoARIMAForecaster:
    """Expanding one-step StatsForecast AutoARIMA forecasts."""

    season_length: int = 12
    error_history_months: int = 240
    origin_cache_dir: str | Path | None = None
    refresh_origin_cache: bool = False
    name: str = "auto_arima"

    def __post_init__(self) -> None:
        if int(self.season_length) < 1:
            raise ValueError("season_length must be positive.")
        if int(self.error_history_months) < 1:
            raise ValueError("error_history_months must be positive.")

    def rolling_forecast(
        self,
        data: PanelData,
        split: RollingSplit,
        execution: ExecutionConfig,
    ) -> BaseForecast:
        point, variance, diagnostics = _rolling_forecast(
            data,
            split,
            execution,
            kind="auto_arima",
            season_length=self.season_length,
            error_history_months=self.error_history_months,
            minimum_history=3,
            model_options={},
            origin_cache_dir=self.origin_cache_dir,
            refresh_origin_cache=self.refresh_origin_cache,
        )
        return BaseForecast(
            name=self.name,
            point=point,
            error=np.asarray(data.target) - point,
            variance=variance,
            diagnostics=diagnostics,
        )


@dataclass(frozen=True, slots=True)
class NBEATSForecaster:
    """Expanding one-step pooled N-BEATS forecasts."""

    season_length: int = 12
    error_history_months: int = 240
    neural_config: NeuralModelConfig = field(default_factory=NeuralModelConfig)
    origin_cache_dir: str | Path | None = None
    refresh_origin_cache: bool = False
    name: str = "nbeats"

    def __post_init__(self) -> None:
        if int(self.season_length) < 1:
            raise ValueError("season_length must be positive.")
        if int(self.error_history_months) < 1:
            raise ValueError("error_history_months must be positive.")
        if not isinstance(self.neural_config, NeuralModelConfig):
            raise TypeError("neural_config must be a NeuralModelConfig.")

    def rolling_forecast(
        self,
        data: PanelData,
        split: RollingSplit,
        execution: ExecutionConfig,
    ) -> BaseForecast:
        input_size = (
            self.neural_config.input_size
            if self.neural_config.input_size > 0
            else max(3, 2 * self.season_length)
        )
        point, variance, diagnostics = _rolling_forecast(
            data,
            split,
            execution,
            kind="nbeats",
            season_length=self.season_length,
            error_history_months=self.error_history_months,
            minimum_history=int(input_size) + 1,
            model_options=asdict(self.neural_config),
            origin_cache_dir=self.origin_cache_dir,
            refresh_origin_cache=self.refresh_origin_cache,
        )
        return BaseForecast(
            name=self.name,
            point=point,
            error=np.asarray(data.target) - point,
            variance=variance,
            diagnostics=diagnostics,
        )
