#!/usr/bin/env python3
"""Run forecasting benchmarks on panel time-series data."""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive, RandomWalkWithDrift, SeasonalNaive
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR

from vara.forecast_data import load_panel_data
from vara.forecast_baselines import (
    FORECAST_VARIANCE_LEVEL,
    auto_ets_baseline_panel,
    seasonal_ets_baseline_panel,
    statsmodels_ets_baseline_panel,
)
from vara.forecast_metrics import forecast_metric_tables

try:
    import ray
except Exception as exc:
    ray = None
    RAY_IMPORT_ERROR = exc
else:
    RAY_IMPORT_ERROR = None

BASE_SUFFIX = "All"
DETAILED_CODE_DEPTH = 3
STATSFORECAST_MODELS = {"naive", "seasonal_naive", "drift", "auto_ets", "auto_arima"}
STATSMODELS_MODELS = {
    "ar1",
    "var1",
    "ar1_snaive",
    "var1_snaive",
    "ar1_auto_ets",
    "var1_auto_ets",
    "ar1_seasonal_ets",
    "var1_seasonal_ets",
    "ar1_statsmodels_ets",
    "var1_statsmodels_ets",
    "statsmodels_ets",
    "seasonal_ets",
}
NEURALFORECAST_MODELS = {"deepar", "nbeats"}
PANEL_MODELS = {"var1", "var1_snaive", "var1_auto_ets", "var1_seasonal_ets", "var1_statsmodels_ets", "deepar", "nbeats"}
ALL_MODELS = STATSFORECAST_MODELS | STATSMODELS_MODELS | NEURALFORECAST_MODELS
DEFAULT_MODELS = "naive,seasonal_naive,drift,auto_ets,auto_arima,ar1,var1"
STATSFORECAST_N_JOBS = 1
FORECAST_MODES = ("recursive", "rolling-origin", "closed-loop")
FORECAST_MODE_ALIASES = {
    "rolling_origin": "rolling-origin",
    "closed_loop": "closed-loop",
}
TASK_MODEL_COL = "_task_model"
TASK_SERIES_COL = "_task_series"
TASK_FORECAST_COL = "_forecast"
TASK_VARIANCE_COL = "_variance"
VARIANCE_SUFFIX = "__variance"
FORECAST_VARIANCE_LONG_FILE = "benchmark_forecast_variance_long.csv"
SUCCESS_MARKER = "_SUCCESS"
_FORECAST_VARIANCE_Z = 1.959963984540054
_INTERVAL_Z_BY_LEVEL = {
    80: 1.2815515655446004,
    90: 1.6448536269514722,
    95: _FORECAST_VARIANCE_Z,
}


@dataclass(frozen=True)
class NeuralModelConfig:
    input_size: int = 0
    max_steps: int = 200
    hidden_size: int = 64
    num_layers: int = 2
    batch_size: int = 32
    learning_rate: float = 1e-3
    scaler_type: str = "robust"
    random_seed: int = 2026

    def legacy_config(self) -> dict[str, Any]:
        return {
            "deepar_input_size": int(self.input_size),
            "deepar_max_steps": int(self.max_steps),
            "deepar_hidden_size": int(self.hidden_size),
            "deepar_lstm_layers": int(self.num_layers),
            "deepar_batch_size": int(self.batch_size),
            "deepar_learning_rate": float(self.learning_rate),
            "deepar_scaler_type": str(self.scaler_type),
            "deepar_random_seed": int(self.random_seed),
            "nbeats_input_size": int(self.input_size),
            "nbeats_max_steps": int(self.max_steps),
            "nbeats_hidden_size": int(self.hidden_size),
            "nbeats_num_layers": int(self.num_layers),
            "nbeats_batch_size": int(self.batch_size),
            "nbeats_learning_rate": float(self.learning_rate),
            "nbeats_scaler_type": str(self.scaler_type),
            "nbeats_random_seed": int(self.random_seed),
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    data_dir: Path = Path("data/tourism")
    output_dir: Path = Path("outputs/tourism_benchmark_statsforecast")
    data_path: Path | None = None
    data_format: str = "auto"
    date_col: str = "date"
    series_col: str = "series_id"
    value_col: str = "value"
    models: str | list[str] = DEFAULT_MODELS
    forecast_mode: str = "rolling-origin"
    horizon: int = 12
    season_length: int = 12
    freq: str = "MS"
    series_depth: int | None = DETAILED_CODE_DEPTH
    series_suffix: str | None = BASE_SUFFIX
    max_series: int = 0
    clip_nonnegative: bool = False
    ray_num_cpus: int = 1
    max_cpus_per_task: int = 1
    ray_debug: bool = False
    resume: bool = True
    resume_csv: Path | None = None
    neural_config: NeuralModelConfig = field(default_factory=NeuralModelConfig)

    @property
    def input_path(self) -> Path:
        return self.data_path if self.data_path is not None else self.data_dir / "tourism.csv"

    @property
    def use_ray(self) -> bool:
        return int(self.ray_num_cpus) > 1

    def run_kwargs(self) -> dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "data_path": self.data_path,
            "data_format": self.data_format,
            "date_col": self.date_col,
            "series_col": self.series_col,
            "value_col": self.value_col,
            "models": self.models,
            "horizon": self.horizon,
            "season_length": self.season_length,
            "freq": self.freq,
            "series_depth": self.series_depth,
            "series_suffix": self.series_suffix,
            "max_series": self.max_series,
            "clip_nonnegative": self.clip_nonnegative,
            "ray_num_cpus": self.ray_num_cpus,
            "max_cpus_per_task": self.max_cpus_per_task,
            "ray_debug": self.ray_debug,
            "resume": self.resume,
            "deepar_input_size": self.neural_config.input_size,
            "deepar_max_steps": self.neural_config.max_steps,
            "deepar_hidden_size": self.neural_config.hidden_size,
            "deepar_lstm_layers": self.neural_config.num_layers,
            "deepar_batch_size": self.neural_config.batch_size,
            "deepar_learning_rate": self.neural_config.learning_rate,
            "deepar_scaler_type": self.neural_config.scaler_type,
            "deepar_random_seed": self.neural_config.random_seed,
        }

    def output_config(self, *, model_names: list[str], checkpoint_csv: Path | None) -> dict[str, Any]:
        return {
            "data_dir": str(self.data_dir),
            "data_path": None if self.data_path is None else str(self.data_path),
            "data_format": str(self.data_format),
            "date_col": str(self.date_col),
            "series_col": str(self.series_col),
            "value_col": str(self.value_col),
            "output_dir": str(self.output_dir),
            "models": model_names,
            "forecast_mode": self.forecast_mode,
            "horizon": int(self.horizon),
            "season_length": int(self.season_length),
            "freq": self.freq,
            "series_depth": -1 if self.series_depth is None else int(self.series_depth),
            "series_suffix": self.series_suffix,
            "max_series": int(self.max_series),
            "clip_nonnegative": bool(self.clip_nonnegative),
            "ray_num_cpus": int(self.ray_num_cpus),
            "max_cpus_per_task": int(self.max_cpus_per_task),
            "ray_debug": bool(self.ray_debug),
            "resume": bool(self.resume),
            "resume_csv": None if checkpoint_csv is None else str(checkpoint_csv),
            "forecast_variance_level": int(FORECAST_VARIANCE_LEVEL),
            **self.neural_config.legacy_config(),
        }


DEFAULT_BENCHMARK_CONFIG = BenchmarkConfig()


def _task_expected_rows(task_series: str, *, horizon: int, n_series: int) -> int:
    return int(horizon) * (int(n_series) if task_series == "panel" else 1)


def _load_completed_task_keys(checkpoint_csv: Path, *, horizon: int, n_series: int) -> set[tuple[str, str]]:
    if not checkpoint_csv.exists() or checkpoint_csv.stat().st_size == 0:
        return set()
    try:
        tasks = pd.read_csv(checkpoint_csv, usecols=[TASK_MODEL_COL, TASK_SERIES_COL, TASK_FORECAST_COL, TASK_VARIANCE_COL])
    except ValueError:
        return set()
    if tasks.empty:
        return set()
    completed = set()
    counts = tasks.groupby([TASK_MODEL_COL, TASK_SERIES_COL], dropna=False).size()
    for (model_name, task_series), n_rows in counts.items():
        task_series = str(task_series)
        if int(n_rows) >= _task_expected_rows(task_series, horizon=horizon, n_series=n_series):
            completed.add((str(model_name), task_series))
    return completed


def _append_task_checkpoint(checkpoint_csv: Path, model_name: str, task_series: str, frame: pd.DataFrame) -> None:
    checkpoint_csv.parent.mkdir(parents=True, exist_ok=True)
    if model_name not in frame.columns:
        raise ValueError(f"Task frame for model {model_name!r} is missing its forecast column.")
    key_cols = [col for col in ("origin", "train_end", "series_id", "date") if col in frame.columns]
    var_col = _variance_col(model_name)
    out = frame.loc[:, [*key_cols, model_name]].rename(columns={model_name: TASK_FORECAST_COL})
    out[TASK_VARIANCE_COL] = frame[var_col].to_numpy(dtype=float) if var_col in frame.columns else np.nan
    out.insert(0, TASK_SERIES_COL, str(task_series))
    out.insert(0, TASK_MODEL_COL, str(model_name))
    write_header = not checkpoint_csv.exists() or checkpoint_csv.stat().st_size == 0
    out.to_csv(checkpoint_csv, mode="a", header=write_header, index=False)


def _load_task_checkpoint_forecasts(checkpoint_csv: Path, model_names: list[str]) -> pd.DataFrame:
    if not checkpoint_csv.exists() or checkpoint_csv.stat().st_size == 0:
        raise ValueError(f"No checkpoint forecasts found at {checkpoint_csv}.")
    raw = pd.read_csv(checkpoint_csv)
    if raw.empty:
        raise ValueError(f"Checkpoint forecasts are empty: {checkpoint_csv}.")
    if "date" in raw.columns:
        raw["date"] = pd.to_datetime(raw["date"])
    if "train_end" in raw.columns:
        raw["train_end"] = pd.to_datetime(raw["train_end"])

    key_cols = ["series_id", "date"]
    if "origin" in raw.columns:
        key_cols = ["origin", "train_end", *key_cols]
    raw = raw.drop_duplicates([TASK_MODEL_COL, *key_cols], keep="last")

    frames = []
    for model_name in model_names:
        frame = raw.loc[raw[TASK_MODEL_COL].astype(str) == model_name].copy()
        if frame.empty:
            raise ValueError(f"Checkpoint is missing completed forecasts for model {model_name!r}.")
        if TASK_FORECAST_COL not in frame.columns:
            raise ValueError(f"Checkpoint rows for model {model_name!r} do not contain {TASK_FORECAST_COL!r}.")
        cols = [*key_cols, TASK_FORECAST_COL]
        rename = {TASK_FORECAST_COL: model_name}
        if TASK_VARIANCE_COL in frame.columns:
            cols.append(TASK_VARIANCE_COL)
            rename[TASK_VARIANCE_COL] = _variance_col(model_name)
        frame = frame.loc[:, cols].rename(columns=rename)
        if _variance_col(model_name) not in frame.columns:
            frame[_variance_col(model_name)] = np.nan
        frames.append(frame.reset_index(drop=True))
    return _merge_forecast_frames(frames)


def _archive_checkpoint_outputs(checkpoint_csv: Path, forecast_long_csv: Path) -> None:
    if checkpoint_csv.exists() and checkpoint_csv.stat().st_size > 0:
        pd.read_csv(checkpoint_csv).to_parquet(checkpoint_csv.with_suffix(".parquet"), index=False)
    if forecast_long_csv.exists() and forecast_long_csv.stat().st_size > 0:
        pd.read_csv(forecast_long_csv).to_parquet(forecast_long_csv.with_suffix(".parquet"), index=False)


def _print_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _variance_col(model_name: str) -> str:
    return f"{model_name}{VARIANCE_SUFFIX}"


def _clean_variance(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.where(np.isfinite(arr) & (arr >= 0.0), arr, np.nan)


def _interval_variance(lower: np.ndarray, upper: np.ndarray, *, level: int = FORECAST_VARIANCE_LEVEL) -> np.ndarray:
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    z = _INTERVAL_Z_BY_LEVEL.get(int(level))
    if z is None:
        return np.full(lo.shape, np.nan, dtype=float)
    return _clean_variance(((hi - lo) / (2.0 * z)) ** 2)


def _forecast_variance_from_interval_columns(
    frame: pd.DataFrame,
    model_name: str,
    *,
    levels: tuple[int, ...] = (FORECAST_VARIANCE_LEVEL, 90, 80),
) -> np.ndarray:
    for level in levels:
        lo_col = f"{model_name}-lo-{level}"
        hi_col = f"{model_name}-hi-{level}"
        if lo_col in frame.columns and hi_col in frame.columns:
            return _interval_variance(
                frame[lo_col].to_numpy(dtype=float),
                frame[hi_col].to_numpy(dtype=float),
                level=level,
            )
    return np.full(len(frame), np.nan, dtype=float)


def _combine_variance_arrays(*arrays: np.ndarray | None) -> np.ndarray | None:
    valid_arrays = [np.asarray(arr, dtype=float) for arr in arrays if arr is not None]
    if not valid_arrays:
        return None
    out = np.zeros_like(valid_arrays[0], dtype=float)
    observed = np.zeros_like(valid_arrays[0], dtype=bool)
    for arr in valid_arrays:
        arr = _clean_variance(arr)
        mask = np.isfinite(arr)
        out[mask] += arr[mask]
        observed |= mask
    out[~observed] = np.nan
    return out


def _key_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in ("origin", "train_end", "series_id", "date") if col in frame.columns]


def _point_forecast_frame(long_forecasts: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    return long_forecasts.loc[:, [*_key_columns(long_forecasts), *model_names]].copy()


def _variance_frame(long_forecasts: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    out = long_forecasts.loc[:, _key_columns(long_forecasts)].copy()
    for model in model_names:
        var_col = _variance_col(model)
        out[model] = long_forecasts[var_col].to_numpy(dtype=float) if var_col in long_forecasts.columns else np.nan
    return out


def _series_code(series_id: str, suffix: str = BASE_SUFFIX) -> str:
    return series_id[: -len(suffix)] if suffix and series_id.endswith(suffix) else series_id


def _series_depth(series_id: str, suffix: str = BASE_SUFFIX) -> int:
    code = _series_code(series_id, suffix)
    return 0 if code == "Total" else len(code)


def _base_series_columns(
    wide: pd.DataFrame,
    series_depth: int | None = DEFAULT_BENCHMARK_CONFIG.series_depth,
    series_suffix: str | None = DEFAULT_BENCHMARK_CONFIG.series_suffix,
) -> list[str]:
    cols = [c for c in wide.columns if c != "date"]
    if series_suffix:
        cols = [c for c in cols if c.endswith(series_suffix)]
    if series_depth is not None and series_suffix:
        cols = [c for c in cols if _series_depth(c, series_suffix) == series_depth]
    if not cols:
        suffix_msg = f" ending with {series_suffix!r}" if series_suffix else ""
        raise ValueError(f"No series columns{suffix_msg} found for series_depth={series_depth}.")
    return cols


def _normalize_forecast_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("_", "-")
    normalized = FORECAST_MODE_ALIASES.get(normalized, normalized)
    if normalized not in set(FORECAST_MODES):
        valid = ", ".join(FORECAST_MODES)
        raise ValueError(f"forecast_mode must be one of {valid}. Got {mode!r}.")
    return normalized


def _parse_forecast_modes(value: str) -> list[str]:
    modes = [_normalize_forecast_mode(part) for part in str(value).split(",") if part.strip()]
    if not modes:
        raise ValueError("forecast_mode must contain at least one mode.")
    out = []
    seen = set()
    for mode in modes:
        if mode not in seen:
            out.append(mode)
            seen.add(mode)
    return out


def _parse_model_names(models: str | list[str]) -> list[str]:
    names = models if isinstance(models, list) else [item.strip() for item in str(models).split(",")]
    aliases = {
        "seasonalnaive": "seasonal_naive",
        "snaive": "seasonal_naive",
        "rwd": "drift",
        "random_walk_with_drift": "drift",
        "ets": "auto_ets",
        "autoets": "auto_ets",
        "arima": "auto_arima",
        "autoarima": "auto_arima",
        "default_ets": "statsmodels_ets",
        "plain_ets": "statsmodels_ets",
        "sm_ets": "statsmodels_ets",
        "fixed_seasonal_ets": "seasonal_ets",
        "seasonalets": "seasonal_ets",
        "hw": "seasonal_ets",
        "holt_winters": "seasonal_ets",
        "ar": "ar1",
        "ar_1": "ar1",
        "ar1_seasonal_naive": "ar1_snaive",
        "ar1_seasonal_naive_resid": "ar1_snaive",
        "ar1_snaive_resid": "ar1_snaive",
        "ar1_autoets": "ar1_auto_ets",
        "ar1_auto_ets_resid": "ar1_auto_ets",
        "ar1_seasonalets": "ar1_seasonal_ets",
        "ar1_seasonal_ets_resid": "ar1_seasonal_ets",
        "ar1_fixed_seasonal_ets": "ar1_seasonal_ets",
        "ar1_statsmodels_ets_resid": "ar1_statsmodels_ets",
        "var": "var1",
        "var_1": "var1",
        "var1_seasonal_naive": "var1_snaive",
        "var1_seasonal_naive_resid": "var1_snaive",
        "var1_snaive_resid": "var1_snaive",
        "var1_autoets": "var1_auto_ets",
        "var1_auto_ets_resid": "var1_auto_ets",
        "var1_seasonalets": "var1_seasonal_ets",
        "var1_seasonal_ets_resid": "var1_seasonal_ets",
        "var1_fixed_seasonal_ets": "var1_seasonal_ets",
        "var1_statsmodels_ets_resid": "var1_statsmodels_ets",
        "deep_ar": "deepar",
        "neural_deepar": "deepar",
        "n_beats": "nbeats",
        "neural_nbeats": "nbeats",
    }
    out = []
    for name in names:
        normalized = aliases.get(name.strip().lower().replace("-", "_"), name.strip().lower().replace("-", "_"))
        if normalized:
            out.append(normalized)
    if not out:
        raise ValueError("No benchmark models selected.")
    unknown = sorted(set(out) - ALL_MODELS)
    if unknown:
        raise ValueError(f"Unknown benchmark model names: {unknown}")
    return out


def _make_models(model_names: list[str], season_length: int):
    models = []
    for name in model_names:
        if name == "naive":
            models.append(Naive(alias=name))
        elif name == "seasonal_naive":
            models.append(SeasonalNaive(season_length=season_length, alias=name))
        elif name == "drift":
            models.append(RandomWalkWithDrift(alias=name))
        elif name == "auto_ets":
            models.append(AutoETS(season_length=season_length, alias=name))
        elif name == "auto_arima":
            models.append(AutoARIMA(season_length=season_length, alias=name))
        else:
            raise ValueError(f"Unknown benchmark model name: {name}")
    return models


def _wide_to_statsforecast_long(wide: pd.DataFrame, series_cols: list[str]) -> pd.DataFrame:
    long = wide.loc[:, ["date", *series_cols]].melt(id_vars="date", var_name="unique_id", value_name="y")
    long = long.rename(columns={"date": "ds"})
    long["ds"] = pd.to_datetime(long["ds"])
    long["y"] = long["y"].astype(float)
    return long


def _load_deepar():
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import DeepAR
    except Exception as exc:
        raise RuntimeError(
            "DeepAR benchmark requires neuralforecast and its PyTorch dependencies. "
            "The current import failed; install/repair neuralforecast dependencies "
            f"before running --models deepar. Original error: {exc!r}"
        ) from exc
    return NeuralForecast, DeepAR


def _load_nbeats():
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NBEATS
    except Exception as exc:
        raise RuntimeError(
            "NBEATS benchmark requires neuralforecast and its PyTorch dependencies. "
            "The current import failed; install/repair neuralforecast dependencies "
            f"before running --models nbeats. Original error: {exc!r}"
        ) from exc
    return NeuralForecast, NBEATS


def _neural_input_size(h: int, season_length: int, requested: int) -> int:
    if requested > 0:
        return int(requested)
    return int(max(3 * h, 2 * season_length, 1))


def _neural_common_model_kwargs(
    *,
    h: int,
    season_length: int,
    neural_config: NeuralModelConfig,
    alias: str,
) -> dict[str, Any]:
    return {
        "h": int(h),
        "input_size": _neural_input_size(h, season_length, neural_config.input_size),
        "max_steps": int(neural_config.max_steps),
        "learning_rate": float(neural_config.learning_rate),
        "batch_size": int(neural_config.batch_size),
        "scaler_type": str(neural_config.scaler_type),
        "random_seed": int(neural_config.random_seed),
        "alias": alias,
        "accelerator": "cpu",
        "devices": 1,
        "enable_progress_bar": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "logger": False,
    }


def _forecast_neural_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    freq: str,
    neural_forecast_cls: Any,
    model: Any,
    model_name: str,
    model_label: str,
) -> pd.DataFrame:
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    nf = neural_forecast_cls(models=[model], freq=freq)
    train_long = _wide_to_statsforecast_long(train_wide, series_cols)
    nf.fit(df=train_long, verbose=False)
    out = nf.predict(verbose=False).rename(columns={"unique_id": "series_id", "ds": "date"})

    dates = pd.to_datetime(pd.Series(forecast_dates)).reset_index(drop=True)
    expected = pd.MultiIndex.from_product(
        [series_cols, dates], names=["series_id", "date"]
    ).to_frame(index=False)
    interval_cols = [
        col for col in out.columns
        if col.startswith(f"{model_name}-lo-") or col.startswith(f"{model_name}-hi-")
    ]
    out = expected.merge(
        out.loc[:, ["series_id", "date", model_name, *interval_cols]],
        on=["series_id", "date"],
        how="left",
    )
    if out[model_name].isna().any():
        raise RuntimeError(f"{model_label} did not return forecasts for all requested series/date pairs.")
    out[_variance_col(model_name)] = _forecast_variance_from_interval_columns(out, model_name)
    return out.loc[:, ["series_id", "date", model_name, _variance_col(model_name)]]


def _forecast_deepar_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
    season_length: int,
    freq: str,
    neural_config: NeuralModelConfig,
) -> pd.DataFrame:
    NeuralForecast, DeepAR = _load_deepar()
    model = DeepAR(
        **_neural_common_model_kwargs(
            h=h,
            season_length=season_length,
            neural_config=neural_config,
            alias="deepar",
        ),
        lstm_n_layers=int(neural_config.num_layers),
        lstm_hidden_size=int(neural_config.hidden_size),
        lstm_dropout=0.0,
    )
    return _forecast_neural_once(
        train_wide,
        series_cols,
        forecast_dates=forecast_dates,
        freq=freq,
        neural_forecast_cls=NeuralForecast,
        model=model,
        model_name="deepar",
        model_label="DeepAR",
    )


def _nbeats_stack_config(
    h: int,
    hidden_size: int,
    num_layers: int,
) -> tuple[list[str], list[int], list[list[int]]]:
    layers = [int(hidden_size)] * max(int(num_layers), 2)
    if int(h) <= 1:
        return ["identity"], [1], [layers.copy()]
    return (
        ["identity", "trend", "seasonality"],
        [1, 1, 1],
        [layers.copy(), layers.copy(), layers.copy()],
    )


def _forecast_nbeats_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
    season_length: int,
    freq: str,
    neural_config: NeuralModelConfig,
) -> pd.DataFrame:
    NeuralForecast, NBEATS = _load_nbeats()
    stack_types, n_blocks, mlp_units = _nbeats_stack_config(h, neural_config.hidden_size, neural_config.num_layers)
    model = NBEATS(
        **_neural_common_model_kwargs(
            h=h,
            season_length=season_length,
            neural_config=neural_config,
            alias="nbeats",
        ),
        stack_types=stack_types,
        n_blocks=n_blocks,
        mlp_units=mlp_units,
    )
    return _forecast_neural_once(
        train_wide,
        series_cols,
        forecast_dates=forecast_dates,
        freq=freq,
        neural_forecast_cls=NeuralForecast,
        model=model,
        model_name="nbeats",
        model_label="NBEATS",
    )


def _forecast_statsforecast_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    model_names: list[str],
    season_length: int,
    freq: str,
    h: int,
    n_jobs: int,
) -> pd.DataFrame:
    models = _make_models(model_names, season_length)
    sf = StatsForecast(models=models, freq=freq, n_jobs=n_jobs)
    out = sf.forecast(df=_wide_to_statsforecast_long(train_wide, series_cols), h=h, level=[FORECAST_VARIANCE_LEVEL])
    out = out.rename(columns={"unique_id": "series_id", "ds": "date"})
    keep_cols = ["series_id", "date"]
    for model_name in model_names:
        keep_cols.append(model_name)
        lo_col = f"{model_name}-lo-{FORECAST_VARIANCE_LEVEL}"
        hi_col = f"{model_name}-hi-{FORECAST_VARIANCE_LEVEL}"
        var_col = _variance_col(model_name)
        if lo_col in out.columns and hi_col in out.columns:
            out[var_col] = _interval_variance(
                out[lo_col].to_numpy(dtype=float),
                out[hi_col].to_numpy(dtype=float),
                level=FORECAST_VARIANCE_LEVEL,
            )
        else:
            out[var_col] = np.nan
        keep_cols.append(var_col)
    return out.loc[:, keep_cols]



def _seasonal_naive_array(values: np.ndarray, h: int, season_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[0] < season_length:
        raise ValueError("Seasonal naive filtering requires at least season_length observations.")
    history = [row.copy() for row in values]
    forecasts = []
    for _ in range(int(h)):
        y_hat = history[-season_length].copy()
        history.append(y_hat)
        forecasts.append(y_hat)
    return np.vstack(forecasts)


def _seasonal_residual_array(values: np.ndarray, season_length: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[0] <= season_length:
        raise ValueError("Seasonal naive filtering requires more than season_length observations.")
    return values[season_length:] - values[:-season_length]


def _validate_forecast_dates(
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
    model_label: str,
) -> pd.Series:
    dates = pd.to_datetime(pd.Series(forecast_dates)).reset_index(drop=True)
    if len(dates) != h:
        raise RuntimeError(f"{model_label} forecast_dates length does not match h.")
    return dates


def _forecast_array_to_long(
    forecast: np.ndarray,
    dates: pd.Series,
    series_cols: list[str],
    model_name: str,
    variance: np.ndarray | None = None,
) -> pd.DataFrame:
    values = np.asarray(forecast, dtype=float)
    if values.ndim == 1:
        values = values[:, None]
    expected_shape = (len(dates), len(series_cols))
    if values.shape != expected_shape:
        raise RuntimeError(f"{model_name} forecast has shape {values.shape}; expected {expected_shape}.")
    wide = pd.DataFrame(values, columns=series_cols)
    wide.insert(0, "date", pd.to_datetime(dates).to_numpy())
    out = wide.melt(id_vars="date", var_name="series_id", value_name=model_name).loc[
        :, ["series_id", "date", model_name]
    ]
    var_col = _variance_col(model_name)
    if variance is None:
        out[var_col] = np.nan
    else:
        var_values = np.asarray(variance, dtype=float)
        if var_values.ndim == 1:
            var_values = var_values[:, None]
        if var_values.shape != expected_shape:
            raise RuntimeError(f"{model_name} variance has shape {var_values.shape}; expected {expected_shape}.")
        var_wide = pd.DataFrame(_clean_variance(var_values), columns=series_cols)
        var_wide.insert(0, "date", pd.to_datetime(dates).to_numpy())
        out[var_col] = var_wide.melt(id_vars="date", var_name="series_id", value_name=var_col)[var_col].to_numpy(dtype=float)
    return out


def _fallback_ar1_values(y: np.ndarray, h: int, fallback: str, model_label: str) -> np.ndarray:
    if fallback == "zero":
        return np.zeros(h, dtype=float)
    if y.shape[0] > 0:
        return np.repeat(y[-1], h).astype(float)
    raise ValueError(f"{model_label} requires at least two training observations.")


def _forecast_ar1_mean_variance(
    values: np.ndarray,
    h: int,
    *,
    fallback: str,
    model_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(values, dtype=float)
    if y.shape[0] <= 1:
        return _fallback_ar1_values(y, h, fallback, model_label), np.full(h, np.nan, dtype=float)
    try:
        fit = AutoReg(y, lags=1, trend="c", old_names=False).fit()
        pred = fit.get_prediction(start=len(y), end=len(y) + int(h) - 1).summary_frame()
        mean = pred["mean"].to_numpy(dtype=float)
        if "mean_se" in pred.columns:
            variance = pred["mean_se"].to_numpy(dtype=float) ** 2
        else:
            variance = np.full(int(h), np.nan, dtype=float)
        return mean, _clean_variance(variance)
    except Exception:
        return _fallback_ar1_values(y, h, fallback, model_label), np.full(h, np.nan, dtype=float)


def _forecast_ar1_values(
    values: np.ndarray,
    h: int,
    *,
    fallback: str,
    model_label: str,
) -> np.ndarray:
    return _forecast_ar1_mean_variance(values, h, fallback=fallback, model_label=model_label)[0]


def _forecast_var1_mean_variance(
    values: np.ndarray,
    series_cols: list[str],
    h: int,
    *,
    fallback_zero: bool,
    model_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(values, dtype=float)
    if y.shape[0] <= 1:
        if fallback_zero:
            return (
                np.zeros((h, len(series_cols)), dtype=float),
                np.full((h, len(series_cols)), np.nan, dtype=float),
            )
        raise ValueError(f"{model_label} requires at least two training observations.")
    try:
        fit = VAR(pd.DataFrame(y, columns=series_cols)).fit(maxlags=1, trend="c")
        forecast = np.asarray(fit.forecast(y[-fit.k_ar:], steps=h), dtype=float)
        forecast_cov = np.asarray(fit.forecast_cov(steps=h), dtype=float)
        variance = np.diagonal(forecast_cov, axis1=1, axis2=2)
        return forecast, _clean_variance(variance)
    except Exception:
        if fallback_zero:
            return (
                np.zeros((h, len(series_cols)), dtype=float),
                np.full((h, len(series_cols)), np.nan, dtype=float),
            )
        raise


def _forecast_var1_values(
    values: np.ndarray,
    series_cols: list[str],
    h: int,
    *,
    fallback_zero: bool,
    model_label: str,
) -> np.ndarray:
    return _forecast_var1_mean_variance(
        values,
        series_cols,
        h,
        fallback_zero=fallback_zero,
        model_label=model_label,
    )[0]


def _forecast_seasonal_ets_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
    season_length: int,
) -> pd.DataFrame:
    dates = _validate_forecast_dates(forecast_dates, h, "seasonal ETS")
    baseline = seasonal_ets_baseline_panel(
        train_wide.loc[:, series_cols].to_numpy(dtype=float),
        horizon=h,
        seasonal_lag=season_length,
        series_ids=series_cols,
    )
    return _forecast_array_to_long(
        baseline["forecast"],
        dates,
        series_cols,
        "seasonal_ets",
        variance=baseline.get("forecast_variance"),
    )


def _forecast_statsmodels_ets_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
) -> pd.DataFrame:
    dates = _validate_forecast_dates(forecast_dates, h, "statsmodels ETS")
    baseline = statsmodels_ets_baseline_panel(
        train_wide.loc[:, series_cols].to_numpy(dtype=float),
        horizon=h,
        seasonal_lag=1,
        series_ids=series_cols,
    )
    return _forecast_array_to_long(
        baseline["forecast"],
        dates,
        series_cols,
        "statsmodels_ets",
        variance=baseline.get("forecast_variance"),
    )


def _forecast_ar1_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
) -> pd.DataFrame:
    dates = _validate_forecast_dates(forecast_dates, h, "AR(1)")
    results = [
        _forecast_ar1_mean_variance(
            train_wide.loc[:, series_id].to_numpy(dtype=float),
            h,
            fallback="last",
            model_label="AR(1)",
        )
        for series_id in series_cols
    ]
    forecasts = [item[0] for item in results]
    variances = [item[1] for item in results]
    return _forecast_array_to_long(
        np.column_stack(forecasts),
        dates,
        series_cols,
        "ar1",
        variance=np.column_stack(variances),
    )


def _filtered_baseline_panel(
    values: np.ndarray,
    h: int,
    season_length: int,
    series_cols: list[str],
    filter_name: str,
) -> dict[str, Any]:
    if filter_name == "auto_ets":
        return auto_ets_baseline_panel(values, h, season_length, series_cols)
    if filter_name == "seasonal_ets":
        return seasonal_ets_baseline_panel(values, h, season_length, series_cols)
    if filter_name == "statsmodels_ets":
        return statsmodels_ets_baseline_panel(values, h, season_length, series_cols)
    raise ValueError(f"Unknown residual filter: {filter_name!r}.")


def _forecast_ar1_filtered_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
    season_length: int,
    filter_name: str,
    model_name: str,
) -> pd.DataFrame:
    dates = _validate_forecast_dates(forecast_dates, h, f"AR(1) {filter_name} residual")
    values = train_wide.loc[:, series_cols].to_numpy(dtype=float)
    baseline = _filtered_baseline_panel(values, h, season_length, series_cols, filter_name)
    residuals = baseline["residuals"]
    results = [
        _forecast_ar1_mean_variance(
            residuals[:, idx],
            h,
            fallback="zero",
            model_label=f"AR(1) {filter_name} residual",
        )
        for idx in range(len(series_cols))
    ]
    forecasts = [item[0] for item in results]
    variances = [item[1] for item in results]
    forecast = baseline["forecast"] + np.column_stack(forecasts)
    variance = _combine_variance_arrays(baseline.get("forecast_variance"), np.column_stack(variances))
    return _forecast_array_to_long(forecast, dates, series_cols, model_name, variance=variance)


def _forecast_var1_filtered_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
    season_length: int,
    filter_name: str,
    model_name: str,
) -> pd.DataFrame:
    dates = _validate_forecast_dates(forecast_dates, h, f"VAR(1) {filter_name} residual")
    values = train_wide.loc[:, series_cols].to_numpy(dtype=float)
    baseline = _filtered_baseline_panel(values, h, season_length, series_cols, filter_name)
    residual_forecast, residual_variance = _forecast_var1_mean_variance(
        baseline["residuals"],
        series_cols,
        h,
        fallback_zero=True,
        model_label=f"VAR(1) {filter_name} residual",
    )
    forecast = baseline["forecast"] + residual_forecast
    variance = _combine_variance_arrays(baseline.get("forecast_variance"), residual_variance)
    return _forecast_array_to_long(forecast, dates, series_cols, model_name, variance=variance)


def _forecast_ar1_snaive_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
    season_length: int,
) -> pd.DataFrame:
    dates = _validate_forecast_dates(forecast_dates, h, "AR(1) seasonal-naive")
    forecasts = []
    variances = []
    for series_id in series_cols:
        y = train_wide.loc[:, series_id].to_numpy(dtype=float)
        baseline = _seasonal_naive_array(y[:, None], h, season_length)[:, 0]
        residual = _seasonal_residual_array(y[:, None], season_length)[:, 0]
        residual_forecast, residual_variance = _forecast_ar1_mean_variance(
            residual,
            h,
            fallback="zero",
            model_label="AR(1) seasonal-naive residual",
        )
        forecasts.append(baseline + residual_forecast)
        variances.append(residual_variance)
    return _forecast_array_to_long(
        np.column_stack(forecasts),
        dates,
        series_cols,
        "ar1_snaive",
        variance=np.column_stack(variances),
    )


def _forecast_var1_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
) -> pd.DataFrame:
    dates = _validate_forecast_dates(forecast_dates, h, "VAR(1)")
    forecast, variance = _forecast_var1_mean_variance(
        train_wide.loc[:, series_cols].to_numpy(dtype=float),
        series_cols,
        h,
        fallback_zero=False,
        model_label="VAR(1)",
    )
    return _forecast_array_to_long(forecast, dates, series_cols, "var1", variance=variance)


def _forecast_var1_snaive_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
    season_length: int,
) -> pd.DataFrame:
    dates = _validate_forecast_dates(forecast_dates, h, "VAR(1) seasonal-naive")
    y = train_wide.loc[:, series_cols].to_numpy(dtype=float)
    baseline = _seasonal_naive_array(y, h, season_length)
    residual = _seasonal_residual_array(y, season_length)
    residual_forecast, residual_variance = _forecast_var1_mean_variance(
        residual,
        series_cols,
        h,
        fallback_zero=True,
        model_label="VAR(1) seasonal-naive residual",
    )
    return _forecast_array_to_long(
        baseline + residual_forecast,
        dates,
        series_cols,
        "var1_snaive",
        variance=residual_variance,
    )


def _forecast_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    model_names: list[str],
    season_length: int,
    freq: str,
    h: int,
    n_jobs: int,
    forecast_dates: pd.Series | list[pd.Timestamp],
    neural_config: NeuralModelConfig,
) -> pd.DataFrame:
    frames = []
    statsforecast_names = [name for name in model_names if name in STATSFORECAST_MODELS]
    if statsforecast_names:
        frames.append(
            _forecast_statsforecast_once(
                train_wide,
                series_cols,
                model_names=statsforecast_names,
                season_length=season_length,
                freq=freq,
                h=h,
                n_jobs=n_jobs,
            )
        )
    if "seasonal_ets" in model_names:
        frames.append(
            _forecast_seasonal_ets_once(
                train_wide,
                series_cols,
                forecast_dates=forecast_dates,
                h=h,
                season_length=season_length,
            )
        )
    if "statsmodels_ets" in model_names:
        frames.append(_forecast_statsmodels_ets_once(train_wide, series_cols, forecast_dates=forecast_dates, h=h))
    for model_name, filter_name in (
        ("ar1_auto_ets", "auto_ets"),
        ("ar1_seasonal_ets", "seasonal_ets"),
        ("ar1_statsmodels_ets", "statsmodels_ets"),
    ):
        if model_name in model_names:
            frames.append(
                _forecast_ar1_filtered_once(
                    train_wide,
                    series_cols,
                    forecast_dates=forecast_dates,
                    h=h,
                    season_length=season_length,
                    filter_name=filter_name,
                    model_name=model_name,
                )
            )
    if "ar1" in model_names:
        frames.append(_forecast_ar1_once(train_wide, series_cols, forecast_dates=forecast_dates, h=h))
    if "ar1_snaive" in model_names:
        frames.append(
            _forecast_ar1_snaive_once(
                train_wide,
                series_cols,
                forecast_dates=forecast_dates,
                h=h,
                season_length=season_length,
            )
        )
    for model_name, filter_name in (
        ("var1_auto_ets", "auto_ets"),
        ("var1_seasonal_ets", "seasonal_ets"),
        ("var1_statsmodels_ets", "statsmodels_ets"),
    ):
        if model_name in model_names:
            frames.append(
                _forecast_var1_filtered_once(
                    train_wide,
                    series_cols,
                    forecast_dates=forecast_dates,
                    h=h,
                    season_length=season_length,
                    filter_name=filter_name,
                    model_name=model_name,
                )
            )
    if "var1" in model_names:
        frames.append(_forecast_var1_once(train_wide, series_cols, forecast_dates=forecast_dates, h=h))
    if "var1_snaive" in model_names:
        frames.append(
            _forecast_var1_snaive_once(
                train_wide,
                series_cols,
                forecast_dates=forecast_dates,
                h=h,
                season_length=season_length,
            )
        )
    if "deepar" in model_names:
        frames.append(
            _forecast_deepar_once(
                train_wide,
                series_cols,
                forecast_dates=forecast_dates,
                h=h,
                season_length=season_length,
                freq=freq,
                neural_config=neural_config,
            )
        )
    if "nbeats" in model_names:
        frames.append(
            _forecast_nbeats_once(
                train_wide,
                series_cols,
                forecast_dates=forecast_dates,
                h=h,
                season_length=season_length,
                freq=freq,
                neural_config=neural_config,
            )
        )

    if not frames:
        raise ValueError("No forecast models selected.")

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=["series_id", "date"], how="outer", sort=False)
    return out


def _forecast_recursive(
    wide: pd.DataFrame,
    series_cols: list[str],
    *,
    horizon: int,
    model_names: list[str],
    season_length: int,
    freq: str,
    n_jobs: int,
    neural_config: NeuralModelConfig,
) -> pd.DataFrame:
    train = wide.iloc[: len(wide) - horizon]
    out = _forecast_once(
        train,
        series_cols,
        model_names=model_names,
        season_length=season_length,
        freq=freq,
        h=horizon,
        n_jobs=n_jobs,
        forecast_dates=wide["date"].iloc[-horizon:],
        neural_config=neural_config,
    )
    return out


def _forecast_rolling_origin(
    wide: pd.DataFrame,
    series_cols: list[str],
    *,
    horizon: int,
    model_names: list[str],
    season_length: int,
    freq: str,
    n_jobs: int,
    neural_config: NeuralModelConfig,
) -> pd.DataFrame:
    rows = []
    for step in range(horizon):
        origin_end = len(wide) - horizon + step
        train = wide.iloc[:origin_end]
        out = _forecast_once(
            train,
            series_cols,
            model_names=model_names,
            season_length=season_length,
            freq=freq,
            h=1,
            n_jobs=n_jobs,
            forecast_dates=[wide["date"].iloc[origin_end]],
            neural_config=neural_config,
        )
        out.insert(0, "origin", step + 1)
        out.insert(1, "train_end", wide["date"].iloc[origin_end - 1])
        rows.append(out)
    return pd.concat(rows, ignore_index=True)



def _append_forecast_to_history(
    history: pd.DataFrame,
    forecast: pd.DataFrame,
    *,
    model_name: str,
    series_cols: list[str],
    forecast_date,
) -> pd.DataFrame:
    if model_name not in forecast.columns:
        raise ValueError(f"Closed-loop forecast frame is missing model column {model_name!r}.")
    values = forecast.set_index("series_id")[model_name].reindex(series_cols)
    if values.isna().any():
        missing = values.index[values.isna()].tolist()
        raise RuntimeError(f"Closed-loop forecast missing series values for {model_name!r}: {missing[:5]}")
    next_row = {"date": pd.to_datetime(forecast_date)}
    next_row.update({series_id: float(values.loc[series_id]) for series_id in series_cols})
    return pd.concat([history, pd.DataFrame([next_row])], ignore_index=True)


def _forecast_closed_loop_single_model(
    wide: pd.DataFrame,
    series_cols: list[str],
    *,
    horizon: int,
    model_name: str,
    season_length: int,
    freq: str,
    n_jobs: int,
    neural_config: NeuralModelConfig,
) -> pd.DataFrame:
    history = wide.iloc[: len(wide) - horizon].loc[:, ["date", *series_cols]].copy().reset_index(drop=True)
    rows = []
    for step in range(horizon):
        origin_end = len(wide) - horizon + step
        forecast_date = wide["date"].iloc[origin_end]
        train_end = history["date"].iloc[-1]
        out = _forecast_once(
            history,
            series_cols,
            model_names=[model_name],
            season_length=season_length,
            freq=freq,
            h=1,
            n_jobs=n_jobs,
            forecast_dates=[forecast_date],
            neural_config=neural_config,
        )
        out.insert(0, "origin", step + 1)
        out.insert(1, "train_end", train_end)
        rows.append(out)
        history = _append_forecast_to_history(
            history,
            out,
            model_name=model_name,
            series_cols=series_cols,
            forecast_date=forecast_date,
        )
    return pd.concat(rows, ignore_index=True)


def _forecast_closed_loop(
    wide: pd.DataFrame,
    series_cols: list[str],
    *,
    horizon: int,
    model_names: list[str],
    season_length: int,
    freq: str,
    n_jobs: int,
    neural_config: NeuralModelConfig,
) -> pd.DataFrame:
    frames = [
        _forecast_closed_loop_single_model(
            wide,
            series_cols,
            horizon=horizon,
            model_name=model_name,
            season_length=season_length,
            freq=freq,
            n_jobs=n_jobs,
            neural_config=neural_config,
        )
        for model_name in model_names
    ]
    return _merge_forecast_frames(frames)


def _merge_forecast_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        raise ValueError("No forecast frames to merge.")
    keys = ["series_id", "date"]
    if all("origin" in frame.columns for frame in frames):
        keys = ["origin", "train_end", *keys]

    out = frames[0]
    for frame in frames[1:]:
        out = out.merge(frame, on=keys, how="outer", sort=False)
    return out


def _forecast_by_mode(
    wide: pd.DataFrame,
    series_cols: list[str],
    *,
    model_names: list[str],
    forecast_mode: str,
    horizon: int,
    season_length: int,
    freq: str,
    n_jobs: int,
    neural_config: NeuralModelConfig,
) -> pd.DataFrame:
    forecast_mode = _normalize_forecast_mode(forecast_mode)
    if forecast_mode == "recursive":
        return _forecast_recursive(
            wide,
            series_cols,
            horizon=horizon,
            model_names=model_names,
            season_length=season_length,
            freq=freq,
            n_jobs=n_jobs,
            neural_config=neural_config,
        )
    if forecast_mode == "closed-loop":
        if len(model_names) == 1:
            return _forecast_closed_loop_single_model(
                wide,
                series_cols,
                horizon=horizon,
                model_name=model_names[0],
                season_length=season_length,
                freq=freq,
                n_jobs=n_jobs,
                neural_config=neural_config,
            )
        return _forecast_closed_loop(
            wide,
            series_cols,
            horizon=horizon,
            model_names=model_names,
            season_length=season_length,
            freq=freq,
            n_jobs=n_jobs,
            neural_config=neural_config,
        )
    return _forecast_rolling_origin(
        wide,
        series_cols,
        horizon=horizon,
        model_names=model_names,
        season_length=season_length,
        freq=freq,
        n_jobs=n_jobs,
        neural_config=neural_config,
    )


def _forecast_single_model(
    wide: pd.DataFrame,
    series_cols: list[str],
    *,
    model_name: str,
    forecast_mode: str,
    horizon: int,
    season_length: int,
    freq: str,
    n_jobs: int,
    neural_config: NeuralModelConfig,
) -> pd.DataFrame:
    return _forecast_by_mode(
        wide,
        series_cols,
        model_names=[model_name],
        forecast_mode=forecast_mode,
        horizon=horizon,
        season_length=season_length,
        freq=freq,
        n_jobs=n_jobs,
        neural_config=neural_config,
    )


def _forecast_model_task_remote(
    wide: pd.DataFrame,
    task_series_cols: list[str],
    *,
    model_name: str,
    task_series: str,
    forecast_mode: str,
    horizon: int,
    season_length: int,
    freq: str,
    n_jobs: int,
    neural_config: NeuralModelConfig,
    ray_debug: bool,
) -> tuple[str, str, pd.DataFrame]:
    task_label = "panel" if task_series == "panel" else f"series={task_series}"
    if ray_debug:
        _print_progress(f"[Ray benchmark start] model={model_name}, {task_label}")
    start = time.time()
    frame = _forecast_single_model(
        wide,
        task_series_cols,
        model_name=model_name,
        forecast_mode=forecast_mode,
        horizon=horizon,
        season_length=season_length,
        freq=freq,
        n_jobs=n_jobs,
        neural_config=neural_config,
    )
    if ray_debug:
        _print_progress(
            f"[Ray benchmark done] model={model_name}, {task_label}, "
            f"elapsed_sec={time.time() - start:.2f}"
        )
    return model_name, task_series, frame


def _forecast_models_with_ray(
    wide: pd.DataFrame,
    series_cols: list[str],
    *,
    model_names: list[str],
    forecast_mode: str,
    horizon: int,
    season_length: int,
    freq: str,
    n_jobs: int,
    max_cpus_per_task: int,
    neural_config: NeuralModelConfig,
    ray_debug: bool,
    checkpoint_csv: Path | None = None,
) -> pd.DataFrame:
    univariate_models = [model_name for model_name in model_names if model_name not in PANEL_MODELS]
    panel_models = [model_name for model_name in model_names if model_name in PANEL_MODELS]
    remote_task = ray.remote(num_cpus=max_cpus_per_task)(_forecast_model_task_remote)
    wide_arg = ray.put(wide)

    checkpoint_csv = Path(checkpoint_csv) if checkpoint_csv is not None else None
    completed_keys = (
        _load_completed_task_keys(checkpoint_csv, horizon=horizon, n_series=len(series_cols))
        if checkpoint_csv is not None
        else set()
    )

    futures = []
    submit_times = {}
    skipped = 0
    for model_name in univariate_models:
        for series_id in series_cols:
            task_key = (model_name, str(series_id))
            if task_key in completed_keys:
                skipped += 1
                continue
            ref = remote_task.remote(
                wide_arg,
                [series_id],
                model_name=model_name,
                task_series=str(series_id),
                forecast_mode=forecast_mode,
                horizon=horizon,
                season_length=season_length,
                freq=freq,
                n_jobs=n_jobs,
                neural_config=neural_config,
                ray_debug=ray_debug,
            )
            futures.append(ref)
            submit_times[ref] = time.time()

    for model_name in panel_models:
        task_key = (model_name, "panel")
        if task_key in completed_keys:
            skipped += 1
            continue
        ref = remote_task.remote(
            wide_arg,
            series_cols,
            model_name=model_name,
            task_series="panel",
            forecast_mode=forecast_mode,
            horizon=horizon,
            season_length=season_length,
            freq=freq,
            n_jobs=n_jobs,
            neural_config=neural_config,
            ray_debug=ray_debug,
        )
        futures.append(ref)
        submit_times[ref] = time.time()

    _print_progress(
        "Submitted Ray benchmark tasks: "
        f"univariate_tasks={len(univariate_models) * len(series_cols)}, "
        f"panel_tasks={len(panel_models)}, total={len(futures)}, skipped={skipped}"
    )
    if not futures:
        if checkpoint_csv is not None:
            return _load_task_checkpoint_forecasts(checkpoint_csv, model_names)
        raise ValueError("No benchmark tasks were submitted.")

    pending = list(futures)
    completed = 0
    frames_by_model = {model_name: [] for model_name in model_names}
    while pending:
        ready, pending = ray.wait(pending, num_returns=1, timeout=60.0)
        if not ready:
            now = time.time()
            slowest_wait = now - min(submit_times.values()) if submit_times else 0.0
            _print_progress(
                "Ray benchmark wait: "
                f"done={completed}/{len(futures)}, pending={len(pending)}, "
                f"slowest_pending_sec={slowest_wait:.1f}, available={ray.available_resources()}"
            )
            continue

        for ref in ready:
            model_name, task_series, frame = ray.get(ref)
            if checkpoint_csv is not None:
                _append_task_checkpoint(checkpoint_csv, model_name, task_series, frame)
            else:
                frames_by_model[model_name].append(frame)
            submit_times.pop(ref, None)
            completed += 1
            if ray_debug:
                _print_progress(
                    f"Ray benchmark collected: model={model_name}, series={task_series}, "
                    f"done={completed}/{len(futures)}"
                )

    if checkpoint_csv is not None:
        return _load_task_checkpoint_forecasts(checkpoint_csv, model_names)

    model_frames = []
    for model_name in model_names:
        frame = pd.concat(frames_by_model[model_name], ignore_index=True, sort=False)
        sort_cols = [col for col in ("origin", "date", "series_id") if col in frame.columns]
        if sort_cols:
            frame = frame.sort_values(sort_cols).reset_index(drop=True)
        model_frames.append(frame)
    return _merge_forecast_frames(model_frames)


def _wide_value_frame(long_forecasts: pd.DataFrame, value_col: str, series_cols: list[str]) -> pd.DataFrame:
    wide = long_forecasts.pivot(index="date", columns="series_id", values=value_col).reset_index()
    wide.columns.name = None
    return wide.loc[:, ["date", *series_cols]]


def _wide_model_frame(long_forecasts: pd.DataFrame, model: str, series_cols: list[str]) -> pd.DataFrame:
    return _wide_value_frame(long_forecasts, model, series_cols)


def _compute_metrics(
    long_forecasts: pd.DataFrame,
    test_wide: pd.DataFrame,
    train_wide: pd.DataFrame,
    series_cols: list[str],
    model_names: list[str],
    *,
    season_length: int,
    forecast_mode: str,
    horizon: int,
    summary_context: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = test_wide.loc[:, series_cols].to_numpy(dtype=float)
    train = train_wide.loc[:, series_cols].to_numpy(dtype=float)
    forecasts = {
        model: _wide_model_frame(long_forecasts, model, series_cols).loc[:, series_cols].to_numpy(dtype=float)
        for model in model_names
    }
    forecast_variances = {
        model: _wide_value_frame(long_forecasts, _variance_col(model), series_cols).loc[:, series_cols].to_numpy(dtype=float)
        for model in model_names
        if _variance_col(model) in long_forecasts.columns
    }
    return forecast_metric_tables(
        forecasts,
        test,
        train,
        test_wide["date"],
        series_cols,
        model_names=model_names,
        season_length=season_length,
        forecast_mode=forecast_mode,
        horizon=horizon,
        summary_context=summary_context,
        forecast_variances=forecast_variances,
    )


def _save_config(output_dir: Path, config: dict[str, Any]) -> Path:
    path = output_dir / "config_benchmark.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def _save_benchmark_outputs(
    output_dir: Path = DEFAULT_BENCHMARK_CONFIG.output_dir,
    *,
    long_forecasts: pd.DataFrame,
    test_wide: pd.DataFrame,
    by_series: pd.DataFrame,
    by_origin: pd.DataFrame,
    summary: pd.DataFrame,
    series_cols: list[str],
    model_names: list[str],
    config: dict[str, Any],
    checkpoint_csv: Path | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_long_csv = output_dir / "benchmark_forecasts_long.csv"
    point_forecasts = _point_forecast_frame(long_forecasts, model_names)
    variance_forecasts = _variance_frame(long_forecasts, model_names)
    point_forecasts.to_csv(forecast_long_csv, index=False)
    variance_forecasts.to_csv(output_dir / FORECAST_VARIANCE_LONG_FILE, index=False)
    if checkpoint_csv is not None:
        _archive_checkpoint_outputs(checkpoint_csv, forecast_long_csv)
    test_wide.to_csv(output_dir / "test.csv", index=False)

    test_values = test_wide.loc[:, series_cols].to_numpy(dtype=float)
    for model in model_names:
        forecast_wide_raw = _wide_model_frame(point_forecasts, model, series_cols)
        forecast_wide = pd.concat(
            [
                forecast_wide_raw.loc[:, ["date"]].reset_index(drop=True),
                pd.DataFrame(
                    forecast_wide_raw.loc[:, series_cols].to_numpy(dtype=float),
                    columns=series_cols,
                ),
            ],
            axis=1,
        )
        forecast_wide.to_csv(output_dir / f"forecast_{model}.csv", index=False)
        variance_wide_raw = _wide_model_frame(variance_forecasts, model, series_cols)
        variance_wide = pd.concat(
            [
                variance_wide_raw.loc[:, ["date"]].reset_index(drop=True),
                pd.DataFrame(
                    variance_wide_raw.loc[:, series_cols].to_numpy(dtype=float),
                    columns=series_cols,
                ),
            ],
            axis=1,
        )
        variance_wide.to_csv(output_dir / f"forecast_variance_{model}.csv", index=False)
        error_wide = forecast_wide.copy()
        error_wide.loc[:, series_cols] = forecast_wide.loc[:, series_cols].to_numpy(dtype=float) - test_values
        error_wide.to_csv(output_dir / f"errors_{model}.csv", index=False)

    by_series.to_csv(output_dir / "metrics_by_series.csv", index=False)
    by_origin.to_csv(output_dir / "metrics_by_origin.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    _save_config(output_dir, config)
    (output_dir / SUCCESS_MARKER).touch()


def run_forecast_benchmark(
    *,
    data_dir: Path = DEFAULT_BENCHMARK_CONFIG.data_dir,
    output_dir: Path,
    data_path: Path | None = None,
    data_format: str = DEFAULT_BENCHMARK_CONFIG.data_format,
    date_col: str = DEFAULT_BENCHMARK_CONFIG.date_col,
    series_col: str = DEFAULT_BENCHMARK_CONFIG.series_col,
    value_col: str = DEFAULT_BENCHMARK_CONFIG.value_col,
    models: str | list[str] = DEFAULT_BENCHMARK_CONFIG.models,
    forecast_mode: str = DEFAULT_BENCHMARK_CONFIG.forecast_mode,
    horizon: int = DEFAULT_BENCHMARK_CONFIG.horizon,
    season_length: int = DEFAULT_BENCHMARK_CONFIG.season_length,
    freq: str = DEFAULT_BENCHMARK_CONFIG.freq,
    series_depth: int | None = DEFAULT_BENCHMARK_CONFIG.series_depth,
    series_suffix: str | None = DEFAULT_BENCHMARK_CONFIG.series_suffix,
    max_series: int = DEFAULT_BENCHMARK_CONFIG.max_series,
    clip_nonnegative: bool = DEFAULT_BENCHMARK_CONFIG.clip_nonnegative,
    ray_num_cpus: int = DEFAULT_BENCHMARK_CONFIG.ray_num_cpus,
    max_cpus_per_task: int = DEFAULT_BENCHMARK_CONFIG.max_cpus_per_task,
    ray_debug: bool = DEFAULT_BENCHMARK_CONFIG.ray_debug,
    resume: bool = DEFAULT_BENCHMARK_CONFIG.resume,
    resume_csv: Path | None = None,
    deepar_input_size: int = DEFAULT_BENCHMARK_CONFIG.neural_config.input_size,
    deepar_max_steps: int = DEFAULT_BENCHMARK_CONFIG.neural_config.max_steps,
    deepar_hidden_size: int = DEFAULT_BENCHMARK_CONFIG.neural_config.hidden_size,
    deepar_lstm_layers: int = DEFAULT_BENCHMARK_CONFIG.neural_config.num_layers,
    deepar_batch_size: int = DEFAULT_BENCHMARK_CONFIG.neural_config.batch_size,
    deepar_learning_rate: float = DEFAULT_BENCHMARK_CONFIG.neural_config.learning_rate,
    deepar_scaler_type: str = DEFAULT_BENCHMARK_CONFIG.neural_config.scaler_type,
    deepar_random_seed: int = DEFAULT_BENCHMARK_CONFIG.neural_config.random_seed,
) -> pd.DataFrame:
    forecast_mode = _normalize_forecast_mode(forecast_mode)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    data_path = None if data_path is None else Path(data_path)
    resume_csv = None if resume_csv is None else Path(resume_csv)
    series_suffix = None if series_suffix == "" else series_suffix
    neural_config = NeuralModelConfig(
        input_size=int(deepar_input_size),
        max_steps=int(deepar_max_steps),
        hidden_size=int(deepar_hidden_size),
        num_layers=int(deepar_lstm_layers),
        batch_size=int(deepar_batch_size),
        learning_rate=float(deepar_learning_rate),
        scaler_type=str(deepar_scaler_type),
        random_seed=int(deepar_random_seed),
    )
    benchmark_config = BenchmarkConfig(
        data_dir=data_dir,
        output_dir=output_dir,
        data_path=data_path,
        data_format=data_format,
        date_col=date_col,
        series_col=series_col,
        value_col=value_col,
        models=models,
        forecast_mode=forecast_mode,
        horizon=horizon,
        season_length=season_length,
        freq=freq,
        series_depth=series_depth,
        series_suffix=series_suffix,
        max_series=max_series,
        clip_nonnegative=clip_nonnegative,
        ray_num_cpus=ray_num_cpus,
        max_cpus_per_task=max_cpus_per_task,
        ray_debug=ray_debug,
        resume=resume,
        resume_csv=resume_csv,
        neural_config=neural_config,
    )
    success_path = output_dir / SUCCESS_MARKER
    summary_path = output_dir / "metrics_summary.csv"
    variance_path = output_dir / FORECAST_VARIANCE_LONG_FILE
    if resume and success_path.exists() and summary_path.exists() and variance_path.exists():
        _print_progress(f"Benchmark output already complete: {success_path}; skipping run")
        return pd.read_csv(summary_path)

    if ray_num_cpus < 1:
        raise ValueError("ray_num_cpus must be >= 1.")
    if max_cpus_per_task < 1:
        raise ValueError("max_cpus_per_task must be >= 1.")
    use_ray = benchmark_config.use_ray
    if use_ray and ray is None:
        raise RuntimeError(f"Ray could not be imported: {RAY_IMPORT_ERROR!r}")
    if use_ray and max_cpus_per_task > ray_num_cpus:
        raise ValueError("max_cpus_per_task cannot exceed ray_num_cpus in Ray mode.")
    input_path = benchmark_config.input_path

    wide = load_panel_data(
        input_path,
        data_format=data_format,
        date_col=date_col,
        series_col=series_col,
        value_col=value_col,
    )
    all_series = _base_series_columns(wide, series_depth=series_depth, series_suffix=series_suffix)
    series_cols = all_series[: min(max_series, len(all_series))] if max_series > 0 else all_series
    if horizon < 1 or horizon >= len(wide):
        raise ValueError("horizon must be positive and smaller than the number of observations.")
    if len(series_cols) < 1:
        raise ValueError("No series selected.")

    model_names = _parse_model_names(models)
    checkpoint_csv = None
    if resume and use_ray:
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_csv = Path(resume_csv) if resume_csv is not None else output_dir / "benchmark_task_results.csv"
        _print_progress(f"Benchmark resume checkpoint: {checkpoint_csv}")

    ray_started_here = False
    if use_ray and not ray.is_initialized():
        _print_progress(f"Starting Ray: ray_num_cpus={ray_num_cpus}, max_cpus_per_task={max_cpus_per_task}")
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=ray_num_cpus,
            _node_ip_address="127.0.0.1",
        )
        ray_started_here = True
        _print_progress("Ray started")
    if use_ray:
        available = ray.available_resources()
        available_cpus = float(available.get("CPU", 0.0))
        if available_cpus < float(max_cpus_per_task):
            raise RuntimeError(
                f"Ray has no available CPUs after initialization: {available}. "
                "Stop stale Ray workers with `ray stop --force` or reduce other Ray jobs."
            )

    try:
        if use_ray:
            long_forecasts = _forecast_models_with_ray(
                wide,
                series_cols,
                model_names=model_names,
                forecast_mode=forecast_mode,
                horizon=horizon,
                season_length=season_length,
                freq=freq,
                n_jobs=STATSFORECAST_N_JOBS,
                max_cpus_per_task=max_cpus_per_task,
                neural_config=neural_config,
                ray_debug=ray_debug,
                checkpoint_csv=checkpoint_csv,
            )
        else:
            long_forecasts = _forecast_by_mode(
                wide,
                series_cols,
                model_names=model_names,
                forecast_mode=forecast_mode,
                horizon=horizon,
                season_length=season_length,
                freq=freq,
                n_jobs=STATSFORECAST_N_JOBS,
                neural_config=neural_config,
            )

        if clip_nonnegative:
            long_forecasts.loc[:, model_names] = long_forecasts.loc[:, model_names].clip(lower=0.0)

        test_wide = wide.iloc[-horizon:].loc[:, ["date", *series_cols]].reset_index(drop=True)
        train_wide = wide.iloc[: len(wide) - horizon].loc[:, ["date", *series_cols]].reset_index(drop=True)

        by_series, by_origin, summary = _compute_metrics(
            long_forecasts,
            test_wide,
            train_wide,
            series_cols,
            model_names,
            season_length=season_length,
            forecast_mode=forecast_mode,
            horizon=horizon,
            summary_context={
                "data": str(input_path),
                "series_depth": -1 if series_depth is None else int(series_depth),
                "season_length": int(season_length),
                "train_start": str(train_wide["date"].iloc[0].date()),
                "train_end": str(train_wide["date"].iloc[-1].date()),
                "test_start": str(test_wide["date"].iloc[0].date()),
                "test_end": str(test_wide["date"].iloc[-1].date()),
            },
        )

        _save_benchmark_outputs(
            output_dir,
            long_forecasts=long_forecasts,
            test_wide=test_wide,
            by_series=by_series,
            by_origin=by_origin,
            summary=summary,
            series_cols=series_cols,
            model_names=model_names,
            checkpoint_csv=checkpoint_csv,
            config=benchmark_config.output_config(model_names=model_names, checkpoint_csv=checkpoint_csv),
        )

        return summary
    finally:
        if ray_started_here:
            _print_progress("Shutting down Ray")
            ray.shutdown()
            _print_progress("Ray shut down")


def _load_config_defaults(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    for key in ("data_dir", "data_path", "output_dir"):
        if key in config and config[key] is not None:
            config[key] = Path(config[key])
    if "models" in config and isinstance(config["models"], list):
        config["models"] = ",".join(config["models"])
    return config


def _benchmark_config_from_args(args: argparse.Namespace) -> BenchmarkConfig:
    series_depth = None if args.series_depth is None or int(args.series_depth) < 0 else int(args.series_depth)
    series_suffix = None if args.series_suffix in (None, "") else args.series_suffix
    return BenchmarkConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        data_path=args.data_path,
        data_format=args.data_format,
        date_col=args.date_col,
        series_col=args.series_col,
        value_col=args.value_col,
        models=args.models,
        forecast_mode=args.forecast_mode,
        horizon=args.horizon,
        season_length=args.season_length,
        freq=args.freq,
        series_depth=series_depth,
        series_suffix=series_suffix,
        max_series=args.max_series,
        clip_nonnegative=args.clip_nonnegative,
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
        ray_debug=args.ray_debug,
        resume=args.resume,
        resume_csv=args.resume_csv,
        neural_config=NeuralModelConfig(
            input_size=args.deepar_input_size,
            max_steps=args.deepar_max_steps,
            hidden_size=args.deepar_hidden_size,
            num_layers=args.deepar_lstm_layers,
            batch_size=args.deepar_batch_size,
            learning_rate=args.deepar_learning_rate,
            scaler_type=args.deepar_scaler_type,
            random_seed=args.deepar_random_seed,
        ),
    )


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config-json", type=Path, default=None)
    known, _ = pre.parse_known_args()
    defaults = _load_config_defaults(known.config_json)
    default_config = DEFAULT_BENCHMARK_CONFIG
    default_neural = default_config.neural_config

    parser = argparse.ArgumentParser(description="Run forecasting benchmarks on panel time-series data.", parents=[pre])
    parser.add_argument("--data-dir", type=Path, default=default_config.data_dir)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=default_config.data_path,
        help="CSV/parquet path in wide or long panel format. Overrides --data-dir/tourism.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=default_config.output_dir)
    parser.add_argument("--data-format", choices=["auto", "wide", "long"], default=default_config.data_format)
    parser.add_argument("--date-col", type=str, default=default_config.date_col)
    parser.add_argument("--series-col", type=str, default=default_config.series_col)
    parser.add_argument("--value-col", type=str, default=default_config.value_col)
    parser.add_argument("--models", type=str, default=default_config.models)
    parser.add_argument(
        "--forecast-mode",
        type=str,
        default=default_config.forecast_mode,
        help="Forecast mode, or comma-separated modes: recursive, rolling-origin, closed-loop.",
    )
    parser.add_argument("--horizon", type=int, default=default_config.horizon)
    parser.add_argument("--season-length", type=int, default=default_config.season_length)
    parser.add_argument("--freq", type=str, default=default_config.freq)
    parser.add_argument("--series-depth", type=int, default=-1 if default_config.series_depth is None else default_config.series_depth)
    parser.add_argument(
        "--series-suffix",
        type=str,
        default="" if default_config.series_suffix is None else default_config.series_suffix,
        help="Series-column suffix used for selection; pass '' to use all non-date columns.",
    )
    parser.add_argument("--max-series", type=int, default=default_config.max_series)
    parser.add_argument("--clip-nonnegative", action="store_true", default=default_config.clip_nonnegative)
    parser.add_argument("--ray-num-cpus", type=int, default=default_config.ray_num_cpus)
    parser.add_argument("--max-cpus-per-task", type=int, default=default_config.max_cpus_per_task)
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable Ray task checkpoint resume.")
    parser.set_defaults(resume=default_config.resume)
    parser.add_argument("--resume-csv", type=Path, default=default_config.resume_csv, help="Checkpoint CSV path; defaults to OUTPUT_DIR/benchmark_task_results.csv.")
    parser.add_argument(
        "--deepar-input-size",
        type=int,
        default=default_neural.input_size,
        help="DeepAR context length; 0 uses max(3*h, 2*season_length).",
    )
    parser.add_argument("--deepar-max-steps", type=int, default=default_neural.max_steps)
    parser.add_argument("--deepar-hidden-size", type=int, default=default_neural.hidden_size)
    parser.add_argument("--deepar-lstm-layers", type=int, default=default_neural.num_layers)
    parser.add_argument("--deepar-batch-size", type=int, default=default_neural.batch_size)
    parser.add_argument("--deepar-learning-rate", type=float, default=default_neural.learning_rate)
    parser.add_argument("--deepar-scaler-type", type=str, default=default_neural.scaler_type)
    parser.add_argument("--deepar-random-seed", type=int, default=default_neural.random_seed)
    parser.add_argument(
        "--nbeats-input-size",
        dest="deepar_input_size",
        type=int,
        default=argparse.SUPPRESS,
        help="NBEATS context length; 0 uses max(3*h, 2*season_length).",
    )
    parser.add_argument("--nbeats-max-steps", dest="deepar_max_steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--nbeats-hidden-size", dest="deepar_hidden_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--nbeats-num-layers", dest="deepar_lstm_layers", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--nbeats-batch-size", dest="deepar_batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--nbeats-learning-rate", dest="deepar_learning_rate", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--nbeats-scaler-type", dest="deepar_scaler_type", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--nbeats-random-seed", dest="deepar_random_seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--ray-debug",
        action="store_true",
        help="Print Ray benchmark model task start/end messages.",
    )
    parser.set_defaults(**defaults)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        forecast_modes = _parse_forecast_modes(args.forecast_mode)
    except ValueError as exc:
        raise SystemExit(f"run_benchmarks.py: error: {exc}") from exc

    benchmark_config = _benchmark_config_from_args(args)
    run_kwargs = benchmark_config.run_kwargs()

    if len(forecast_modes) == 1:
        summary = run_forecast_benchmark(
            output_dir=benchmark_config.output_dir,
            forecast_mode=forecast_modes[0],
            resume_csv=benchmark_config.resume_csv,
            **run_kwargs,
        )
    else:
        benchmark_config.output_dir.mkdir(parents=True, exist_ok=True)
        summary_frames = []
        by_series_frames = []
        by_origin_frames = []
        forecast_frames = []
        variance_frames = []
        for forecast_mode in forecast_modes:
            mode_output_dir = benchmark_config.output_dir / forecast_mode
            mode_resume_csv = None
            if benchmark_config.resume_csv is not None:
                mode_resume_csv = benchmark_config.resume_csv.with_name(
                    f"{benchmark_config.resume_csv.stem}_{forecast_mode}{benchmark_config.resume_csv.suffix}"
                )
            _print_progress(f"Running benchmark forecast_mode={forecast_mode}, output_dir={mode_output_dir}")
            summary_frames.append(run_forecast_benchmark(
                output_dir=mode_output_dir,
                forecast_mode=forecast_mode,
                resume_csv=mode_resume_csv,
                **run_kwargs,
            ))
            by_series_frames.append(pd.read_csv(mode_output_dir / "metrics_by_series.csv"))
            by_origin_frames.append(pd.read_csv(mode_output_dir / "metrics_by_origin.csv"))
            forecasts = pd.read_csv(mode_output_dir / "benchmark_forecasts_long.csv")
            forecasts.insert(0, "forecast_mode", forecast_mode)
            forecast_frames.append(forecasts)
            variances = pd.read_csv(mode_output_dir / FORECAST_VARIANCE_LONG_FILE)
            variances.insert(0, "forecast_mode", forecast_mode)
            variance_frames.append(variances)

        summary = pd.concat(summary_frames, ignore_index=True, sort=False)
        summary.to_csv(benchmark_config.output_dir / "metrics_summary.csv", index=False)
        pd.concat(by_series_frames, ignore_index=True, sort=False).to_csv(
            benchmark_config.output_dir / "metrics_by_series.csv", index=False
        )
        pd.concat(by_origin_frames, ignore_index=True, sort=False).to_csv(
            benchmark_config.output_dir / "metrics_by_origin.csv", index=False
        )
        pd.concat(forecast_frames, ignore_index=True, sort=False).to_csv(
            benchmark_config.output_dir / "benchmark_forecasts_long.csv", index=False
        )
        pd.concat(variance_frames, ignore_index=True, sort=False).to_csv(
            benchmark_config.output_dir / FORECAST_VARIANCE_LONG_FILE, index=False
        )

    print("Benchmark summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
