#!/usr/bin/env python3
"""Run forecasting benchmarks on panel time-series data."""

from __future__ import annotations

import argparse
import json
import logging
import time
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
STATSMODELS_MODELS = {"ar1", "var1", "ar1_snaive", "var1_snaive"}
NEURALFORECAST_MODELS = {"deepar", "nbeats"}
PANEL_MODELS = {"var1", "var1_snaive", "deepar", "nbeats"}
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


def _task_expected_rows(task_series: str, *, horizon: int, n_series: int) -> int:
    return int(horizon) * (int(n_series) if task_series == "panel" else 1)


def _load_completed_task_keys(checkpoint_csv: Path, *, horizon: int, n_series: int) -> set[tuple[str, str]]:
    if not checkpoint_csv.exists() or checkpoint_csv.stat().st_size == 0:
        return set()
    try:
        tasks = pd.read_csv(checkpoint_csv, usecols=[TASK_MODEL_COL, TASK_SERIES_COL])
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
    out = frame.loc[:, [*key_cols, model_name]].rename(columns={model_name: TASK_FORECAST_COL})
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
        frame = frame.loc[:, [*key_cols, TASK_FORECAST_COL]].rename(columns={TASK_FORECAST_COL: model_name})
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


def _series_code(series_id: str, suffix: str = BASE_SUFFIX) -> str:
    return series_id[: -len(suffix)] if suffix and series_id.endswith(suffix) else series_id


def _series_depth(series_id: str, suffix: str = BASE_SUFFIX) -> int:
    code = _series_code(series_id, suffix)
    return 0 if code == "Total" else len(code)


def _base_series_columns(
    wide: pd.DataFrame,
    series_depth: int | None = DETAILED_CODE_DEPTH,
    series_suffix: str | None = BASE_SUFFIX,
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
        "ar": "ar1",
        "ar_1": "ar1",
        "ar1_seasonal_naive": "ar1_snaive",
        "ar1_seasonal_naive_resid": "ar1_snaive",
        "ar1_snaive_resid": "ar1_snaive",
        "var": "var1",
        "var_1": "var1",
        "var1_seasonal_naive": "var1_snaive",
        "var1_seasonal_naive_resid": "var1_snaive",
        "var1_snaive_resid": "var1_snaive",
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


def _deepar_input_size(h: int, season_length: int, requested: int) -> int:
    if requested > 0:
        return int(requested)
    return int(max(3 * h, 2 * season_length, 1))


def _forecast_deepar_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
    season_length: int,
    freq: str,
    deepar_input_size: int,
    deepar_max_steps: int,
    deepar_hidden_size: int,
    deepar_lstm_layers: int,
    deepar_batch_size: int,
    deepar_learning_rate: float,
    deepar_scaler_type: str,
    deepar_random_seed: int,
) -> pd.DataFrame:
    NeuralForecast, DeepAR = _load_deepar()
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    input_size = _deepar_input_size(h, season_length, deepar_input_size)
    model = DeepAR(
        h=h,
        input_size=input_size,
        lstm_n_layers=int(deepar_lstm_layers),
        lstm_hidden_size=int(deepar_hidden_size),
        lstm_dropout=0.0,
        max_steps=int(deepar_max_steps),
        learning_rate=float(deepar_learning_rate),
        batch_size=int(deepar_batch_size),
        scaler_type=str(deepar_scaler_type),
        random_seed=int(deepar_random_seed),
        alias="deepar",
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )
    nf = NeuralForecast(models=[model], freq=freq)
    train_long = _wide_to_statsforecast_long(train_wide, series_cols)
    nf.fit(df=train_long, verbose=False)
    out = nf.predict(verbose=False).rename(columns={"unique_id": "series_id", "ds": "date"})

    dates = pd.to_datetime(pd.Series(forecast_dates)).reset_index(drop=True)
    expected = pd.MultiIndex.from_product(
        [series_cols, dates], names=["series_id", "date"]
    ).to_frame(index=False)
    out = expected.merge(out.loc[:, ["series_id", "date", "deepar"]], on=["series_id", "date"], how="left")
    if out["deepar"].isna().any():
        raise RuntimeError("DeepAR did not return forecasts for all requested series/date pairs.")
    return out


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
    nbeats_input_size: int,
    nbeats_max_steps: int,
    nbeats_hidden_size: int,
    nbeats_num_layers: int,
    nbeats_batch_size: int,
    nbeats_learning_rate: float,
    nbeats_scaler_type: str,
    nbeats_random_seed: int,
) -> pd.DataFrame:
    NeuralForecast, NBEATS = _load_nbeats()
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
    try:
        import torch
        torch.set_num_threads(1)
    except Exception:
        pass

    input_size = _deepar_input_size(h, season_length, nbeats_input_size)
    stack_types, n_blocks, mlp_units = _nbeats_stack_config(h, nbeats_hidden_size, nbeats_num_layers)
    model = NBEATS(
        h=h,
        input_size=input_size,
        stack_types=stack_types,
        n_blocks=n_blocks,
        mlp_units=mlp_units,
        max_steps=int(nbeats_max_steps),
        learning_rate=float(nbeats_learning_rate),
        batch_size=int(nbeats_batch_size),
        scaler_type=str(nbeats_scaler_type),
        random_seed=int(nbeats_random_seed),
        alias="nbeats",
        accelerator="cpu",
        devices=1,
        enable_progress_bar=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        logger=False,
    )
    nf = NeuralForecast(models=[model], freq=freq)
    train_long = _wide_to_statsforecast_long(train_wide, series_cols)
    nf.fit(df=train_long, verbose=False)
    out = nf.predict(verbose=False).rename(columns={"unique_id": "series_id", "ds": "date"})

    dates = pd.to_datetime(pd.Series(forecast_dates)).reset_index(drop=True)
    expected = pd.MultiIndex.from_product(
        [series_cols, dates], names=["series_id", "date"]
    ).to_frame(index=False)
    out = expected.merge(out.loc[:, ["series_id", "date", "nbeats"]], on=["series_id", "date"], how="left")
    if out["nbeats"].isna().any():
        raise RuntimeError("NBEATS did not return forecasts for all requested series/date pairs.")
    return out


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
    out = sf.forecast(df=_wide_to_statsforecast_long(train_wide, series_cols), h=h)
    return out.rename(columns={"unique_id": "series_id", "ds": "date"})



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


def _forecast_ar1_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
) -> pd.DataFrame:
    dates = pd.to_datetime(pd.Series(forecast_dates)).reset_index(drop=True)
    if len(dates) != h:
        raise RuntimeError("AR(1) forecast_dates length does not match h.")

    rows = []
    for series_id in series_cols:
        y = train_wide.loc[:, series_id].to_numpy(dtype=float)
        if y.shape[0] <= 1:
            raise ValueError("AR(1) requires at least two training observations.")
        try:
            forecast = AutoReg(y, lags=1, trend="c", old_names=False).fit().forecast(steps=h)
        except Exception:
            forecast = np.repeat(y[-1], h)
        for i, date in enumerate(dates):
            rows.append({"series_id": series_id, "date": date, "ar1": float(forecast[i])})
    return pd.DataFrame(rows)



def _forecast_ar1_snaive_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
    season_length: int,
) -> pd.DataFrame:
    dates = pd.to_datetime(pd.Series(forecast_dates)).reset_index(drop=True)
    if len(dates) != h:
        raise RuntimeError("AR(1) seasonal-naive forecast_dates length does not match h.")

    rows = []
    for series_id in series_cols:
        y = train_wide.loc[:, series_id].to_numpy(dtype=float)
        baseline = _seasonal_naive_array(y[:, None], h, season_length)[:, 0]
        residual = _seasonal_residual_array(y[:, None], season_length)[:, 0]
        if residual.shape[0] <= 1:
            residual_forecast = np.zeros(h, dtype=float)
        else:
            try:
                residual_forecast = AutoReg(residual, lags=1, trend="c", old_names=False).fit().forecast(steps=h)
            except Exception:
                residual_forecast = np.zeros(h, dtype=float)
        forecast = baseline + np.asarray(residual_forecast, dtype=float)
        for i, date in enumerate(dates):
            rows.append({"series_id": series_id, "date": date, "ar1_snaive": float(forecast[i])})
    return pd.DataFrame(rows)


def _forecast_var1_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
) -> pd.DataFrame:
    Y = train_wide.loc[:, series_cols].to_numpy(dtype=float)
    if Y.shape[0] <= 1:
        raise ValueError("VAR(1) requires at least two training observations.")

    fit = VAR(pd.DataFrame(Y, columns=series_cols)).fit(maxlags=1, trend="c")
    forecast = fit.forecast(Y[-fit.k_ar:], steps=h)
    dates = pd.to_datetime(pd.Series(forecast_dates)).reset_index(drop=True)
    if len(dates) != h:
        raise RuntimeError("VAR(1) forecast_dates length does not match h.")

    rows = []
    for j, series_id in enumerate(series_cols):
        for i, date in enumerate(dates):
            rows.append({"series_id": series_id, "date": date, "var1": float(forecast[i, j])})
    return pd.DataFrame(rows)



def _forecast_var1_snaive_once(
    train_wide: pd.DataFrame,
    series_cols: list[str],
    *,
    forecast_dates: pd.Series | list[pd.Timestamp],
    h: int,
    season_length: int,
) -> pd.DataFrame:
    Y = train_wide.loc[:, series_cols].to_numpy(dtype=float)
    baseline = _seasonal_naive_array(Y, h, season_length)
    residual = _seasonal_residual_array(Y, season_length)
    if residual.shape[0] <= 1:
        residual_forecast = np.zeros((h, len(series_cols)), dtype=float)
    else:
        try:
            fit = VAR(pd.DataFrame(residual, columns=series_cols)).fit(maxlags=1, trend="c")
            residual_forecast = fit.forecast(residual[-fit.k_ar:], steps=h)
        except Exception:
            residual_forecast = np.zeros((h, len(series_cols)), dtype=float)
    forecast = baseline + np.asarray(residual_forecast, dtype=float)

    dates = pd.to_datetime(pd.Series(forecast_dates)).reset_index(drop=True)
    if len(dates) != h:
        raise RuntimeError("VAR(1) seasonal-naive forecast_dates length does not match h.")

    rows = []
    for j, series_id in enumerate(series_cols):
        for i, date in enumerate(dates):
            rows.append({"series_id": series_id, "date": date, "var1_snaive": float(forecast[i, j])})
    return pd.DataFrame(rows)


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
    deepar_input_size: int,
    deepar_max_steps: int,
    deepar_hidden_size: int,
    deepar_lstm_layers: int,
    deepar_batch_size: int,
    deepar_learning_rate: float,
    deepar_scaler_type: str,
    deepar_random_seed: int,
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
                deepar_input_size=deepar_input_size,
                deepar_max_steps=deepar_max_steps,
                deepar_hidden_size=deepar_hidden_size,
                deepar_lstm_layers=deepar_lstm_layers,
                deepar_batch_size=deepar_batch_size,
                deepar_learning_rate=deepar_learning_rate,
                deepar_scaler_type=deepar_scaler_type,
                deepar_random_seed=deepar_random_seed,
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
                nbeats_input_size=deepar_input_size,
                nbeats_max_steps=deepar_max_steps,
                nbeats_hidden_size=deepar_hidden_size,
                nbeats_num_layers=deepar_lstm_layers,
                nbeats_batch_size=deepar_batch_size,
                nbeats_learning_rate=deepar_learning_rate,
                nbeats_scaler_type=deepar_scaler_type,
                nbeats_random_seed=deepar_random_seed,
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
    deepar_input_size: int,
    deepar_max_steps: int,
    deepar_hidden_size: int,
    deepar_lstm_layers: int,
    deepar_batch_size: int,
    deepar_learning_rate: float,
    deepar_scaler_type: str,
    deepar_random_seed: int,
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
        deepar_input_size=deepar_input_size,
        deepar_max_steps=deepar_max_steps,
        deepar_hidden_size=deepar_hidden_size,
        deepar_lstm_layers=deepar_lstm_layers,
        deepar_batch_size=deepar_batch_size,
        deepar_learning_rate=deepar_learning_rate,
        deepar_scaler_type=deepar_scaler_type,
        deepar_random_seed=deepar_random_seed,
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
    deepar_input_size: int,
    deepar_max_steps: int,
    deepar_hidden_size: int,
    deepar_lstm_layers: int,
    deepar_batch_size: int,
    deepar_learning_rate: float,
    deepar_scaler_type: str,
    deepar_random_seed: int,
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
            deepar_input_size=deepar_input_size,
            deepar_max_steps=deepar_max_steps,
            deepar_hidden_size=deepar_hidden_size,
            deepar_lstm_layers=deepar_lstm_layers,
            deepar_batch_size=deepar_batch_size,
            deepar_learning_rate=deepar_learning_rate,
            deepar_scaler_type=deepar_scaler_type,
            deepar_random_seed=deepar_random_seed,
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
    deepar_input_size: int,
    deepar_max_steps: int,
    deepar_hidden_size: int,
    deepar_lstm_layers: int,
    deepar_batch_size: int,
    deepar_learning_rate: float,
    deepar_scaler_type: str,
    deepar_random_seed: int,
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
            deepar_input_size=deepar_input_size,
            deepar_max_steps=deepar_max_steps,
            deepar_hidden_size=deepar_hidden_size,
            deepar_lstm_layers=deepar_lstm_layers,
            deepar_batch_size=deepar_batch_size,
            deepar_learning_rate=deepar_learning_rate,
            deepar_scaler_type=deepar_scaler_type,
            deepar_random_seed=deepar_random_seed,
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
    deepar_input_size: int,
    deepar_max_steps: int,
    deepar_hidden_size: int,
    deepar_lstm_layers: int,
    deepar_batch_size: int,
    deepar_learning_rate: float,
    deepar_scaler_type: str,
    deepar_random_seed: int,
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
            deepar_input_size=deepar_input_size,
            deepar_max_steps=deepar_max_steps,
            deepar_hidden_size=deepar_hidden_size,
            deepar_lstm_layers=deepar_lstm_layers,
            deepar_batch_size=deepar_batch_size,
            deepar_learning_rate=deepar_learning_rate,
            deepar_scaler_type=deepar_scaler_type,
            deepar_random_seed=deepar_random_seed,
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
    deepar_input_size: int,
    deepar_max_steps: int,
    deepar_hidden_size: int,
    deepar_lstm_layers: int,
    deepar_batch_size: int,
    deepar_learning_rate: float,
    deepar_scaler_type: str,
    deepar_random_seed: int,
) -> pd.DataFrame:
    if forecast_mode == "recursive":
        return _forecast_recursive(
            wide,
            series_cols,
            horizon=horizon,
            model_names=[model_name],
            season_length=season_length,
            freq=freq,
            n_jobs=n_jobs,
            deepar_input_size=deepar_input_size,
            deepar_max_steps=deepar_max_steps,
            deepar_hidden_size=deepar_hidden_size,
            deepar_lstm_layers=deepar_lstm_layers,
            deepar_batch_size=deepar_batch_size,
            deepar_learning_rate=deepar_learning_rate,
            deepar_scaler_type=deepar_scaler_type,
            deepar_random_seed=deepar_random_seed,
        )
    if forecast_mode == "closed-loop":
        return _forecast_closed_loop_single_model(
            wide,
            series_cols,
            horizon=horizon,
            model_name=model_name,
            season_length=season_length,
            freq=freq,
            n_jobs=n_jobs,
            deepar_input_size=deepar_input_size,
            deepar_max_steps=deepar_max_steps,
            deepar_hidden_size=deepar_hidden_size,
            deepar_lstm_layers=deepar_lstm_layers,
            deepar_batch_size=deepar_batch_size,
            deepar_learning_rate=deepar_learning_rate,
            deepar_scaler_type=deepar_scaler_type,
            deepar_random_seed=deepar_random_seed,
        )
    return _forecast_rolling_origin(
        wide,
        series_cols,
        horizon=horizon,
        model_names=[model_name],
        season_length=season_length,
        freq=freq,
        n_jobs=n_jobs,
        deepar_input_size=deepar_input_size,
        deepar_max_steps=deepar_max_steps,
        deepar_hidden_size=deepar_hidden_size,
        deepar_lstm_layers=deepar_lstm_layers,
        deepar_batch_size=deepar_batch_size,
        deepar_learning_rate=deepar_learning_rate,
        deepar_scaler_type=deepar_scaler_type,
        deepar_random_seed=deepar_random_seed,
    )


def _forecast_single_model_remote(
    wide: pd.DataFrame,
    series_cols: list[str],
    *,
    model_name: str,
    forecast_mode: str,
    horizon: int,
    season_length: int,
    freq: str,
    n_jobs: int,
    deepar_input_size: int,
    deepar_max_steps: int,
    deepar_hidden_size: int,
    deepar_lstm_layers: int,
    deepar_batch_size: int,
    deepar_learning_rate: float,
    deepar_scaler_type: str,
    deepar_random_seed: int,
    ray_debug: bool,
) -> tuple[str, pd.DataFrame]:
    if ray_debug:
        _print_progress(f"[Ray benchmark start] model={model_name}")
    start = time.time()
    frame = _forecast_single_model(
        wide,
        series_cols,
        model_name=model_name,
        forecast_mode=forecast_mode,
        horizon=horizon,
        season_length=season_length,
        freq=freq,
        n_jobs=n_jobs,
        deepar_input_size=deepar_input_size,
        deepar_max_steps=deepar_max_steps,
        deepar_hidden_size=deepar_hidden_size,
        deepar_lstm_layers=deepar_lstm_layers,
        deepar_batch_size=deepar_batch_size,
        deepar_learning_rate=deepar_learning_rate,
        deepar_scaler_type=deepar_scaler_type,
        deepar_random_seed=deepar_random_seed,
    )
    if ray_debug:
        _print_progress(f"[Ray benchmark done] model={model_name}, elapsed_sec={time.time() - start:.2f}")
    return model_name, frame


def _forecast_univariate_series_remote(
    wide: pd.DataFrame,
    series_id: str,
    *,
    model_name: str,
    forecast_mode: str,
    horizon: int,
    season_length: int,
    freq: str,
    n_jobs: int,
    deepar_input_size: int,
    deepar_max_steps: int,
    deepar_hidden_size: int,
    deepar_lstm_layers: int,
    deepar_batch_size: int,
    deepar_learning_rate: float,
    deepar_scaler_type: str,
    deepar_random_seed: int,
    ray_debug: bool,
) -> tuple[str, str, pd.DataFrame]:
    if ray_debug:
        _print_progress(f"[Ray benchmark start] model={model_name}, series={series_id}")
    start = time.time()
    frame = _forecast_single_model(
        wide,
        [series_id],
        model_name=model_name,
        forecast_mode=forecast_mode,
        horizon=horizon,
        season_length=season_length,
        freq=freq,
        n_jobs=n_jobs,
        deepar_input_size=deepar_input_size,
        deepar_max_steps=deepar_max_steps,
        deepar_hidden_size=deepar_hidden_size,
        deepar_lstm_layers=deepar_lstm_layers,
        deepar_batch_size=deepar_batch_size,
        deepar_learning_rate=deepar_learning_rate,
        deepar_scaler_type=deepar_scaler_type,
        deepar_random_seed=deepar_random_seed,
    )
    if ray_debug:
        _print_progress(
            f"[Ray benchmark done] model={model_name}, series={series_id}, "
            f"elapsed_sec={time.time() - start:.2f}"
        )
    return model_name, series_id, frame


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
    deepar_input_size: int,
    deepar_max_steps: int,
    deepar_hidden_size: int,
    deepar_lstm_layers: int,
    deepar_batch_size: int,
    deepar_learning_rate: float,
    deepar_scaler_type: str,
    deepar_random_seed: int,
    ray_debug: bool,
    checkpoint_csv: Path | None = None,
) -> pd.DataFrame:
    univariate_models = [model_name for model_name in model_names if model_name not in PANEL_MODELS]
    panel_models = [model_name for model_name in model_names if model_name in PANEL_MODELS]
    remote_univariate_task = ray.remote(num_cpus=max_cpus_per_task)(_forecast_univariate_series_remote)
    remote_panel_task = ray.remote(num_cpus=max_cpus_per_task)(_forecast_single_model_remote)
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
            ref = remote_univariate_task.remote(
                wide_arg,
                series_id,
                model_name=model_name,
                forecast_mode=forecast_mode,
                horizon=horizon,
                season_length=season_length,
                freq=freq,
                n_jobs=n_jobs,
                deepar_input_size=deepar_input_size,
                deepar_max_steps=deepar_max_steps,
                deepar_hidden_size=deepar_hidden_size,
                deepar_lstm_layers=deepar_lstm_layers,
                deepar_batch_size=deepar_batch_size,
                deepar_learning_rate=deepar_learning_rate,
                deepar_scaler_type=deepar_scaler_type,
                deepar_random_seed=deepar_random_seed,
                ray_debug=ray_debug,
            )
            futures.append(ref)
            submit_times[ref] = time.time()

    for model_name in panel_models:
        task_key = (model_name, "panel")
        if task_key in completed_keys:
            skipped += 1
            continue
        ref = remote_panel_task.remote(
            wide_arg,
            series_cols,
            model_name=model_name,
            forecast_mode=forecast_mode,
            horizon=horizon,
            season_length=season_length,
            freq=freq,
            n_jobs=n_jobs,
            deepar_input_size=deepar_input_size,
            deepar_max_steps=deepar_max_steps,
            deepar_hidden_size=deepar_hidden_size,
            deepar_lstm_layers=deepar_lstm_layers,
            deepar_batch_size=deepar_batch_size,
            deepar_learning_rate=deepar_learning_rate,
            deepar_scaler_type=deepar_scaler_type,
            deepar_random_seed=deepar_random_seed,
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
            result = ray.get(ref)
            if len(result) == 3:
                model_name, series_id, frame = result
                task_series = str(series_id)
            else:
                model_name, frame = result
                task_series = "panel"
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


def _wide_model_frame(long_forecasts: pd.DataFrame, model: str, series_cols: list[str]) -> pd.DataFrame:
    wide = long_forecasts.pivot(index="date", columns="series_id", values=model).reset_index()
    wide.columns.name = None
    return wide.loc[:, ["date", *series_cols]]


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
    )


def _save_config(output_dir: Path, config: dict[str, Any]) -> Path:
    path = output_dir / "config_tourism_benchmark.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def run_tourism_benchmark(
    *,
    data_dir: Path,
    output_dir: Path,
    data_path: Path | None = None,
    data_format: str = "auto",
    date_col: str = "date",
    series_col: str = "series_id",
    value_col: str = "value",
    models: str | list[str] = DEFAULT_MODELS,
    forecast_mode: str = "rolling-origin",
    horizon: int = 12,
    season_length: int = 12,
    freq: str = "MS",
    series_depth: int | None = DETAILED_CODE_DEPTH,
    series_suffix: str | None = BASE_SUFFIX,
    max_series: int = 0,
    clip_nonnegative: bool = False,
    ray_num_cpus: int = 1,
    max_cpus_per_task: int = 1,
    ray_debug: bool = False,
    resume: bool = True,
    resume_csv: Path | None = None,
    deepar_input_size: int = 0,
    deepar_max_steps: int = 200,
    deepar_hidden_size: int = 64,
    deepar_lstm_layers: int = 2,
    deepar_batch_size: int = 32,
    deepar_learning_rate: float = 1e-3,
    deepar_scaler_type: str = "robust",
    deepar_random_seed: int = 2026,
) -> pd.DataFrame:
    forecast_mode = _normalize_forecast_mode(forecast_mode)
    if ray_num_cpus < 1:
        raise ValueError("ray_num_cpus must be >= 1.")
    if max_cpus_per_task < 1:
        raise ValueError("max_cpus_per_task must be >= 1.")
    use_ray = ray_num_cpus > 1
    if use_ray and ray is None:
        raise RuntimeError(f"Ray could not be imported: {RAY_IMPORT_ERROR!r}")
    if use_ray and max_cpus_per_task > ray_num_cpus:
        raise ValueError("max_cpus_per_task cannot exceed ray_num_cpus in Ray mode.")
    input_path = data_path if data_path is not None else data_dir / "tourism.csv"
    series_suffix = None if series_suffix == "" else series_suffix

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
                deepar_input_size=deepar_input_size,
                deepar_max_steps=deepar_max_steps,
                deepar_hidden_size=deepar_hidden_size,
                deepar_lstm_layers=deepar_lstm_layers,
                deepar_batch_size=deepar_batch_size,
                deepar_learning_rate=deepar_learning_rate,
                deepar_scaler_type=deepar_scaler_type,
                deepar_random_seed=deepar_random_seed,
                ray_debug=ray_debug,
                checkpoint_csv=checkpoint_csv,
            )
        elif forecast_mode == "recursive":
            long_forecasts = _forecast_recursive(
                wide,
                series_cols,
                horizon=horizon,
                model_names=model_names,
                season_length=season_length,
                freq=freq,
                n_jobs=STATSFORECAST_N_JOBS,
                deepar_input_size=deepar_input_size,
                deepar_max_steps=deepar_max_steps,
                deepar_hidden_size=deepar_hidden_size,
                deepar_lstm_layers=deepar_lstm_layers,
                deepar_batch_size=deepar_batch_size,
                deepar_learning_rate=deepar_learning_rate,
                deepar_scaler_type=deepar_scaler_type,
                deepar_random_seed=deepar_random_seed,
            )
        elif forecast_mode == "closed-loop":
            long_forecasts = _forecast_closed_loop(
                wide,
                series_cols,
                horizon=horizon,
                model_names=model_names,
                season_length=season_length,
                freq=freq,
                n_jobs=STATSFORECAST_N_JOBS,
                deepar_input_size=deepar_input_size,
                deepar_max_steps=deepar_max_steps,
                deepar_hidden_size=deepar_hidden_size,
                deepar_lstm_layers=deepar_lstm_layers,
                deepar_batch_size=deepar_batch_size,
                deepar_learning_rate=deepar_learning_rate,
                deepar_scaler_type=deepar_scaler_type,
                deepar_random_seed=deepar_random_seed,
            )
        else:
            long_forecasts = _forecast_rolling_origin(
                wide,
                series_cols,
                horizon=horizon,
                model_names=model_names,
                season_length=season_length,
                freq=freq,
                n_jobs=STATSFORECAST_N_JOBS,
                deepar_input_size=deepar_input_size,
                deepar_max_steps=deepar_max_steps,
                deepar_hidden_size=deepar_hidden_size,
                deepar_lstm_layers=deepar_lstm_layers,
                deepar_batch_size=deepar_batch_size,
                deepar_learning_rate=deepar_learning_rate,
                deepar_scaler_type=deepar_scaler_type,
                deepar_random_seed=deepar_random_seed,
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

        output_dir.mkdir(parents=True, exist_ok=True)
        forecast_long_csv = output_dir / "tourism_benchmark_forecasts_long.csv"
        long_forecasts.to_csv(forecast_long_csv, index=False)
        if checkpoint_csv is not None:
            _archive_checkpoint_outputs(checkpoint_csv, forecast_long_csv)
        test_wide.to_csv(output_dir / "tourism_test.csv", index=False)
        for model in model_names:
            forecast_wide_raw = _wide_model_frame(long_forecasts, model, series_cols)
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
            forecast_wide.to_csv(output_dir / f"tourism_forecast_{model}.csv", index=False)
            error_wide = forecast_wide.copy()
            error_wide.loc[:, series_cols] = forecast_wide.loc[:, series_cols].to_numpy(dtype=float) - test_wide.loc[:, series_cols].to_numpy(dtype=float)
            error_wide.to_csv(output_dir / f"tourism_errors_{model}.csv", index=False)

        by_series.to_csv(output_dir / "metrics_by_series.csv", index=False)
        by_origin.to_csv(output_dir / "metrics_by_origin.csv", index=False)
        summary.to_csv(output_dir / "metrics_summary.csv", index=False)
        _save_config(output_dir, {
            "data_dir": str(data_dir),
            "data_path": None if data_path is None else str(data_path),
            "data_format": str(data_format),
            "date_col": str(date_col),
            "series_col": str(series_col),
            "value_col": str(value_col),
            "output_dir": str(output_dir),
            "models": model_names,
            "forecast_mode": forecast_mode,
            "horizon": int(horizon),
            "season_length": int(season_length),
            "freq": freq,
            "series_depth": -1 if series_depth is None else int(series_depth),
            "series_suffix": series_suffix,
            "max_series": int(max_series),
            "clip_nonnegative": bool(clip_nonnegative),
            "ray_num_cpus": int(ray_num_cpus),
            "max_cpus_per_task": int(max_cpus_per_task),
            "ray_debug": bool(ray_debug),
            "resume": bool(resume),
            "resume_csv": None if checkpoint_csv is None else str(checkpoint_csv),
            "deepar_input_size": int(deepar_input_size),
            "deepar_max_steps": int(deepar_max_steps),
            "deepar_hidden_size": int(deepar_hidden_size),
            "deepar_lstm_layers": int(deepar_lstm_layers),
            "deepar_batch_size": int(deepar_batch_size),
            "deepar_learning_rate": float(deepar_learning_rate),
            "deepar_scaler_type": str(deepar_scaler_type),
            "deepar_random_seed": int(deepar_random_seed),
            "nbeats_input_size": int(deepar_input_size),
            "nbeats_max_steps": int(deepar_max_steps),
            "nbeats_hidden_size": int(deepar_hidden_size),
            "nbeats_num_layers": int(deepar_lstm_layers),
            "nbeats_batch_size": int(deepar_batch_size),
            "nbeats_learning_rate": float(deepar_learning_rate),
            "nbeats_scaler_type": str(deepar_scaler_type),
            "nbeats_random_seed": int(deepar_random_seed),
        })
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


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config-json", type=Path, default=None)
    known, _ = pre.parse_known_args()
    defaults = _load_config_defaults(known.config_json)

    parser = argparse.ArgumentParser(description="Run forecasting benchmarks on panel time-series data.", parents=[pre])
    parser.add_argument("--data-dir", type=Path, default=Path("data/tourism"))
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="CSV/parquet path in wide or long panel format. Overrides --data-dir/tourism.csv.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tourism_benchmark_statsforecast"))
    parser.add_argument("--data-format", choices=["auto", "wide", "long"], default="auto")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--series-col", type=str, default="series_id")
    parser.add_argument("--value-col", type=str, default="value")
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS)
    parser.add_argument(
        "--forecast-mode",
        type=str,
        default="rolling-origin",
        help="Forecast mode, or comma-separated modes: recursive, rolling-origin, closed-loop.",
    )
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--freq", type=str, default="MS")
    parser.add_argument("--series-depth", type=int, default=DETAILED_CODE_DEPTH)
    parser.add_argument(
        "--series-suffix",
        type=str,
        default=BASE_SUFFIX,
        help="Series-column suffix used for selection; pass '' to use all non-date columns.",
    )
    parser.add_argument("--max-series", type=int, default=0)
    parser.add_argument("--clip-nonnegative", action="store_true")
    parser.add_argument("--ray-num-cpus", type=int, default=1)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Disable Ray task checkpoint resume.")
    parser.set_defaults(resume=True)
    parser.add_argument("--resume-csv", type=Path, default=None, help="Checkpoint CSV path; defaults to OUTPUT_DIR/benchmark_task_results.csv.")
    parser.add_argument(
        "--deepar-input-size",
        type=int,
        default=0,
        help="DeepAR context length; 0 uses max(3*h, 2*season_length).",
    )
    parser.add_argument("--deepar-max-steps", type=int, default=200)
    parser.add_argument("--deepar-hidden-size", type=int, default=64)
    parser.add_argument("--deepar-lstm-layers", type=int, default=2)
    parser.add_argument("--deepar-batch-size", type=int, default=32)
    parser.add_argument("--deepar-learning-rate", type=float, default=1e-3)
    parser.add_argument("--deepar-scaler-type", type=str, default="robust")
    parser.add_argument("--deepar-random-seed", type=int, default=2026)
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

    run_kwargs = {
        "data_dir": args.data_dir,
        "data_path": args.data_path,
        "data_format": args.data_format,
        "date_col": args.date_col,
        "series_col": args.series_col,
        "value_col": args.value_col,
        "models": args.models,
        "horizon": args.horizon,
        "season_length": args.season_length,
        "freq": args.freq,
        "series_depth": None if args.series_depth < 0 else args.series_depth,
        "series_suffix": None if args.series_suffix == "" else args.series_suffix,
        "max_series": args.max_series,
        "clip_nonnegative": args.clip_nonnegative,
        "ray_num_cpus": args.ray_num_cpus,
        "max_cpus_per_task": args.max_cpus_per_task,
        "ray_debug": args.ray_debug,
        "resume": args.resume,
        "deepar_input_size": args.deepar_input_size,
        "deepar_max_steps": args.deepar_max_steps,
        "deepar_hidden_size": args.deepar_hidden_size,
        "deepar_lstm_layers": args.deepar_lstm_layers,
        "deepar_batch_size": args.deepar_batch_size,
        "deepar_learning_rate": args.deepar_learning_rate,
        "deepar_scaler_type": args.deepar_scaler_type,
        "deepar_random_seed": args.deepar_random_seed,
    }

    if len(forecast_modes) == 1:
        summary = run_tourism_benchmark(
            output_dir=args.output_dir,
            forecast_mode=forecast_modes[0],
            resume_csv=args.resume_csv,
            **run_kwargs,
        )
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_frames = []
        by_series_frames = []
        by_origin_frames = []
        forecast_frames = []
        for forecast_mode in forecast_modes:
            mode_output_dir = args.output_dir / forecast_mode
            mode_resume_csv = None
            if args.resume_csv is not None:
                mode_resume_csv = args.resume_csv.with_name(f"{args.resume_csv.stem}_{forecast_mode}{args.resume_csv.suffix}")
            _print_progress(f"Running benchmark forecast_mode={forecast_mode}, output_dir={mode_output_dir}")
            summary_frames.append(run_tourism_benchmark(
                output_dir=mode_output_dir,
                forecast_mode=forecast_mode,
                resume_csv=mode_resume_csv,
                **run_kwargs,
            ))
            by_series_frames.append(pd.read_csv(mode_output_dir / "metrics_by_series.csv"))
            by_origin_frames.append(pd.read_csv(mode_output_dir / "metrics_by_origin.csv"))
            forecasts = pd.read_csv(mode_output_dir / "tourism_benchmark_forecasts_long.csv")
            forecasts.insert(0, "forecast_mode", forecast_mode)
            forecast_frames.append(forecasts)

        summary = pd.concat(summary_frames, ignore_index=True, sort=False)
        summary.to_csv(args.output_dir / "metrics_summary.csv", index=False)
        pd.concat(by_series_frames, ignore_index=True, sort=False).to_csv(
            args.output_dir / "metrics_by_series.csv", index=False
        )
        pd.concat(by_origin_frames, ignore_index=True, sort=False).to_csv(
            args.output_dir / "metrics_by_origin.csv", index=False
        )
        pd.concat(forecast_frames, ignore_index=True, sort=False).to_csv(
            args.output_dir / "tourism_benchmark_forecasts_long.csv", index=False
        )

    print("Benchmark summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
