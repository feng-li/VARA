#!/usr/bin/env python3
"""Run forecasting benchmarks on the tourism dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, Naive, RandomWalkWithDrift, SeasonalNaive
from statsmodels.tsa.api import VAR

BASE_SUFFIX = "All"
DETAILED_CODE_DEPTH = 3
STATSFORECAST_MODELS = {"naive", "seasonal_naive", "drift", "auto_ets", "auto_arima"}
STATSMODELS_MODELS = {"var1"}
ALL_MODELS = STATSFORECAST_MODELS | STATSMODELS_MODELS
DEFAULT_MODELS = "naive,seasonal_naive,drift,auto_ets,auto_arima,var1"
STATSFORECAST_N_JOBS = 1


def _series_code(series_id: str, suffix: str = BASE_SUFFIX) -> str:
    return series_id[: -len(suffix)] if suffix and series_id.endswith(suffix) else series_id


def _series_depth(series_id: str, suffix: str = BASE_SUFFIX) -> int:
    code = _series_code(series_id, suffix)
    return 0 if code == "Total" else len(code)


def _base_series_columns(wide: pd.DataFrame, series_depth: int | None = DETAILED_CODE_DEPTH) -> list[str]:
    cols = [c for c in wide.columns if c != "date" and c.endswith(BASE_SUFFIX)]
    if series_depth is not None:
        cols = [c for c in cols if _series_depth(c) == series_depth]
    if not cols:
        raise ValueError(f"No tourism series ending with {BASE_SUFFIX!r} found for series_depth={series_depth}.")
    return cols


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
        "var": "var1",
        "var_1": "var1",
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
    if "var1" in model_names:
        frames.append(_forecast_var1_once(train_wide, series_cols, forecast_dates=forecast_dates, h=h))

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
        )
        out.insert(0, "origin", step + 1)
        out.insert(1, "train_end", wide["date"].iloc[origin_end - 1])
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def _wide_model_frame(long_forecasts: pd.DataFrame, model: str, series_cols: list[str]) -> pd.DataFrame:
    wide = long_forecasts.pivot(index="date", columns="series_id", values=model).reset_index()
    wide.columns.name = None
    return wide.loc[:, ["date", *series_cols]]


def _safe_pct(numer: np.ndarray, denom: np.ndarray) -> np.ndarray:
    return numer / np.where(np.abs(denom) > 1e-12, np.abs(denom), np.nan)


def _seasonal_scales(train: np.ndarray, season_length: int) -> tuple[np.ndarray, np.ndarray]:
    if train.shape[0] <= season_length:
        raise ValueError("Training data must have more rows than season_length for MASE/RMSSE.")
    diff = train[season_length:] - train[:-season_length]
    mae_scale = np.nanmean(np.abs(diff), axis=0)
    rmse_scale = np.nanmean(diff ** 2, axis=0)
    mae_scale = np.where(mae_scale > 1e-12, mae_scale, np.nan)
    rmse_scale = np.where(rmse_scale > 1e-12, rmse_scale, np.nan)
    return mae_scale, rmse_scale


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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    test = test_wide.loc[:, series_cols].to_numpy(dtype=float)
    train = train_wide.loc[:, series_cols].to_numpy(dtype=float)
    mase_scale, rmsse_scale = _seasonal_scales(train, season_length)

    by_series_rows = []
    by_origin_rows = []
    summary_rows = []
    for model in model_names:
        forecast = _wide_model_frame(long_forecasts, model, series_cols).loc[:, series_cols].to_numpy(dtype=float)
        error = forecast - test
        abs_error = np.abs(error)
        squared_error = error ** 2
        pct_error = _safe_pct(abs_error, test)
        smape = 2.0 * abs_error / np.where(np.abs(test) + np.abs(forecast) > 1e-12, np.abs(test) + np.abs(forecast), np.nan)

        series_rmse = np.sqrt(np.nanmean(squared_error, axis=0))
        series_mae = np.nanmean(abs_error, axis=0)
        series_rows = pd.DataFrame({
            "model": model,
            "series_id": series_cols,
            "rmse": series_rmse,
            "mae": series_mae,
            "mape_pct": np.nanmean(pct_error, axis=0) * 100.0,
            "smape_pct": np.nanmean(smape, axis=0) * 100.0,
            "mase": series_mae / mase_scale,
            "rmsse": np.sqrt(np.nanmean(squared_error, axis=0) / rmsse_scale),
            "bias": np.nanmean(error, axis=0),
            "mean_actual": np.nanmean(test, axis=0),
            "mean_forecast": np.nanmean(forecast, axis=0),
        })
        by_series_rows.append(series_rows)

        by_origin_rows.append(pd.DataFrame({
            "model": model,
            "date": test_wide["date"],
            "rmse": np.sqrt(np.nanmean(squared_error, axis=1)),
            "mae": np.nanmean(abs_error, axis=1),
            "mape_pct": np.nanmean(pct_error, axis=1) * 100.0,
            "smape_pct": np.nanmean(smape, axis=1) * 100.0,
            "bias": np.nanmean(error, axis=1),
        }))

        summary_rows.append({
            "model": model,
            "forecast_mode": forecast_mode,
            "horizon": int(horizon),
            "n_series": len(series_cols),
            "n_test": int(test.shape[0]),
            "rmse": float(np.sqrt(np.nanmean(squared_error))),
            "mae": float(np.nanmean(abs_error)),
            "mape_pct": float(np.nanmean(pct_error) * 100.0),
            "smape_pct": float(np.nanmean(smape) * 100.0),
            "mase": float(np.nanmean(series_rows["mase"])),
            "rmsse": float(np.nanmean(series_rows["rmsse"])),
            "bias": float(np.nanmean(error)),
        })

    return (
        pd.concat(by_series_rows, ignore_index=True),
        pd.concat(by_origin_rows, ignore_index=True),
        pd.DataFrame(summary_rows),
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
    models: str | list[str] = DEFAULT_MODELS,
    forecast_mode: str = "rolling-origin",
    horizon: int = 12,
    season_length: int = 12,
    freq: str = "MS",
    series_depth: int | None = DETAILED_CODE_DEPTH,
    max_series: int = 0,
    clip_nonnegative: bool = False,
) -> pd.DataFrame:
    if forecast_mode not in {"recursive", "rolling-origin"}:
        raise ValueError("forecast_mode must be 'recursive' or 'rolling-origin'.")

    wide = pd.read_csv(data_dir / "tourism.csv")
    wide["date"] = pd.to_datetime(wide["date"])
    all_series = _base_series_columns(wide, series_depth=series_depth)
    series_cols = all_series[: min(max_series, len(all_series))] if max_series > 0 else all_series
    if horizon < 1 or horizon >= len(wide):
        raise ValueError("horizon must be positive and smaller than the number of months.")
    if len(series_cols) < 1:
        raise ValueError("No tourism series selected.")

    model_names = _parse_model_names(models)
    if forecast_mode == "recursive":
        long_forecasts = _forecast_recursive(
            wide,
            series_cols,
            horizon=horizon,
            model_names=model_names,
            season_length=season_length,
            freq=freq,
            n_jobs=STATSFORECAST_N_JOBS,
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
    )
    summary.insert(1, "data", str(data_dir / "tourism.csv"))
    summary.insert(2, "series_depth", -1 if series_depth is None else int(series_depth))
    summary.insert(3, "season_length", int(season_length))
    summary.insert(4, "train_start", str(train_wide["date"].iloc[0].date()))
    summary.insert(5, "train_end", str(train_wide["date"].iloc[-1].date()))
    summary.insert(6, "test_start", str(test_wide["date"].iloc[0].date()))
    summary.insert(7, "test_end", str(test_wide["date"].iloc[-1].date()))

    output_dir.mkdir(parents=True, exist_ok=True)
    long_forecasts.to_csv(output_dir / "tourism_benchmark_forecasts_long.csv", index=False)
    test_wide.to_csv(output_dir / "tourism_test.csv", index=False)
    for model in model_names:
        forecast_wide = _wide_model_frame(long_forecasts, model, series_cols)
        forecast_wide.to_csv(output_dir / f"tourism_forecast_{model}.csv", index=False)
        error_wide = forecast_wide.copy()
        error_wide.loc[:, series_cols] = forecast_wide.loc[:, series_cols].to_numpy(dtype=float) - test_wide.loc[:, series_cols].to_numpy(dtype=float)
        error_wide.to_csv(output_dir / f"tourism_errors_{model}.csv", index=False)

    by_series.to_csv(output_dir / "metrics_by_series.csv", index=False)
    by_origin.to_csv(output_dir / "metrics_by_origin.csv", index=False)
    summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    _save_config(output_dir, {
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "models": model_names,
        "forecast_mode": forecast_mode,
        "horizon": int(horizon),
        "season_length": int(season_length),
        "freq": freq,
        "series_depth": -1 if series_depth is None else int(series_depth),
        "max_series": int(max_series),
        "clip_nonnegative": bool(clip_nonnegative),
    })
    return summary


def _load_config_defaults(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as fh:
        config = json.load(fh)
    for key in ("data_dir", "output_dir"):
        if key in config:
            config[key] = Path(config[key])
    if "models" in config and isinstance(config["models"], list):
        config["models"] = ",".join(config["models"])
    return config


def _parse_args() -> argparse.Namespace:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config-json", type=Path, default=None)
    known, _ = pre.parse_known_args()
    defaults = _load_config_defaults(known.config_json)

    parser = argparse.ArgumentParser(description="Run forecasting benchmarks on tourism data.", parents=[pre])
    parser.add_argument("--data-dir", type=Path, default=Path("data/tourism"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/tourism_benchmark_statsforecast"))
    parser.add_argument("--models", type=str, default=DEFAULT_MODELS)
    parser.add_argument("--forecast-mode", choices=["recursive", "rolling-origin"], default="rolling-origin")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--freq", type=str, default="MS")
    parser.add_argument("--series-depth", type=int, default=DETAILED_CODE_DEPTH)
    parser.add_argument("--max-series", type=int, default=0)
    parser.add_argument("--clip-nonnegative", action="store_true")
    parser.set_defaults(**defaults)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_tourism_benchmark(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models=args.models,
        forecast_mode=args.forecast_mode,
        horizon=args.horizon,
        season_length=args.season_length,
        freq=args.freq,
        series_depth=None if args.series_depth < 0 else args.series_depth,
        max_series=args.max_series,
        clip_nonnegative=args.clip_nonnegative,
    )
    print("Tourism benchmark summary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
