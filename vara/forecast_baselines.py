"""Shared baseline forecast helpers for residual VARA workflows."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


AUTO_ETS_ALIAS = "auto_ets"


def auto_ets_config(seasonal_lag: int) -> dict[str, Any]:
    return {
        "model": "AutoETS",
        "alias": AUTO_ETS_ALIAS,
        "season_length": max(1, int(seasonal_lag)),
        "freq": 1,
    }


def _statsforecast_long_from_panel(values: np.ndarray, series_ids: Sequence[str]) -> pd.DataFrame:
    values = np.asarray(values, dtype=float)
    frame = pd.DataFrame(values, columns=list(series_ids))
    frame.insert(0, "ds", np.arange(values.shape[0]))
    return frame.melt(id_vars="ds", var_name="unique_id", value_name="y")


def _statsforecast_wide(frame: pd.DataFrame, series_ids: Sequence[str], value_col: str) -> np.ndarray:
    wide = frame.pivot(index="ds", columns="unique_id", values=value_col).sort_index()
    return wide.loc[:, list(series_ids)].to_numpy(dtype=float)


def auto_ets_baseline_panel(
    history: np.ndarray,
    horizon: int,
    seasonal_lag: int,
    series_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit AutoETS on ``history`` and return fitted, residual, and forecast panels.

    ``history`` has shape ``(T, p)`` and all returned panels use the same series
    order as ``series_ids``. This function is intentionally the single source of
    AutoETS configuration for residual feature extraction and forecast recovery.
    """
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS

    values = np.asarray(history, dtype=float)
    if values.ndim != 2:
        raise ValueError("history must have shape (T, p).")
    if values.shape[0] < 2:
        raise ValueError("AutoETS baseline requires at least two training observations.")
    if values.shape[1] != len(series_ids):
        raise ValueError("history column count must match series_ids.")

    config = auto_ets_config(seasonal_lag)
    long = _statsforecast_long_from_panel(values, series_ids)
    sf = StatsForecast(
        models=[AutoETS(season_length=config["season_length"], alias=AUTO_ETS_ALIAS)],
        freq=config["freq"],
        n_jobs=1,
    )
    forecast_long = sf.forecast(h=int(horizon), df=long, fitted=True)
    fitted_long = sf.forecast_fitted_values()
    fitted = _statsforecast_wide(fitted_long, series_ids, AUTO_ETS_ALIAS)
    forecast = _statsforecast_wide(forecast_long, series_ids, AUTO_ETS_ALIAS)
    return {
        "fitted": fitted,
        "residuals": values - fitted,
        "forecast": forecast,
        "config": config,
    }
