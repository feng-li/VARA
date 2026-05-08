"""Stationarity diagnostics for forecasting panels and fitted VARA dynamics."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss

from .attention import TopKAttentionCache, build_attention

PANEL_TRANSFORMS = (
    "level",
    "log1p",
    "diff",
    "log1p-diff",
    "seasonal-diff",
    "log1p-seasonal-diff",
)


def transform_panel(Y: np.ndarray, transform: str = "level", seasonal_lag: int = 12) -> np.ndarray:
    """Apply a simple stationarity transform to a ``(T, p)`` panel."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional array with shape (T, p).")
    if seasonal_lag < 1:
        raise ValueError("seasonal_lag must be positive.")

    if transform == "level":
        return Y
    if transform == "log1p":
        return np.log1p(np.maximum(Y, 0.0))
    if transform == "diff":
        return np.diff(Y, axis=0)
    if transform == "log1p-diff":
        return np.diff(np.log1p(np.maximum(Y, 0.0)), axis=0)
    if transform == "seasonal-diff":
        if Y.shape[0] <= seasonal_lag:
            raise ValueError("Y must have more rows than seasonal_lag.")
        return Y[seasonal_lag:] - Y[:-seasonal_lag]
    if transform == "log1p-seasonal-diff":
        if Y.shape[0] <= seasonal_lag:
            raise ValueError("Y must have more rows than seasonal_lag.")
        base = np.log1p(np.maximum(Y, 0.0))
        return base[seasonal_lag:] - base[:-seasonal_lag]
    raise ValueError(f"Unknown transform: {transform}")


def standardize_panel(Y: np.ndarray) -> np.ndarray:
    """Column-standardize a panel, leaving constant columns centered only."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional array with shape (T, p).")
    std = Y.std(axis=0, keepdims=True)
    std = np.where(std > 0.0, std, 1.0)
    return (Y - Y.mean(axis=0, keepdims=True)) / std


def _warning_text(caught: Sequence[warnings.WarningMessage]) -> str:
    return "; ".join(str(w.message).replace("\n", " ") for w in caught)


def _safe_adf(x: np.ndarray, regression: str, alpha: float) -> dict:
    out = {
        "adf_stat": np.nan,
        "adf_pvalue": np.nan,
        "adf_lag": np.nan,
        "adf_nobs": np.nan,
        "adf_reject_5": False,
        "adf_error": "",
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stat, pvalue, lag, nobs, *_ = adfuller(x, regression=regression, autolag="AIC")
        out.update({
            "adf_stat": float(stat),
            "adf_pvalue": float(pvalue),
            "adf_lag": int(lag),
            "adf_nobs": int(nobs),
            "adf_reject_5": bool(pvalue < alpha),
            "adf_error": _warning_text(caught),
        })
    except Exception as exc:
        out["adf_error"] = str(exc)
    return out


def _safe_kpss(x: np.ndarray, regression: str, alpha: float) -> dict:
    out = {
        "kpss_stat": np.nan,
        "kpss_pvalue": np.nan,
        "kpss_lags": np.nan,
        "kpss_reject_5": False,
        "kpss_error": "",
    }
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stat, pvalue, lags, _ = kpss(x, regression=regression, nlags="auto")
        out.update({
            "kpss_stat": float(stat),
            "kpss_pvalue": float(pvalue),
            "kpss_lags": int(lags),
            "kpss_reject_5": bool(pvalue < alpha),
            "kpss_error": _warning_text(caught),
        })
    except Exception as exc:
        out["kpss_error"] = str(exc)
    return out


def stationarity_rows(
    Y: np.ndarray,
    series_ids: Optional[Sequence[str]] = None,
    transform: str = "level",
    regression: str = "c",
    alpha: float = 0.05,
    min_obs: int = 20,
) -> list[dict]:
    """Return ADF/KPSS stationarity diagnostics for each panel column."""
    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a two-dimensional array with shape (T, p).")
    if series_ids is None:
        series_ids = [f"series_{j}" for j in range(Y.shape[1])]
    if len(series_ids) != Y.shape[1]:
        raise ValueError("series_ids length must match Y.shape[1].")

    rows = []
    for j, series_id in enumerate(series_ids):
        x = np.asarray(Y[:, j], dtype=float)
        x = x[np.isfinite(x)]
        row = {
            "series_id": str(series_id),
            "transform": transform,
            "n_obs": int(x.size),
            "std": float(np.std(x)) if x.size else np.nan,
        }
        if x.size < min_obs or np.nanstd(x) <= 0.0:
            row.update({
                "adf_stat": np.nan,
                "adf_pvalue": np.nan,
                "adf_lag": np.nan,
                "adf_nobs": np.nan,
                "adf_reject_5": False,
                "adf_error": "Too few observations or constant series.",
                "kpss_stat": np.nan,
                "kpss_pvalue": np.nan,
                "kpss_lags": np.nan,
                "kpss_reject_5": False,
                "kpss_error": "Too few observations or constant series.",
                "stationary_5": False,
            })
        else:
            row.update(_safe_adf(x, regression, alpha))
            row.update(_safe_kpss(x, regression, alpha))
            row["stationary_5"] = bool(row["adf_reject_5"] and not row["kpss_reject_5"])
        rows.append(row)
    return rows


def summarize_stationarity_rows(rows: Sequence[dict]) -> dict:
    """Summarize per-series stationarity diagnostics."""
    if not rows:
        return {
            "adf_reject_5_rate": np.nan,
            "kpss_reject_5_rate": np.nan,
            "stationary_5_rate": np.nan,
            "n_stationary_5": 0,
        }
    adf = np.asarray([bool(row["adf_reject_5"]) for row in rows], dtype=float)
    kpss_ = np.asarray([bool(row["kpss_reject_5"]) for row in rows], dtype=float)
    stationary = np.asarray([bool(row["stationary_5"]) for row in rows], dtype=float)
    return {
        "adf_reject_5_rate": float(adf.mean()),
        "kpss_reject_5_rate": float(kpss_.mean()),
        "stationary_5_rate": float(stationary.mean()),
        "n_stationary_5": int(stationary.sum()),
    }


def check_panel_stationarity(
    Y: np.ndarray,
    series_ids: Optional[Sequence[str]] = None,
    transform: str = "level",
    seasonal_lag: int = 12,
    regression: str = "c",
    alpha: float = 0.05,
    min_obs: int = 20,
) -> tuple[np.ndarray, list[dict], dict]:
    """Transform a panel, run ADF/KPSS by column, and return a summary."""
    Y_transformed = transform_panel(Y, transform=transform, seasonal_lag=seasonal_lag)
    rows = stationarity_rows(
        Y_transformed,
        series_ids=series_ids,
        transform=transform,
        regression=regression,
        alpha=alpha,
        min_obs=min_obs,
    )
    return Y_transformed, rows, summarize_stationarity_rows(rows)


def vara_transition_matrix(
    alpha: np.ndarray,
    beta: np.ndarray,
    Z: np.ndarray,
    eta: float,
    params_scale: Optional[dict] = None,
    attention_mode: str = "dense",
    attention_top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
) -> np.ndarray:
    """Return the fitted linear transition matrix ``diag(alpha)+diag(beta)W``."""
    scale = 1.0 if params_scale is None else float(params_scale.get("eta_a", 1.0))
    W = build_attention(
        jnp.asarray(Z),
        float(eta),
        scale,
        attention_mode=attention_mode,
        top_k=attention_top_k,
        attention_cache=attention_cache,
    )
    return np.diag(np.asarray(alpha, dtype=float)) + np.diag(np.asarray(beta, dtype=float)) @ np.asarray(W)


def vara_stability_from_params(
    alpha: np.ndarray,
    beta: np.ndarray,
    Z: np.ndarray,
    eta: float,
    params_scale: Optional[dict] = None,
    attention_mode: str = "dense",
    attention_top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
) -> dict:
    """Return spectral-radius stability diagnostics for fitted VARA parameters."""
    A = vara_transition_matrix(
        alpha,
        beta,
        Z,
        eta,
        params_scale=params_scale,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        attention_cache=attention_cache,
    )
    eigvals = np.linalg.eigvals(A)
    spectral_radius = float(np.max(np.abs(eigvals)))
    return {
        "vara_spectral_radius": spectral_radius,
        "vara_stable": bool(spectral_radius < 1.0),
    }
