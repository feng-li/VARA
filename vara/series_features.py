"""Time-series feature extraction utilities for VARA forecasting."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import tsfeatures as tsf

from .stationarity import PANEL_TRANSFORMS, transform_panel


META_COLUMNS = ("series_id", "transform", "n_obs", "train_start", "train_end")


def _series_code(series_id: str, suffix: str | None = None) -> str:
    if suffix and series_id.endswith(suffix):
        return series_id[: -len(suffix)]
    return series_id


def _series_depth(series_id: str, suffix: str | None = None) -> int:
    code = _series_code(series_id, suffix)
    return 0 if code == "Total" else len(code)


def select_series_columns(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    series_cols: Sequence[str] | None = None,
    series_suffix: str | None = None,
    series_depth: int | None = None,
) -> list[str]:
    """Select wide-panel series columns with optional suffix/depth filters."""
    if series_cols is not None:
        cols = [str(col) for col in series_cols]
    else:
        cols = [str(col) for col in df.columns if col != date_col]

    if series_suffix:
        cols = [col for col in cols if col.endswith(series_suffix)]
    if series_depth is not None:
        cols = [col for col in cols if _series_depth(col, series_suffix) == series_depth]

    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"Selected series columns are not in the input data: {missing[:5]}")
    if not cols:
        raise ValueError("No series columns selected.")
    return cols


def resolve_tsfeatures_functions(features: str | Sequence[str | Callable] | None) -> list[Callable] | None:
    """Resolve comma-separated ``tsfeatures`` function names to callables.

    ``None`` or ``"default"`` uses the default feature list from the package.
    """
    if features is None or features == "default":
        return None
    if isinstance(features, str):
        names: Sequence[str | Callable] = [name.strip() for name in features.split(",") if name.strip()]
    else:
        names = features

    out = []
    for item in names:
        if callable(item):
            out.append(item)
            continue
        if not hasattr(tsf, str(item)):
            raise ValueError(f"Unknown tsfeatures function: {item}")
        fn = getattr(tsf, str(item))
        if not callable(fn):
            raise ValueError(f"tsfeatures.{item} is not callable.")
        out.append(fn)
    if not out:
        raise ValueError("No tsfeatures functions selected.")
    return out


def _date_slice_for_transform(dates: pd.Series, transform: str, seasonal_lag: int) -> pd.Series:
    if transform in ("level", "log1p"):
        return dates.reset_index(drop=True)
    if transform in ("diff", "log1p-diff"):
        return dates.iloc[1:].reset_index(drop=True)
    if transform in ("seasonal-diff", "log1p-seasonal-diff"):
        return dates.iloc[seasonal_lag:].reset_index(drop=True)
    raise ValueError(f"Unknown transform: {transform}")


def wide_panel_to_tsfeatures_long(
    df: pd.DataFrame,
    *,
    series_cols: Sequence[str],
    date_col: str = "date",
    transform: str = "level",
    seasonal_lag: int = 12,
) -> tuple[pd.DataFrame, int, str, str]:
    """Convert a wide panel to the long format expected by ``tsfeatures``."""
    if transform not in PANEL_TRANSFORMS:
        raise ValueError(f"Unknown transform: {transform}")
    if date_col not in df.columns:
        raise ValueError(f"date_col={date_col!r} is not in the input data.")

    Y_raw = df.loc[:, list(series_cols)].to_numpy(dtype=float)
    if not np.isfinite(Y_raw).all():
        raise ValueError("Selected series contain non-finite values.")

    Y = transform_panel(Y_raw, transform=transform, seasonal_lag=seasonal_lag)
    dates = _date_slice_for_transform(pd.to_datetime(df[date_col]), transform, seasonal_lag)
    if len(dates) != Y.shape[0]:
        raise RuntimeError("Transformed date index length does not match transformed panel length.")

    p = len(series_cols)
    long_df = pd.DataFrame({
        "unique_id": np.repeat(np.asarray(series_cols, dtype=object), Y.shape[0]),
        "ds": np.tile(dates.to_numpy(), p),
        "y": Y.T.reshape(-1),
    })
    return long_df, int(Y.shape[0]), str(dates.iloc[0].date()), str(dates.iloc[-1].date())


def clean_feature_frame(
    features: pd.DataFrame,
    *,
    standardize_output: bool = False,
    meta_columns: Sequence[str] = META_COLUMNS,
) -> pd.DataFrame:
    """Replace non-finite numeric features and optionally standardize columns."""
    out = features.copy()
    feature_cols = [col for col in out.columns if col not in set(meta_columns)]
    numeric_cols = [col for col in feature_cols if pd.api.types.is_numeric_dtype(out[col])]

    if numeric_cols:
        out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
        all_missing = [col for col in numeric_cols if out[col].isna().all()]
        if all_missing:
            out = out.drop(columns=all_missing)
            numeric_cols = [col for col in numeric_cols if col not in set(all_missing)]
        for col in numeric_cols:
            median = out[col].median(skipna=True)
            out[col] = out[col].fillna(0.0 if pd.isna(median) else float(median))
        if standardize_output:
            mean = out[numeric_cols].mean(axis=0)
            std = out[numeric_cols].std(axis=0, ddof=0).replace(0.0, 1.0)
            out[numeric_cols] = (out[numeric_cols] - mean) / std
    return out


def extract_series_tsfeatures(
    df: pd.DataFrame,
    *,
    date_col: str = "date",
    series_cols: Sequence[str] | None = None,
    series_suffix: str | None = None,
    series_depth: int | None = None,
    holdout: int = 0,
    transform: str = "level",
    seasonal_lag: int = 12,
    freq: int = 12,
    features: str | Sequence[str | Callable] | None = "default",
    scale_series: bool = True,
    standardize_output: bool = False,
    clean: bool = True,
    threads: int | None = 1,
) -> pd.DataFrame:
    """Extract ``tsfeatures`` rows for each selected series in a wide panel.

    The holdout period is removed before feature extraction, which prevents
    leakage when features are used in forecasting. The ``features`` argument can
    be ``"default"``, a comma-separated string such as ``"acf_features"``, or a
    callable list such as ``[tsfeatures.acf_features]``.
    """
    if holdout < 0 or holdout >= len(df):
        raise ValueError("holdout must be nonnegative and smaller than the number of rows.")

    cols = select_series_columns(
        df,
        date_col=date_col,
        series_cols=series_cols,
        series_suffix=series_suffix,
        series_depth=series_depth,
    )
    train = df.iloc[: len(df) - holdout].copy() if holdout else df.copy()
    long_df, n_obs, train_start, train_end = wide_panel_to_tsfeatures_long(
        train,
        series_cols=cols,
        date_col=date_col,
        transform=transform,
        seasonal_lag=seasonal_lag,
    )

    feature_functions = resolve_tsfeatures_functions(features)
    kwargs = {} if feature_functions is None else {"features": feature_functions}
    extracted = tsf.tsfeatures(long_df, freq=freq, scale=scale_series, threads=threads, **kwargs)
    extracted = extracted.rename(columns={"unique_id": "series_id"})

    meta = pd.DataFrame({
        "series_id": cols,
        "transform": transform,
        "n_obs": n_obs,
        "train_start": train_start,
        "train_end": train_end,
    })
    out = meta.merge(extracted, on="series_id", how="left", sort=False)
    if clean:
        out = clean_feature_frame(out, standardize_output=standardize_output)
    return out


def extract_series_tsfeatures_csv(
    input_path: Path,
    output_path: Path,
    **kwargs,
) -> pd.DataFrame:
    """Read a wide CSV, extract per-series ``tsfeatures``, and save a CSV."""
    df = pd.read_csv(input_path)
    features = extract_series_tsfeatures(df, **kwargs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_path, index=False)
    return features
