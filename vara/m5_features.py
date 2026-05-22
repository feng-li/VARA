"""Series-level M5 feature extraction for VARA attention."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .series_features import META_COLUMNS, clean_feature_frame


M5_META_COLUMNS = ("id", "item_id", "dept_id", "cat_id", "store_id", "state_id")
M5_ATTENTION_META_COLUMNS = (
    *META_COLUMNS,
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
)
M5_CATEGORICAL_FEATURES = ("state_id", "store_id", "cat_id", "dept_id")
DEFAULT_RECENT_WINDOWS = (7, 28, 56)


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported table suffix {path.suffix!r}; use CSV or parquet.")


def infer_m5_day_columns(columns: Sequence[str]) -> list[str]:
    days = [str(col) for col in columns if str(col).startswith("d_") and str(col)[2:].isdigit()]
    return sorted(days, key=lambda col: int(col[2:]))


def _require_columns(df: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _safe_std(values: np.ndarray, axis: int) -> np.ndarray:
    return np.nan_to_num(np.nanstd(values, axis=axis), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_mean(values: np.ndarray, axis: int) -> np.ndarray:
    return np.nan_to_num(np.nanmean(values, axis=axis), nan=0.0, posinf=0.0, neginf=0.0)


def _safe_ratio(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    out = np.zeros_like(np.asarray(num, dtype=float), dtype=float)
    den = np.asarray(den, dtype=float)
    mask = np.isfinite(num) & np.isfinite(den) & (np.abs(den) > 1e-12)
    out[mask] = np.asarray(num, dtype=float)[mask] / den[mask]
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _window_mean(values: np.ndarray, window: int) -> np.ndarray:
    if values.shape[1] == 0:
        return np.zeros(values.shape[0], dtype=float)
    width = min(int(window), values.shape[1])
    return _safe_mean(values[:, -width:], axis=1)


def _calendar_for_days(calendar: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    _require_columns(calendar, ["d", "date", "wm_yr_wk"], "calendar")
    cal = calendar.copy()
    cal["d"] = cal["d"].astype(str)
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.set_index("d").reindex(day_cols)
    if cal["date"].isna().any():
        missing = cal.loc[cal["date"].isna()].index.tolist()[:5]
        raise ValueError(f"Calendar is missing dates for day columns, e.g. {missing}")
    return cal.reset_index()


def _calendar_lift(values: np.ndarray, mask: np.ndarray, base_mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not bool(np.any(mask)):
        zeros = np.zeros(values.shape[0], dtype=float)
        return zeros, zeros
    in_mean = _safe_mean(values[:, mask], axis=1)
    out_mean = _safe_mean(values[:, ~mask], axis=1) if bool(np.any(~mask)) else base_mean
    return _safe_ratio(in_mean, base_mean) - 1.0, _safe_ratio(in_mean, out_mean) - 1.0


def _demand_feature_frame(sales: pd.DataFrame, calendar_days: pd.DataFrame, day_cols: list[str]) -> pd.DataFrame:
    values = sales.loc[:, day_cols].to_numpy(dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    log_values = np.log1p(values)
    n_series, n_obs = values.shape

    mean_sales = _safe_mean(values, axis=1)
    std_sales = _safe_std(values, axis=1)
    nonzero = values > 0.0
    zero_fraction = 1.0 - nonzero.mean(axis=1)

    out = pd.DataFrame({
        "series_id": sales["id"].astype(str).to_numpy(),
        "sales_mean": mean_sales,
        "sales_std": std_sales,
        "sales_cv": _safe_ratio(std_sales, mean_sales),
        "sales_zero_fraction": zero_fraction,
        "sales_nonzero_fraction": nonzero.mean(axis=1),
        "sales_log_mean": _safe_mean(log_values, axis=1),
        "sales_log_std": _safe_std(log_values, axis=1),
        "sales_last": values[:, -1] if n_obs else np.zeros(n_series),
        "sales_recent_log_last": log_values[:, -1] if n_obs else np.zeros(n_series),
    })

    for window in DEFAULT_RECENT_WINDOWS:
        recent = _window_mean(values, window)
        out[f"sales_recent_mean_{window}"] = recent
        out[f"sales_recent_rel_mean_{window}"] = _safe_ratio(recent, mean_sales) - 1.0

    if n_obs >= 2:
        out["sales_log_growth_1"] = log_values[:, -1] - log_values[:, -2]
    else:
        out["sales_log_growth_1"] = 0.0
    if n_obs > 28:
        out["sales_log_growth_28"] = log_values[:, -1] - log_values[:, -29]
    else:
        out["sales_log_growth_28"] = 0.0
    if n_obs >= 14:
        first = _safe_mean(values[:, : max(1, n_obs // 2)], axis=1)
        second = _safe_mean(values[:, n_obs // 2 :], axis=1)
        out["sales_half_trend_rel"] = _safe_ratio(second, first) - 1.0
    else:
        out["sales_half_trend_rel"] = 0.0

    dates = pd.to_datetime(calendar_days["date"])
    dow = dates.dt.dayofweek.to_numpy(dtype=int)
    for day in range(7):
        mask = dow == day
        day_mean = _safe_mean(values[:, mask], axis=1) if bool(np.any(mask)) else np.zeros(n_series)
        out[f"dow_{day}_rel_mean"] = _safe_ratio(day_mean, mean_sales) - 1.0

    weekend = dow >= 5
    out["weekend_lift_rel"], out["weekend_vs_weekday_rel"] = _calendar_lift(values, weekend, mean_sales)

    event_cols = [col for col in ["event_name_1", "event_name_2", "event_type_1", "event_type_2"] if col in calendar_days.columns]
    if event_cols:
        event_mask = calendar_days.loc[:, event_cols].notna().any(axis=1).to_numpy()
        out["event_lift_rel"], out["event_vs_nonevent_rel"] = _calendar_lift(values, event_mask, mean_sales)
    else:
        out["event_lift_rel"] = 0.0
        out["event_vs_nonevent_rel"] = 0.0

    snap_lift = np.zeros(n_series, dtype=float)
    snap_vs_no_snap = np.zeros(n_series, dtype=float)
    for state in sorted(sales["state_id"].astype(str).unique()):
        snap_col = f"snap_{state}"
        if snap_col not in calendar_days.columns:
            continue
        series_mask = sales["state_id"].astype(str).to_numpy() == state
        snap_mask = calendar_days[snap_col].fillna(0).to_numpy(dtype=int) == 1
        lift, contrast = _calendar_lift(values[series_mask], snap_mask, mean_sales[series_mask])
        snap_lift[series_mask] = lift
        snap_vs_no_snap[series_mask] = contrast
    out["snap_lift_rel"] = snap_lift
    out["snap_vs_no_snap_rel"] = snap_vs_no_snap
    return out


def _price_feature_frame(
    sales: pd.DataFrame,
    prices: pd.DataFrame,
    calendar_days: pd.DataFrame,
) -> pd.DataFrame:
    _require_columns(prices, ["store_id", "item_id", "wm_yr_wk", "sell_price"], "prices")
    max_week = int(calendar_days["wm_yr_wk"].max())
    selected = sales.loc[:, ["store_id", "item_id"]].drop_duplicates()
    price = prices.merge(selected, on=["store_id", "item_id"], how="inner")
    price = price.loc[price["wm_yr_wk"].astype(int) <= max_week].copy()
    if price.empty:
        return pd.DataFrame({"series_id": sales["id"].astype(str)})

    price = price.sort_values(["store_id", "item_id", "wm_yr_wk"]).reset_index(drop=True)
    group_cols = ["store_id", "item_id"]
    group = price.groupby(group_cols, sort=False)
    price["price_momentum"] = price["sell_price"] / group["sell_price"].shift(1)
    agg = group.agg(
        price_mean=("sell_price", "mean"),
        price_std=("sell_price", "std"),
        price_min=("sell_price", "min"),
        price_max=("sell_price", "max"),
        price_nunique=("sell_price", "nunique"),
        price_weeks_observed=("sell_price", "size"),
        release_week=("wm_yr_wk", "min"),
    ).reset_index()
    recent = group.tail(1).loc[:, [*group_cols, "sell_price", "price_momentum"]].rename(
        columns={"sell_price": "price_recent", "price_momentum": "price_momentum_recent"}
    )
    agg = agg.merge(recent, on=group_cols, how="left", sort=False)
    agg["price_recent_norm"] = agg["price_recent"] / agg["price_max"]
    agg["price_range_rel"] = (agg["price_max"] - agg["price_min"]) / agg["price_mean"]
    agg["release_age_weeks"] = max_week - agg["release_week"]

    merged = sales.loc[:, ["id", "store_id", "item_id"]].merge(agg, on=group_cols, how="left", sort=False)
    return merged.rename(columns={"id": "series_id"}).drop(columns=group_cols)


def make_m5_attention_features(
    sales: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    store_id: str | None = None,
    holdout: int = 28,
    max_series: int = 0,
    standardize_output: bool = False,
) -> pd.DataFrame:
    """Build one M5 attention feature row per item-store series.

    ``holdout`` removes the last H sales days before computing demand, price, and
    calendar-response features, matching the forecasting train/test split.
    """
    _require_columns(sales, M5_META_COLUMNS, "sales")
    day_cols = infer_m5_day_columns(sales.columns)
    if not day_cols:
        raise ValueError("sales must contain d_* columns.")
    if holdout < 0 or holdout >= len(day_cols):
        raise ValueError("holdout must be nonnegative and smaller than the number of d_* columns.")

    sales_data = sales.copy()
    price_data = prices.copy()
    if store_id:
        sales_data = sales_data.loc[sales_data["store_id"].astype(str) == str(store_id)].copy()
        price_data = price_data.loc[price_data["store_id"].astype(str) == str(store_id)].copy()
    if max_series > 0:
        sales_data = sales_data.iloc[:max_series].copy()
    if sales_data.empty:
        raise ValueError("No M5 sales rows remain after filtering.")

    train_day_cols = day_cols[:-holdout] if holdout else day_cols
    calendar_days = _calendar_for_days(calendar, train_day_cols)
    dates = pd.to_datetime(calendar_days["date"])

    meta = sales_data.loc[:, list(M5_META_COLUMNS)].copy().rename(columns={"id": "series_id"})
    meta["series_id"] = meta["series_id"].astype(str)
    meta.insert(1, "transform", "m5_attention")
    meta.insert(2, "n_obs", len(train_day_cols))
    meta.insert(3, "train_start", str(dates.iloc[0].date()))
    meta.insert(4, "train_end", str(dates.iloc[-1].date()))

    demand = _demand_feature_frame(sales_data, calendar_days, train_day_cols)
    price = _price_feature_frame(sales_data, price_data, calendar_days)
    features = meta.merge(demand, on="series_id", how="left", sort=False).merge(price, on="series_id", how="left", sort=False)
    features = pd.get_dummies(features, columns=list(M5_CATEGORICAL_FEATURES), prefix=list(M5_CATEGORICAL_FEATURES), dtype=float)
    features = clean_feature_frame(
        features,
        standardize_output=standardize_output,
        meta_columns=M5_ATTENTION_META_COLUMNS,
    )
    return features


def make_m5_attention_features_from_files(
    *,
    sales_path: Path,
    prices_path: Path,
    calendar_path: Path,
    output_path: Path,
    store_id: str | None = None,
    holdout: int = 28,
    max_series: int = 0,
    standardize_output: bool = False,
    overwrite: bool = False,
) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass overwrite=True or --overwrite.")
    sales = read_table(sales_path)
    prices = read_table(prices_path)
    calendar = read_table(calendar_path)
    features = make_m5_attention_features(
        sales,
        prices,
        calendar,
        store_id=store_id,
        holdout=holdout,
        max_series=max_series,
        standardize_output=standardize_output,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        features.to_csv(output_path, index=False)
    else:
        features.to_parquet(output_path, index=False)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sales_path": str(sales_path),
        "prices_path": str(prices_path),
        "calendar_path": str(calendar_path),
        "output_path": str(output_path),
        "store_id": "" if store_id is None else str(store_id),
        "holdout": int(holdout),
        "max_series": int(max_series),
        "standardize_output": bool(standardize_output),
        "n_series": int(len(features)),
        "n_columns": int(len(features.columns)),
        "feature_columns": [col for col in features.columns if col not in set(M5_ATTENTION_META_COLUMNS)],
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata
