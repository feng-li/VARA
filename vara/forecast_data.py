"""Shared loading helpers for panel forecasting scripts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PANEL_DATA_FORMATS = {"auto", "wide", "long"}


def read_table(path: Path) -> pd.DataFrame:
    """Read a CSV or parquet table."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data file suffix {path.suffix!r}; use CSV or parquet.")


def wide_to_long_panel(
    wide: pd.DataFrame,
    *,
    date_col: str = "date",
    series_col: str = "series_id",
    value_col: str = "value",
) -> pd.DataFrame:
    """Convert a wide panel table to long format."""
    if date_col not in wide.columns:
        raise ValueError(f"Wide panel table must contain date column {date_col!r}.")
    series_cols = [col for col in wide.columns if col != date_col]
    if not series_cols:
        raise ValueError("Wide panel table must contain at least one series column.")
    out = wide.loc[:, [date_col, *series_cols]].melt(
        id_vars=date_col,
        var_name=series_col,
        value_name=value_col,
    )
    out[date_col] = pd.to_datetime(out[date_col])
    out[series_col] = out[series_col].astype(str)
    out[value_col] = pd.to_numeric(out[value_col], errors="raise")
    return out.reset_index(drop=True)


def long_to_wide_panel(
    long: pd.DataFrame,
    *,
    date_col: str = "date",
    series_col: str = "series_id",
    value_col: str = "value",
) -> pd.DataFrame:
    """Convert a long panel table to the wide shape used by forecast routines."""
    required = {date_col, series_col, value_col}
    missing = sorted(required - set(long.columns))
    if missing:
        raise ValueError(f"Long panel table is missing required columns: {missing}")

    data = long.loc[:, [date_col, series_col, value_col]].copy()
    data[date_col] = pd.to_datetime(data[date_col])
    data[series_col] = data[series_col].astype(str)
    data[value_col] = pd.to_numeric(data[value_col], errors="raise")
    series_order = pd.Index(data[series_col].drop_duplicates())

    dup = data.duplicated([date_col, series_col])
    if dup.any():
        example = data.loc[dup, [date_col, series_col]].head(5).to_dict("records")
        raise ValueError(f"Long panel table has duplicate date/series rows, e.g. {example}")

    wide = data.pivot(index=date_col, columns=series_col, values=value_col).sort_index()
    wide = wide.reindex(columns=series_order)
    if wide.isna().any().any():
        missing_count = int(wide.isna().sum().sum())
        raise ValueError(f"Long panel table is not a complete balanced panel; missing values={missing_count}.")
    wide = wide.reset_index().rename_axis(None, axis=1)
    wide = wide.rename(columns={date_col: "date"})
    return wide


def load_panel_data(
    path: Path,
    *,
    data_format: str = "auto",
    date_col: str = "date",
    series_col: str = "series_id",
    value_col: str = "value",
) -> pd.DataFrame:
    """Load a wide or long panel table and return a wide table with a date column."""
    if data_format not in PANEL_DATA_FORMATS:
        raise ValueError(f"data_format must be one of {sorted(PANEL_DATA_FORMATS)}.")
    raw = read_table(path)

    if data_format == "auto":
        if {date_col, series_col, value_col}.issubset(raw.columns):
            data_format = "long"
        elif date_col in raw.columns:
            data_format = "wide"
        else:
            raise ValueError(
                "Could not infer data format. Expected either a wide table with "
                f"{date_col!r}, or a long table with {date_col!r}, {series_col!r}, {value_col!r}."
            )

    if data_format == "long":
        wide = long_to_wide_panel(raw, date_col=date_col, series_col=series_col, value_col=value_col)
    else:
        if date_col not in raw.columns:
            raise ValueError(f"Wide panel table must contain date column {date_col!r}.")
        wide = raw.copy()
        if date_col != "date":
            wide = wide.rename(columns={date_col: "date"})
        wide["date"] = pd.to_datetime(wide["date"])
        series_cols = [col for col in wide.columns if col != "date"]
        if not series_cols:
            raise ValueError("Wide panel table must contain at least one series column.")
        wide.loc[:, series_cols] = wide.loc[:, series_cols].apply(pd.to_numeric, errors="raise")
        wide = wide.sort_values("date").reset_index(drop=True)

    return wide
