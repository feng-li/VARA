#!/usr/bin/env python3
"""Prepare the M4 yearly dataset for VARA forecasting experiments.

Expected raw files are the official M4 CSVs:

- ``Yearly-train.csv``
- ``Yearly-test.csv``
- optional ``M4-info.csv``

The script writes long train/test/full files and a balanced tail-wide panel.
The wide panel aligns each series by relative time using the last
``tail_length`` training observations followed by the official test horizon.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_DIR = Path("data/m4/raw")
OUTPUT_DIR = Path("data/m4/yearly")
YEARLY_TRAIN_NAMES = ("Yearly-train.csv", "yearly-train.csv", "Yearly_train.csv", "YearlyTrain.csv")
YEARLY_TEST_NAMES = ("Yearly-test.csv", "yearly-test.csv", "Yearly_test.csv", "YearlyTest.csv")
INFO_NAMES = ("M4-info.csv", "m4-info.csv", "M4Info.csv", "M4-info.csv")


def _find_first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.exists():
            return path
    return None


def _resolve_path(path: Path | None, raw_dir: Path, names: tuple[str, ...], label: str, required: bool) -> Path | None:
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"{label} file does not exist: {path}")
        return path

    found = _find_first_existing(raw_dir, names)
    if found is None and required:
        candidates = ", ".join(str(raw_dir / name) for name in names)
        raise FileNotFoundError(f"Could not find {label} file. Expected one of: {candidates}")
    return found


def _read_m4_matrix(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(path)
    if raw.shape[1] < 2:
        raise ValueError(f"{path} must contain a series id column and at least one value column.")

    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for record in raw.itertuples(index=False, name=None):
        series_id = str(record[0])
        values = pd.to_numeric(pd.Series(record[1:]), errors="coerce").dropna().to_numpy(dtype=float)
        meta_rows.append({"series_id": series_id, "length": int(values.size)})
        for i, value in enumerate(values, start=1):
            rows.append({"series_id": series_id, "t": i, "y": float(value)})

    return pd.DataFrame(rows), pd.DataFrame(meta_rows)


def _normalize_info(info_path: Path | None) -> pd.DataFrame:
    if info_path is None:
        return pd.DataFrame(columns=["series_id"])

    info = pd.read_csv(info_path)
    rename = {}
    for col in info.columns:
        key = str(col).strip().lower().replace(" ", "").replace("_", "")
        if key in {"m4id", "id", "seriesid", "uniqueid"}:
            rename[col] = "series_id"
        elif key == "category":
            rename[col] = "category"
        elif key == "frequency":
            rename[col] = "frequency"
        elif key == "horizon":
            rename[col] = "horizon"
        elif key in {"sp", "seasonalperiod", "seasonality"}:
            rename[col] = "seasonal_period"
        elif key == "startingdate":
            rename[col] = "starting_date"
    info = info.rename(columns=rename)
    if "series_id" not in info.columns:
        raise ValueError(f"{info_path} does not contain an M4 id column.")

    keep = [col for col in ["series_id", "category", "frequency", "horizon", "seasonal_period", "starting_date"] if col in info.columns]
    info = info.loc[:, keep].copy()
    if "frequency" in info.columns:
        yearly = info["frequency"].astype(str).str.lower().eq("yearly")
        if yearly.any():
            info = info.loc[yearly].copy()
    return info


def _start_year(value: object) -> int | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit() and len(text) <= 4:
        return int(text)
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.notna(parsed):
        return int(parsed.year)
    for token in text.replace("/", "-").split("-"):
        if token.isdigit() and len(token) == 4:
            return int(token)
    return None


def _add_dates(long_df: pd.DataFrame, info: pd.DataFrame, *, default_start_year: int = 1900) -> pd.DataFrame:
    out = long_df.copy()
    if "starting_date" in info.columns:
        starts = info.loc[:, ["series_id", "starting_date"]].copy()
        starts["start_year"] = starts["starting_date"].map(_start_year)
        out = out.merge(starts.loc[:, ["series_id", "start_year"]], on="series_id", how="left", sort=False)
    else:
        out["start_year"] = np.nan

    out["start_year"] = out["start_year"].fillna(default_start_year).astype(int)
    years = out["start_year"].to_numpy(dtype=int) + out["t"].to_numpy(dtype=int) - 1
    out["date"] = pd.Series(years.astype(str)) + "-01-01"
    return out.loc[:, ["series_id", "t", "date", "y"]]


def _make_full_long(train_long: pd.DataFrame, test_long: pd.DataFrame, lengths: pd.DataFrame) -> pd.DataFrame:
    train = train_long.copy()
    train["split"] = "train"
    test = test_long.merge(lengths.rename(columns={"length": "train_length"}), on="series_id", how="left", sort=False)
    test["t"] = test["t"] + test["train_length"].astype(int)
    test = test.drop(columns=["train_length"])
    test["split"] = "test"
    return pd.concat([train, test], ignore_index=True, sort=False)


def _make_tail_wide(
    train_long: pd.DataFrame,
    test_long: pd.DataFrame,
    train_lengths: pd.DataFrame,
    test_lengths: pd.DataFrame,
    *,
    tail_length: int,
    max_series: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    horizon_values = sorted(test_lengths["length"].unique().tolist())
    if len(horizon_values) != 1:
        raise ValueError(f"Expected one common test horizon for yearly data, got {horizon_values}.")
    horizon = int(horizon_values[0])

    eligible = train_lengths.loc[train_lengths["length"] >= tail_length, "series_id"].tolist()
    if max_series > 0:
        eligible = eligible[:max_series]
    if not eligible:
        raise ValueError("No M4 yearly series have enough observations for the requested tail_length.")

    total_length = int(tail_length + horizon)
    dates = pd.date_range("1900-01-01", periods=total_length, freq="YS")
    columns: dict[str, Any] = {"date": dates}
    meta_rows = []
    train_map = {row.series_id: int(row.length) for row in train_lengths.itertuples(index=False)}
    train_values_by_id = {
        str(series_id): group["y"].to_numpy(dtype=float)
        for series_id, group in train_long.groupby("series_id", sort=False)
    }
    test_values_by_id = {
        str(series_id): group["y"].to_numpy(dtype=float)
        for series_id, group in test_long.groupby("series_id", sort=False)
    }

    for series_id in eligible:
        train_values = train_values_by_id.get(series_id, np.asarray([], dtype=float))
        test_values = test_values_by_id.get(series_id, np.asarray([], dtype=float))
        if train_values.size < tail_length or test_values.size != horizon:
            continue
        values = np.concatenate([train_values[-tail_length:], test_values])
        columns[series_id] = values
        meta_rows.append({
            "series_id": series_id,
            "train_length": train_map[series_id],
            "tail_length": int(tail_length),
            "horizon": horizon,
        })

    if len(meta_rows) < 2:
        raise ValueError("Balanced tail-wide output needs at least two eligible series.")
    return pd.DataFrame(columns), pd.DataFrame(meta_rows)


def prepare_m4_yearly(
    *,
    raw_dir: Path = RAW_DIR,
    output_dir: Path = OUTPUT_DIR,
    train_path: Path | None = None,
    test_path: Path | None = None,
    info_path: Path | None = None,
    tail_length: int = 30,
    max_series: int = 0,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    train_path = _resolve_path(train_path, raw_dir, YEARLY_TRAIN_NAMES, "M4 yearly train", required=True)
    test_path = _resolve_path(test_path, raw_dir, YEARLY_TEST_NAMES, "M4 yearly test", required=True)
    info_path = _resolve_path(info_path, raw_dir, INFO_NAMES, "M4 info", required=False)
    if tail_length < 2:
        raise ValueError("tail_length must be at least 2.")
    if max_series < 0:
        raise ValueError("max_series must be nonnegative.")

    train_long, train_lengths = _read_m4_matrix(train_path)
    test_long, test_lengths = _read_m4_matrix(test_path)
    info = _normalize_info(info_path)

    known_ids = set(train_lengths["series_id"])
    test_ids = set(test_lengths["series_id"])
    if known_ids != test_ids:
        missing_in_test = sorted(known_ids - test_ids)[:5]
        missing_in_train = sorted(test_ids - known_ids)[:5]
        raise ValueError(f"Train/test series ids do not match. missing_in_test={missing_in_test}, missing_in_train={missing_in_train}")

    train_long = _add_dates(train_long, info)
    full_long = _make_full_long(train_long, test_long, train_lengths)
    full_long = _add_dates(full_long.drop(columns=["date"], errors="ignore"), info)
    test_long_aligned = full_long.loc[full_long["t"] > full_long.merge(train_lengths.rename(columns={"length": "train_length"}), on="series_id", how="left")["train_length"].to_numpy()].copy()

    series_info = train_lengths.rename(columns={"length": "train_length"}).merge(
        test_lengths.rename(columns={"length": "test_length"}),
        on="series_id",
        how="left",
        sort=False,
    )
    if not info.empty:
        series_info = series_info.merge(info, on="series_id", how="left", sort=False)
    if "starting_date" in series_info.columns:
        series_info["start_year"] = series_info["starting_date"].map(_start_year)
    else:
        series_info["start_year"] = np.nan
    series_info["horizon"] = series_info["test_length"]

    tail_wide, tail_meta = _make_tail_wide(
        train_long,
        test_long,
        train_lengths,
        test_lengths,
        tail_length=tail_length,
        max_series=max_series,
    )
    tail_meta = tail_meta.merge(series_info, on=["series_id", "train_length", "horizon"], how="left", sort=False)

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train_long": output_dir / "m4_yearly_train_long.csv",
        "test_long": output_dir / "m4_yearly_test_long.csv",
        "full_long": output_dir / "m4_yearly_full_long.csv",
        "series_info": output_dir / "m4_yearly_series_info.csv",
        "tail_wide": output_dir / f"m4_yearly_tail{tail_length}_wide.csv",
        "tail_info": output_dir / f"m4_yearly_tail{tail_length}_series_info.csv",
        "metadata": output_dir / "metadata.json",
    }
    train_long.to_csv(paths["train_long"], index=False)
    test_long_aligned.to_csv(paths["test_long"], index=False)
    full_long.to_csv(paths["full_long"], index=False)
    series_info.to_csv(paths["series_info"], index=False)
    tail_wide.to_csv(paths["tail_wide"], index=False)
    tail_meta.to_csv(paths["tail_info"], index=False)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "M4 yearly",
        "raw_dir": str(raw_dir),
        "train_path": str(train_path),
        "test_path": str(test_path),
        "info_path": "" if info_path is None else str(info_path),
        "n_series": int(len(series_info)),
        "n_train_rows": int(len(train_long)),
        "n_test_rows": int(len(test_long_aligned)),
        "test_horizon_values": sorted(int(x) for x in test_lengths["length"].unique().tolist()),
        "tail_length": int(tail_length),
        "tail_n_series": int(tail_wide.shape[1] - 1),
        "tail_total_rows": int(tail_wide.shape[0]),
        "max_series": int(max_series),
        "outputs": {key: str(value) for key, value in paths.items() if key != "metadata"},
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare official M4 yearly files for VARA experiments.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--train-path", type=Path, default=None)
    parser.add_argument("--test-path", type=Path, default=None)
    parser.add_argument("--info-path", type=Path, default=None)
    parser.add_argument("--tail-length", type=int, default=30)
    parser.add_argument("--max-series", type=int, default=0, help="0 keeps all eligible series in the tail-wide output.")
    args = parser.parse_args()

    metadata = prepare_m4_yearly(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        train_path=args.train_path,
        test_path=args.test_path,
        info_path=args.info_path,
        tail_length=args.tail_length,
        max_series=args.max_series,
    )
    print("M4 yearly organization complete:")
    for key, value in metadata.items():
        if key == "outputs":
            print("outputs:")
            for name, path in value.items():
                print(f"  {name}: {path}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
