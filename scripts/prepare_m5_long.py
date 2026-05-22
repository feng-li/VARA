#!/usr/bin/env python3
"""Convert raw M5 sales history to forecasting-ready long parquet files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


RAW_DIR = Path("data/m5")
OUTPUT_DIR = Path("data/m5/long")
SPLIT_FILES = {
    "evaluation": "sales_train_evaluation.parquet",
    "validation": "sales_train_validation.parquet",
}
META_COLUMNS = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
OUTPUT_COLUMNS = ["date", "series_id", "value", "d", "item_id", "dept_id", "cat_id", "store_id", "state_id"]


def _day_columns(columns: list[str]) -> list[str]:
    days = [col for col in columns if col.startswith("d_") and col[2:].isdigit()]
    return sorted(days, key=lambda col: int(col[2:]))


def _load_calendar(calendar_path: Path, day_cols: list[str]) -> pd.DataFrame:
    calendar = pd.read_parquet(calendar_path, columns=["d", "date"])
    calendar["date"] = pd.to_datetime(calendar["date"])
    calendar = calendar.loc[calendar["d"].isin(day_cols), ["d", "date"]].copy()
    missing_days = sorted(set(day_cols) - set(calendar["d"]), key=lambda col: int(col[2:]))
    if missing_days:
        raise ValueError(f"Calendar is missing day ids: {missing_days[:5]}")
    return calendar


def _write_parquet_chunks(chunks: list[pd.DataFrame], output_path: Path, *, compression: str) -> int:
    writer: pq.ParquetWriter | None = None
    n_rows = 0
    try:
        for chunk in chunks:
            table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression=compression)
            writer.write_table(table)
            n_rows += int(table.num_rows)
    finally:
        if writer is not None:
            writer.close()
    return n_rows


def _iter_long_chunks(
    sales: pd.DataFrame,
    calendar: pd.DataFrame,
    day_cols: list[str],
    *,
    chunk_series: int,
):
    if chunk_series < 1:
        raise ValueError("chunk_series must be positive.")

    for start in range(0, len(sales), chunk_series):
        chunk = sales.iloc[start:start + chunk_series].copy()
        long = chunk.melt(
            id_vars=META_COLUMNS,
            value_vars=day_cols,
            var_name="d",
            value_name="value",
        )
        long = long.merge(calendar, on="d", how="left", sort=False)
        if long["date"].isna().any():
            raise ValueError("Converted M5 long chunk contains missing dates after calendar merge.")
        long = long.rename(columns={"id": "series_id"})
        long["series_id"] = long["series_id"].astype(str)
        long["value"] = pd.to_numeric(long["value"], errors="raise").astype("int32")
        for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id", "d"]:
            long[col] = long[col].astype(str)
        yield long.loc[:, OUTPUT_COLUMNS]


def convert_m5_sales_to_long(
    *,
    sales_path: Path,
    calendar_path: Path,
    output_path: Path,
    series_metadata_path: Path,
    chunk_series: int = 512,
    compression: str = "zstd",
    max_series: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    sales_path = Path(sales_path)
    calendar_path = Path(calendar_path)
    output_path = Path(output_path)
    series_metadata_path = Path(series_metadata_path)
    if not sales_path.exists():
        raise FileNotFoundError(f"Missing M5 sales file: {sales_path}")
    if not calendar_path.exists():
        raise FileNotFoundError(f"Missing M5 calendar file: {calendar_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass overwrite=True or --overwrite.")
    if series_metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {series_metadata_path}; pass overwrite=True or --overwrite.")
    if max_series < 0:
        raise ValueError("max_series must be nonnegative.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    series_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    if series_metadata_path.exists():
        series_metadata_path.unlink()

    sales = pd.read_parquet(sales_path)
    missing_meta = [col for col in META_COLUMNS if col not in sales.columns]
    if missing_meta:
        raise ValueError(f"M5 sales file is missing metadata columns: {missing_meta}")
    if max_series > 0:
        sales = sales.iloc[:max_series].copy()

    day_cols = _day_columns(list(sales.columns))
    if not day_cols:
        raise ValueError(f"M5 sales file has no d_* day columns: {sales_path}")
    calendar = _load_calendar(calendar_path, day_cols)

    series_metadata = sales.loc[:, META_COLUMNS].rename(columns={"id": "series_id"}).copy()
    series_metadata.to_parquet(series_metadata_path, index=False, compression=compression)

    n_rows = _write_parquet_chunks(
        _iter_long_chunks(sales, calendar, day_cols, chunk_series=chunk_series),
        output_path,
        compression=compression,
    )
    expected_rows = int(len(sales) * len(day_cols))
    if n_rows != expected_rows:
        raise RuntimeError(f"Unexpected row count: wrote {n_rows}, expected {expected_rows}")

    return {
        "sales_path": str(sales_path),
        "calendar_path": str(calendar_path),
        "output_path": str(output_path),
        "series_metadata_path": str(series_metadata_path),
        "n_series": int(len(sales)),
        "n_days": int(len(day_cols)),
        "n_rows": int(n_rows),
        "date_start": str(calendar["date"].min().date()),
        "date_end": str(calendar["date"].max().date()),
        "first_day": day_cols[0],
        "last_day": day_cols[-1],
        "chunk_series": int(chunk_series),
        "compression": compression,
        "max_series": int(max_series),
    }


def prepare_m5_long(
    *,
    raw_dir: Path = RAW_DIR,
    output_dir: Path = OUTPUT_DIR,
    splits: tuple[str, ...] = ("evaluation", "validation"),
    chunk_series: int = 512,
    compression: str = "zstd",
    max_series: int = 0,
    overwrite: bool = False,
) -> dict[str, Any]:
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    calendar_path = raw_dir / "calendar.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for split in splits:
        if split not in SPLIT_FILES:
            raise ValueError(f"Unknown M5 split {split!r}; expected one of {sorted(SPLIT_FILES)}")
        stem = Path(SPLIT_FILES[split]).stem
        suffix = f"_head{max_series}" if max_series > 0 else ""
        summaries[split] = convert_m5_sales_to_long(
            sales_path=raw_dir / SPLIT_FILES[split],
            calendar_path=calendar_path,
            output_path=output_dir / f"{stem}_long{suffix}.parquet",
            series_metadata_path=output_dir / f"{stem}_series_metadata{suffix}.parquet",
            chunk_series=chunk_series,
            compression=compression,
            max_series=max_series,
            overwrite=overwrite,
        )

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "M5 forecasting accuracy",
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "splits": list(splits),
        "long_schema": OUTPUT_COLUMNS,
        "summaries": summaries,
    }
    metadata_path = output_dir / ("metadata_head%d.json" % max_series if max_series > 0 else "metadata.json")
    if metadata_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {metadata_path}; pass overwrite=True or --overwrite.")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def _parse_splits(text: str) -> tuple[str, ...]:
    splits = tuple(item.strip() for item in text.split(",") if item.strip())
    if not splits:
        raise ValueError("At least one split must be provided.")
    return splits


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw M5 sales history to long parquet files.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--splits", type=str, default="evaluation,validation", help="Comma-separated subset of evaluation,validation.")
    parser.add_argument("--chunk-series", type=int, default=512)
    parser.add_argument("--compression", type=str, default="zstd")
    parser.add_argument("--max-series", type=int, default=0, help="0 converts all series; positive values are useful for smoke tests.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = prepare_m5_long(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        splits=_parse_splits(args.splits),
        chunk_series=args.chunk_series,
        compression=args.compression,
        max_series=args.max_series,
        overwrite=args.overwrite,
    )
    print("M5 long conversion complete:")
    for split, summary in metadata["summaries"].items():
        print(
            f"  {split}: rows={summary['n_rows']}, series={summary['n_series']}, "
            f"days={summary['n_days']}, output={summary['output_path']}"
        )
    print(f"  metadata: {metadata['metadata_path']}")


if __name__ == "__main__":
    main()
