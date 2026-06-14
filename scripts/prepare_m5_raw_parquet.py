#!/usr/bin/env python3
"""Convert extracted M5 CSV files to raw parquet inputs used by VARA."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


CSV_DIR = Path("data/m5/csv")
OUTPUT_DIR = Path("data/m5")
DEFAULT_FILES = ("sales_train_evaluation", "calendar", "sell_prices")
KNOWN_FILES = {
    "sales_train_evaluation": "sales_train_evaluation.csv",
    "sales_train_validation": "sales_train_validation.csv",
    "calendar": "calendar.csv",
    "sell_prices": "sell_prices.csv",
    "sample_submission": "sample_submission.csv",
}


def _normalise_file_name(name: str) -> str:
    stem = name.strip()
    for suffix in (".csv.gz", ".csv", ".parquet"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    if not stem:
        raise ValueError("Empty M5 file name.")
    return stem


def _parse_files(text: str) -> tuple[str, ...]:
    files = tuple(_normalise_file_name(item) for item in text.split(",") if item.strip())
    if not files:
        raise ValueError("At least one file stem must be provided.")
    return files


def _find_csv(csv_dir: Path, stem: str) -> Path:
    base_name = KNOWN_FILES.get(stem, f"{stem}.csv")
    candidates = [
        csv_dir / base_name,
        csv_dir / f"{Path(base_name).stem}.csv.gz",
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Missing extracted M5 CSV for {stem!r}. Tried: {tried}")


def _validate_columns(stem: str, df: pd.DataFrame) -> None:
    required = {
        "calendar": {"date", "d", "wm_yr_wk"},
        "sell_prices": {"store_id", "item_id", "wm_yr_wk", "sell_price"},
        "sales_train_evaluation": {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d_1"},
        "sales_train_validation": {"id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "d_1"},
        "sample_submission": {"id"},
    }.get(stem)
    if not required:
        return
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{stem}.csv is missing required columns: {missing}")


def convert_m5_csv_to_parquet(
    *,
    csv_dir: Path = CSV_DIR,
    output_dir: Path = OUTPUT_DIR,
    files: tuple[str, ...] = DEFAULT_FILES,
    compression: str = "zstd",
    overwrite: bool = False,
) -> dict[str, Any]:
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"Missing M5 CSV directory: {csv_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, Any] = {}
    for stem in files:
        input_path = _find_csv(csv_dir, stem)
        output_path = output_dir / f"{stem}.parquet"
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output already exists: {output_path}; pass --overwrite to replace it.")

        df = pd.read_csv(input_path, low_memory=False)
        _validate_columns(stem, df)
        if stem == "calendar" and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        df.to_parquet(output_path, index=False, compression=compression)
        summaries[stem] = {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "compression": compression,
        }
        print(f"Converted {input_path} -> {output_path} ({len(df)} rows, {len(df.columns)} columns)")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": "M5 Forecasting Accuracy Kaggle CSV files",
        "csv_dir": str(csv_dir),
        "output_dir": str(output_dir),
        "files": list(files),
        "summaries": summaries,
    }
    metadata_path = output_dir / "m5_raw_parquet_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    print(f"Wrote metadata: {metadata_path}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert extracted M5 CSV files to raw parquet files.")
    parser.add_argument("--csv-dir", type=Path, default=CSV_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--files",
        default=",".join(DEFAULT_FILES),
        help="Comma-separated CSV stems to convert. Defaults to the raw files needed by the VARA M5 application.",
    )
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    convert_m5_csv_to_parquet(
        csv_dir=args.csv_dir,
        output_dir=args.output_dir,
        files=_parse_files(args.files),
        compression=args.compression,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
