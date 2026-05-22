#!/usr/bin/env python3
"""Prepare series-level M5 features for VARA attention."""

from __future__ import annotations

import argparse
from pathlib import Path

from vara.m5_features import make_m5_attention_features_from_files


DEFAULT_INPUT = Path("data/m5")
DEFAULT_OUTPUT = Path("data/m5/m5_attention_features_CA_1.parquet")


def _find_table(input_dir: Path, stem: str) -> Path:
    for suffix in (".parquet", ".csv"):
        path = input_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {stem}.parquet or {stem}.csv in {input_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare series-level M5 features for VARA attention.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sales", type=Path, default=None)
    parser.add_argument("--prices", type=Path, default=None)
    parser.add_argument("--calendar", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--store-id", default="CA_1", help="Use an empty value to prepare all stores.")
    parser.add_argument("--holdout", type=int, default=28, help="Last H sales days excluded from feature computation.")
    parser.add_argument("--max-series", type=int, default=0, help="0 keeps all filtered series; positive values are useful for smoke tests.")
    parser.add_argument("--standardize-output", action="store_true", help="Standardize numeric features before saving.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = make_m5_attention_features_from_files(
        sales_path=args.sales or _find_table(args.input_dir, "sales_train_evaluation"),
        prices_path=args.prices or _find_table(args.input_dir, "sell_prices"),
        calendar_path=args.calendar or _find_table(args.input_dir, "calendar"),
        output_path=args.out,
        store_id=args.store_id or None,
        holdout=args.holdout,
        max_series=args.max_series,
        standardize_output=args.standardize_output,
        overwrite=args.overwrite,
    )
    print("M5 attention feature preparation complete:")
    print(f"  output: {metadata['output_path']}")
    print(f"  metadata: {metadata['metadata_path']}")
    print(f"  n_series: {metadata['n_series']}")
    print(f"  n_columns: {metadata['n_columns']}")
    print(f"  holdout: {metadata['holdout']}")
    print(f"  store_id: {metadata['store_id'] or 'all stores'}")


if __name__ == "__main__":
    main()
