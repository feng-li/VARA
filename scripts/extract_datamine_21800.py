#!/usr/bin/env python3
"""Extract the frozen 21,800-series Accounting-Signal Risk input panel.

The output keeps the source long-table schema and every available observation
for each selected signal.  Missing panel cells therefore remain absent rows and
are reconstructed as NaN by the normal DataMine application loader.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
from typing import Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    import _bootstrap  # type: ignore[import-not-found]  # noqa: F401
except ModuleNotFoundError:  # Imported as scripts.extract_datamine_21800 in tests.
    from scripts import _bootstrap  # noqa: F401

from vara.datamine import (
    audit_return_file,
    build_dense_panel,
    load_signal_metadata,
    relative_panel_split,
    select_frozen_universe,
    sha256_file,
)


DEFAULT_SOURCE = Path("data/DataMine/DataMinedLongShortReturnsEW.parquet")
DEFAULT_METADATA = Path("data/DataMine/DataMinedSignalList.parquet")
DEFAULT_OUTPUT = Path("data/DataMine/DataMinedLongShortReturnsEW_21800.parquet")
EXPECTED_SERIES = 21_800


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select the frozen 21,800 Accounting-Signal Risk series and write "
            "their original long-format rows to a standalone Parquet file."
        )
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file after the new file validates.",
    )
    return parser


def frozen_signal_ids(
    source: Path,
    metadata_path: Path,
    *,
    batch_size: int,
) -> tuple[list[str], dict[str, object]]:
    """Reproduce the paper's pre-validation frozen-universe selection."""
    metadata = load_signal_metadata(metadata_path, weighting="ew")
    audit = audit_return_file(
        source,
        metadata,
        return_unit="percent",
        chunksize=batch_size,
    )
    full_panel = build_dense_panel(
        source,
        metadata,
        audit,
        weighting="ew",
        return_unit="percent",
        chunksize=batch_size,
    )
    split = relative_panel_split(
        full_panel.dates,
        validation_months=60,
        test_months=120,
    )
    selected, diagnostics = select_frozen_universe(
        full_panel,
        split,
        universe_window_months=240,
        min_observed_fraction=0.99,
        min_median_nstock=10.0,
        min_nstock=5.0,
        min_nstock_fraction=0.99,
        max_missing_run=1,
        max_series=0,
        seed=202_407,
        selection_mode="quality_persistence",
    )
    if len(selected.signal_ids) != EXPECTED_SERIES:
        raise ValueError(
            f"Frozen rules selected {len(selected.signal_ids):,} series; "
            f"expected {EXPECTED_SERIES:,}. Refusing to write a mislabeled file."
        )
    selection = {
        "selection_start": diagnostics["selection_start"].iloc[0].date().isoformat(),
        "selection_end": diagnostics["selection_end"].iloc[0].date().isoformat(),
        "universe_series": int(len(diagnostics)),
        "selected_series": int(diagnostics["selected"].sum()),
        "rules": {
            "universe_window_months": 240,
            "min_observed_fraction": 0.99,
            "min_median_nstock": 10.0,
            "min_nstock": 5.0,
            "min_nstock_fraction": 0.99,
            "max_missing_run": 1,
        },
    }
    return selected.signal_ids, selection


def extract_signal_rows(
    source: Path,
    output: Path,
    signal_ids: Sequence[str],
    *,
    batch_size: int,
    compression: str,
    force: bool,
) -> int:
    """Stream selected signal rows to an atomically installed Parquet file."""
    if source.resolve() == output.resolve():
        raise ValueError("Source and output paths must differ.")
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists: {output}. Use --force to replace it.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

    parquet = pq.ParquetFile(source)
    signal_field = parquet.schema_arrow.field("signalid")
    if not pa.types.is_integer(signal_field.type):
        raise TypeError(
            "The frozen DataMine extractor expects an integer signalid column; "
            f"found {signal_field.type}."
        )
    selected_values = pa.array([int(value) for value in signal_ids], type=signal_field.type)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    writer: pq.ParquetWriter | None = None
    rows_written = 0
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
        temporary = Path(temporary_name)
        writer = pq.ParquetWriter(
            temporary,
            parquet.schema_arrow,
            compression=compression,
        )
        for batch in parquet.iter_batches(batch_size=batch_size):
            keep = pc.is_in(batch.column("signalid"), value_set=selected_values)
            selected_batch = batch.filter(keep)
            if selected_batch.num_rows:
                writer.write_batch(selected_batch)
                rows_written += selected_batch.num_rows
        writer.close()
        writer = None

        extracted_ids: set[int] = set()
        extracted_parquet = pq.ParquetFile(temporary)
        for batch in extracted_parquet.iter_batches(
            batch_size=batch_size,
            columns=["signalid"],
        ):
            extracted_ids.update(pc.unique(batch.column("signalid")).to_pylist())
        expected_ids = {int(value) for value in signal_ids}
        if extracted_ids != expected_ids:
            missing = sorted(expected_ids.difference(extracted_ids))[:10]
            extra = sorted(extracted_ids.difference(expected_ids))[:10]
            raise ValueError(
                "Extracted signal IDs do not match the frozen universe; "
                f"missing={missing}, extra={extra}."
            )
        temporary.replace(output)
        output.chmod(source.stat().st_mode & 0o666)
    finally:
        if writer is not None:
            writer.close()
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return rows_written


def main() -> None:
    args = _parser().parse_args()
    if not args.source.exists():
        raise FileNotFoundError(args.source)
    if not args.metadata.exists():
        raise FileNotFoundError(args.metadata)

    print(f"Selecting the frozen universe from {args.source}", flush=True)
    signal_ids, selection = frozen_signal_ids(
        args.source,
        args.metadata,
        batch_size=args.batch_size,
    )
    print(f"Extracting {len(signal_ids):,} series to {args.output}", flush=True)
    rows = extract_signal_rows(
        args.source,
        args.output,
        signal_ids,
        batch_size=args.batch_size,
        compression=args.compression,
        force=args.force,
    )

    manifest = {
        "schema": "datamine_frozen_return_panel_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": args.source.as_posix(),
        "source_sha256": sha256_file(args.source),
        "metadata": args.metadata.as_posix(),
        "metadata_sha256": sha256_file(args.metadata),
        "output": args.output.as_posix(),
        "output_sha256": sha256_file(args.output),
        "rows": rows,
        **selection,
        "row_policy": "all source rows for selected signal IDs; no imputation",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {rows:,} rows for {len(signal_ids):,} series "
        f"({args.output.stat().st_size:,} bytes).",
        flush=True,
    )
    print(f"Manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
