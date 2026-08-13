from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pytest

from scripts.extract_datamine_21800 import extract_signal_rows


def test_extract_signal_rows_preserves_schema_order_and_missing_rows(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "selected.parquet"
    frame = pd.DataFrame({
        "signalid": [1, 1, 2, 3, 3],
        "year": [2020, 2020, 2020, 2020, 2020],
        "month": [1, 2, 1, 1, 3],
        "ret": [1.0, 2.0, 3.0, 4.0, 6.0],
        "nstock": [10, 11, 12, 13, 15],
    })
    frame.to_parquet(source, index=False)

    rows = extract_signal_rows(
        source,
        output,
        ["1", "3"],
        batch_size=2,
        compression="zstd",
        force=False,
    )

    result = pd.read_parquet(output)
    assert rows == 4
    pd.testing.assert_frame_equal(result, frame.loc[frame.signalid.isin([1, 3])].reset_index(drop=True))
    assert pq.read_schema(output) == pq.read_schema(source)
    assert not ((result.signalid == 3) & (result.month == 2)).any()


def test_extract_signal_rows_refuses_to_overwrite_without_force(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "selected.parquet"
    pd.DataFrame({
        "signalid": [1],
        "year": [2020],
        "month": [1],
        "ret": [1.0],
        "nstock": [10],
    }).to_parquet(source, index=False)
    output.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="--force"):
        extract_signal_rows(
            source,
            output,
            ["1"],
            batch_size=2,
            compression="zstd",
            force=False,
        )
