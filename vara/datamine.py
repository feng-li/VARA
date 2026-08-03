"""Leakage-safe helpers for the Accounting-Signal Risk VARA application."""

from __future__ import annotations

from dataclasses import dataclass
from functools import wraps
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from vara.forecast_metrics import add_composite_rank_score, normal_crps


RETURN_REQUIRED_COLUMNS = {"signalid", "year", "month", "ret"}
SIGNAL_COLUMN_ALIASES = {
    "signalid": ("signalid", "signal_id"),
    "formula_type": ("formula_type", "signal_form", "form"),
    "numerator": ("numerator", "v1", "x1"),
    "denominator": ("denominator", "v2", "x2"),
}
RETURN_UNITS = {"percent", "decimal"}
TARGET_TYPES = {"log_squared_return", "raw_return"}
BASELINE_TYPES = {"har_ridge", "ar1_ridge"}
SERIES_SELECTION_MODES = {"stratified_random", "quality_persistence"}


@dataclass(frozen=True)
class PanelSplit:
    """Relative validation/test split for a monthly panel."""

    universe_selection_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    validation_indices: np.ndarray
    test_indices: np.ndarray


@dataclass
class DataMinePanel:
    """Dense monthly panel in one frozen component order."""

    dates: pd.DatetimeIndex
    component_ids: list[str]
    signal_ids: list[str]
    returns: np.ndarray
    nstock: np.ndarray
    metadata: pd.DataFrame
    weighting: str
    coverage_source: str

    def validate(self) -> None:
        if self.returns.ndim != 2 or self.nstock.ndim != 2:
            raise ValueError("returns and nstock must be two-dimensional arrays.")
        if self.returns.shape != self.nstock.shape:
            raise ValueError("returns and nstock must have the same shape.")
        if self.returns.shape != (len(self.dates), len(self.component_ids)):
            raise ValueError("Panel arrays do not match the date/component dimensions.")
        if len(self.signal_ids) != len(self.component_ids):
            raise ValueError("signal_ids and component_ids must have the same length.")
        if list(self.metadata["component_id"].astype(str)) != self.component_ids:
            raise ValueError("Metadata rows must follow the frozen component order.")


@dataclass(frozen=True)
class RobustPanelScaler:
    """Componentwise median/MAD response transformation."""

    center: np.ndarray
    scale: np.ndarray
    scale_floor: float
    clip: float | None

    def transform(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        transformed = (values - self.center) / self.scale
        if self.clip is not None:
            transformed = np.clip(transformed, -float(self.clip), float(self.clip))
        return transformed

    def inverse(self, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=float)
        return self.center + self.scale * values


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def write_json(path: str | Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    """Write deterministic, human-readable JSON."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest."""
    digest = sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(int(chunk_bytes)):
            digest.update(block)
    return digest.hexdigest()


def _canonical_signal_ids(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.notna().all() and np.all(np.equal(numeric, np.floor(numeric))):
        return numeric.astype(np.int64).astype(str)
    out = values.astype("string").str.strip()
    if out.isna().any() or (out == "").any():
        raise ValueError("signalid contains missing or empty values.")
    return out.astype(str)


def _resolve_alias(columns: Iterable[str], canonical: str) -> str:
    available = set(columns)
    matches = [alias for alias in SIGNAL_COLUMN_ALIASES[canonical] if alias in available]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one column for {canonical!r}; found {matches or 'none'} "
            f"among aliases {SIGNAL_COLUMN_ALIASES[canonical]}."
        )
    return matches[0]


def canonical_signal_metadata(frame: pd.DataFrame, weighting: str) -> pd.DataFrame:
    """Map public signal-list variants to the application metadata schema."""
    rename = {
        _resolve_alias(frame.columns, canonical): canonical
        for canonical in SIGNAL_COLUMN_ALIASES
    }
    metadata = frame.rename(columns=rename).loc[
        :, ["signalid", "formula_type", "numerator", "denominator"]
    ].copy()
    metadata["signalid"] = _canonical_signal_ids(metadata["signalid"])
    for column in ("formula_type", "numerator", "denominator"):
        metadata[column] = metadata[column].astype("string").str.strip()
        if metadata[column].isna().any() or (metadata[column] == "").any():
            raise ValueError(f"Signal metadata column {column!r} contains missing values.")
        metadata[column] = metadata[column].astype(str)
    if metadata["signalid"].duplicated().any():
        duplicated = metadata.loc[metadata["signalid"].duplicated(), "signalid"].head().tolist()
        raise ValueError(f"Signal metadata must have one row per signalid; duplicates: {duplicated}")

    weighting = str(weighting).lower()
    metadata["weighting"] = weighting
    metadata["component_id"] = weighting.upper() + "_" + metadata["signalid"]
    ratio = metadata["formula_type"].eq("v1/v2")
    change = metadata["formula_type"].eq("diff(v1)/lag(v2)")
    labels = np.where(
        ratio,
        metadata["numerator"] + " / " + metadata["denominator"],
        np.where(
            change,
            "delta12(" + metadata["numerator"] + ") / lag12(" + metadata["denominator"] + ")",
            metadata["formula_type"] + "(" + metadata["numerator"] + "," + metadata["denominator"] + ")",
        ),
    )
    metadata["formula_label"] = labels

    numeric_ids = pd.to_numeric(metadata["signalid"], errors="coerce")
    if numeric_ids.notna().all():
        metadata = metadata.assign(_sort_id=numeric_ids).sort_values("_sort_id").drop(columns="_sort_id")
    else:
        metadata = metadata.sort_values("signalid")
    return metadata.reset_index(drop=True)


def load_signal_metadata(path: str | Path, weighting: str) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        frame = pd.read_parquet(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError(f"Unsupported signal metadata format: {source.suffix}")
    return canonical_signal_metadata(frame, weighting=weighting)


def _source_columns(path: str | Path) -> list[str]:
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        return list(pq.ParquetFile(source).schema_arrow.names)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source, nrows=0).columns.astype(str).tolist()
    raise ValueError(f"Unsupported return-data format: {source.suffix}")


def _iter_source_frames(
    path: str | Path,
    *,
    batch_size: int,
) -> Iterable[pd.DataFrame]:
    """Yield bounded-size pandas batches from Parquet or legacy CSV input."""
    source = Path(path)
    if source.suffix.lower() in {".parquet", ".pq"}:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(source)
        for batch in parquet.iter_batches(batch_size=int(batch_size), use_threads=True):
            yield batch.to_pandas()
        return
    if source.suffix.lower() == ".csv":
        yield from pd.read_csv(source, chunksize=int(batch_size))
        return
    raise ValueError(f"Unsupported return-data format: {source.suffix}")


def _coverage_columns(columns: Iterable[str]) -> tuple[tuple[str, ...], str]:
    available = set(columns)
    if {"nstock_long", "nstock_short"}.issubset(available):
        return ("nstock_long", "nstock_short"), "min(nstock_long,nstock_short)"
    if "nstock" in available:
        return ("nstock",), "nstock"
    raise ValueError("Return data need nstock or both nstock_long and nstock_short.")


def _coverage_values(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    if tuple(columns) == ("nstock",):
        return pd.to_numeric(frame["nstock"], errors="coerce").to_numpy(dtype=float)
    long = pd.to_numeric(frame[columns[0]], errors="coerce").to_numpy(dtype=float)
    short = pd.to_numeric(frame[columns[1]], errors="coerce").to_numpy(dtype=float)
    return np.minimum(long, short)


def audit_return_file(
    path: str | Path,
    metadata: pd.DataFrame,
    *,
    return_unit: str,
    chunksize: int = 500_000,
) -> dict[str, Any]:
    """Stream a return dataset and validate its keys, values, and metadata join."""
    if return_unit not in RETURN_UNITS:
        raise ValueError(f"return_unit must be one of {sorted(RETURN_UNITS)}.")
    source = Path(path)
    source_columns = _source_columns(source)
    missing = RETURN_REQUIRED_COLUMNS.difference(source_columns)
    if missing:
        raise ValueError(f"Missing required return columns: {sorted(missing)}")
    coverage_columns, coverage_source = _coverage_columns(source_columns)

    row_count = 0
    finite_count = 0
    ret_sum = 0.0
    ret_sum_sq = 0.0
    ret_min = np.inf
    ret_max = -np.inf
    nstock_min = np.inf
    nstock_max = -np.inf
    date_codes: set[int] = set()
    observed_signalids: set[str] = set()
    first_key: int | None = None
    previous_key: int | None = None
    duplicate_count = 0
    sorted_keys = True
    invalid_months = 0
    invalid_years = 0
    nonfinite_returns = 0
    negative_coverage = 0

    for chunk in _iter_source_frames(source, batch_size=int(chunksize)):
        signalid = _canonical_signal_ids(chunk["signalid"])
        signal_numeric = pd.to_numeric(signalid, errors="raise").to_numpy(dtype=np.int64)
        year = pd.to_numeric(chunk["year"], errors="coerce").to_numpy(dtype=float)
        month = pd.to_numeric(chunk["month"], errors="coerce").to_numpy(dtype=float)
        invalid_years += int((~np.isfinite(year)).sum())
        invalid_months += int((~np.isfinite(month) | (month < 1) | (month > 12)).sum())
        if invalid_years or invalid_months:
            raise ValueError("Return data contain invalid year or month values.")
        year_i = year.astype(np.int64)
        month_i = month.astype(np.int64)
        date_code = year_i * 12 + month_i
        key = signal_numeric * 1_000_000 + year_i * 100 + month_i

        duplicate_count += int(pd.Series(key).duplicated().sum())
        if previous_key is not None:
            duplicate_count += int(key[0] == previous_key)
            sorted_keys = sorted_keys and bool(key[0] >= previous_key)
        if len(key) > 1:
            sorted_keys = sorted_keys and bool(np.all(key[1:] >= key[:-1]))
        if first_key is None:
            first_key = int(key[0])
        previous_key = int(key[-1])

        returns = pd.to_numeric(chunk["ret"], errors="coerce").to_numpy(dtype=float)
        finite = np.isfinite(returns)
        nonfinite_returns += int((~finite).sum())
        finite_values = returns[finite]
        if finite_values.size:
            finite_count += int(finite_values.size)
            ret_sum += float(finite_values.sum())
            ret_sum_sq += float(np.square(finite_values).sum())
            ret_min = min(ret_min, float(finite_values.min()))
            ret_max = max(ret_max, float(finite_values.max()))

        coverage = _coverage_values(chunk, coverage_columns)
        finite_coverage = coverage[np.isfinite(coverage)]
        negative_coverage += int((finite_coverage < 0).sum())
        if finite_coverage.size:
            nstock_min = min(nstock_min, float(finite_coverage.min()))
            nstock_max = max(nstock_max, float(finite_coverage.max()))

        row_count += len(chunk)
        date_codes.update(np.unique(date_code).tolist())
        observed_signalids.update(signalid.unique().tolist())

    if row_count == 0:
        raise ValueError(f"Return file is empty: {source}")
    if duplicate_count:
        raise ValueError(f"Return file contains {duplicate_count} duplicate date-signal keys.")
    if not sorted_keys:
        raise ValueError("Return file must be sorted by signalid, year, and month for streaming validation.")
    if nonfinite_returns:
        raise ValueError(f"Return file contains {nonfinite_returns} non-finite returns.")
    if negative_coverage:
        raise ValueError(f"Return file contains {negative_coverage} negative stock counts.")

    metadata_ids = set(metadata["signalid"].astype(str))
    missing_metadata = sorted(observed_signalids.difference(metadata_ids))
    absent_returns = sorted(metadata_ids.difference(observed_signalids))
    codes = np.asarray(sorted(date_codes), dtype=np.int64)
    gaps = int(np.sum(np.diff(codes) != 1)) if len(codes) > 1 else 0
    mean = ret_sum / finite_count
    variance = max(ret_sum_sq / finite_count - mean**2, 0.0)
    min_code = int(codes.min())
    max_code = int(codes.max())

    return {
        "path": source.as_posix(),
        "bytes": source.stat().st_size,
        "rows": row_count,
        "signals": len(observed_signalids),
        "signalid_min": min(int(value) for value in observed_signalids),
        "signalid_max": max(int(value) for value in observed_signalids),
        "date_min": f"{(min_code - 1) // 12:04d}-{(min_code - 1) % 12 + 1:02d}",
        "date_max": f"{(max_code - 1) // 12:04d}-{(max_code - 1) % 12 + 1:02d}",
        "distinct_months": len(codes),
        "calendar_gaps": gaps,
        "return_unit_source": return_unit,
        "return_unit_processed": "decimal",
        "return_mean_source_units": mean,
        "return_sd_source_units": float(np.sqrt(variance)),
        "return_min_source_units": ret_min,
        "return_max_source_units": ret_max,
        "coverage_source": coverage_source,
        "coverage_min": nstock_min,
        "coverage_max": nstock_max,
        "missing_metadata_signalids": missing_metadata,
        "metadata_signalids_without_returns": absent_returns,
        "keys_sorted": sorted_keys,
        "duplicate_keys": duplicate_count,
        "nonfinite_returns": nonfinite_returns,
    }


def relative_panel_split(
    dates: Sequence[Any],
    *,
    validation_months: int,
    test_months: int,
) -> PanelSplit:
    """Create a split relative to the final month without inspecting outcomes."""
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    if not index.is_monotonic_increasing or index.has_duplicates:
        raise ValueError("dates must be unique and increasing.")
    expected = pd.date_range(index[0], index[-1], freq="ME")
    if not index.equals(expected):
        raise ValueError("dates must form a consecutive month-end index.")
    validation_months = int(validation_months)
    test_months = int(test_months)
    if validation_months < 1 or test_months < 1:
        raise ValueError("validation_months and test_months must be positive.")
    if len(index) <= validation_months + test_months:
        raise ValueError("Panel is too short for the requested validation and test periods.")

    test_start_idx = len(index) - test_months
    validation_start_idx = test_start_idx - validation_months
    validation_indices = np.arange(validation_start_idx, test_start_idx, dtype=np.int64)
    test_indices = np.arange(test_start_idx, len(index), dtype=np.int64)
    return PanelSplit(
        universe_selection_end=index[validation_start_idx - 1],
        validation_start=index[validation_start_idx],
        validation_end=index[test_start_idx - 1],
        test_start=index[test_start_idx],
        test_end=index[-1],
        validation_indices=validation_indices,
        test_indices=test_indices,
    )


def _month_end_from_code(code: int) -> pd.Timestamp:
    year = (int(code) - 1) // 12
    month = (int(code) - 1) % 12 + 1
    return pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(0)


def build_dense_panel(
    return_path: str | Path,
    metadata: pd.DataFrame,
    audit: Mapping[str, Any],
    *,
    weighting: str,
    return_unit: str,
    chunksize: int = 500_000,
) -> DataMinePanel:
    """Stream the return dataset into dense return and coverage matrices."""
    if return_unit not in RETURN_UNITS:
        raise ValueError(f"return_unit must be one of {sorted(RETURN_UNITS)}.")
    source = Path(return_path)
    source_columns = _source_columns(source)
    coverage_columns, coverage_source = _coverage_columns(source_columns)
    observed = set(str(value) for value in audit["metadata_signalids_without_returns"])
    observed = set(metadata["signalid"].astype(str)).difference(observed)
    panel_meta = metadata.loc[metadata["signalid"].isin(observed)].copy()
    panel_meta = panel_meta.reset_index(drop=True)
    signal_ids = panel_meta["signalid"].astype(str).tolist()
    component_ids = panel_meta["component_id"].astype(str).tolist()

    start = pd.Period(str(audit["date_min"]), freq="M")
    end = pd.Period(str(audit["date_max"]), freq="M")
    dates = pd.period_range(start, end, freq="M").to_timestamp("M")
    returns = np.full((len(dates), len(signal_ids)), np.nan, dtype=np.float32)
    nstock = np.full_like(returns, np.nan)

    numeric_ids = pd.to_numeric(pd.Series(signal_ids), errors="coerce")
    dense_indexer: np.ndarray | None = None
    id_to_col: dict[str, int] | None = None
    if numeric_ids.notna().all() and int(numeric_ids.max()) < 10_000_000:
        dense_indexer = np.full(int(numeric_ids.max()) + 1, -1, dtype=np.int32)
        dense_indexer[numeric_ids.to_numpy(dtype=np.int64)] = np.arange(len(signal_ids), dtype=np.int32)
    else:
        id_to_col = {signalid: idx for idx, signalid in enumerate(signal_ids)}

    start_code = int(start.year * 12 + start.month)
    for chunk in _iter_source_frames(source, batch_size=int(chunksize)):
        ids = _canonical_signal_ids(chunk["signalid"])
        if dense_indexer is not None:
            numeric = pd.to_numeric(ids, errors="raise").to_numpy(dtype=np.int64)
            valid_id = (numeric >= 0) & (numeric < len(dense_indexer))
            col = np.full(len(chunk), -1, dtype=np.int32)
            col[valid_id] = dense_indexer[numeric[valid_id]]
        else:
            assert id_to_col is not None
            col = ids.map(id_to_col).fillna(-1).to_numpy(dtype=np.int32)
        keep = col >= 0
        if not keep.any():
            continue
        year = pd.to_numeric(chunk.loc[keep, "year"], errors="raise").to_numpy(dtype=np.int64)
        month = pd.to_numeric(chunk.loc[keep, "month"], errors="raise").to_numpy(dtype=np.int64)
        row = year * 12 + month - start_code
        col_keep = col[keep]
        values = pd.to_numeric(chunk.loc[keep, "ret"], errors="raise").to_numpy(dtype=float)
        if return_unit == "percent":
            values = values / 100.0
        coverage = _coverage_values(chunk.loc[keep], coverage_columns)
        if np.isfinite(returns[row, col_keep]).any():
            raise ValueError("Duplicate panel cell encountered while building dense matrices.")
        returns[row, col_keep] = values.astype(np.float32)
        nstock[row, col_keep] = coverage.astype(np.float32)

    panel = DataMinePanel(
        dates=dates,
        component_ids=component_ids,
        signal_ids=signal_ids,
        returns=returns,
        nstock=nstock,
        metadata=panel_meta,
        weighting=str(weighting).lower(),
        coverage_source=coverage_source,
    )
    panel.validate()
    return panel


def _longest_true_run(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("mask must have shape (time, components).")
    current = np.zeros(mask.shape[1], dtype=np.int32)
    longest = np.zeros(mask.shape[1], dtype=np.int32)
    for row in mask:
        current = np.where(row, current + 1, 0)
        longest = np.maximum(longest, current)
    return longest


def _stratified_pilot_indices(
    metadata: pd.DataFrame,
    eligible_indices: np.ndarray,
    *,
    max_series: int,
    seed: int,
    scores: np.ndarray | None = None,
) -> np.ndarray:
    if max_series <= 0 or len(eligible_indices) <= max_series:
        return np.asarray(eligible_indices, dtype=np.int64)
    eligible_meta = metadata.iloc[eligible_indices].copy()
    eligible_meta["_index"] = eligible_indices
    if scores is not None:
        values = np.asarray(scores, dtype=float)
        if values.shape != (len(metadata),):
            raise ValueError("scores must have one value per metadata row.")
        eligible_meta["_score"] = values[eligible_indices]
    groups = list(eligible_meta.groupby("formula_type", sort=True, observed=True))
    total = len(eligible_meta)
    raw = np.asarray([max_series * len(group) / total for _, group in groups], dtype=float)
    quota = np.floor(raw).astype(int)
    quota = np.minimum(quota, np.asarray([len(group) for _, group in groups], dtype=int))
    remaining = max_series - int(quota.sum())
    order = np.argsort(-(raw - quota))
    for group_pos in order:
        if remaining == 0:
            break
        if quota[group_pos] < len(groups[group_pos][1]):
            quota[group_pos] += 1
            remaining -= 1

    chosen: list[int] = []
    for (_, group), take in zip(groups, quota, strict=True):
        if take <= 0:
            continue
        hashes = group["component_id"].map(
            lambda value: sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()
        )
        ranked = group.assign(_hash=hashes)
        if scores is None:
            picked = ranked.sort_values("_hash").head(int(take))
        else:
            picked = ranked.sort_values(
                ["_score", "_hash"], ascending=[False, True]
            ).head(int(take))
        chosen.extend(picked["_index"].astype(int).tolist())
    return np.asarray(sorted(chosen), dtype=np.int64)


def prevalidation_selection_scores(
    returns: np.ndarray,
    nstock: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    """Score series using only the frozen pre-validation selection window."""
    if mode not in SERIES_SELECTION_MODES:
        raise ValueError(f"mode must be one of {sorted(SERIES_SELECTION_MODES)}.")
    values = np.asarray(returns, dtype=float)
    coverage = np.asarray(nstock, dtype=float)
    if values.ndim != 2 or coverage.shape != values.shape:
        raise ValueError("returns and nstock must be equally shaped two-dimensional arrays.")
    if mode == "stratified_random":
        return np.zeros(values.shape[1], dtype=float)

    observed = np.isfinite(values)
    median_nstock = np.nanmedian(np.where(observed, coverage, np.nan), axis=0)
    squared_return_acf = _panel_acf(np.square(values), lag=1)
    squared_return_acf = np.where(np.isfinite(squared_return_acf), squared_return_acf, 0.0)

    quality_rank = pd.Series(median_nstock).rank(method="average", pct=True).fillna(0.0).to_numpy()
    low_persistence_rank = pd.Series(-squared_return_acf).rank(
        method="average", pct=True
    ).fillna(0.0).to_numpy()
    return np.asarray(quality_rank + low_persistence_rank, dtype=float)


def select_frozen_universe(
    panel: DataMinePanel,
    split: PanelSplit,
    *,
    universe_window_months: int,
    min_observed_fraction: float,
    min_median_nstock: float,
    min_nstock: float,
    min_nstock_fraction: float,
    max_missing_run: int,
    max_series: int,
    seed: int,
    selection_mode: str = "stratified_random",
) -> tuple[DataMinePanel, pd.DataFrame]:
    """Select a universe using only the trailing pre-validation period."""
    selection_end = int(np.flatnonzero(panel.dates <= split.universe_selection_end)[-1])
    selection_start = max(0, selection_end - int(universe_window_months) + 1)
    returns = np.asarray(panel.returns[selection_start : selection_end + 1], dtype=float)
    nstock = np.asarray(panel.nstock[selection_start : selection_end + 1], dtype=float)
    observed = np.isfinite(returns)
    observed_fraction = observed.mean(axis=0)
    median_nstock = np.nanmedian(np.where(observed, nstock, np.nan), axis=0)
    nstock_fraction = (observed & np.isfinite(nstock) & (nstock >= float(min_nstock))).mean(axis=0)
    longest_missing = _longest_true_run(~observed)

    eligible = (
        (observed_fraction >= float(min_observed_fraction))
        & (median_nstock >= float(min_median_nstock))
        & (nstock_fraction >= float(min_nstock_fraction))
        & (longest_missing <= int(max_missing_run))
    )
    eligible_indices = np.flatnonzero(eligible)
    if eligible_indices.size == 0:
        raise ValueError("Frozen-universe rules selected zero components.")
    selection_scores = prevalidation_selection_scores(
        returns,
        nstock,
        mode=selection_mode,
    )
    chosen = _stratified_pilot_indices(
        panel.metadata,
        eligible_indices,
        max_series=int(max_series),
        seed=int(seed),
        scores=None if selection_mode == "stratified_random" else selection_scores,
    )

    diagnostics = panel.metadata.loc[
        :, ["signalid", "component_id", "formula_type", "numerator", "denominator"]
    ].copy()
    diagnostics["selection_start"] = panel.dates[selection_start]
    diagnostics["selection_end"] = panel.dates[selection_end]
    diagnostics["observed_return_fraction"] = observed_fraction
    diagnostics["median_nstock"] = median_nstock
    diagnostics["fraction_months_nstock_ge_threshold"] = nstock_fraction
    diagnostics["longest_missing_run"] = longest_missing
    diagnostics["selection_mode"] = str(selection_mode)
    diagnostics["selection_score"] = selection_scores
    diagnostics["eligible"] = eligible
    diagnostics["selected"] = False
    diagnostics.loc[chosen, "selected"] = True

    selected_meta = panel.metadata.iloc[chosen].copy().reset_index(drop=True)
    selected_panel = DataMinePanel(
        dates=panel.dates.copy(),
        component_ids=selected_meta["component_id"].astype(str).tolist(),
        signal_ids=selected_meta["signalid"].astype(str).tolist(),
        returns=np.asarray(panel.returns[:, chosen], dtype=np.float32),
        nstock=np.asarray(panel.nstock[:, chosen], dtype=np.float32),
        metadata=selected_meta,
        weighting=panel.weighting,
        coverage_source=panel.coverage_source,
    )
    selected_panel.validate()
    return selected_panel, diagnostics


def save_panel(panel: DataMinePanel, output_dir: str | Path) -> None:
    """Save matrix artifacts without changing their component order."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "returns.npy", np.asarray(panel.returns, dtype=np.float32), allow_pickle=False)
    np.save(output / "nstock.npy", np.asarray(panel.nstock, dtype=np.float32), allow_pickle=False)
    np.save(output / "dates.npy", panel.dates.to_numpy(dtype="datetime64[ns]"), allow_pickle=False)
    panel.metadata.to_parquet(output / "strategy_metadata.parquet", index=False)
    pd.DataFrame({
        "column_index": np.arange(len(panel.component_ids), dtype=int),
        "component_id": panel.component_ids,
        "signalid": panel.signal_ids,
    }).to_csv(output / "component_order.csv", index=False)


def load_panel(output_dir: str | Path, *, mmap_mode: str | None = "r") -> DataMinePanel:
    """Load a prepared panel and enforce its saved component order."""
    source = Path(output_dir)
    order = pd.read_csv(source / "component_order.csv", dtype={"component_id": str, "signalid": str})
    metadata = pd.read_parquet(source / "strategy_metadata.parquet")
    metadata["signalid"] = metadata["signalid"].astype(str)
    metadata["component_id"] = metadata["component_id"].astype(str)
    metadata = order[["component_id"]].merge(metadata, on="component_id", how="left", validate="one_to_one")
    manifest_path = source / "panel_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    panel = DataMinePanel(
        dates=pd.DatetimeIndex(np.load(source / "dates.npy", mmap_mode=mmap_mode, allow_pickle=False)),
        component_ids=order["component_id"].astype(str).tolist(),
        signal_ids=order["signalid"].astype(str).tolist(),
        returns=np.load(source / "returns.npy", mmap_mode=mmap_mode, allow_pickle=False),
        nstock=np.load(source / "nstock.npy", mmap_mode=mmap_mode, allow_pickle=False),
        metadata=metadata,
        weighting=str(manifest.get("weighting", "unknown")),
        coverage_source=str(manifest.get("coverage_source", "nstock")),
    )
    panel.validate()
    return panel


def canonical_return_chunk(
    frame: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    weighting: str,
    return_unit: str,
) -> pd.DataFrame:
    """Convert one raw chunk to the documented long-table schema."""
    missing = RETURN_REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required return columns: {sorted(missing)}")
    coverage_columns, _ = _coverage_columns(frame.columns)
    out = pd.DataFrame()
    out["signalid"] = _canonical_signal_ids(frame["signalid"])
    out["date"] = pd.to_datetime({
        "year": pd.to_numeric(frame["year"], errors="raise").astype(int),
        "month": pd.to_numeric(frame["month"], errors="raise").astype(int),
        "day": 1,
    }) + pd.offsets.MonthEnd(0)
    out["weighting"] = str(weighting).lower()
    out["component_id"] = str(weighting).upper() + "_" + out["signalid"]
    values = pd.to_numeric(frame["ret"], errors="coerce")
    out["return_decimal"] = values / 100.0 if return_unit == "percent" else values
    out["nstock"] = _coverage_values(frame, coverage_columns)
    static = metadata.loc[
        :, ["signalid", "numerator", "denominator", "formula_type", "formula_label"]
    ]
    out = out.merge(static, on="signalid", how="left", validate="many_to_one")
    if out[["numerator", "denominator", "formula_type"]].isna().any().any():
        raise ValueError("Return rows failed to join to signal metadata.")
    return out.loc[
        :,
        [
            "date",
            "signalid",
            "weighting",
            "component_id",
            "return_decimal",
            "nstock",
            "numerator",
            "denominator",
            "formula_type",
            "formula_label",
        ],
    ]


def write_canonical_long_parquet(
    return_path: str | Path,
    metadata: pd.DataFrame,
    output_path: str | Path,
    *,
    weighting: str,
    return_unit: str,
    chunksize: int = 250_000,
) -> None:
    """Stream raw Parquet or legacy CSV input to the canonical long table."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for chunk in _iter_source_frames(return_path, batch_size=int(chunksize)):
            canonical = canonical_return_chunk(
                chunk,
                metadata,
                weighting=weighting,
                return_unit=return_unit,
            )
            table = pa.Table.from_pandas(canonical, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def make_target(
    returns: np.ndarray,
    *,
    target_type: str,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Construct the risk or raw-return target without filling missing cells."""
    if target_type not in TARGET_TYPES:
        raise ValueError(f"target_type must be one of {sorted(TARGET_TYPES)}.")
    values = np.asarray(returns, dtype=float)
    if target_type == "raw_return":
        return values.copy()
    if epsilon <= 0:
        raise ValueError("epsilon must be positive for log_squared_return.")
    return np.log(float(epsilon) + np.square(values))


def _window_design(target: np.ndarray, t: int, baseline_type: str) -> np.ndarray:
    p = target.shape[1]
    if baseline_type == "ar1_ridge":
        return np.column_stack((np.ones(p), target[t - 1]))
    if baseline_type != "har_ridge":
        raise ValueError(f"Unknown baseline_type: {baseline_type}")
    return np.column_stack((
        np.ones(p),
        target[t - 1],
        np.mean(target[t - 3 : t], axis=0),
        np.mean(target[t - 12 : t], axis=0),
    ))


def rolling_ridge_forecast(
    target: np.ndarray,
    *,
    baseline_type: str,
    ridge: float,
    minimum_observations: int,
) -> np.ndarray:
    """Generate expanding-window genuine one-step ridge forecasts.

    For target month ``t``, sufficient statistics include responses only through
    ``t - 1``. Missing component-level regression rows are skipped.
    """
    if baseline_type not in BASELINE_TYPES:
        raise ValueError(f"baseline_type must be one of {sorted(BASELINE_TYPES)}.")
    if ridge < 0:
        raise ValueError("ridge must be nonnegative.")
    values = np.asarray(target, dtype=float)
    if values.ndim != 2:
        raise ValueError("target must have shape (time, components).")
    T, p = values.shape
    minimum_lag = 12 if baseline_type == "har_ridge" else 1
    dimension = 4 if baseline_type == "har_ridge" else 2
    forecast = np.full((T, p), np.nan, dtype=float)
    sxx = np.zeros((p, dimension, dimension), dtype=float)
    sxy = np.zeros((p, dimension), dtype=float)
    count = np.zeros(p, dtype=np.int32)
    penalty = np.eye(dimension, dtype=float) * float(ridge)
    penalty[0, 0] = 0.0

    for t in range(minimum_lag + 1, T):
        regression_month = t - 1
        x_previous = _window_design(values, regression_month, baseline_type)
        y_previous = values[regression_month]
        valid_previous = np.isfinite(y_previous) & np.isfinite(x_previous).all(axis=1)
        if valid_previous.any():
            xp = x_previous[valid_previous]
            yp = y_previous[valid_previous]
            sxx[valid_previous] += xp[:, :, None] * xp[:, None, :]
            sxy[valid_previous] += xp * yp[:, None]
            count[valid_previous] += 1

        x_target = _window_design(values, t, baseline_type)
        eligible = (count >= int(minimum_observations)) & np.isfinite(x_target).all(axis=1)
        if not eligible.any():
            continue
        indices = np.flatnonzero(eligible)
        systems = sxx[indices] + penalty[None, :, :]
        rhs = sxy[indices]
        try:
            coefficients = np.linalg.solve(systems, rhs[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            coefficients = np.stack([
                np.linalg.lstsq(system, vector, rcond=None)[0]
                for system, vector in zip(systems, rhs, strict=True)
            ])
        forecast[t, indices] = np.sum(x_target[indices] * coefficients, axis=1)
    return forecast


def fixed_garch11_grid_forecast(
    returns: np.ndarray,
    *,
    estimation_end_index: int,
    alpha_candidates: Sequence[float] = (0.03, 0.05, 0.08, 0.12, 0.18, 0.25),
    beta_candidates: Sequence[float] = (0.50, 0.65, 0.75, 0.85, 0.90, 0.94),
    variance_floor: float = 1e-10,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Forecast second moments with pre-validation fitted GARCH(1,1) models.

    A Gaussian quasi-likelihood grid selects component-specific ``alpha`` and
    ``beta``. Means, unconditional variances, and parameters are estimated only
    through ``estimation_end_index`` and then remain frozen. Conditional
    variances are recursively updated with each subsequently observed return.
    """
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("returns must have shape (time, components).")
    end = int(estimation_end_index)
    if end < 24 or end >= len(values):
        raise ValueError("estimation_end_index must leave at least 24 estimation rows.")
    if variance_floor <= 0.0:
        raise ValueError("variance_floor must be positive.")

    candidates = [
        (float(alpha), float(beta))
        for alpha in alpha_candidates
        for beta in beta_candidates
        if float(alpha) >= 0.0
        and float(beta) >= 0.0
        and float(alpha) + float(beta) < 0.999
    ]
    if not candidates:
        raise ValueError("The GARCH grid contains no stationary parameter pairs.")

    estimation = values[: end + 1]
    mean = np.nanmean(estimation, axis=0)
    centered = estimation - mean[None, :]
    scale2 = np.nanmean(np.square(centered), axis=0)
    finite_scale = scale2[np.isfinite(scale2) & (scale2 > float(variance_floor))]
    fallback_scale = float(np.median(finite_scale)) if finite_scale.size else 1.0
    scale2 = np.where(
        np.isfinite(scale2) & (scale2 > float(variance_floor)),
        scale2,
        max(fallback_scale, float(variance_floor)),
    )
    standardized_squared = np.square(centered) / scale2[None, :]

    alpha_grid = np.asarray([pair[0] for pair in candidates], dtype=float)[:, None]
    beta_grid = np.asarray([pair[1] for pair in candidates], dtype=float)[:, None]
    omega_grid = 1.0 - alpha_grid - beta_grid
    conditional = np.ones((len(candidates), values.shape[1]), dtype=float)
    objective = np.zeros_like(conditional)
    counts = np.zeros_like(conditional, dtype=np.int32)
    for time_index in range(1, end + 1):
        previous = standardized_squared[time_index - 1]
        previous = np.where(np.isfinite(previous), previous, 1.0)
        conditional = omega_grid + alpha_grid * previous[None, :] + beta_grid * conditional
        conditional = np.maximum(conditional, float(variance_floor))
        current = standardized_squared[time_index]
        observed = np.isfinite(current)[None, :]
        loss = np.log(conditional) + current[None, :] / conditional
        objective += np.where(observed, loss, 0.0)
        counts += observed
    objective = np.divide(
        objective,
        counts,
        out=np.full_like(objective, np.inf),
        where=counts > 0,
    )
    best = np.argmin(objective, axis=0)
    columns = np.arange(values.shape[1])
    alpha = alpha_grid[:, 0][best]
    beta = beta_grid[:, 0][best]
    omega = 1.0 - alpha - beta

    full_centered_squared = np.square(values - mean[None, :]) / scale2[None, :]
    conditional = np.ones(values.shape[1], dtype=float)
    second_moment = np.full_like(values, np.nan, dtype=float)
    for time_index in range(1, len(values)):
        previous = full_centered_squared[time_index - 1]
        previous = np.where(np.isfinite(previous), previous, 1.0)
        conditional = omega + alpha * previous + beta * conditional
        conditional = np.maximum(conditional, float(variance_floor))
        second_moment[time_index] = np.square(mean) + scale2 * conditional

    parameters = pd.DataFrame({
        "mean_return": mean,
        "unconditional_variance": scale2,
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "gaussian_qlike": objective[best, columns],
        "estimation_observations": np.isfinite(estimation).sum(axis=0),
    })
    return second_moment, parameters


def rolling_mean_forecast(target: np.ndarray, window: int) -> np.ndarray:
    """Forecast each component with its trailing finite-window mean."""
    values = np.asarray(target, dtype=float)
    output = np.full_like(values, np.nan, dtype=float)
    for t in range(1, len(values)):
        history = values[max(0, t - int(window)) : t]
        count = np.isfinite(history).sum(axis=0)
        total = np.where(np.isfinite(history), history, 0.0).sum(axis=0)
        output[t] = np.divide(total, count, out=np.full(values.shape[1], np.nan), where=count > 0)
    return output


def rolling_shrinkage_mean_forecast(
    target: np.ndarray,
    *,
    window: int,
    weight: float,
) -> np.ndarray:
    """Shrink each trailing component mean toward the trailing panel mean."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1].")
    component = rolling_mean_forecast(target, window=window)
    output = np.full_like(component, np.nan)
    for t in range(1, len(component)):
        history = np.asarray(target[max(0, t - int(window)) : t], dtype=float)
        panel_mean = float(np.nanmean(history)) if np.isfinite(history).any() else np.nan
        output[t] = float(weight) * component[t] + (1.0 - float(weight)) * panel_mean
    return output


def select_ridge_forecast(
    target: np.ndarray,
    validation_indices: Sequence[int],
    *,
    baseline_type: str,
    ridge_candidates: Sequence[float],
    minimum_observations: int,
) -> tuple[float, np.ndarray, pd.DataFrame]:
    """Select one common ridge penalty using pooled validation squared error."""
    rows: list[dict[str, float]] = []
    forecasts: dict[float, np.ndarray] = {}
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    for candidate in ridge_candidates:
        ridge = float(candidate)
        candidate_forecast = rolling_ridge_forecast(
            target,
            baseline_type=baseline_type,
            ridge=ridge,
            minimum_observations=minimum_observations,
        )
        forecasts[ridge] = candidate_forecast
        error = candidate_forecast[validation_indices] - np.asarray(target)[validation_indices]
        finite = np.isfinite(error)
        mse = float(np.mean(np.square(error[finite]))) if finite.any() else np.inf
        rows.append({
            "baseline_type": baseline_type,
            "ridge": ridge,
            "validation_cells": int(finite.sum()),
            "validation_rmse": float(np.sqrt(mse)),
            "validation_mae": float(np.mean(np.abs(error[finite]))) if finite.any() else np.inf,
        })
    selection = pd.DataFrame(rows).sort_values(["validation_rmse", "ridge"]).reset_index(drop=True)
    selected = float(selection.iloc[0]["ridge"])
    selection["selected"] = selection["ridge"].eq(selected)
    return selected, forecasts[selected], selection


def fit_robust_panel_scaler(
    values: np.ndarray,
    *,
    clip: float | None,
    floor_quantile: float = 0.05,
) -> RobustPanelScaler:
    """Fit a componentwise median/MAD scaler on historical values only."""
    history = np.asarray(values, dtype=float)
    if history.ndim != 2 or history.shape[0] < 2:
        raise ValueError("values must have shape (time >= 2, components).")
    center = np.nanmedian(history, axis=0)
    mad = np.nanmedian(np.abs(history - center[None, :]), axis=0)
    raw_scale = 1.4826 * mad
    positive = raw_scale[np.isfinite(raw_scale) & (raw_scale > 0.0)]
    scale_floor = float(np.quantile(positive, float(floor_quantile))) if positive.size else 1.0
    scale = np.where(np.isfinite(raw_scale) & (raw_scale >= scale_floor), raw_scale, scale_floor)
    center = np.where(np.isfinite(center), center, 0.0)
    return RobustPanelScaler(
        center=np.asarray(center, dtype=float),
        scale=np.asarray(scale, dtype=float),
        scale_floor=scale_floor,
        clip=None if clip is None else float(clip),
    )


def estimate_global_lambda(
    baseline_error: np.ndarray,
    correction: np.ndarray,
    *,
    ridge: float = 0.0,
) -> dict[str, float]:
    """Estimate a pooled correction coefficient by LSE and clip to [0, 1]."""
    error = np.asarray(baseline_error, dtype=float)
    adjustment = np.asarray(correction, dtype=float)
    if error.shape != adjustment.shape:
        raise ValueError("baseline_error and correction must have the same shape.")
    valid = np.isfinite(error) & np.isfinite(adjustment)
    numerator = float(np.sum(error[valid] * adjustment[valid]))
    denominator = float(np.sum(np.square(adjustment[valid])) + float(ridge))
    raw = numerator / denominator if denominator > 0.0 else 0.0
    return {
        "lambda": float(np.clip(raw, 0.0, 1.0)),
        "lambda_unbounded": float(raw),
        "numerator": numerator,
        "denominator": denominator,
        "ridge": float(ridge),
        "validation_cells": int(valid.sum()),
    }


def estimate_group_lambdas(
    baseline_error: np.ndarray,
    correction: np.ndarray,
    metadata: pd.DataFrame,
    *,
    group_cols: Sequence[str] = (),
    ridge: float = 0.0,
    min_group_series: int = 1,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    """Estimate bounded pooled lambdas by metadata group with global fallback."""
    error = np.asarray(baseline_error, dtype=float)
    adjustment = np.asarray(correction, dtype=float)
    if error.shape != adjustment.shape or error.ndim != 2:
        raise ValueError("baseline_error and correction must have the same 2D shape.")
    metadata_frame = metadata.reset_index(drop=True)
    if len(metadata_frame) != error.shape[1]:
        raise ValueError("metadata rows must match the panel component dimension.")
    columns = [str(column) for column in group_cols]
    missing = sorted(set(columns).difference(metadata_frame.columns))
    if missing:
        raise ValueError(f"Unknown lambda group columns: {missing}")
    if int(min_group_series) < 1:
        raise ValueError("min_group_series must be positive.")

    global_info = estimate_global_lambda(error, adjustment, ridge=ridge)
    lambda_values = np.full(error.shape[1], global_info["lambda"], dtype=float)
    if not columns:
        table = pd.DataFrame([{
            "group": "__global__",
            "series": int(error.shape[1]),
            "fallback_global": False,
            **global_info,
        }])
        return lambda_values, table, global_info

    group_key = columns[0] if len(columns) == 1 else columns
    rows: list[dict[str, Any]] = []
    grouped = metadata_frame.groupby(group_key, sort=True, dropna=False, observed=True)
    for raw_key, indices in grouped.indices.items():
        key = raw_key if isinstance(raw_key, tuple) else (raw_key,)
        component_indices = np.asarray(indices, dtype=np.int64)
        fallback = len(component_indices) < int(min_group_series)
        info = (
            global_info
            if fallback
            else estimate_global_lambda(
                error[:, component_indices],
                adjustment[:, component_indices],
                ridge=ridge,
            )
        )
        lambda_values[component_indices] = info["lambda"]
        row: dict[str, Any] = {
            column: value for column, value in zip(columns, key, strict=True)
        }
        row.update({
            "group": " | ".join(str(value) for value in key),
            "series": int(len(component_indices)),
            "fallback_global": bool(fallback),
            **info,
        })
        rows.append(row)
    return lambda_values, pd.DataFrame(rows), global_info


def rolling_group_lambdas(
    baseline_error: np.ndarray,
    correction: np.ndarray,
    metadata: pd.DataFrame,
    evaluation_indices: Sequence[int],
    *,
    window_months: int,
    min_history_months: int,
    group_cols: Sequence[str] = (),
    ridge: float = 0.0,
    min_group_series: int = 1,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Estimate leakage-safe lambdas from prior rolling forecast errors only."""
    error = np.asarray(baseline_error, dtype=float)
    adjustment = np.asarray(correction, dtype=float)
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    if error.shape != adjustment.shape or error.ndim != 2:
        raise ValueError("baseline_error and correction must have the same 2D shape.")
    if evaluation.size == 0 or np.any(np.diff(evaluation) <= 0):
        raise ValueError("evaluation_indices must be a nonempty increasing sequence.")
    if int(window_months) < 1:
        raise ValueError("window_months must be positive.")
    if int(min_history_months) < 1 or int(min_history_months) > int(window_months):
        raise ValueError("min_history_months must be in [1, window_months].")
    metadata_frame = metadata.reset_index(drop=True)
    if len(metadata_frame) != error.shape[1]:
        raise ValueError("metadata rows must match the panel component dimension.")
    columns = [str(column) for column in group_cols]
    missing = sorted(set(columns).difference(metadata_frame.columns))
    if missing:
        raise ValueError(f"Unknown lambda group columns: {missing}")
    if int(min_group_series) < 1:
        raise ValueError("min_group_series must be positive.")

    values = np.full(error.shape, np.nan, dtype=float)
    rows: list[pd.DataFrame] = []
    for position, target_index in enumerate(evaluation):
        prior = evaluation[max(0, position - int(window_months)):position]
        if len(prior) < int(min_history_months):
            values[int(target_index)] = 0.0
            continue
        target_lambdas, table, _ = estimate_group_lambdas(
            error[prior],
            adjustment[prior],
            metadata,
            group_cols=group_cols,
            ridge=ridge,
            min_group_series=min_group_series,
        )
        values[int(target_index)] = target_lambdas
        table.insert(0, "target_index", int(target_index))
        table.insert(1, "lambda_training_start_index", int(prior[0]))
        table.insert(2, "lambda_training_end_index", int(prior[-1]))
        rows.append(table)
    return values, pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def rolling_refit_group_lambdas(
    baseline_error: np.ndarray,
    correction: np.ndarray,
    metadata: pd.DataFrame,
    evaluation_indices: Sequence[int],
    refit_index: Sequence[int],
    *,
    window_months: int,
    min_history_months: int,
    group_cols: Sequence[str] = (),
    ridge: float = 0.0,
    min_group_series: int = 1,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Estimate one leakage-safe LSE lambda at the start of every refit block.

    The coefficient for a block uses only genuine forecast errors and VARA
    corrections from evaluation origins strictly before that block.  It is
    then held fixed until the next VARA refit.  This makes the lambda schedule
    coincide with the rolling model windows without allowing a target outcome
    to tune its own forecast.
    """
    error = np.asarray(baseline_error, dtype=float)
    adjustment = np.asarray(correction, dtype=float)
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    refits = np.asarray(refit_index, dtype=np.int64)
    if error.shape != adjustment.shape or error.ndim != 2:
        raise ValueError("baseline_error and correction must have the same 2D shape.")
    if evaluation.size == 0 or np.any(np.diff(evaluation) <= 0):
        raise ValueError("evaluation_indices must be a nonempty increasing sequence.")
    if refits.ndim != 1 or len(refits) != len(error):
        raise ValueError("refit_index must have one entry per panel row.")
    evaluation_refits = refits[evaluation]
    if np.any(evaluation_refits < 0) or np.any(np.diff(evaluation_refits) < 0):
        raise ValueError("Evaluation refit indices must be nonnegative and nondecreasing.")
    if int(window_months) < 1:
        raise ValueError("window_months must be positive.")
    if int(min_history_months) < 1 or int(min_history_months) > int(window_months):
        raise ValueError("min_history_months must be in [1, window_months].")
    metadata_frame = metadata.reset_index(drop=True)
    if len(metadata_frame) != error.shape[1]:
        raise ValueError("metadata rows must match the panel component dimension.")
    columns = [str(column) for column in group_cols]
    missing = sorted(set(columns).difference(metadata_frame.columns))
    if missing:
        raise ValueError(f"Unknown lambda group columns: {missing}")
    if int(min_group_series) < 1:
        raise ValueError("min_group_series must be positive.")

    values = np.full(error.shape, np.nan, dtype=float)
    rows: list[pd.DataFrame] = []
    current_lambdas: np.ndarray | None = None
    current_refit: int | None = None
    for position, (target_index, target_refit) in enumerate(
        zip(evaluation, evaluation_refits, strict=True)
    ):
        block_start = current_refit is None or int(target_refit) != current_refit
        if block_start:
            current_refit = int(target_refit)
            prior = evaluation[max(0, position - int(window_months)):position]
            sufficient = len(prior) >= int(min_history_months)
            if sufficient:
                lambda_error = error[prior]
                lambda_adjustment = adjustment[prior]
            else:
                lambda_error = error[:0]
                lambda_adjustment = adjustment[:0]
            current_lambdas, table, _ = estimate_group_lambdas(
                lambda_error,
                lambda_adjustment,
                metadata_frame,
                group_cols=columns,
                ridge=ridge,
                min_group_series=min_group_series,
            )
            table = table.rename(
                columns={"validation_cells": "lambda_training_cells"}
            )
            table.insert(0, "lambda_refit_target_index", int(target_index))
            table.insert(1, "refit_index", current_refit)
            table.insert(
                2,
                "lambda_training_start_index",
                int(prior[0]) if len(prior) else -1,
            )
            table.insert(
                3,
                "lambda_training_end_index",
                int(prior[-1]) if len(prior) else -1,
            )
            table.insert(4, "lambda_training_months", int(len(prior)))
            table.insert(5, "lambda_history_sufficient", bool(sufficient))
            rows.append(table)
        assert current_lambdas is not None
        values[int(target_index)] = current_lambdas
    return values, pd.concat(rows, ignore_index=True)


def estimate_series_lambdas(
    baseline_error: np.ndarray,
    correction: np.ndarray,
    *,
    ridge: float = 0.0,
    prior_strength_months: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Estimate series lambdas with an empirical-Bayes penalty toward global."""
    error = np.asarray(baseline_error, dtype=float)
    adjustment = np.asarray(correction, dtype=float)
    if error.shape != adjustment.shape or error.ndim != 2:
        raise ValueError("baseline_error and correction must have the same 2D shape.")
    if float(prior_strength_months) < 0.0:
        raise ValueError("prior_strength_months must be nonnegative.")
    global_info = estimate_global_lambda(error, adjustment, ridge=ridge)
    valid = np.isfinite(error) & np.isfinite(adjustment)
    numerator = np.sum(np.where(valid, error * adjustment, 0.0), axis=0)
    denominator = np.sum(np.where(valid, np.square(adjustment), 0.0), axis=0)
    pooled_cells = max(int(valid.sum()), 1)
    correction_energy_per_cell = float(np.sum(denominator) / pooled_cells)
    prior_penalty = float(prior_strength_months) * correction_energy_per_cell
    raw = np.divide(
        numerator + prior_penalty * global_info["lambda"],
        denominator + prior_penalty + float(ridge),
        out=np.full(error.shape[1], global_info["lambda"], dtype=float),
        where=(denominator + prior_penalty + float(ridge)) > 0.0,
    )
    lambdas = np.clip(raw, 0.0, 1.0)
    return lambdas, {
        **global_info,
        "prior_strength_months": float(prior_strength_months),
        "correction_energy_per_cell": correction_energy_per_cell,
        "prior_penalty": prior_penalty,
        "series_lambda_mean": float(np.mean(lambdas)),
        "series_lambda_median": float(np.median(lambdas)),
        "series_lambda_min": float(np.min(lambdas)),
        "series_lambda_max": float(np.max(lambdas)),
        "zero_lambda_series_fraction": float(np.mean(lambdas == 0.0)),
    }


def formula_feature_blocks(
    metadata: pd.DataFrame,
    *,
    svd_dim: int,
    svd_drop_components: int = 0,
    encoding: str = "svd",
    role_embedding_dim: int = 32,
    seed: int,
) -> tuple[dict[str, tuple[np.ndarray, list[str]]], dict[str, Any]]:
    """Construct leakage-free formula features from static strategy metadata."""
    from scipy import sparse
    from sklearn.decomposition import TruncatedSVD

    required = {"numerator", "denominator", "formula_type"}
    missing = required.difference(metadata.columns)
    if missing:
        raise ValueError(f"Metadata are missing formula columns: {sorted(missing)}")
    n = len(metadata)
    categories: dict[str, list[str]] = {
        field: sorted(metadata[field].astype(str).unique().tolist())
        for field in ("numerator", "denominator", "formula_type")
    }
    offsets: dict[str, int] = {}
    width = 0
    for field in ("numerator", "denominator", "formula_type"):
        offsets[field] = width
        width += len(categories[field])
    rows: list[int] = []
    columns: list[int] = []
    incidence_names: list[str] = []
    for field in ("numerator", "denominator", "formula_type"):
        mapping = {value: idx for idx, value in enumerate(categories[field])}
        rows.extend(range(n))
        columns.extend(offsets[field] + mapping[value] for value in metadata[field].astype(str))
        incidence_names.extend(f"{field}={value}" for value in categories[field])
    incidence = sparse.csr_matrix(
        (np.ones(len(rows), dtype=float), (rows, columns)),
        shape=(n, width),
    )
    formula_types = categories["formula_type"]
    type_values = np.column_stack([
        metadata["formula_type"].astype(str).eq(value).to_numpy(dtype=float)
        for value in formula_types
    ])
    blocks: dict[str, tuple[np.ndarray, list[str]]] = {
        "formula_type": (
            type_values,
            [f"formula_type={value}" for value in formula_types],
        ),
    }
    provenance = {
        "encoding": str(encoding),
        "incidence_columns": incidence_names,
        "incidence_shape": list(incidence.shape),
        "seed": int(seed),
    }
    if encoding == "svd":
        maximum_dim = max(1, min(n - 1, width - 1))
        dropped_components = int(svd_drop_components)
        if dropped_components < 0:
            raise ValueError("svd_drop_components must be nonnegative.")
        if dropped_components >= maximum_dim:
            raise ValueError(
                "svd_drop_components must leave at least one formula SVD component."
            )
        requested_dimension = max(1, int(svd_dim))
        retained_dimension = min(requested_dimension, maximum_dim - dropped_components)
        fitted_dimension = retained_dimension + dropped_components
        svd = TruncatedSVD(n_components=fitted_dimension, random_state=int(seed))
        fitted_embedding = svd.fit_transform(incidence)
        embedding = fitted_embedding[:, dropped_components:]
        blocks = {
            "formula_svd": (
                np.asarray(embedding, dtype=float),
                [
                    f"formula_svd_{idx + 1:02d}"
                    for idx in range(dropped_components, fitted_dimension)
                ],
            ),
            **blocks,
        }
        provenance.update({
            "svd_requested_dimension": requested_dimension,
            "svd_dimension": retained_dimension,
            "svd_fitted_dimension": fitted_dimension,
            "svd_dropped_components": dropped_components,
            "svd_explained_variance_ratio": (
                svd.explained_variance_ratio_[dropped_components:].tolist()
            ),
            "svd_singular_values": svd.singular_values_[dropped_components:].tolist(),
            "svd_fitted_explained_variance_ratio": svd.explained_variance_ratio_.tolist(),
            "svd_fitted_singular_values": svd.singular_values_.tolist(),
        })
    elif encoding == "role_embedding":
        dimension = int(role_embedding_dim)
        if dimension < 2:
            raise ValueError("role_embedding_dim must be at least 2.")
        role_details: dict[str, Any] = {}
        role_blocks: dict[str, tuple[np.ndarray, list[str]]] = {}
        for role_index, field in enumerate(("numerator", "denominator")):
            role_seed = np.random.SeedSequence([int(seed), role_index + 1])
            rng = np.random.default_rng(role_seed)
            codes = rng.standard_normal((len(categories[field]), dimension))
            codes /= np.linalg.norm(codes, axis=1, keepdims=True)
            mapping = {value: idx for idx, value in enumerate(categories[field])}
            row_codes = np.asarray([
                mapping[value] for value in metadata[field].astype(str)
            ], dtype=np.int64)
            block_name = f"formula_{field}_embedding"
            role_blocks[block_name] = (
                codes[row_codes],
                [f"{block_name}_{idx + 1:02d}" for idx in range(dimension)],
            )
            role_details[field] = {
                "categories": categories[field],
                "category_count": len(categories[field]),
                "dimension": dimension,
                "seed_entropy": [int(seed), role_index + 1],
            }
        blocks = {**role_blocks, **blocks}
        provenance["role_embeddings"] = role_details
    else:
        raise ValueError("encoding must be 'svd' or 'role_embedding'.")
    return blocks, provenance


def economic_category_block(
    metadata: pd.DataFrame,
    ingredient_dictionary: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Map a reviewed ingredient dictionary to separate numerator/denominator features."""
    if "ingredient" not in ingredient_dictionary.columns:
        raise ValueError("Ingredient dictionary must contain an 'ingredient' column.")
    dictionary = ingredient_dictionary.copy()
    dictionary["ingredient"] = dictionary["ingredient"].astype(str)
    if dictionary["ingredient"].duplicated().any():
        raise ValueError("Ingredient dictionary must have one row per ingredient.")
    category_columns = [
        column for column in dictionary.columns
        if column != "ingredient" and not column.endswith("_description")
    ]
    if not category_columns:
        raise ValueError("Ingredient dictionary contains no category columns.")
    lookup = dictionary.set_index("ingredient")
    feature_columns: list[np.ndarray] = []
    feature_names: list[str] = []
    for role, ingredient_column in (("num", "numerator"), ("den", "denominator")):
        ingredients = metadata[ingredient_column].astype(str)
        for category_column in category_columns:
            values = ingredients.map(lookup[category_column]).fillna("unknown").astype(str)
            for category in sorted(values.unique()):
                feature_columns.append(values.eq(category).to_numpy(dtype=float))
                feature_names.append(f"{role}_{category_column}={category}")
    return np.column_stack(feature_columns), feature_names


def _nan_mad(values: np.ndarray, axis: int = 0) -> np.ndarray:
    center = np.nanmedian(values, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(values - center), axis=axis)


def _suppress_empty_feature_warnings(function):
    """Silence NumPy warnings for deliberately retained all-missing feature windows."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            return function(*args, **kwargs)
    return wrapped


def _finite_condition_fraction(values: np.ndarray, condition: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    count = finite.sum(axis=0)
    numerator = (finite & condition).sum(axis=0)
    return np.divide(
        numerator,
        count,
        out=np.full(values.shape[1], np.nan),
        where=count > 0,
    )


def _panel_acf(values: np.ndarray, lag: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.shape[0] <= int(lag):
        return np.full(values.shape[1], np.nan)
    left = values[:-lag]
    right = values[lag:]
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(axis=0)
    left_mean = np.divide(
        np.where(valid, left, 0.0).sum(axis=0),
        count,
        out=np.zeros(values.shape[1]),
        where=count > 0,
    )
    right_mean = np.divide(
        np.where(valid, right, 0.0).sum(axis=0),
        count,
        out=np.zeros(values.shape[1]),
        where=count > 0,
    )
    left_centered = np.where(valid, left - left_mean, 0.0)
    right_centered = np.where(valid, right - right_mean, 0.0)
    numerator = np.sum(left_centered * right_centered, axis=0)
    denominator = np.sqrt(
        np.sum(np.square(left_centered), axis=0)
        * np.sum(np.square(right_centered), axis=0)
    )
    return np.divide(
        numerator,
        denominator,
        out=np.full(values.shape[1], np.nan),
        where=(count >= 3) & (denominator > 1e-12),
    )


def _finite_window(values: np.ndarray, end: int, window: int) -> np.ndarray:
    start = max(0, int(end) - int(window) + 1)
    return np.asarray(values[start : int(end) + 1], dtype=float)


def _window_moments(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.nanmean(values, axis=0)
    centered = values - mean[None, :]
    variance = np.nanmean(np.square(centered), axis=0)
    scale = np.sqrt(variance)
    skew = np.divide(
        np.nanmean(centered**3, axis=0),
        np.maximum(scale, 1e-12) ** 3,
        out=np.full(values.shape[1], np.nan),
        where=scale > 1e-12,
    )
    kurtosis = np.divide(
        np.nanmean(centered**4, axis=0),
        np.maximum(variance, 1e-12) ** 2,
        out=np.full(values.shape[1], np.nan),
        where=variance > 1e-12,
    ) - 3.0
    return skew, kurtosis


def _factor_loading_features(
    values: np.ndarray,
    *,
    n_components: int,
    seed: int,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    from sklearn.decomposition import PCA

    panel = np.asarray(values, dtype=float)
    p = panel.shape[1]
    maximum = min(panel.shape[0] - 1, p - 1, int(n_components))
    if maximum < 1:
        return np.empty((p, 0)), [], {"dimension": 0}
    center = np.nanmedian(panel, axis=0)
    scale = 1.4826 * _nan_mad(panel, axis=0)
    positive = scale[np.isfinite(scale) & (scale > 0)]
    floor = float(np.quantile(positive, 0.05)) if positive.size else 1.0
    scale = np.where(np.isfinite(scale) & (scale >= floor), scale, floor)
    standardized = (panel - center[None, :]) / scale[None, :]
    standardized = np.where(np.isfinite(standardized), standardized, 0.0)
    pca = PCA(n_components=maximum, svd_solver="randomized", random_state=int(seed))
    scores = pca.fit_transform(standardized)
    loadings = pca.components_.T
    contribution = np.sum(
        pca.explained_variance_[None, :] * np.square(loadings),
        axis=1,
    )
    total_variance = np.var(standardized, axis=0, ddof=1)
    idiosyncratic_share = 1.0 - np.divide(
        contribution,
        total_variance,
        out=np.zeros(p),
        where=total_variance > 1e-12,
    )
    panel_mean = np.mean(standardized, axis=1)
    panel_mean_corr = _correlation_with_vector(standardized, panel_mean)
    matrix = np.column_stack((loadings, panel_mean_corr, idiosyncratic_share))
    names = [f"risk_pc_loading_{idx + 1:02d}" for idx in range(maximum)]
    names.extend(("correlation_with_panel_mean", "idiosyncratic_variance_share"))
    return matrix, names, {
        "dimension": maximum,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "scale_floor": floor,
        "score_shape": list(scores.shape),
    }


def _correlation_with_vector(values: np.ndarray, vector: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    reference = np.asarray(vector, dtype=float)
    valid = np.isfinite(matrix) & np.isfinite(reference[:, None])
    count = valid.sum(axis=0)
    matrix_mean = np.divide(
        np.where(valid, matrix, 0.0).sum(axis=0),
        count,
        out=np.zeros(matrix.shape[1]),
        where=count > 0,
    )
    reference_mean = np.divide(
        np.where(valid, reference[:, None], 0.0).sum(axis=0),
        count,
        out=np.zeros(matrix.shape[1]),
        where=count > 0,
    )
    x = np.where(valid, matrix - matrix_mean[None, :], 0.0)
    y = np.where(valid, reference[:, None] - reference_mean[None, :], 0.0)
    denom = np.sqrt(np.sum(x * x, axis=0) * np.sum(y * y, axis=0))
    return np.divide(
        np.sum(x * y, axis=0),
        denom,
        out=np.full(matrix.shape[1], np.nan),
        where=(count >= 3) & (denom > 1e-12),
    )


def _predictive_error_factor_features(
    errors: np.ndarray,
    *,
    n_components: int,
    seed: int,
    ridge: float = 1e-3,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    from sklearn.decomposition import PCA

    panel = np.asarray(errors, dtype=float)
    p = panel.shape[1]
    maximum = min(panel.shape[0] - 2, p - 1, int(n_components))
    if maximum < 1:
        return np.empty((p, 0)), [], {"dimension": 0}
    scaler = fit_robust_panel_scaler(panel, clip=8.0)
    standardized = scaler.transform(panel)
    standardized = np.where(np.isfinite(standardized), standardized, 0.0)
    pca = PCA(n_components=maximum, svd_solver="randomized", random_state=int(seed))
    factors = pca.fit_transform(standardized)
    x = factors[:-1]
    y = standardized[1:]
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    system = x.T @ x + float(ridge) * np.eye(maximum)
    coefficients = np.linalg.solve(system, x.T @ y).T
    names = [f"lagged_error_factor_response_{idx + 1:02d}" for idx in range(maximum)]
    return coefficients, names, {
        "dimension": maximum,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "ridge": float(ridge),
        "response_scale_floor": scaler.scale_floor,
    }


@_suppress_empty_feature_warnings
def dynamic_feature_blocks(
    *,
    returns: np.ndarray,
    target: np.ndarray,
    nstock: np.ndarray,
    baseline_errors: np.ndarray,
    origin_index: int,
    feature_window_months: int,
    risk_factor_dim: int,
    predictive_factor_dim: int,
    seed: int,
    coverage_threshold: float,
) -> tuple[dict[str, tuple[np.ndarray, list[str]]], dict[str, Any]]:
    """Build origin-specific return, coverage, and genuine-error feature blocks."""
    origin = int(origin_index)
    if origin < 12:
        raise ValueError("At least 12 months are required for dynamic features.")
    raw = _finite_window(returns, origin, feature_window_months)
    target_window = _finite_window(target, origin, feature_window_months)
    coverage = _finite_window(nstock, origin, feature_window_months)
    errors = _finite_window(baseline_errors, origin, feature_window_months)
    p = raw.shape[1]

    dynamics: list[np.ndarray] = []
    dynamic_names: list[str] = []
    for window in (12, 36, 60, 120):
        sample = raw[-min(window, len(raw)) :]
        dynamics.extend((
            np.nanmean(sample, axis=0),
            np.nanmedian(sample, axis=0),
            _nan_mad(sample, axis=0),
            np.nanstd(sample, axis=0),
        ))
        dynamic_names.extend((
            f"mean_return_{window}",
            f"median_return_{window}",
            f"mad_return_{window}",
            f"sd_return_{window}",
        ))
    for window in (12, 36, 60):
        sample = raw[-min(window, len(raw)) :]
        dynamics.extend((
            np.nanmean(np.abs(sample), axis=0),
            np.nanmean(np.square(sample), axis=0),
        ))
        dynamic_names.extend((f"mean_abs_return_{window}", f"mean_squared_return_{window}"))
    short = raw[-min(12, len(raw)) :]
    long = raw[-min(60, len(raw)) :]
    long_mad = _nan_mad(long, axis=0)
    dynamics.append((np.nanmean(short, axis=0) - np.nanmean(long, axis=0)) / (long_mad + 1e-8))
    dynamics.append(np.log((_nan_mad(short, axis=0) + 1e-8) / (long_mad + 1e-8)))
    dynamic_names.extend(("mean_shift_12_60", "mad_shift_12_60"))

    tail = raw[-min(120, len(raw)) :]
    skew, kurtosis = _window_moments(tail)
    negative = np.where(tail < 0.0, tail, np.nan)
    positive = np.where(tail > 0.0, tail, np.nan)
    q05, q25, q75, q95 = np.nanquantile(tail, (0.05, 0.25, 0.75, 0.95), axis=0)
    dynamics.extend((
        _finite_condition_fraction(tail, tail > 0.0),
        np.sqrt(np.nanmean(np.square(negative), axis=0)),
        np.sqrt(np.nanmean(np.square(positive), axis=0)),
        skew,
        kurtosis,
        q05,
        q25,
        q75,
        q95,
        np.abs(q05) - np.abs(q95),
        np.nanmin(tail, axis=0),
        np.nanmax(tail, axis=0),
    ))
    dynamic_names.extend((
        "positive_return_fraction",
        "downside_semideviation",
        "upside_semideviation",
        "return_skewness",
        "return_excess_kurtosis",
        "return_q05",
        "return_q25",
        "return_q75",
        "return_q95",
        "tail_asymmetry",
        "maximum_one_month_loss",
        "maximum_one_month_gain",
    ))
    for lag in (1, 3, 6, 12):
        dynamics.extend((
            _panel_acf(raw, lag),
            _panel_acf(np.abs(raw), lag),
            _panel_acf(np.square(raw), lag),
        ))
        dynamic_names.extend((
            f"acf_return_lag{lag}",
            f"acf_abs_return_lag{lag}",
            f"acf_squared_return_lag{lag}",
        ))
    dynamic_matrix = np.column_stack(dynamics)

    observed = np.isfinite(raw)
    first_seen = np.argmax(np.isfinite(np.asarray(returns[: origin + 1], dtype=float)), axis=0)
    ever_seen = np.isfinite(np.asarray(returns[: origin + 1], dtype=float)).any(axis=0)
    age = np.where(ever_seen, origin - first_seen + 1, 0)
    recent_nstock = np.full(p, np.nan)
    for row in range(len(coverage) - 1, -1, -1):
        fill = ~np.isfinite(recent_nstock) & np.isfinite(coverage[row])
        recent_nstock[fill] = coverage[row, fill]
    coverage_matrix = np.column_stack((
        np.nanmedian(coverage, axis=0),
        np.nanmin(coverage, axis=0),
        recent_nstock,
        np.nanmean(coverage >= float(coverage_threshold), axis=0),
        observed.mean(axis=0),
        age,
        _longest_true_run(~observed),
        np.log1p(np.nanmedian(coverage, axis=0)),
    ))
    coverage_names = [
        "median_nstock",
        "min_nstock",
        "recent_nstock",
        f"fraction_months_nstock_ge_{float(coverage_threshold):g}",
        "return_observed_fraction",
        "strategy_age_months",
        "longest_missing_run",
        "log1p_median_nstock",
    ]

    error_features: list[np.ndarray] = [
        np.nanmedian(errors, axis=0),
        _nan_mad(errors, axis=0),
    ]
    error_names = ["error_median", "error_mad"]
    for window in (12, 36, 60):
        sample = errors[-min(window, len(errors)) :]
        error_features.extend((np.nanmean(sample, axis=0), _nan_mad(sample, axis=0)))
        error_names.extend((f"error_mean_{window}", f"error_mad_{window}"))
    error_long_mad = _nan_mad(errors[-min(60, len(errors)) :], axis=0)
    error_features.append(
        (np.nanmean(errors[-min(12, len(errors)) :], axis=0) - np.nanmean(errors[-min(60, len(errors)) :], axis=0))
        / (error_long_mad + 1e-8)
    )
    error_features.append(
        np.log((_nan_mad(errors[-min(12, len(errors)) :], axis=0) + 1e-8) / (error_long_mad + 1e-8))
    )
    error_names.extend(("error_mean_shift_12_60", "error_scale_shift_12_60"))
    for lag in (1, 3, 6, 12):
        error_features.append(_panel_acf(errors, lag))
        error_names.append(f"error_acf_lag{lag}")
    error_features.extend((
        _finite_condition_fraction(errors, errors > 0.0),
        _finite_condition_fraction(errors, errors < 0.0),
    ))
    error_names.extend(("error_positive_fraction", "error_negative_fraction"))
    error_matrix = np.column_stack(error_features)

    factor_matrix, factor_names, factor_provenance = _factor_loading_features(
        target_window,
        n_components=int(risk_factor_dim),
        seed=int(seed),
    )
    predictive_matrix, predictive_names, predictive_provenance = _predictive_error_factor_features(
        errors,
        n_components=int(predictive_factor_dim),
        seed=int(seed),
    )
    blocks: dict[str, tuple[np.ndarray, list[str]]] = {
        "return_risk_dynamics": (dynamic_matrix, dynamic_names),
        "coverage": (coverage_matrix, coverage_names),
        "baseline_error_dynamics": (error_matrix, error_names),
    }
    if factor_names:
        blocks["risk_factor_loadings"] = (factor_matrix, factor_names)
    if predictive_names:
        blocks["predictive_error_response"] = (predictive_matrix, predictive_names)
    return blocks, {
        "origin_index": origin,
        "feature_window_months": int(feature_window_months),
        "history_start_index": max(0, origin - int(feature_window_months) + 1),
        "risk_factor": factor_provenance,
        "predictive_error_factor": predictive_provenance,
    }


def preprocess_feature_blocks(
    blocks: Mapping[str, tuple[np.ndarray, Sequence[str]]],
    *,
    block_weights: Mapping[str, float] | None = None,
    winsor_quantiles: tuple[float, float] = (0.01, 0.99),
    row_normalize: bool = True,
) -> tuple[np.ndarray, list[str], list[str], dict[str, Any]]:
    """Clean, robustly scale, balance, and optionally row-normalize features."""
    if not blocks:
        raise ValueError("At least one feature block is required.")
    weights = dict(block_weights or {})
    processed: list[np.ndarray] = []
    feature_names: list[str] = []
    block_names: list[str] = []
    provenance: dict[str, Any] = {"blocks": {}, "row_normalized": bool(row_normalize)}
    expected_rows: int | None = None
    lower_q, upper_q = winsor_quantiles

    for block_name, (raw_values, raw_names) in blocks.items():
        values = np.asarray(raw_values, dtype=float)
        names = [str(name) for name in raw_names]
        if values.ndim != 2 or values.shape[1] != len(names):
            raise ValueError(f"Feature block {block_name!r} has inconsistent shape/names.")
        if expected_rows is None:
            expected_rows = values.shape[0]
        elif values.shape[0] != expected_rows:
            raise ValueError("All feature blocks must have the same number of rows.")

        columns: list[np.ndarray] = []
        kept_names: list[str] = []
        column_info: list[dict[str, Any]] = []
        for column_index, name in enumerate(names):
            column = values[:, column_index].copy()
            finite = np.isfinite(column)
            missing_fraction = float(1.0 - finite.mean())
            if finite.sum() < 2 or missing_fraction > 0.95:
                continue
            finite_values = column[finite]
            lower = float(np.quantile(finite_values, lower_q))
            upper = float(np.quantile(finite_values, upper_q))
            column[finite] = np.clip(finite_values, lower, upper)
            median = float(np.median(column[finite]))
            mad_scale = float(1.4826 * np.median(np.abs(column[finite] - median)))
            fallback = "mad"
            scale = mad_scale
            if not np.isfinite(scale) or scale <= 1e-12:
                scale = float(np.std(column[finite], ddof=0))
                fallback = "std"
            if not np.isfinite(scale) or scale <= 1e-12:
                continue
            standardized = (column - median) / scale
            standardized[~finite] = 0.0
            columns.append(standardized)
            kept_names.append(name)
            column_info.append({
                "name": name,
                "missing_fraction": missing_fraction,
                "winsor_lower": lower,
                "winsor_upper": upper,
                "center": median,
                "scale": scale,
                "scale_method": fallback,
            })
            if missing_fraction > 0.0:
                indicator = (~finite).astype(float)
                indicator_center = float(indicator.mean())
                indicator_scale = float(indicator.std(ddof=0))
                if indicator_scale > 1e-12:
                    columns.append((indicator - indicator_center) / indicator_scale)
                    kept_names.append(f"{name}__missing")
                    column_info.append({
                        "name": f"{name}__missing",
                        "missing_fraction": 0.0,
                        "center": indicator_center,
                        "scale": indicator_scale,
                        "scale_method": "std",
                    })
        if not columns:
            provenance["blocks"][block_name] = {"dropped": True, "reason": "no usable columns"}
            continue
        matrix = np.column_stack(columns)
        block_weight = float(weights.get(block_name, 1.0))
        multiplier = block_weight / np.sqrt(matrix.shape[1])
        matrix *= multiplier
        processed.append(matrix)
        feature_names.extend(kept_names)
        block_names.extend([block_name] * matrix.shape[1])
        provenance["blocks"][block_name] = {
            "input_columns": len(names),
            "output_columns": matrix.shape[1],
            "weight": block_weight,
            "balance_multiplier": multiplier,
            "columns": column_info,
        }

    if not processed:
        raise ValueError("Feature preprocessing removed every column.")
    combined = np.column_stack(processed)
    row_norms = np.linalg.norm(combined, axis=1)
    if row_normalize:
        safe_norms = np.where(row_norms > 1e-12, row_norms, 1.0)
        combined = combined / safe_norms[:, None]
    provenance["feature_count"] = combined.shape[1]
    provenance["row_norm_before_min"] = float(row_norms.min())
    provenance["row_norm_before_median"] = float(np.median(row_norms))
    provenance["row_norm_before_max"] = float(row_norms.max())
    provenance["zero_row_count"] = int((row_norms <= 1e-12).sum())
    return combined, feature_names, block_names, provenance


def variance_from_log_risk(
    forecast: np.ndarray,
    *,
    epsilon: float,
    variance_floor: float,
) -> np.ndarray:
    values = np.asarray(forecast, dtype=float)
    variance = np.exp(np.clip(values, -50.0, 50.0)) - float(epsilon)
    return np.maximum(float(variance_floor), variance)


def fit_variance_calibration(
    actual_return: np.ndarray,
    log_variance_forecast: np.ndarray,
    *,
    epsilon: float,
    variance_floor: float,
) -> float:
    """Fit the positive multiplicative QLIKE calibration on validation cells."""
    realized = np.square(np.asarray(actual_return, dtype=float))
    variance = variance_from_log_risk(
        log_variance_forecast,
        epsilon=epsilon,
        variance_floor=variance_floor,
    )
    valid = np.isfinite(realized) & np.isfinite(variance) & (variance > 0.0)
    if not valid.any():
        return 1.0
    factor = float(np.mean(realized[valid] / variance[valid]))
    return max(factor, 1e-8)


def _cross_sectional_correlations(
    actual: np.ndarray,
    forecast: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.stats import rankdata

    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    pearson = np.full(actual.shape[0], np.nan)
    spearman = np.full(actual.shape[0], np.nan)
    for idx, (observed, predicted) in enumerate(zip(actual, forecast, strict=True)):
        valid = np.isfinite(observed) & np.isfinite(predicted)
        if valid.sum() < 3:
            continue
        x = observed[valid]
        y = predicted[valid]
        if np.std(x) > 1e-12 and np.std(y) > 1e-12:
            pearson[idx] = float(np.corrcoef(x, y)[0, 1])
            spearman[idx] = float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    return pearson, spearman


def evaluate_risk_forecasts(
    *,
    target: np.ndarray,
    returns: np.ndarray,
    dates: Sequence[Any],
    forecasts: Mapping[str, np.ndarray],
    validation_indices: Sequence[int],
    test_indices: Sequence[int],
    benchmark_name: str,
    epsilon: float,
    variance_floor: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Evaluate log-risk and calibrated variance forecasts on matched test cells.

    ``absolute_r2`` is the conventional pooled coefficient of determination on
    the out-of-sample cells.  Its denominator is the total sum of squares about
    the observed test-cell mean, so it does not use the supplied benchmark.
    ``oos_r2_vs_benchmark`` is retained as a backward-compatible diagnostic.
    """
    if benchmark_name not in forecasts:
        raise ValueError(f"benchmark_name {benchmark_name!r} is not in forecasts.")
    target = np.asarray(target, dtype=float)
    returns = np.asarray(returns, dtype=float)
    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    test_indices = np.asarray(test_indices, dtype=np.int64)
    calibrations: dict[str, float] = {}
    for name, forecast in forecasts.items():
        calibrations[name] = fit_variance_calibration(
            returns[validation_indices],
            np.asarray(forecast)[validation_indices],
            epsilon=epsilon,
            variance_floor=variance_floor,
        )

    actual_log = target[test_indices]
    actual_return = returns[test_indices]
    actual_variance = np.square(actual_return)
    benchmark = np.asarray(forecasts[benchmark_name], dtype=float)[test_indices]
    benchmark_error = benchmark - actual_log
    matched_test_cells = np.isfinite(actual_log)
    for full_forecast in forecasts.values():
        matched_test_cells &= np.isfinite(
            np.asarray(full_forecast, dtype=float)[test_indices]
        )
    if matched_test_cells.any():
        absolute_r2_observed_mean = float(np.mean(actual_log[matched_test_cells]))
        absolute_r2_total_sum_of_squares = float(
            np.sum(
                np.square(
                    actual_log[matched_test_cells] - absolute_r2_observed_mean
                )
            )
        )
    else:
        absolute_r2_total_sum_of_squares = np.nan
    summary_rows: list[dict[str, Any]] = []
    monthly_rows: list[pd.DataFrame] = []
    for name, full_forecast in forecasts.items():
        log_forecast = np.asarray(full_forecast, dtype=float)[test_indices]
        valid = matched_test_cells
        error = log_forecast - actual_log
        squared_error = np.square(error)
        benchmark_squared_error = np.square(benchmark_error)
        variance = calibrations[name] * variance_from_log_risk(
            log_forecast,
            epsilon=epsilon,
            variance_floor=variance_floor,
        )
        gaussian_return_crps = normal_crps(0.0, variance, actual_return)
        qlike = actual_variance / variance + np.log(variance)
        variance_squared_error = np.square(actual_variance - variance)
        sse = float(np.sum(squared_error[valid]))
        benchmark_sse = float(np.sum(benchmark_squared_error[valid]))
        absolute_r2_sse = float(
            np.sum(squared_error[matched_test_cells])
        )
        summary_rows.append({
            "model": name,
            "target_type": "log_squared_return",
            "benchmark": benchmark_name,
            "test_cells": int(valid.sum()),
            "log_risk_rmse": float(np.sqrt(np.mean(squared_error[valid]))),
            "log_risk_mae": float(np.mean(np.abs(error[valid]))),
            "log_risk_bias": float(np.mean(error[valid])),
            "gaussian_return_crps": float(np.mean(gaussian_return_crps[valid])),
            "qlike": float(np.mean(qlike[valid])),
            "variance_mse": float(np.mean(variance_squared_error[valid])),
            "absolute_r2": (
                1.0 - absolute_r2_sse / absolute_r2_total_sum_of_squares
                if absolute_r2_total_sum_of_squares > 0.0
                else np.nan
            ),
            "oos_r2_vs_benchmark": 1.0 - sse / benchmark_sse if benchmark_sse > 0.0 else np.nan,
            "variance_calibration": calibrations[name],
            "mean_predicted_variance": float(np.mean(variance[valid])),
            "mean_realized_squared_return": float(np.mean(actual_variance[valid])),
        })
        monthly_valid = valid.sum(axis=1)
        monthly_rows.append(pd.DataFrame({
            "date": pd.DatetimeIndex(pd.to_datetime(dates))[test_indices],
            "model": name,
            "observed_cells": monthly_valid,
            "squared_error": np.divide(
                np.where(valid, squared_error, 0.0).sum(axis=1),
                monthly_valid,
                out=np.full(len(test_indices), np.nan),
                where=monthly_valid > 0,
            ),
            "benchmark_squared_error": np.divide(
                np.where(valid, benchmark_squared_error, 0.0).sum(axis=1),
                monthly_valid,
                out=np.full(len(test_indices), np.nan),
                where=monthly_valid > 0,
            ),
            "squared_error_differential": np.divide(
                np.where(valid, squared_error - benchmark_squared_error, 0.0).sum(axis=1),
                monthly_valid,
                out=np.full(len(test_indices), np.nan),
                where=monthly_valid > 0,
            ),
            "log_risk_bias": np.divide(
                np.where(valid, error, 0.0).sum(axis=1),
                monthly_valid,
                out=np.full(len(test_indices), np.nan),
                where=monthly_valid > 0,
            ),
            "gaussian_return_crps": np.divide(
                np.where(valid, gaussian_return_crps, 0.0).sum(axis=1),
                monthly_valid,
                out=np.full(len(test_indices), np.nan),
                where=monthly_valid > 0,
            ),
            "qlike": np.divide(
                np.where(valid, qlike, 0.0).sum(axis=1),
                monthly_valid,
                out=np.full(len(test_indices), np.nan),
                where=monthly_valid > 0,
            ),
        }))
    summary = add_composite_rank_score(
        pd.DataFrame(summary_rows),
        lower_is_better=(
            "log_risk_rmse",
            "log_risk_mae",
            "gaussian_return_crps",
        ),
        closest_to_zero=("log_risk_bias",),
    )
    return summary, pd.concat(monthly_rows, ignore_index=True), calibrations


def evaluate_raw_return_forecasts(
    *,
    returns: np.ndarray,
    dates: Sequence[Any],
    forecasts: Mapping[str, np.ndarray],
    test_indices: Sequence[int],
    benchmark_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate raw-return forecasts against a feasible benchmark."""
    actual = np.asarray(returns, dtype=float)[np.asarray(test_indices, dtype=np.int64)]
    benchmark = np.asarray(forecasts[benchmark_name], dtype=float)[np.asarray(test_indices, dtype=np.int64)]
    benchmark_error = benchmark - actual
    rows: list[dict[str, Any]] = []
    monthly: list[pd.DataFrame] = []
    for name, full_forecast in forecasts.items():
        forecast = np.asarray(full_forecast, dtype=float)[np.asarray(test_indices, dtype=np.int64)]
        valid = np.isfinite(actual) & np.isfinite(forecast) & np.isfinite(benchmark)
        error = forecast - actual
        squared = np.square(error)
        benchmark_squared = np.square(benchmark_error)
        pearson, spearman = _cross_sectional_correlations(actual, forecast)
        sse = float(np.sum(squared[valid]))
        benchmark_sse = float(np.sum(benchmark_squared[valid]))
        rows.append({
            "model": name,
            "target_type": "raw_return",
            "benchmark": benchmark_name,
            "test_cells": int(valid.sum()),
            "rmse": float(np.sqrt(np.mean(squared[valid]))),
            "mae": float(np.mean(np.abs(error[valid]))),
            "oos_r2_vs_benchmark": 1.0 - sse / benchmark_sse if benchmark_sse > 0.0 else np.nan,
            "mean_monthly_pearson": float(np.nanmean(pearson)),
            "mean_monthly_spearman": float(np.nanmean(spearman)),
            "sign_accuracy": float(np.mean(np.sign(forecast[valid]) == np.sign(actual[valid]))),
        })
        count = valid.sum(axis=1)
        monthly.append(pd.DataFrame({
            "date": pd.DatetimeIndex(pd.to_datetime(dates))[np.asarray(test_indices, dtype=np.int64)],
            "model": name,
            "observed_cells": count,
            "squared_error": np.divide(
                np.where(valid, squared, 0.0).sum(axis=1),
                count,
                out=np.full(len(count), np.nan),
                where=count > 0,
            ),
            "benchmark_squared_error": np.divide(
                np.where(valid, benchmark_squared, 0.0).sum(axis=1),
                count,
                out=np.full(len(count), np.nan),
                where=count > 0,
            ),
            "squared_error_differential": np.divide(
                np.where(valid, squared - benchmark_squared, 0.0).sum(axis=1),
                count,
                out=np.full(len(count), np.nan),
                where=count > 0,
            ),
            "pearson": pearson,
            "spearman": spearman,
        }))
    return pd.DataFrame(rows), pd.concat(monthly, ignore_index=True)


def correction_diagnostics(
    baseline_error: np.ndarray,
    correction: np.ndarray,
) -> dict[str, float]:
    error = np.asarray(baseline_error, dtype=float)
    adjustment = np.asarray(correction, dtype=float)
    valid = np.isfinite(error) & np.isfinite(adjustment)
    if valid.sum() < 2:
        return {
            "cells": int(valid.sum()),
            "correlation": np.nan,
            "correction_sd_over_error_sd": np.nan,
            "implied_lambda": np.nan,
        }
    e = error[valid]
    c = adjustment[valid]
    denominator = float(np.sum(c * c))
    return {
        "cells": int(valid.sum()),
        "correlation": float(np.corrcoef(e, c)[0, 1]) if np.std(e) > 0 and np.std(c) > 0 else np.nan,
        "correction_sd_over_error_sd": float(np.std(c) / np.std(e)) if np.std(e) > 0 else np.nan,
        "implied_lambda": float(np.sum(e * c) / denominator) if denominator > 0 else np.nan,
    }


@dataclass
class RollingVARAResult:
    """Outputs from scheduled-refit rolling one-step VARA forecasts."""

    standardized_forecast: np.ndarray
    correction: np.ndarray
    refit_index: np.ndarray
    fit_summary: pd.DataFrame
    parameter_summary: pd.DataFrame
    neighbors: pd.DataFrame


@dataclass
class RollingGroupVARAResult:
    """Outputs from a multi-head hard-group VARA correction model."""

    standardized_forecast: np.ndarray
    correction: np.ndarray
    refit_index: np.ndarray
    fit_summary: pd.DataFrame


@dataclass
class RollingHARVARAResult:
    """Outputs from a HAR-lag extension of the VARA correction model."""

    standardized_forecast: np.ndarray
    correction: np.ndarray
    refit_index: np.ndarray
    fit_summary: pd.DataFrame


def _fixed_smallest_topk_cache(cache: Any) -> Any:
    """Return a top-k cache whose support is always the smallest-score set.

    The generic top-k implementation changes support with the sign of eta.
    For the Accounting-Signal Risk specification the sparse graph is defined
    first by the 500 smallest formula scores, and eta is then estimated as the
    softmax temperature on that fixed graph.  Reusing the negative-score cache
    for both signs makes that profile objective continuous at eta zero.
    """
    from vara.attention import TopKAttentionCache

    if cache.indices_neg is None or cache.scores_neg is None:
        raise ValueError("A smallest-score attention cache is required.")
    return TopKAttentionCache(
        indices_pos=cache.indices_neg,
        scores_pos=cache.scores_neg,
        indices_neg=cache.indices_neg,
        scores_neg=cache.scores_neg,
        mode=cache.mode,
        cluster_count=cache.cluster_count,
        cluster_probe=cache.cluster_probe,
        cluster_candidate_size=cache.cluster_candidate_size,
    )


def _har_training_regressors(
    response: np.ndarray,
    peer_response: np.ndarray,
    *,
    windows: Sequence[int],
    zero_own_lag_coefficients: bool,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    """Build the target and componentwise HAR regressors at a fixed eta."""
    values = np.asarray(response, dtype=float)
    peers = np.asarray(peer_response, dtype=float)
    if values.shape != peers.shape or values.ndim != 2:
        raise ValueError("response and peer_response must have the same 2D shape.")
    window_values = tuple(int(window) for window in windows)
    if not window_values or min(window_values) < 1 or max(window_values) >= len(values):
        raise ValueError("HAR windows must be positive and shorter than the response history.")

    target_rows = np.arange(max(window_values), len(values), dtype=np.int64)
    cumulative_response = np.vstack(
        (np.zeros((1, values.shape[1])), np.cumsum(values, axis=0))
    )
    cumulative_peer = np.vstack(
        (np.zeros((1, values.shape[1])), np.cumsum(peers, axis=0))
    )
    own = [
        (cumulative_response[target_rows] - cumulative_response[target_rows - window])
        / window
        for window in window_values
    ]
    peer = [
        (cumulative_peer[target_rows] - cumulative_peer[target_rows - window]) / window
        for window in window_values
    ]
    if bool(zero_own_lag_coefficients):
        regressors = peer
        names = [f"peer_mean_{window}" for window in window_values]
    else:
        regressors = own + peer
        names = [f"own_mean_{window}" for window in window_values]
        names += [f"peer_mean_{window}" for window in window_values]
    return values[target_rows], regressors, names


def _fit_har_vara_profile(
    response: np.ndarray,
    peer_response: np.ndarray,
    *,
    windows: Sequence[int],
    ridge: float,
    zero_own_lag_coefficients: bool,
    variance_floor: float = 1e-12,
) -> dict[str, Any]:
    """Profile component slopes and variances for one candidate eta.

    The criterion is the ridge-penalized concentrated Gaussian
    quasi-likelihood.  At ridge zero it reduces to the ordinary concentrated
    QMLE used by the two-stage VARA estimator.
    """
    now, regressors, regressor_names = _har_training_regressors(
        response,
        peer_response,
        windows=windows,
        zero_own_lag_coefficients=zero_own_lag_coefficients,
    )
    regressor_count = len(regressors)
    p = now.shape[1]
    gram = np.empty((p, regressor_count, regressor_count), dtype=float)
    rhs = np.empty((p, regressor_count), dtype=float)
    for left in range(regressor_count):
        rhs[:, left] = np.sum(regressors[left] * now, axis=0)
        for right in range(left, regressor_count):
            product = np.sum(regressors[left] * regressors[right], axis=0)
            gram[:, left, right] = product
            gram[:, right, left] = product
    penalized_gram = gram + float(ridge) * np.eye(regressor_count)[None, :, :]
    coefficients = np.linalg.solve(penalized_gram, rhs[..., None]).squeeze(axis=2)
    response_sse = np.sum(np.square(now), axis=0)
    fitted_cross_product = np.sum(coefficients * rhs, axis=1)
    fitted_sse = np.einsum(
        "pi,pij,pj->p",
        coefficients,
        gram,
        coefficients,
        optimize=True,
    )
    residual_sse = np.maximum(
        response_sse - 2.0 * fitted_cross_product + fitted_sse,
        0.0,
    )
    penalty = float(ridge) * np.sum(np.square(coefficients), axis=1)
    profiled_variance = np.maximum(
        (residual_sse + penalty) / now.shape[0],
        float(variance_floor),
    )
    objective = -0.5 * now.shape[0] * float(np.sum(np.log(profiled_variance)))
    return {
        "coefficients": coefficients,
        "regressor_names": regressor_names,
        "residual_variance": residual_sse / now.shape[0],
        "profiled_variance": profiled_variance,
        "objective": objective,
        "effective_rows": int(now.shape[0]),
    }


def estimate_har_vara_eta(
    *,
    response: np.ndarray,
    feature_matrix: np.ndarray,
    attention_cache: Any,
    attention_top_k: int,
    attention_batch_size: int,
    ridge: float,
    eta_bounds: tuple[float, float],
    eta_grid_size: int,
    eta_local_tolerance: float,
    windows: Sequence[int] = (1, 3, 12),
    zero_own_lag_coefficients: bool = False,
) -> tuple[float, dict[str, Any], dict[str, Any]]:
    """Estimate eta by the HAR-specific concentrated quasi-likelihood."""
    import jax.numpy as jnp

    from vara.attention import (
        attention_weighted_lags,
        cached_topk_attention_weighted_lags,
    )
    from vara.simulation import golden_section_maximize

    lower, upper = (float(eta_bounds[0]), float(eta_bounds[1]))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("eta_bounds must contain two finite increasing values.")
    if int(eta_grid_size) < 2:
        raise ValueError("eta_grid_size must be at least two.")
    if float(eta_local_tolerance) <= 0.0:
        raise ValueError("eta_local_tolerance must be positive.")

    values = np.asarray(response, dtype=float)
    Z = np.asarray(feature_matrix, dtype=float)
    profile_cache: dict[float, dict[str, Any]] = {}

    def profile(eta_value: float) -> dict[str, Any]:
        key = float(eta_value)
        if key not in profile_cache:
            if getattr(attention_cache, "mode", "") == "exact-fixed-smallest":
                peer_response = cached_topk_attention_weighted_lags(
                    values,
                    attention_cache,
                    key,
                )
            else:
                peer_response = np.asarray(attention_weighted_lags(
                    jnp.asarray(values),
                    jnp.asarray(Z),
                    key,
                    {"eta_a": 1.0},
                    batch_size=int(attention_batch_size),
                    dense_threshold=0,
                    attention_mode="topk",
                    attention_top_k=int(attention_top_k),
                    attention_cache=attention_cache,
                ))
            profile_cache[key] = _fit_har_vara_profile(
                values,
                peer_response,
                windows=windows,
                ridge=ridge,
                zero_own_lag_coefficients=zero_own_lag_coefficients,
            )
        return profile_cache[key]

    def objective(eta_value: float) -> float:
        return float(profile(float(eta_value))["objective"])

    grid = np.linspace(lower, upper, int(eta_grid_size), dtype=float)
    grid_objectives = np.asarray([objective(value) for value in grid], dtype=float)
    best_index = int(np.argmax(grid_objectives))
    left_index = max(0, best_index - 1)
    right_index = min(len(grid) - 1, best_index + 1)
    if best_index == 0:
        right_index = 1
    elif best_index == len(grid) - 1:
        left_index = len(grid) - 2
    refined_eta, _ = golden_section_maximize(
        objective,
        float(grid[left_index]),
        float(grid[right_index]),
        tol=float(eta_local_tolerance),
    )
    refined_value = float(refined_eta)

    candidates = [float(value) for value in grid]
    candidates.append(refined_value)
    candidate_objectives = np.asarray([objective(value) for value in candidates])
    selected_position = int(np.argmax(candidate_objectives))
    eta_hat = float(candidates[selected_position])
    selected_profile = profile(eta_hat)
    tolerance = max(float(eta_local_tolerance), 1e-10)
    diagnostics = {
        "eta_hat": eta_hat,
        "eta_on_boundary": bool(
            abs(eta_hat - lower) <= tolerance or abs(eta_hat - upper) <= tolerance
        ),
        "eta_grid_best": float(grid[best_index]),
        "eta_grid_best_objective": float(grid_objectives[best_index]),
        "eta_objective": float(selected_profile["objective"]),
        "eta_bounds_lower": lower,
        "eta_bounds_upper": upper,
        "eta_grid_size": int(eta_grid_size),
        "eta_local_tolerance": float(eta_local_tolerance),
        "eta_estimation": "ridge_penalized_concentrated_qmle",
        "attention_support": (
            "deterministic_fixed_smallest_scores"
            if getattr(attention_cache, "mode", "") == "exact-fixed-smallest"
            else "fixed_smallest_scores"
        ),
    }
    return eta_hat, diagnostics, selected_profile


def _group_mean_excluding_self(values: np.ndarray, labels: Sequence[Any]) -> np.ndarray:
    matrix = np.asarray(values, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(labels):
        raise ValueError("values and labels have inconsistent component dimensions.")
    codes, uniques = pd.factorize(pd.Series(labels, dtype="object"), sort=True)
    result = np.zeros_like(matrix)
    for group_index in range(len(uniques)):
        members = np.flatnonzero(codes == group_index)
        if len(members) <= 1:
            continue
        total = np.sum(matrix[:, members], axis=1, keepdims=True)
        result[:, members] = (total - matrix[:, members]) / (len(members) - 1)
    return result


def rolling_group_vara_corrections(
    *,
    baseline_errors: np.ndarray,
    evaluation_indices: Sequence[int],
    metadata: pd.DataFrame,
    group_cols: Sequence[str],
    include_global_head: bool,
    error_window_months: int,
    response_clip: float | None,
    refit_frequency_months: int,
    ridge: float,
) -> RollingGroupVARAResult:
    """Forecast baseline errors with own-lag and hard group-attention heads."""
    errors = np.asarray(baseline_errors, dtype=float)
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    metadata_frame = metadata.reset_index(drop=True)
    columns = [str(column) for column in group_cols]
    missing = sorted(set(columns).difference(metadata_frame.columns))
    if errors.ndim != 2 or errors.shape[1] != len(metadata_frame):
        raise ValueError("baseline_errors and metadata have inconsistent shapes.")
    if evaluation.size == 0 or np.any(np.diff(evaluation) <= 0):
        raise ValueError("evaluation_indices must be a nonempty increasing sequence.")
    if missing:
        raise ValueError(f"Unknown group VARA columns: {missing}")
    if not columns and not include_global_head:
        raise ValueError("At least one group or global attention head is required.")
    if int(refit_frequency_months) < 1:
        raise ValueError("refit_frequency_months must be positive.")

    standardized_forecast = np.full_like(errors, np.nan, dtype=float)
    correction = np.full_like(errors, np.nan, dtype=float)
    refit_index = np.full(len(errors), -1, dtype=np.int64)
    fit_rows: list[dict[str, Any]] = []
    state: dict[str, Any] | None = None
    labels = [metadata_frame[column].astype(str).to_numpy() for column in columns]
    global_labels = np.zeros(errors.shape[1], dtype=np.int8)

    for evaluation_position, target_index in enumerate(evaluation):
        origin_index = int(target_index) - 1
        should_refit = state is None or evaluation_position % int(refit_frequency_months) == 0
        if should_refit:
            start = max(0, origin_index - int(error_window_months) + 1)
            history = np.asarray(errors[start:origin_index + 1], dtype=float)
            history = history[np.isfinite(history).any(axis=1)]
            if history.shape[0] < 24:
                raise ValueError(
                    f"Group VARA refit at index {origin_index} has only {history.shape[0]} rows."
                )
            scaler = fit_robust_panel_scaler(history, clip=response_clip)
            response = scaler.transform(history)
            imputed_cells = int((~np.isfinite(response)).sum())
            response = np.where(np.isfinite(response), response, 0.0)
            lag = response[:-1]
            now = response[1:]
            regressors = [lag]
            regressor_names = ["own_lag"]
            for column, group_labels in zip(columns, labels, strict=True):
                regressors.append(_group_mean_excluding_self(lag, group_labels))
                regressor_names.append(f"{column}_attention")
            if include_global_head:
                regressors.append(_group_mean_excluding_self(lag, global_labels))
                regressor_names.append("global_attention")
            regressor_count = len(regressors)
            p = errors.shape[1]
            gram = np.empty((p, regressor_count, regressor_count), dtype=float)
            rhs = np.empty((p, regressor_count), dtype=float)
            for left in range(regressor_count):
                rhs[:, left] = np.sum(regressors[left] * now, axis=0)
                for right in range(left, regressor_count):
                    product = np.sum(regressors[left] * regressors[right], axis=0)
                    gram[:, left, right] = product
                    gram[:, right, left] = product
            gram += float(ridge) * np.eye(regressor_count)[None, :, :]
            coefficients = np.linalg.solve(gram, rhs[..., None]).squeeze(axis=2)
            fitted = sum(
                regressors[index] * coefficients[None, :, index]
                for index in range(regressor_count)
            )
            residual = now - fitted
            row: dict[str, Any] = {
                "refit_index": origin_index,
                "error_window_start_index": start,
                "error_rows": int(history.shape[0]),
                "imputed_error_cells": imputed_cells,
                "imputed_error_fraction": imputed_cells / response.size,
                "response_scale_floor": scaler.scale_floor,
                "residual_variance_mean": float(np.mean(np.square(residual))),
            }
            for index, name in enumerate(regressor_names):
                row[f"{name}_coefficient_mean"] = float(np.mean(coefficients[:, index]))
                row[f"{name}_coefficient_abs_mean"] = float(
                    np.mean(np.abs(coefficients[:, index]))
                )
                row[f"{name}_coefficient_abs_max"] = float(
                    np.max(np.abs(coefficients[:, index]))
                )
            fit_rows.append(row)
            state = {
                "scaler": scaler,
                "coefficients": coefficients,
                "refit_index": origin_index,
            }

        assert state is not None
        current = state["scaler"].transform(errors[origin_index])
        current = np.where(np.isfinite(current), current, 0.0)
        current_regressors = [current]
        for group_labels in labels:
            current_regressors.append(
                _group_mean_excluding_self(current[None, :], group_labels)[0]
            )
        if include_global_head:
            current_regressors.append(
                _group_mean_excluding_self(current[None, :], global_labels)[0]
            )
        coefficients = state["coefficients"]
        zhat = sum(
            current_regressors[index] * coefficients[:, index]
            for index in range(len(current_regressors))
        )
        standardized_forecast[int(target_index)] = zhat
        correction[int(target_index)] = state["scaler"].inverse(zhat)
        refit_index[int(target_index)] = int(state["refit_index"])

    return RollingGroupVARAResult(
        standardized_forecast=standardized_forecast,
        correction=correction,
        refit_index=refit_index,
        fit_summary=pd.DataFrame(fit_rows),
    )


def rolling_har_vara_corrections(
    *,
    baseline_errors: np.ndarray,
    evaluation_indices: Sequence[int],
    feature_matrix: np.ndarray,
    error_window_months: int,
    response_clip: float | None,
    refit_frequency_months: int,
    ridge: float,
    attention_top_k: int,
    attention_cache_mode: str,
    attention_batch_size: int,
    eta_bounds: tuple[float, float] = (-2.0, 2.0),
    eta_grid_size: int = 7,
    eta_local_tolerance: float = 1e-4,
    zero_own_lag_coefficients: bool = False,
    progress_callback: Any | None = None,
) -> RollingHARVARAResult:
    """Forecast errors using rolling QMLE-estimated HAR--VARA corrections.

    When ``zero_own_lag_coefficients`` is true, the own-error HAR coefficients
    are constrained to zero and only attention-weighted peer-error regressors
    are fitted.  This supplies a no-own-lag VARA ablation without changing the
    HAR windows used to aggregate the peer errors. Eta is estimated at every
    scheduled refit by profiling the same componentwise HAR regressions used
    for forecasting.
    """
    import jax.numpy as jnp

    from vara.attention import (
        attention_weighted_lags,
        build_cached_topk_attention_csr,
        build_fixed_smallest_topk_attention_cache,
        build_topk_attention_cache,
    )

    errors = np.asarray(baseline_errors, dtype=float)
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    Z = np.asarray(feature_matrix, dtype=float)
    if errors.ndim != 2 or Z.ndim != 2 or errors.shape[1] != Z.shape[0]:
        raise ValueError("baseline_errors and feature_matrix have inconsistent shapes.")
    if evaluation.size == 0 or np.any(np.diff(evaluation) <= 0):
        raise ValueError("evaluation_indices must be a nonempty increasing sequence.")
    if int(error_window_months) < 24:
        raise ValueError("error_window_months must be at least 24.")
    if int(refit_frequency_months) < 1:
        raise ValueError("refit_frequency_months must be positive.")

    if str(attention_cache_mode) == "exact":
        cache = build_fixed_smallest_topk_attention_cache(
            Z,
            top_k=int(attention_top_k),
            batch_size=int(attention_batch_size),
        )
    else:
        candidate_cache = build_topk_attention_cache(
            jnp.asarray(Z),
            top_k=int(attention_top_k),
            batch_size=int(attention_batch_size),
            include_negative=True,
            cache_mode=str(attention_cache_mode),
        )
        cache = _fixed_smallest_topk_cache(candidate_cache)
    standardized_forecast = np.full_like(errors, np.nan, dtype=float)
    correction = np.full_like(errors, np.nan, dtype=float)
    refit_index = np.full(len(errors), -1, dtype=np.int64)
    fit_rows: list[dict[str, Any]] = []
    state: dict[str, Any] | None = None
    windows = (1, 3, 12)

    for evaluation_position, target_index in enumerate(evaluation):
        origin_index = int(target_index) - 1
        should_refit = state is None or evaluation_position % int(refit_frequency_months) == 0
        if should_refit:
            start = max(0, origin_index - int(error_window_months) + 1)
            history = np.asarray(errors[start:origin_index + 1], dtype=float)
            history = history[np.isfinite(history).any(axis=1)]
            if history.shape[0] < 24:
                raise ValueError(
                    f"HAR-VARA refit at index {origin_index} has only {history.shape[0]} rows."
                )
            scaler = fit_robust_panel_scaler(history, clip=response_clip)
            response = scaler.transform(history)
            imputed_cells = int((~np.isfinite(response)).sum())
            response = np.where(np.isfinite(response), response, 0.0)
            eta_hat, eta_diagnostics, profile_fit = estimate_har_vara_eta(
                response=response,
                feature_matrix=Z,
                attention_cache=cache,
                attention_top_k=int(attention_top_k),
                attention_batch_size=int(attention_batch_size),
                ridge=float(ridge),
                eta_bounds=(float(eta_bounds[0]), float(eta_bounds[1])),
                eta_grid_size=int(eta_grid_size),
                eta_local_tolerance=float(eta_local_tolerance),
                windows=windows,
                zero_own_lag_coefficients=zero_own_lag_coefficients,
            )
            coefficients = np.asarray(profile_fit["coefficients"], dtype=float)
            regressor_names = list(profile_fit["regressor_names"])
            row: dict[str, Any] = {
                "refit_index": origin_index,
                "error_window_start_index": start,
                "error_rows": int(history.shape[0]),
                "imputed_error_cells": imputed_cells,
                "imputed_error_fraction": imputed_cells / response.size,
                "response_scale_floor": scaler.scale_floor,
                "residual_variance_mean": float(
                    np.mean(profile_fit["residual_variance"])
                ),
                "attention_top_k": int(attention_top_k),
                "own_lag_coefficients_fixed_zero": bool(zero_own_lag_coefficients),
            }
            row.update(eta_diagnostics)
            if bool(zero_own_lag_coefficients):
                for window in windows:
                    row[f"own_mean_{window}_coefficient_abs_mean"] = 0.0
                    row[f"own_mean_{window}_coefficient_abs_max"] = 0.0
            for index, name in enumerate(regressor_names):
                row[f"{name}_coefficient_abs_mean"] = float(
                    np.mean(np.abs(coefficients[:, index]))
                )
                row[f"{name}_coefficient_abs_max"] = float(
                    np.max(np.abs(coefficients[:, index]))
                )
            fit_rows.append(row)
            if progress_callback is not None:
                progress_callback(dict(row))
            attention_operator = (
                build_cached_topk_attention_csr(cache, float(eta_hat))
                if getattr(cache, "mode", "") == "exact-fixed-smallest"
                else None
            )
            state = {
                "scaler": scaler,
                "coefficients": coefficients,
                "eta_hat": float(eta_hat),
                "refit_index": origin_index,
                "attention_operator": attention_operator,
            }

        assert state is not None
        recent = np.asarray(errors[max(0, origin_index - 11):origin_index + 1], dtype=float)
        recent = state["scaler"].transform(recent)
        recent = np.where(np.isfinite(recent), recent, 0.0)
        if len(recent) < 12:
            raise ValueError("HAR-VARA prediction requires at least 12 lagged error months.")
        own_regressors = [np.mean(recent[-window:], axis=0) for window in windows]
        own_matrix = np.stack(own_regressors, axis=0)
        if state["attention_operator"] is not None:
            calculation_dtype = np.result_type(
                np.asarray(cache.scores_pos).dtype,
                np.float32,
            )
            peer_matrix = np.asarray(
                state["attention_operator"]
                @ own_matrix.astype(calculation_dtype, copy=False).T
            ).T
        else:
            peer_matrix = np.asarray(attention_weighted_lags(
                jnp.asarray(own_matrix),
                jnp.asarray(Z),
                float(state["eta_hat"]),
                {"eta_a": 1.0},
                batch_size=int(attention_batch_size),
                dense_threshold=0,
                attention_mode="topk",
                attention_top_k=int(attention_top_k),
                attention_cache=cache,
            ))
        peer_regressors = [peer_matrix[index] for index in range(len(windows))]
        prediction_regressors = (
            peer_regressors
            if bool(zero_own_lag_coefficients)
            else own_regressors + peer_regressors
        )
        coefficients = state["coefficients"]
        zhat = sum(
            prediction_regressors[index] * coefficients[:, index]
            for index in range(len(prediction_regressors))
        )
        standardized_forecast[int(target_index)] = zhat
        correction[int(target_index)] = state["scaler"].inverse(zhat)
        refit_index[int(target_index)] = int(state["refit_index"])

    return RollingHARVARAResult(
        standardized_forecast=standardized_forecast,
        correction=correction,
        refit_index=refit_index,
        fit_summary=pd.DataFrame(fit_rows),
    )


def attention_neighbor_diagnostics(
    *,
    feature_matrix: np.ndarray,
    eta: float,
    component_ids: Sequence[str],
    metadata: pd.DataFrame,
    attention_mode: str,
    attention_top_k: int | None,
    attention_cache: Any | None,
    top_n: int = 10,
) -> pd.DataFrame:
    """Return top attended neighbors and component-level similarity diagnostics."""
    import jax.numpy as jnp
    from vara.attention import build_attention, topk_attention_cached_indices_weights

    Z = jnp.asarray(feature_matrix)
    p = Z.shape[0]
    keep = min(int(top_n), max(int(p) - 1, 1))
    if attention_mode == "uniform":
        indices = np.empty((int(p), keep), dtype=np.int64)
        candidates = np.arange(keep + 1, dtype=np.int64)
        for target_index in range(int(p)):
            indices[target_index] = candidates[candidates != target_index][:keep]
        weights = np.full((int(p), keep), 1.0 / max(int(p) - 1, 1), dtype=float)
        full_effective = np.full(int(p), max(int(p) - 1, 1), dtype=float)
    elif attention_mode == "topk" and attention_cache is not None:
        indices_jax, weights_jax = topk_attention_cached_indices_weights(
            attention_cache,
            float(eta),
            1.0,
            0,
            int(p),
        )
        indices = np.asarray(indices_jax)
        weights = np.asarray(weights_jax)
        order = np.argsort(-weights, axis=1)[:, :keep]
        indices = np.take_along_axis(indices, order, axis=1)
        weights = np.take_along_axis(weights, order, axis=1)
        full_weights = np.asarray(weights_jax)
        entropy_terms = np.where(
            full_weights > 0.0,
            full_weights * np.log(np.maximum(full_weights, np.finfo(float).tiny)),
            0.0,
        )
        full_effective = np.exp(-np.sum(entropy_terms, axis=1))
    else:
        W = np.asarray(build_attention(
            Z,
            float(eta),
            1.0,
            attention_mode=attention_mode,
            top_k=attention_top_k,
            attention_cache=attention_cache,
        ))
        order = np.argsort(-W, axis=1)[:, :keep]
        indices = order
        weights = np.take_along_axis(W, order, axis=1)
        entropy_terms = np.where(
            W > 0.0,
            W * np.log(np.maximum(W, np.finfo(float).tiny)),
            0.0,
        )
        full_effective = np.exp(-np.sum(entropy_terms, axis=1))

    metadata = metadata.reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for target_index in range(int(p)):
        for rank in range(keep):
            neighbor_index = int(indices[target_index, rank])
            rows.append({
                "target_index": target_index,
                "component_id": str(component_ids[target_index]),
                "neighbor_rank": rank + 1,
                "neighbor_index": neighbor_index,
                "neighbor_component_id": str(component_ids[neighbor_index]),
                "attention_weight": float(weights[target_index, rank]),
                "effective_neighbors": float(full_effective[target_index]),
                "same_numerator": bool(
                    metadata.loc[target_index, "numerator"] == metadata.loc[neighbor_index, "numerator"]
                ),
                "same_denominator": bool(
                    metadata.loc[target_index, "denominator"] == metadata.loc[neighbor_index, "denominator"]
                ),
                "same_formula_type": bool(
                    metadata.loc[target_index, "formula_type"] == metadata.loc[neighbor_index, "formula_type"]
                ),
            })
    return pd.DataFrame(rows)


def rolling_vara_corrections(
    *,
    baseline_errors: np.ndarray,
    evaluation_indices: Sequence[int],
    feature_builder: Any,
    component_ids: Sequence[str],
    metadata: pd.DataFrame,
    error_window_months: int,
    response_clip: float | None,
    refit_frequency_months: int,
    grid_size: int,
    ridge: float,
    eta_bounds: tuple[float, float],
    attention_mode: str,
    attention_top_k: int,
    attention_cache_mode: str,
    batch_size: int,
    dense_threshold: int,
    top_neighbors: int = 10,
    refit_artifact_callback: Any | None = None,
) -> RollingVARAResult:
    """Forecast genuine baseline errors with scheduled, leakage-safe VARA refits.

    ``feature_builder(origin_index)`` must return ``(Z, feature_info)`` and may
    only use data through ``origin_index``. Evaluation indices identify target
    months, so each corresponding origin is ``target_index - 1``.
    """
    import jax.numpy as jnp

    from vara.attention import build_topk_attention_cache
    from vara.forecast import VARAForecaster

    errors = np.asarray(baseline_errors, dtype=float)
    if errors.ndim != 2:
        raise ValueError("baseline_errors must have shape (time, components).")
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    if evaluation.size == 0 or np.any(np.diff(evaluation) <= 0):
        raise ValueError("evaluation_indices must be a nonempty increasing sequence.")
    if evaluation[0] < 1 or evaluation[-1] >= len(errors):
        raise ValueError("evaluation_indices contain an invalid target month.")
    if errors.shape[1] != len(component_ids) or len(metadata) != len(component_ids):
        raise ValueError("Component order is inconsistent across errors, IDs, and metadata.")
    if refit_frequency_months < 1:
        raise ValueError("refit_frequency_months must be positive.")
    if attention_mode not in {"dense", "topk", "uniform"}:
        raise ValueError("DataMine rolling forecasts support dense, topk, or uniform attention.")

    standardized_forecast = np.full_like(errors, np.nan, dtype=float)
    correction = np.full_like(errors, np.nan, dtype=float)
    refit_index = np.full(len(errors), -1, dtype=np.int64)
    fit_rows: list[dict[str, Any]] = []
    parameter_frames: list[pd.DataFrame] = []
    neighbor_frames: list[pd.DataFrame] = []
    state: dict[str, Any] | None = None
    cached_attention_Z: np.ndarray | None = None
    cached_attention: Any | None = None

    for evaluation_position, target_index in enumerate(evaluation):
        origin_index = int(target_index) - 1
        should_refit = state is None or evaluation_position % int(refit_frequency_months) == 0
        if should_refit:
            start = max(0, origin_index - int(error_window_months) + 1)
            history = np.asarray(errors[start : origin_index + 1], dtype=float)
            usable_rows = np.isfinite(history).any(axis=1)
            history = history[usable_rows]
            if history.shape[0] < 24:
                raise ValueError(
                    f"VARA refit at index {origin_index} has only {history.shape[0]} historical error rows."
                )
            scaler = fit_robust_panel_scaler(history, clip=response_clip)
            response = scaler.transform(history)
            imputed_cells = int((~np.isfinite(response)).sum())
            response = np.where(np.isfinite(response), response, 0.0)
            Z, feature_info = feature_builder(origin_index)
            Z = np.asarray(Z, dtype=float)
            if Z.ndim != 2 or Z.shape[0] != errors.shape[1] or not np.isfinite(Z).all():
                raise ValueError("feature_builder returned an invalid feature matrix.")

            cache = None
            cache_reused = False
            if attention_mode == "topk":
                if (
                    cached_attention_Z is not None
                    and cached_attention_Z.shape == Z.shape
                    and np.array_equal(cached_attention_Z, Z)
                ):
                    cache = cached_attention
                    cache_reused = True
                else:
                    cache = build_topk_attention_cache(
                        jnp.asarray(Z),
                        top_k=int(attention_top_k),
                        batch_size=int(batch_size),
                        include_negative=True,
                        cache_mode=attention_cache_mode,
                    )
                    cached_attention_Z = Z.copy()
                    cached_attention = cache
            forecaster = VARAForecaster(
                eta_bounds=(float(eta_bounds[0]), float(eta_bounds[1])),
                grid_size=int(grid_size),
                ridge=float(ridge),
                batch_size=int(batch_size),
                dense_threshold=int(dense_threshold),
                attention_mode=attention_mode,
                attention_top_k=int(attention_top_k) if attention_mode == "topk" else None,
                attention_cache=cache,
            )
            forecaster.fit(jnp.asarray(response), jnp.asarray(Z))
            diagnostics = forecaster.diagnostics()
            diagnostics.update({
                "refit_index": origin_index,
                "error_window_start_index": start,
                "error_rows": int(history.shape[0]),
                "imputed_error_cells": imputed_cells,
                "imputed_error_fraction": imputed_cells / response.size,
                "response_scale_floor": scaler.scale_floor,
                "attention_cache_reused": cache_reused,
            })
            fit_rows.append(diagnostics)
            parameter_frames.append(pd.DataFrame({
                "refit_index": origin_index,
                "component_id": list(component_ids),
                "alpha_hat": np.asarray(forecaster.alpha_hat),
                "beta_hat": np.asarray(forecaster.beta_hat),
                "sigma2_hat": np.asarray(forecaster.sigma2_hat),
            }))
            neighbors = attention_neighbor_diagnostics(
                feature_matrix=Z,
                eta=float(forecaster.eta_hat),
                component_ids=component_ids,
                metadata=metadata,
                attention_mode=attention_mode,
                attention_top_k=int(attention_top_k) if attention_mode == "topk" else None,
                attention_cache=cache,
                top_n=top_neighbors,
            )
            neighbors.insert(0, "refit_index", origin_index)
            neighbor_frames.append(neighbors)
            state = {
                "forecaster": forecaster,
                "scaler": scaler,
                "refit_index": origin_index,
                "feature_info": feature_info,
                "Z": Z,
            }
            if refit_artifact_callback is not None:
                refit_artifact_callback(
                    origin_index=origin_index,
                    feature_matrix=Z,
                    feature_info=feature_info,
                    scaler=scaler,
                    diagnostics=diagnostics,
                )

        assert state is not None
        current_error = errors[origin_index]
        current_standardized = state["scaler"].transform(current_error)
        current_standardized = np.where(np.isfinite(current_standardized), current_standardized, 0.0)
        zhat = np.asarray(
            state["forecaster"].predict(h=1, y_last=jnp.asarray(current_standardized))[0],
            dtype=float,
        )
        standardized_forecast[target_index] = zhat
        correction[target_index] = state["scaler"].inverse(zhat)
        refit_index[target_index] = int(state["refit_index"])

    return RollingVARAResult(
        standardized_forecast=standardized_forecast,
        correction=correction,
        refit_index=refit_index,
        fit_summary=pd.DataFrame(fit_rows),
        parameter_summary=(
            pd.concat(parameter_frames, ignore_index=True)
            if parameter_frames else pd.DataFrame()
        ),
        neighbors=(
            pd.concat(neighbor_frames, ignore_index=True)
            if neighbor_frames else pd.DataFrame()
        ),
    )
