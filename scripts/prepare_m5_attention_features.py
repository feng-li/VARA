#!/usr/bin/env python3
"""Prepare series-level M5 features for VARA attention."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import trange

from vara.forecast_baselines import auto_ets_config
from vara.hierarchical_features import (
    M5_ATTENTION_META_COLUMNS,
    M5_CATEGORICAL_FEATURES,
    make_m5_attention_features,
    make_m5_attention_features_from_files,
    read_table,
)
from vara.series_features import clean_feature_frame


DEFAULT_INPUT = Path("data/m5")
DEFAULT_OUTPUT = Path("data/m5/m5_attention_features_all_stores.parquet")


def _find_table(input_dir: Path, stem: str) -> Path:
    for suffix in (".parquet", ".csv"):
        path = input_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {stem}.parquet or {stem}.csv in {input_dir}")


def _print_progress(message: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] {message}", flush=True)


def _store_feature_task(
    store_id: str,
    sales: pd.DataFrame,
    prices: pd.DataFrame,
    calendar: pd.DataFrame,
    *,
    holdout: int,
    filter_method: str,
    seasonal_lag: int,
    ray_debug: bool,
) -> tuple[str, pd.DataFrame]:
    if ray_debug:
        _print_progress(f"Ray M5 feature task start: store_id={store_id}, n_series={len(sales)}")
    features = make_m5_attention_features(
        sales,
        prices,
        calendar,
        store_id=None,
        holdout=holdout,
        max_series=0,
        filter_method=filter_method,
        seasonal_lag=seasonal_lag,
        standardize_output=False,
    )
    if ray_debug:
        _print_progress(f"Ray M5 feature task done: store_id={store_id}, n_series={len(features)}")
    return store_id, features


def _write_features_and_metadata(
    features: pd.DataFrame,
    *,
    sales_path: Path,
    prices_path: Path,
    calendar_path: Path,
    output_path: Path,
    store_id: str | None,
    holdout: int,
    max_series: int,
    filter_method: str,
    seasonal_lag: int,
    standardize_output: bool,
    ray_parallel: bool,
    ray_num_cpus: int,
    max_cpus_per_task: int,
    ray_debug: bool,
) -> dict[str, Any]:
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
        "filter_method": str(filter_method),
        "seasonal_lag": int(seasonal_lag),
        "baseline_config": auto_ets_config(seasonal_lag) if str(filter_method) == "auto_ets" else {},
        "standardize_output": bool(standardize_output),
        "n_series": int(len(features)),
        "n_columns": int(len(features.columns)),
        "feature_columns": [col for col in features.columns if col not in set(M5_ATTENTION_META_COLUMNS)],
        "ray_parallel": bool(ray_parallel),
        "ray_num_cpus": int(ray_num_cpus),
        "max_cpus_per_task": int(max_cpus_per_task),
        "ray_debug": bool(ray_debug),
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metadata["metadata_path"] = str(metadata_path)
    return metadata


def _make_features_with_ray(
    *,
    ray_module: Any,
    sales_path: Path,
    prices_path: Path,
    calendar_path: Path,
    output_path: Path,
    store_id: str | None,
    holdout: int,
    max_series: int,
    filter_method: str,
    seasonal_lag: int,
    standardize_output: bool,
    overwrite: bool,
    ray_num_cpus: int,
    max_cpus_per_task: int,
    ray_debug: bool,
) -> dict[str, Any]:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}; pass overwrite=True or --overwrite.")

    sales = read_table(sales_path)
    prices = read_table(prices_path)
    calendar = read_table(calendar_path)
    if store_id:
        sales = sales.loc[sales["store_id"].astype(str) == str(store_id)].copy()
        prices = prices.loc[prices["store_id"].astype(str) == str(store_id)].copy()
    if max_series > 0:
        sales = sales.iloc[:max_series].copy()
    if sales.empty:
        raise ValueError("No M5 sales rows remain after filtering.")

    store_ids = sorted(sales["store_id"].astype(str).unique().tolist())
    remote_task = ray_module.remote(num_cpus=max_cpus_per_task)(_store_feature_task)
    calendar_arg = ray_module.put(calendar)
    futures = []
    submit_times = {}
    for sid in store_ids:
        sales_chunk = sales.loc[sales["store_id"].astype(str) == sid].copy()
        prices_chunk = prices.loc[prices["store_id"].astype(str) == sid].copy()
        ref = remote_task.remote(
            sid,
            sales_chunk,
            prices_chunk,
            calendar_arg,
            holdout=holdout,
            filter_method=filter_method,
            seasonal_lag=seasonal_lag,
            ray_debug=ray_debug,
        )
        futures.append(ref)
        submit_times[ref] = time.time()

    _print_progress(f"Submitted Ray M5 feature tasks: stores={len(store_ids)}, total={len(futures)}")
    pending = list(futures)
    completed = 0
    frames = []
    progress = trange(len(futures), desc="M5 store feature tasks", ncols=100)
    try:
        while pending:
            ready, pending = ray_module.wait(pending, num_returns=1, timeout=60.0)
            if not ready:
                now = time.time()
                slowest_wait = now - min(submit_times.values()) if submit_times else 0.0
                _print_progress(
                    "Ray M5 feature wait: "
                    f"done={completed}/{len(futures)}, pending={len(pending)}, "
                    f"slowest_pending_sec={slowest_wait:.1f}, available={ray_module.available_resources()}"
                )
                continue

            for ref in ready:
                sid, frame = ray_module.get(ref)
                frames.append(frame)
                submit_times.pop(ref, None)
                completed += 1
                progress.update(1)
                if ray_debug:
                    _print_progress(f"Ray M5 feature collected: store_id={sid}, done={completed}/{len(futures)}")
    finally:
        progress.close()

    features = pd.concat(frames, ignore_index=True, sort=False)
    dummy_prefixes = tuple(f"{name}_" for name in M5_CATEGORICAL_FEATURES)
    dummy_cols = [col for col in features.columns if col.startswith(dummy_prefixes)]
    if dummy_cols:
        features.loc[:, dummy_cols] = features.loc[:, dummy_cols].fillna(0.0)

    series_order = {series_id: idx for idx, series_id in enumerate(sales["id"].astype(str).tolist())}
    features["_series_order"] = features["series_id"].astype(str).map(series_order)
    features = features.sort_values("_series_order").drop(columns="_series_order").reset_index(drop=True)
    features = clean_feature_frame(
        features,
        standardize_output=standardize_output,
        meta_columns=M5_ATTENTION_META_COLUMNS,
    )
    return _write_features_and_metadata(
        features,
        sales_path=sales_path,
        prices_path=prices_path,
        calendar_path=calendar_path,
        output_path=output_path,
        store_id=store_id,
        holdout=holdout,
        max_series=max_series,
        filter_method=filter_method,
        seasonal_lag=seasonal_lag,
        standardize_output=standardize_output,
        ray_parallel=True,
        ray_num_cpus=ray_num_cpus,
        max_cpus_per_task=max_cpus_per_task,
        ray_debug=ray_debug,
    )


def _prepare_features(
    *,
    sales_path: Path,
    prices_path: Path,
    calendar_path: Path,
    output_path: Path,
    store_id: str | None,
    holdout: int,
    max_series: int,
    filter_method: str,
    seasonal_lag: int,
    standardize_output: bool,
    overwrite: bool,
    ray_num_cpus: int,
    max_cpus_per_task: int,
    ray_debug: bool,
) -> dict[str, Any]:
    if ray_num_cpus < 1:
        raise ValueError("ray_num_cpus must be >= 1.")
    if max_cpus_per_task < 1:
        raise ValueError("max_cpus_per_task must be >= 1.")
    use_ray = ray_num_cpus > 1
    if use_ray and max_cpus_per_task > ray_num_cpus:
        raise ValueError("max_cpus_per_task cannot exceed ray_num_cpus in Ray mode.")

    ray_module = None
    if use_ray:
        import ray as ray_module

    ray_started_here = False
    if use_ray and not ray_module.is_initialized():
        _print_progress(f"Starting Ray: ray_num_cpus={ray_num_cpus}, max_cpus_per_task={max_cpus_per_task}")
        ray_module.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=ray_num_cpus,
            _node_ip_address="127.0.0.1",
        )
        ray_started_here = True
        _print_progress("Ray started")
    if use_ray:
        available = ray_module.available_resources()
        available_cpus = float(available.get("CPU", 0.0))
        if available_cpus < float(max_cpus_per_task):
            raise RuntimeError(
                f"Ray has no available CPUs after initialization: {available}. "
                "Stop stale Ray workers with `ray stop --force` or reduce other Ray jobs."
            )

    try:
        if use_ray:
            return _make_features_with_ray(
                ray_module=ray_module,
                sales_path=sales_path,
                prices_path=prices_path,
                calendar_path=calendar_path,
                output_path=output_path,
                store_id=store_id,
                holdout=holdout,
                max_series=max_series,
                filter_method=filter_method,
                seasonal_lag=seasonal_lag,
                standardize_output=standardize_output,
                overwrite=overwrite,
                ray_num_cpus=ray_num_cpus,
                max_cpus_per_task=max_cpus_per_task,
                ray_debug=ray_debug,
            )
        return make_m5_attention_features_from_files(
            sales_path=sales_path,
            prices_path=prices_path,
            calendar_path=calendar_path,
            output_path=output_path,
            store_id=store_id,
            holdout=holdout,
            max_series=max_series,
            filter_method=filter_method,
            seasonal_lag=seasonal_lag,
            standardize_output=standardize_output,
            overwrite=overwrite,
        )
    finally:
        if ray_started_here and ray_module is not None:
            _print_progress("Shutting down Ray")
            ray_module.shutdown()
            _print_progress("Ray shut down")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare series-level M5 features for VARA attention.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--sales", type=Path, default=None)
    parser.add_argument("--prices", type=Path, default=None)
    parser.add_argument("--calendar", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--store-id", default=None, help="Optional store id filter; omit to prepare all stores.")
    parser.add_argument("--holdout", type=int, default=28, help="Last H sales days excluded from feature computation.")
    parser.add_argument("--filter-method", choices=["direct", "seasonal_naive", "auto_ets"], default="direct")
    parser.add_argument("--seasonal-lag", type=int, default=7, help="Seasonal lag used by residual filters.")
    parser.add_argument("--max-series", type=int, default=0, help="0 keeps all filtered series; positive values are useful for smoke tests.")
    parser.add_argument("--standardize-output", action="store_true", help="Standardize numeric features before saving.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--ray-num-cpus", type=int, default=1)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    parser.add_argument("--ray-debug", action="store_true")
    args = parser.parse_args()

    metadata = _prepare_features(
        sales_path=args.sales or _find_table(args.input_dir, "sales_train_evaluation"),
        prices_path=args.prices or _find_table(args.input_dir, "sell_prices"),
        calendar_path=args.calendar or _find_table(args.input_dir, "calendar"),
        output_path=args.out,
        store_id=args.store_id or None,
        holdout=args.holdout,
        max_series=args.max_series,
        filter_method=args.filter_method,
        seasonal_lag=args.seasonal_lag,
        standardize_output=args.standardize_output,
        overwrite=args.overwrite,
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
        ray_debug=args.ray_debug,
    )
    print("M5 attention feature preparation complete:")
    print(f"  output: {metadata['output_path']}")
    print(f"  metadata: {metadata['metadata_path']}")
    print(f"  n_series: {metadata['n_series']}")
    print(f"  n_columns: {metadata['n_columns']}")
    print(f"  holdout: {metadata['holdout']}")
    print(f"  filter_method: {metadata['filter_method']}")
    print(f"  seasonal_lag: {metadata['seasonal_lag']}")
    print(f"  store_id: {metadata['store_id'] or 'all stores'}")
    print(f"  ray_parallel: {metadata.get('ray_parallel', False)}")


if __name__ == "__main__":
    main()
