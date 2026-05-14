#!/usr/bin/env python3
"""Extract tsfeatures for each series in a wide CSV panel."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from vara.run_config import save_run_config, select_config_values
from vara.series_features import extract_series_tsfeatures_csv
from vara.stationarity import PANEL_TRANSFORMS

RUN_CONFIG_KEYS = (
    "input",
    "output",
    "date_col",
    "series_cols",
    "series_suffix",
    "series_depth",
    "holdout",
    "transform",
    "seasonal_lag",
    "freq",
    "features",
    "threads",
    "scale_series",
    "standardize_output",
    "clean",
    "config_out",
)


def _parse_series_cols(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _config_path_for_output(output: Path) -> Path:
    return output.with_name(f"config_series_features_{output.stem}.json")


def _load_config_json_defaults(path: str | Path) -> Dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    defaults = {key: config[key] for key in RUN_CONFIG_KEYS if key in config}
    for key in ("input", "output", "config_out"):
        if key in defaults and defaults[key] is not None:
            defaults[key] = Path(defaults[key])
    return defaults


def _run_kwargs_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "input_path": args.input,
        "output_path": args.output,
        "date_col": args.date_col,
        "series_cols": _parse_series_cols(args.series_cols),
        "series_suffix": args.series_suffix,
        "series_depth": args.series_depth,
        "holdout": args.holdout,
        "transform": args.transform,
        "seasonal_lag": args.seasonal_lag,
        "freq": args.freq,
        "features": args.features,
        "scale_series": args.scale_series,
        "standardize_output": args.standardize_output,
        "clean": args.clean,
        "threads": args.threads,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract tsfeatures from a wide time-series panel.")
    parser.add_argument(
        "--config-json",
        type=str,
        default=None,
        help="Load defaults from a saved config_series_features_*.json. CLI flags override JSON values.",
    )
    parser.add_argument("--input", type=Path, default=None, help="Wide CSV input path.")
    parser.add_argument("--output", type=Path, default=None, help="Output feature CSV path.")
    parser.add_argument("--date-col", type=str, default="date")
    parser.add_argument("--series-cols", type=str, default=None, help="Comma-separated series columns.")
    parser.add_argument("--series-suffix", type=str, default=None, help="Optional suffix filter, e.g. All.")
    parser.add_argument(
        "--series-depth",
        type=int,
        default=None,
        help="Optional hierarchy-code depth after removing suffix; use with --series-suffix.",
    )
    parser.add_argument("--holdout", type=int, default=0, help="Rows at the end excluded from feature extraction.")
    parser.add_argument("--transform", type=str, default="level", choices=PANEL_TRANSFORMS)
    parser.add_argument("--seasonal-lag", type=int, default=12)
    parser.add_argument("--freq", type=int, default=12, help="Seasonal frequency passed to tsfeatures.")
    parser.add_argument(
        "--features",
        type=str,
        default="default",
        help="Comma-separated tsfeatures function names, e.g. acf_features,stl_features; use default for package defaults.",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--scale-series",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable tsfeatures per-series scaling.",
    )
    parser.add_argument(
        "--standardize-output",
        action="store_true",
        help="Column-standardize numeric output features across series.",
    )
    parser.add_argument(
        "--clean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable replacing NaN/inf feature values.",
    )
    parser.add_argument(
        "--config-out",
        type=Path,
        default=None,
        help="Output JSON config path. Defaults to config_series_features_<output-stem>.json beside --output.",
    )

    config_args, _ = parser.parse_known_args()
    if config_args.config_json:
        parser.set_defaults(**_load_config_json_defaults(config_args.config_json))
    args = parser.parse_args()
    if args.input is None:
        parser.error("--input is required unless supplied by --config-json")
    if args.output is None:
        parser.error("--output is required unless supplied by --config-json")

    run_tag = datetime.now().strftime("%Y%m%d_%H.%M.%S")
    config_path = args.config_out or _config_path_for_output(args.output)
    features = extract_series_tsfeatures_csv(**_run_kwargs_from_args(args))
    feature_cols = [col for col in features.columns if col not in {"series_id", "transform", "n_obs", "train_start", "train_end"}]

    run_config = select_config_values(vars(args), RUN_CONFIG_KEYS)
    run_config.update({
        "command": " ".join(sys.argv),
        "config_tag": args.output.stem,
        "run_tag": run_tag,
        "output_path": str(args.output),
        "config_path": str(config_path),
        "n_series": int(len(features)),
        "n_feature_columns": int(len(feature_cols)),
        "feature_columns": list(feature_cols),
    })
    config_path.parent.mkdir(parents=True, exist_ok=True)
    save_run_config(config_path, run_config)

    print(f"Saved features: {args.output}")
    print(f"Saved config  : {config_path}")
    print(f"Series: {len(features)}")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Transform: {args.transform}")
    print(f"Holdout rows: {args.holdout}")


if __name__ == "__main__":
    main()
