#!/usr/bin/env python3
"""Run the staged Accounting-Signal Risk VARA forecasting workflow."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

import _bootstrap  # noqa: F401

from vara.datamine import (
    DataMinePanel,
    PanelSplit,
    SERIES_SELECTION_MODES,
    audit_return_file,
    build_dense_panel,
    canonical_return_chunk,
    correction_diagnostics,
    dynamic_feature_blocks,
    economic_category_block,
    estimate_global_lambda,
    evaluate_raw_return_forecasts,
    evaluate_risk_forecasts,
    fixed_garch11_grid_forecast,
    formula_feature_blocks,
    load_panel,
    load_signal_metadata,
    make_target,
    preprocess_feature_blocks,
    relative_panel_split,
    rolling_mean_forecast,
    rolling_shrinkage_mean_forecast,
    rolling_vara_corrections,
    save_panel,
    select_frozen_universe,
    select_ridge_forecast,
    sha256_file,
    variance_from_log_risk,
    write_canonical_long_parquet,
    write_json,
)
from vara.forecast_significance import diebold_mariano_test


STAGES = (
    "audit",
    "prepare",
    "baseline",
    "features",
    "validate",
    "forecast",
    "evaluate",
    "run",
)
FEATURE_SETS = ("formula", "dynamics", "formula_dynamic", "full")
FORMULA_ENCODINGS = ("svd", "role_embedding")


@dataclass
class WorkflowConfig:
    data_dir: Path = Path("data/DataMine")
    return_file: str = "DataMinedLongShortReturnsEW_21800.parquet"
    metadata_file: str = "DataMinedSignalList.parquet"
    output_dir: Path = Path("outputs/datamine_vara/ew_log_risk_pilot")
    weighting: str = "ew"
    return_unit: str = "percent"
    download_date: str | None = None
    chunksize: int = 500_000
    write_long_parquet: bool = True
    max_series: int = 2_000
    seed: int = 202_407
    series_selection_mode: str = "stratified_random"

    validation_months: int = 60
    test_months: int = 120
    universe_window_months: int = 240
    min_observed_fraction: float = 0.99
    min_median_nstock: float = 10.0
    min_nstock: float = 5.0
    min_nstock_fraction: float = 0.99
    max_missing_run: int = 1

    target_type: str = "log_squared_return"
    epsilon: float = 1e-6
    minimum_baseline_observations: int = 120
    baseline_response_normalization: str = "none"
    baseline_ridge_candidates: list[float] = field(
        default_factory=lambda: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]
    )
    benchmark_ar1_ridge_candidates: list[float] = field(default_factory=lambda: [1.0])
    garch_alpha_candidates: list[float] = field(
        default_factory=lambda: [0.03, 0.05, 0.08, 0.12, 0.18, 0.25]
    )
    garch_beta_candidates: list[float] = field(
        default_factory=lambda: [0.50, 0.65, 0.75, 0.85, 0.90, 0.94]
    )

    error_window_months: int = 240
    feature_window_months: int = 120
    formula_svd_dim: int = 32
    formula_svd_drop_components: int = 0
    formula_encoding: str = "svd"
    formula_role_embedding_dim: int = 32
    risk_factor_dim: int = 8
    predictive_factor_dim: int = 8
    feature_set: str = "full"
    ingredient_dictionary: Path | None = None
    feature_block_weights: dict[str, float] = field(default_factory=dict)
    row_normalize_features: bool = True

    response_clip: float | None = 8.0
    refit_frequency_months: int = 12
    grid_size: int = 7
    vara_ridge: float = 1e-3
    eta_bounds: list[float] = field(default_factory=lambda: [-2.0, 2.0])
    attention_mode: str = "auto"
    attention_top_k: int = 50
    attention_cache_mode: str = "cluster"
    attention_batch_size: int = 512
    dense_threshold: int = 2_500
    top_neighbors: int = 10

    lambda_window_months: int = 0
    lambda_ridge: float = 1e-3
    variance_floor: float = 1e-8
    specification_id: str = "datamine_ew_logrisk_har_ooserror_full_v1"

    @property
    def return_path(self) -> Path:
        path = Path(self.return_file)
        return path if path.is_absolute() else self.data_dir / path

    @property
    def metadata_path(self) -> Path:
        path = Path(self.metadata_file)
        return path if path.is_absolute() else self.data_dir / path

    @property
    def processed_dir(self) -> Path:
        return self.output_dir / "processed"

    @property
    def baseline_dir(self) -> Path:
        return self.output_dir / "baseline"

    @property
    def features_dir(self) -> Path:
        return self.output_dir / "features"

    @property
    def forecasts_dir(self) -> Path:
        return self.output_dir / "forecasts"

    @property
    def metrics_dir(self) -> Path:
        return self.output_dir / "metrics"


def _print(message: str) -> None:
    print(f"[DataMine] {message}", flush=True)


def _parse_float_list(value: str) -> list[float]:
    values = [float(item.strip()) for item in str(value).split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected a comma-separated list of floats.")
    return values


def _parse_feature_block_weight(value: str) -> tuple[str, float]:
    name, separator, raw_weight = str(value).partition("=")
    name = name.strip()
    if not separator or not name:
        raise argparse.ArgumentTypeError("Expected BLOCK=WEIGHT.")
    try:
        weight = float(raw_weight)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Feature block weight must be numeric.") from exc
    if not np.isfinite(weight) or weight < 0.0:
        raise argparse.ArgumentTypeError("Feature block weight must be finite and nonnegative.")
    return name, weight


def _config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument("--config", type=Path, default=Path("data/DataMine/primary.yml"))
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--return-file")
    parser.add_argument("--metadata-file")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weighting", choices=("ew", "vw"))
    parser.add_argument("--return-unit", choices=("percent", "decimal"))
    parser.add_argument("--download-date")
    parser.add_argument("--chunksize", type=int)
    parser.add_argument("--write-long-parquet", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-series", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--series-selection-mode", choices=sorted(SERIES_SELECTION_MODES))
    parser.add_argument("--validation-months", type=int)
    parser.add_argument("--test-months", type=int)
    parser.add_argument("--universe-window-months", type=int)
    parser.add_argument("--min-observed-fraction", type=float)
    parser.add_argument("--min-median-nstock", type=float)
    parser.add_argument("--min-nstock", type=float)
    parser.add_argument("--min-nstock-fraction", type=float)
    parser.add_argument("--max-missing-run", type=int)
    parser.add_argument("--target-type", choices=("log_squared_return", "raw_return"))
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--minimum-baseline-observations", type=int)
    parser.add_argument(
        "--baseline-response-normalization",
        choices=("none", "training_zscore"),
    )
    parser.add_argument("--baseline-ridge-candidates", type=_parse_float_list)
    parser.add_argument("--benchmark-ar1-ridge-candidates", type=_parse_float_list)
    parser.add_argument("--garch-alpha-candidates", type=_parse_float_list)
    parser.add_argument("--garch-beta-candidates", type=_parse_float_list)
    parser.add_argument("--error-window-months", type=int)
    parser.add_argument("--feature-window-months", type=int)
    parser.add_argument("--formula-svd-dim", type=int)
    parser.add_argument("--formula-svd-drop-components", type=int)
    parser.add_argument("--formula-encoding", choices=FORMULA_ENCODINGS)
    parser.add_argument("--formula-role-embedding-dim", type=int)
    parser.add_argument("--risk-factor-dim", type=int)
    parser.add_argument("--predictive-factor-dim", type=int)
    parser.add_argument("--feature-set", choices=FEATURE_SETS)
    parser.add_argument(
        "--feature-block-weight",
        action="append",
        type=_parse_feature_block_weight,
        default=None,
        metavar="BLOCK=WEIGHT",
    )
    parser.add_argument("--ingredient-dictionary", type=Path)
    parser.add_argument("--row-normalize-features", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--response-clip", type=float)
    parser.add_argument("--refit-frequency-months", type=int)
    parser.add_argument("--grid-size", type=int)
    parser.add_argument("--vara-ridge", type=float)
    parser.add_argument("--eta-bounds", type=_parse_float_list)
    parser.add_argument("--attention-mode", choices=("auto", "dense", "topk", "uniform"))
    parser.add_argument("--attention-top-k", type=int)
    parser.add_argument("--attention-cache-mode", choices=("exact", "cluster"))
    parser.add_argument("--attention-batch-size", type=int)
    parser.add_argument("--dense-threshold", type=int)
    parser.add_argument("--top-neighbors", type=int)
    parser.add_argument("--lambda-window-months", type=int)
    parser.add_argument("--lambda-ridge", type=float)
    parser.add_argument("--variance-floor", type=float)
    parser.add_argument("--specification-id")
    return parser


def _load_config(args: argparse.Namespace) -> WorkflowConfig:
    defaults = asdict(WorkflowConfig())
    if args.config.exists():
        loaded = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        unknown = sorted(set(loaded).difference(defaults))
        if unknown:
            raise ValueError(f"Unknown configuration keys: {unknown}")
        defaults.update(loaded)
    elif args.config != Path("data/DataMine/primary.yml"):
        raise FileNotFoundError(args.config)
    feature_weight_overrides = args.feature_block_weight
    for key, value in vars(args).items():
        if key in {"stage", "config", "feature_block_weight"} or value is None:
            continue
        defaults[key] = value
    if feature_weight_overrides:
        block_weights = dict(defaults["feature_block_weights"])
        for block_name, weight in feature_weight_overrides:
            if block_name not in block_weights:
                raise ValueError(f"Unknown feature block: {block_name}")
            block_weights[block_name] = float(weight)
        defaults["feature_block_weights"] = block_weights
    for key in ("data_dir", "output_dir"):
        defaults[key] = Path(defaults[key])
    if defaults.get("ingredient_dictionary") not in (None, ""):
        defaults["ingredient_dictionary"] = Path(defaults["ingredient_dictionary"])
    else:
        defaults["ingredient_dictionary"] = None
    for key in (
        "baseline_ridge_candidates",
        "benchmark_ar1_ridge_candidates",
        "garch_alpha_candidates",
        "garch_beta_candidates",
    ):
        defaults[key] = [float(value) for value in defaults[key]]
    defaults["eta_bounds"] = [float(value) for value in defaults["eta_bounds"]]
    if len(defaults["eta_bounds"]) != 2 or defaults["eta_bounds"][0] >= defaults["eta_bounds"][1]:
        raise ValueError("eta_bounds must contain two increasing values.")
    config = WorkflowConfig(**defaults)
    if config.feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}.")
    if config.attention_mode not in {"auto", "dense", "topk", "uniform"}:
        raise ValueError("attention_mode must be auto, dense, topk, or uniform.")
    if config.series_selection_mode not in SERIES_SELECTION_MODES:
        raise ValueError(
            f"series_selection_mode must be one of {sorted(SERIES_SELECTION_MODES)}."
        )
    if config.lambda_window_months < 0:
        raise ValueError("lambda_window_months must be nonnegative.")
    if config.lambda_window_months > config.validation_months:
        raise ValueError("lambda_window_months cannot exceed validation_months.")
    if config.formula_svd_drop_components < 0:
        raise ValueError("formula_svd_drop_components must be nonnegative.")
    if config.formula_encoding not in FORMULA_ENCODINGS:
        raise ValueError(f"formula_encoding must be one of {FORMULA_ENCODINGS}.")
    if config.formula_role_embedding_dim < 2:
        raise ValueError("formula_role_embedding_dim must be at least 2.")
    if config.baseline_response_normalization not in {"none", "training_zscore"}:
        raise ValueError(
            "baseline_response_normalization must be none or training_zscore."
        )
    return config


def _config_payload(config: WorkflowConfig) -> dict[str, Any]:
    payload = asdict(config)
    for key in ("data_dir", "output_dir", "ingredient_dictionary"):
        if payload[key] is not None:
            payload[key] = Path(payload[key]).as_posix()
    payload["return_path"] = config.return_path.as_posix()
    payload["metadata_path"] = config.metadata_path.as_posix()
    return payload


def _audit_path(config: WorkflowConfig) -> Path:
    return config.output_dir / "audit" / "audit.json"


def _load_split(config: WorkflowConfig, panel: DataMinePanel) -> PanelSplit:
    return relative_panel_split(
        panel.dates,
        validation_months=config.validation_months,
        test_months=config.test_months,
    )


def _require(path: Path, stage: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing prerequisite {path}. Run the {stage!r} stage first.")


def stage_audit(config: WorkflowConfig) -> dict[str, Any]:
    _print(f"Auditing {config.return_path}")
    if not config.return_path.exists():
        raise FileNotFoundError(config.return_path)
    if not config.metadata_path.exists():
        raise FileNotFoundError(config.metadata_path)
    metadata = load_signal_metadata(config.metadata_path, weighting=config.weighting)
    audit = audit_return_file(
        config.return_path,
        metadata,
        return_unit=config.return_unit,
        chunksize=config.chunksize,
    )
    audit.update({
        "weighting": config.weighting,
        "metadata_file": config.metadata_path.as_posix(),
        "metadata_rows": len(metadata),
        "metadata_formula_types": metadata["formula_type"].value_counts().sort_index().to_dict(),
        "schema_adaptations": {
            "metadata_filename": config.metadata_path.name,
            "metadata_columns": {
                "signal_form": "formula_type",
                "v1": "numerator",
                "v2": "denominator",
            },
            "coverage": audit["coverage_source"],
        },
        "audit_created_utc": datetime.now(timezone.utc).isoformat(),
    })
    write_json(_audit_path(config), audit)

    source_files = sorted(config.data_dir.glob("DataMined*.csv"))
    source_files.extend(sorted(config.data_dir.glob("DataMined*.parquet")))
    source_files.extend(sorted(config.data_dir.glob("*.pdf")))
    inventory = []
    for path in source_files:
        _print(f"Hashing {path.name}")
        inventory.append({
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "local_mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        })
    manifest = {
        "source_repository": "https://github.com/chenandrewy/flex-mining",
        "source_data_folder": "https://drive.google.com/drive/folders/1SZe_aF4ZNvK4ZRx2jQUE1j19KQvBaqWr",
        "associated_paper": "https://arxiv.org/abs/2212.10317",
        "download_date": config.download_date,
        "local_inventory_date": date.today().isoformat(),
        "files": inventory,
        "processing": {
            "source_return_unit": config.return_unit,
            "return_unit_after_processing": "decimal",
            "primary_weighting": config.weighting,
            "selected_return_file": config.return_path.name,
        },
        "schema_adaptations": audit["schema_adaptations"],
        "data_use_warning": (
            "Strategy returns are public research outputs; underlying CRSP and Compustat "
            "security-level records are proprietary and are not redistributed here."
        ),
    }
    manifest_output = config.output_dir / "audit" / "data_manifest.yml"
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    if "sample" not in config.return_path.name.lower():
        (config.data_dir / "data_manifest.yml").write_text(
            yaml.safe_dump(manifest, sort_keys=False),
            encoding="utf-8",
        )
    write_json(config.output_dir / "config_effective.json", _config_payload(config))
    _print(f"Audit complete: {audit['rows']:,} rows, {audit['signals']:,} signals")
    return audit


def _ingredient_dictionary_template(metadata: pd.DataFrame) -> pd.DataFrame:
    ingredients = sorted(set(metadata["numerator"]).union(metadata["denominator"]))
    return pd.DataFrame({
        "ingredient": ingredients,
        "statement_family": "",
        "economic_family": "",
        "measurement_type": "",
        "sign_convention": "",
        "scale_role": "",
        "accrual_cash": "",
        "operating_financing": "",
    })


def stage_prepare(config: WorkflowConfig) -> DataMinePanel:
    _require(_audit_path(config), "audit")
    audit = json.loads(_audit_path(config).read_text(encoding="utf-8"))
    metadata = load_signal_metadata(config.metadata_path, weighting=config.weighting)
    _print("Building dense monthly matrices")
    full_panel = build_dense_panel(
        config.return_path,
        metadata,
        audit,
        weighting=config.weighting,
        return_unit=config.return_unit,
        chunksize=config.chunksize,
    )
    split = _load_split(config, full_panel)
    panel, diagnostics = select_frozen_universe(
        full_panel,
        split,
        universe_window_months=config.universe_window_months,
        min_observed_fraction=config.min_observed_fraction,
        min_median_nstock=config.min_median_nstock,
        min_nstock=config.min_nstock,
        min_nstock_fraction=config.min_nstock_fraction,
        max_missing_run=config.max_missing_run,
        max_series=config.max_series,
        seed=config.seed,
        selection_mode=config.series_selection_mode,
    )
    save_panel(panel, config.processed_dir)
    diagnostics.to_parquet(config.processed_dir / "universe_diagnostics.parquet", index=False)
    _ingredient_dictionary_template(panel.metadata).to_csv(
        config.processed_dir / "ingredient_dictionary_template.csv",
        index=False,
    )
    validation_observed = np.isfinite(panel.returns[split.validation_indices])
    test_observed = np.isfinite(panel.returns[split.test_indices])
    manifest = {
        "weighting": panel.weighting,
        "coverage_source": panel.coverage_source,
        "return_unit": "decimal",
        "source_return_file": config.return_path.as_posix(),
        "dates": {
            "start": panel.dates[0].date().isoformat(),
            "end": panel.dates[-1].date().isoformat(),
            "months": len(panel.dates),
        },
        "split": {
            "universe_selection_end": split.universe_selection_end.date().isoformat(),
            "validation_start": split.validation_start.date().isoformat(),
            "validation_end": split.validation_end.date().isoformat(),
            "test_start": split.test_start.date().isoformat(),
            "test_end": split.test_end.date().isoformat(),
        },
        "universe": {
            "eligible_components": int(diagnostics["eligible"].sum()),
            "selected_components": len(panel.component_ids),
            "max_series": config.max_series,
            "selection_mode": config.series_selection_mode,
            "selection_uses_validation_or_test": False,
            "selection_window_months": config.universe_window_months,
            "criteria": {
                "min_observed_fraction": config.min_observed_fraction,
                "min_median_nstock": config.min_median_nstock,
                "min_nstock": config.min_nstock,
                "min_nstock_fraction": config.min_nstock_fraction,
                "max_missing_run": config.max_missing_run,
            },
        },
        "post_selection_diagnostics_not_used_for_selection": {
            "validation_observed_fraction": float(validation_observed.mean()),
            "test_observed_fraction": float(test_observed.mean()),
            "validation_missing_cells": int((~validation_observed).sum()),
            "test_missing_cells": int((~test_observed).sum()),
        },
    }
    write_json(config.processed_dir / "panel_manifest.json", manifest)
    if config.write_long_parquet:
        _print("Writing canonical long Parquet table")
        write_canonical_long_parquet(
            config.return_path,
            metadata,
            config.output_dir / "interim" / "returns_long.parquet",
            weighting=config.weighting,
            return_unit=config.return_unit,
            chunksize=min(config.chunksize, 250_000),
        )
    _print(
        f"Prepared {len(panel.component_ids):,} frozen components; "
        f"validation {split.validation_start.date()} to {split.validation_end.date()}, "
        f"test {split.test_start.date()} to {split.test_end.date()}"
    )
    return panel


def _validation_loss_rows(
    target: np.ndarray,
    forecast: np.ndarray,
    validation_indices: np.ndarray,
    name: str,
) -> dict[str, Any]:
    error = np.asarray(forecast)[validation_indices] - np.asarray(target)[validation_indices]
    finite = np.isfinite(error)
    return {
        "baseline_type": name,
        "ridge": np.nan,
        "validation_cells": int(finite.sum()),
        "validation_rmse": float(np.sqrt(np.mean(np.square(error[finite])))),
        "validation_mae": float(np.mean(np.abs(error[finite]))),
        "selected": False,
    }


def stage_baseline(config: WorkflowConfig) -> dict[str, Any]:
    _require(config.processed_dir / "panel_manifest.json", "prepare")
    panel = load_panel(config.processed_dir)
    split = _load_split(config, panel)
    target = make_target(panel.returns, target_type=config.target_type, epsilon=config.epsilon)
    baseline_type = "har_ridge" if config.target_type == "log_squared_return" else "ar1_ridge"
    _print(f"Selecting {baseline_type} penalty from {config.baseline_ridge_candidates}")
    selected_ridge, baseline_forecast, selection = select_ridge_forecast(
        target,
        split.validation_indices,
        baseline_type=baseline_type,
        ridge_candidates=config.baseline_ridge_candidates,
        minimum_observations=config.minimum_baseline_observations,
        response_normalization=config.baseline_response_normalization,
    )
    benchmarks: dict[str, np.ndarray] = {}
    benchmark_manifest: dict[str, Any] = {}
    garch_parameters: pd.DataFrame | None = None
    if config.target_type == "log_squared_return":
        last = np.full_like(target, np.nan)
        last[1:] = target[:-1]
        ar1_ridge, ar1_forecast, _ = select_ridge_forecast(
            target,
            split.validation_indices,
            baseline_type="ar1_ridge",
            ridge_candidates=config.benchmark_ar1_ridge_candidates,
            minimum_observations=config.minimum_baseline_observations,
            response_normalization=config.baseline_response_normalization,
        )
        garch_variance, garch_parameters = fixed_garch11_grid_forecast(
            panel.returns,
            estimation_end_index=int(split.validation_indices[0]) - 1,
            alpha_candidates=config.garch_alpha_candidates,
            beta_candidates=config.garch_beta_candidates,
            variance_floor=config.variance_floor,
        )
        garch_forecast = np.log(
            float(config.epsilon) + np.maximum(garch_variance, config.variance_floor)
        )
        benchmarks = {
            "ar1_forecast": ar1_forecast,
            "garch11_forecast": garch_forecast,
            "last_observation_forecast": last,
            "mean_3_forecast": rolling_mean_forecast(target, 3),
            "mean_12_forecast": rolling_mean_forecast(target, 12),
        }
        benchmark_manifest = {
            "ar1_ridge": ar1_ridge,
            "garch_parameter_estimation_end": panel.dates[
                int(split.validation_indices[0]) - 1
            ].date().isoformat(),
            "garch_alpha_candidates": config.garch_alpha_candidates,
            "garch_beta_candidates": config.garch_beta_candidates,
            "garch_parameters_frozen_during_validation_and_test": True,
        }
    else:
        zero = np.zeros_like(target)
        shrink_candidates: list[tuple[float, int, float, np.ndarray]] = []
        for window in (60, 120, 240):
            for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
                forecast = rolling_shrinkage_mean_forecast(target, window=window, weight=weight)
                error = forecast[split.validation_indices] - target[split.validation_indices]
                finite = np.isfinite(error)
                mse = float(np.mean(np.square(error[finite]))) if finite.any() else np.inf
                shrink_candidates.append((mse, window, weight, forecast))
        _, selected_window, selected_weight, shrink = min(shrink_candidates, key=lambda row: row[0])
        benchmarks = {
            "zero_forecast": zero,
            "shrinkage_mean_forecast": shrink,
        }
        write_json(config.baseline_dir / "shrinkage_mean_selection.json", {
            "window": selected_window,
            "weight": selected_weight,
        })

    config.baseline_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.baseline_dir / "target.npy", target.astype(np.float32), allow_pickle=False)
    np.save(config.baseline_dir / "baseline_forecast.npy", baseline_forecast.astype(np.float32), allow_pickle=False)
    baseline_error = target - baseline_forecast
    np.save(config.baseline_dir / "baseline_error.npy", baseline_error.astype(np.float32), allow_pickle=False)
    for name, values in benchmarks.items():
        np.save(config.baseline_dir / f"{name}.npy", values.astype(np.float32), allow_pickle=False)
        selection = pd.concat([
            selection,
            pd.DataFrame([_validation_loss_rows(target, values, split.validation_indices, name)]),
        ], ignore_index=True)
    if garch_parameters is not None:
        garch_parameters.insert(0, "component_id", panel.component_ids)
        garch_parameters.insert(1, "signalid", panel.signal_ids)
        garch_parameters.to_parquet(
            config.baseline_dir / "garch11_parameters.parquet", index=False
        )
        np.save(
            config.baseline_dir / "garch11_variance_forecast.npy",
            garch_variance.astype(np.float32),
            allow_pickle=False,
        )
    selection.to_csv(config.baseline_dir / "baseline_selection.csv", index=False)
    manifest = {
        "target_type": config.target_type,
        "epsilon": config.epsilon,
        "baseline_type": baseline_type,
        "response_normalization": config.baseline_response_normalization,
        "selected_ridge": selected_ridge,
        "minimum_baseline_observations": config.minimum_baseline_observations,
        "genuine_one_step_errors": True,
        "first_forecast_index": int(np.flatnonzero(np.isfinite(baseline_forecast).any(axis=1))[0]),
        "validation_forecast_coverage": float(np.isfinite(baseline_forecast[split.validation_indices]).mean()),
        "test_forecast_coverage": float(np.isfinite(baseline_forecast[split.test_indices]).mean()),
        "benchmarks": benchmark_manifest,
    }
    write_json(config.baseline_dir / "baseline_manifest.json", manifest)
    _print(f"Selected ridge={selected_ridge:g}; genuine baseline errors saved")
    return manifest


def _load_baseline_arrays(config: WorkflowConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _require(config.baseline_dir / "baseline_manifest.json", "baseline")
    target = np.load(config.baseline_dir / "target.npy", mmap_mode="r", allow_pickle=False)
    forecast = np.load(config.baseline_dir / "baseline_forecast.npy", mmap_mode="r", allow_pickle=False)
    errors = np.load(config.baseline_dir / "baseline_error.npy", mmap_mode="r", allow_pickle=False)
    return target, forecast, errors


def _resolve_ingredient_dictionary(config: WorkflowConfig) -> pd.DataFrame | None:
    if config.ingredient_dictionary is None:
        return None
    path = config.ingredient_dictionary
    if not path.is_absolute() and not path.exists():
        path = config.data_dir / path
    if not path.exists():
        raise FileNotFoundError(path)
    dictionary = pd.read_csv(path)
    populated = [column for column in dictionary.columns if column != "ingredient" and dictionary[column].notna().any()]
    if not populated:
        raise ValueError("Ingredient dictionary has no populated category columns.")
    return dictionary


def _feature_factory(
    config: WorkflowConfig,
    panel: DataMinePanel,
    target: np.ndarray,
    errors: np.ndarray,
):
    formula_blocks, formula_provenance = formula_feature_blocks(
        panel.metadata,
        svd_dim=config.formula_svd_dim,
        svd_drop_components=config.formula_svd_drop_components,
        encoding=config.formula_encoding,
        role_embedding_dim=config.formula_role_embedding_dim,
        seed=config.seed,
    )
    dictionary = _resolve_ingredient_dictionary(config)
    if dictionary is not None:
        values, names = economic_category_block(panel.metadata, dictionary)
        formula_blocks["economic_categories"] = (values, names)

    if config.feature_set == "formula":
        static_Z, feature_names, block_names, preprocessing = preprocess_feature_blocks(
            formula_blocks,
            block_weights=config.feature_block_weights,
            row_normalize=config.row_normalize_features,
        )

        def build_static(origin_index: int) -> tuple[np.ndarray, dict[str, Any]]:
            return static_Z, {
                "origin_index": int(origin_index),
                "feature_set": config.feature_set,
                "feature_names": feature_names,
                "block_names": block_names,
                "formula": formula_provenance,
                "dynamic": {"skipped": True, "reason": "formula-only feature set"},
                "preprocessing": preprocessing,
                "ingredient_dictionary": (
                    None
                    if config.ingredient_dictionary is None
                    else config.ingredient_dictionary.as_posix()
                ),
            }

        return build_static

    def build(origin_index: int) -> tuple[np.ndarray, dict[str, Any]]:
        dynamic_blocks, dynamic_provenance = dynamic_feature_blocks(
            returns=panel.returns,
            target=target,
            nstock=panel.nstock,
            baseline_errors=errors,
            origin_index=origin_index,
            feature_window_months=config.feature_window_months,
            risk_factor_dim=config.risk_factor_dim,
            predictive_factor_dim=config.predictive_factor_dim,
            seed=config.seed,
            coverage_threshold=config.min_nstock,
        )
        if config.feature_set == "dynamics":
            selected = dict(dynamic_blocks)
        elif config.feature_set == "formula_dynamic":
            selected = dict(formula_blocks)
            for name in ("return_risk_dynamics", "coverage", "risk_factor_loadings"):
                if name in dynamic_blocks:
                    selected[name] = dynamic_blocks[name]
        else:
            selected = dict(formula_blocks)
            selected.update(dynamic_blocks)
        Z, feature_names, block_names, preprocessing = preprocess_feature_blocks(
            selected,
            block_weights=config.feature_block_weights,
            row_normalize=config.row_normalize_features,
        )
        info = {
            "origin_index": int(origin_index),
            "feature_set": config.feature_set,
            "feature_names": feature_names,
            "block_names": block_names,
            "formula": formula_provenance,
            "dynamic": dynamic_provenance,
            "preprocessing": preprocessing,
            "ingredient_dictionary": (
                None if config.ingredient_dictionary is None else config.ingredient_dictionary.as_posix()
            ),
        }
        return Z, info

    return build


def _save_feature_snapshot(
    output_dir: Path,
    *,
    origin_index: int,
    origin_date: pd.Timestamp,
    Z: np.ndarray,
    info: Mapping[str, Any],
    component_ids: Sequence[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "Z.npy", np.asarray(Z, dtype=np.float32), allow_pickle=False)
    pd.DataFrame({
        "column_index": np.arange(Z.shape[1]),
        "feature_name": info["feature_names"],
        "feature_block": info["block_names"],
    }).to_csv(output_dir / "feature_columns.csv", index=False)
    pd.DataFrame({
        "row_index": np.arange(Z.shape[0]),
        "component_id": list(component_ids),
    }).to_csv(output_dir / "component_order.csv", index=False)
    payload = dict(info)
    payload.update({
        "origin_index": int(origin_index),
        "origin_date": origin_date.date().isoformat(),
        "shape": list(Z.shape),
    })
    write_json(output_dir / "feature_provenance.json", payload)


def stage_features(config: WorkflowConfig) -> dict[str, Any]:
    panel = load_panel(config.processed_dir)
    target, _, errors = _load_baseline_arrays(config)
    split = _load_split(config, panel)
    origin_index = int(split.validation_indices[0]) - 1
    _print(f"Building feature contract snapshot at {panel.dates[origin_index].date()}")
    build = _feature_factory(config, panel, target, errors)
    Z, info = build(origin_index)
    snapshot = config.features_dir / f"snapshot_{panel.dates[origin_index]:%Y%m}"
    _save_feature_snapshot(
        snapshot,
        origin_index=origin_index,
        origin_date=panel.dates[origin_index],
        Z=Z,
        info=info,
        component_ids=panel.component_ids,
    )
    _print(f"Feature snapshot saved: {Z.shape[0]:,} components x {Z.shape[1]:,} features")
    return {"shape": list(Z.shape), "origin_index": origin_index, "path": snapshot.as_posix()}


def _resolved_attention_mode(config: WorkflowConfig, p: int) -> str:
    if config.attention_mode == "auto":
        return "dense" if p <= config.dense_threshold else "topk"
    return config.attention_mode


def _write_forecast_table(
    config: WorkflowConfig,
    panel: DataMinePanel,
    split: PanelSplit,
    *,
    target: np.ndarray,
    baseline: np.ndarray,
    standardized_correction: np.ndarray,
    correction: np.ndarray,
    final: np.ndarray,
    lambda_value: float,
    refit_index: np.ndarray,
    evaluation_indices: Sequence[int],
) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    output = config.forecasts_dir / "forecasts.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    evaluation = np.asarray(evaluation_indices, dtype=np.int64)
    validation_set = set(np.asarray(split.validation_indices, dtype=np.int64).tolist())
    try:
        for target_index in evaluation:
            p = len(panel.component_ids)
            refit = int(refit_index[target_index])
            variance = (
                variance_from_log_risk(final[target_index], epsilon=config.epsilon, variance_floor=config.variance_floor)
                if config.target_type == "log_squared_return" else np.full(p, np.nan)
            )
            frame = pd.DataFrame({
                "origin": panel.dates[target_index - 1],
                "target_date": panel.dates[target_index],
                "component_id": panel.component_ids,
                "signalid": panel.signal_ids,
                "weighting": panel.weighting,
                "target_type": config.target_type,
                "actual_target": np.asarray(target[target_index], dtype=float),
                "actual_return": np.asarray(panel.returns[target_index], dtype=float),
                "baseline_forecast": np.asarray(baseline[target_index], dtype=float),
                "vara_standardized_correction": np.asarray(standardized_correction[target_index], dtype=float),
                "vara_correction": np.asarray(correction[target_index], dtype=float),
                "lambda": float(lambda_value),
                "final_forecast": np.asarray(final[target_index], dtype=float),
                "variance_forecast": variance,
                "vara_refit_date": panel.dates[refit] if refit >= 0 else pd.NaT,
                "feature_refit_date": panel.dates[refit] if refit >= 0 else pd.NaT,
                "error_window_months": config.error_window_months,
                "specification_id": config.specification_id,
                "evaluation_period": "validation" if int(target_index) in validation_set else "test",
            })
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()


def stage_forecast(
    config: WorkflowConfig,
    *,
    validation_only: bool = False,
) -> dict[str, Any]:
    panel = load_panel(config.processed_dir)
    target, baseline, errors = _load_baseline_arrays(config)
    split = _load_split(config, panel)
    feature_build = _feature_factory(config, panel, target, errors)
    evaluation = (
        np.asarray(split.validation_indices, dtype=np.int64)
        if validation_only
        else np.concatenate((split.validation_indices, split.test_indices))
    )
    attention_mode = _resolved_attention_mode(config, len(panel.component_ids))
    _print(
        f"Forecasting {len(evaluation)} rolling origins with {attention_mode} attention "
        f"and {config.refit_frequency_months}-month refits"
    )

    def save_refit_artifacts(**kwargs) -> None:
        origin_index = int(kwargs["origin_index"])
        refit_dir = config.features_dir / "refits" / f"origin_{panel.dates[origin_index]:%Y%m}"
        _save_feature_snapshot(
            refit_dir,
            origin_index=origin_index,
            origin_date=panel.dates[origin_index],
            Z=kwargs["feature_matrix"],
            info=kwargs["feature_info"],
            component_ids=panel.component_ids,
        )
        np.save(refit_dir / "response_center.npy", kwargs["scaler"].center.astype(np.float32), allow_pickle=False)
        np.save(refit_dir / "response_scale.npy", kwargs["scaler"].scale.astype(np.float32), allow_pickle=False)
        write_json(refit_dir / "fit_diagnostics.json", kwargs["diagnostics"])

    result = rolling_vara_corrections(
        baseline_errors=errors,
        evaluation_indices=evaluation,
        feature_builder=feature_build,
        component_ids=panel.component_ids,
        metadata=panel.metadata,
        error_window_months=config.error_window_months,
        response_clip=config.response_clip,
        refit_frequency_months=config.refit_frequency_months,
        grid_size=config.grid_size,
        ridge=config.vara_ridge,
        eta_bounds=(config.eta_bounds[0], config.eta_bounds[1]),
        attention_mode=attention_mode,
        attention_top_k=config.attention_top_k,
        attention_cache_mode=config.attention_cache_mode,
        batch_size=config.attention_batch_size,
        dense_threshold=config.dense_threshold,
        top_neighbors=config.top_neighbors,
        refit_artifact_callback=save_refit_artifacts,
    )
    lambda_indices = split.validation_indices
    if config.lambda_window_months > 0:
        lambda_indices = lambda_indices[-int(config.lambda_window_months) :]
    lambda_info = estimate_global_lambda(
        np.asarray(errors)[lambda_indices],
        result.correction[lambda_indices],
        ridge=config.lambda_ridge,
    )
    lambda_value = lambda_info["lambda"]
    final = np.asarray(baseline, dtype=float) + lambda_value * result.correction
    unshrunk = np.asarray(baseline, dtype=float) + result.correction
    config.forecasts_dir.mkdir(parents=True, exist_ok=True)
    np.save(config.forecasts_dir / "vara_standardized_correction.npy", result.standardized_forecast.astype(np.float32), allow_pickle=False)
    np.save(config.forecasts_dir / "vara_correction.npy", result.correction.astype(np.float32), allow_pickle=False)
    np.save(config.forecasts_dir / "vara_final_forecast.npy", final.astype(np.float32), allow_pickle=False)
    np.save(config.forecasts_dir / "vara_unshrunk_forecast.npy", unshrunk.astype(np.float32), allow_pickle=False)
    np.save(config.forecasts_dir / "vara_refit_index.npy", result.refit_index, allow_pickle=False)
    result.fit_summary.assign(
        refit_date=lambda frame: pd.to_datetime([panel.dates[int(idx)] for idx in frame["refit_index"]])
    ).to_csv(config.forecasts_dir / "vara_fit_summary.csv", index=False)
    result.parameter_summary.to_parquet(config.forecasts_dir / "parameter_summary.parquet", index=False)
    result.neighbors.to_parquet(config.forecasts_dir / "attention_neighbors.parquet", index=False)
    write_json(config.forecasts_dir / "lambda.json", lambda_info)
    run_contract = _config_payload(config)
    run_contract.update({
        "resolved_attention_mode": attention_mode,
        "lambda": lambda_value,
        "forecast_scope": "validation" if validation_only else "validation_and_test",
        "lambda_estimation_period": {
            "start": panel.dates[int(lambda_indices[0])].date().isoformat(),
            "end": panel.dates[int(lambda_indices[-1])].date().isoformat(),
            "months": int(len(lambda_indices)),
        },
    })
    if validation_only:
        write_json(config.forecasts_dir / "validation_candidate_config.json", run_contract)
    else:
        run_contract["test_configuration_locked_before_metric_evaluation"] = True
        write_json(config.forecasts_dir / "locked_test_config.json", run_contract)
    _write_forecast_table(
        config,
        panel,
        split,
        target=target,
        baseline=baseline,
        standardized_correction=result.standardized_forecast,
        correction=result.correction,
        final=final,
        lambda_value=lambda_value,
        refit_index=result.refit_index,
        evaluation_indices=evaluation,
    )
    scope = "validation" if validation_only else "validation and test"
    _print(f"VARA {scope} forecasts saved; validation lambda={lambda_value:.6f}")
    return {
        "lambda": lambda_value,
        "lambda_unbounded": lambda_info["lambda_unbounded"],
        "refits": len(result.fit_summary),
        "attention_mode": attention_mode,
    }


def _raw_dm_table(monthly: pd.DataFrame, focal_model: str, benchmark_model: str) -> pd.DataFrame:
    rows = []
    for loss_column in ("squared_error", "gaussian_return_crps"):
        if loss_column not in monthly.columns:
            continue
        focal = monthly.loc[monthly["model"].eq(focal_model), ["date", loss_column]].rename(
            columns={loss_column: "focal"}
        )
        benchmark = monthly.loc[monthly["model"].eq(benchmark_model), ["date", loss_column]].rename(
            columns={loss_column: "benchmark"}
        )
        matched = focal.merge(benchmark, on="date", how="inner").dropna()
        if len(matched) < 3:
            continue
        test = diebold_mariano_test(matched["focal"], matched["benchmark"], forecast_horizon=1)
        rows.append({
            "focal_model": focal_model,
            "comparator_model": benchmark_model,
            "loss": loss_column,
            **test,
        })
    return pd.DataFrame(rows)


def stage_evaluate(config: WorkflowConfig) -> pd.DataFrame:
    _require(config.forecasts_dir / "locked_test_config.json", "forecast")
    panel = load_panel(config.processed_dir)
    target, baseline, errors = _load_baseline_arrays(config)
    split = _load_split(config, panel)
    final = np.load(config.forecasts_dir / "vara_final_forecast.npy", mmap_mode="r", allow_pickle=False)
    unshrunk = np.load(config.forecasts_dir / "vara_unshrunk_forecast.npy", mmap_mode="r", allow_pickle=False)
    correction = np.load(config.forecasts_dir / "vara_correction.npy", mmap_mode="r", allow_pickle=False)
    forecasts = {
        "baseline": baseline,
        "vara_validation_shrunk": final,
        "vara_unshrunk": unshrunk,
    }
    benchmark_files = (
        ("ar1", "ar1_forecast.npy"),
        ("garch11", "garch11_forecast.npy"),
        ("last_observation", "last_observation_forecast.npy"),
        ("mean_3", "mean_3_forecast.npy"),
        ("mean_12", "mean_12_forecast.npy"),
        ("zero", "zero_forecast.npy"),
        ("shrinkage_mean", "shrinkage_mean_forecast.npy"),
    )
    for model_name, filename in benchmark_files:
        path = config.baseline_dir / filename
        if path.exists():
            forecasts[model_name] = np.load(path, mmap_mode="r", allow_pickle=False)
    config.metrics_dir.mkdir(parents=True, exist_ok=True)
    if config.target_type == "log_squared_return":
        metrics, monthly, calibrations = evaluate_risk_forecasts(
            target=target,
            returns=panel.returns,
            dates=panel.dates,
            forecasts=forecasts,
            validation_indices=split.validation_indices,
            test_indices=split.test_indices,
            benchmark_name="baseline",
            epsilon=config.epsilon,
            variance_floor=config.variance_floor,
        )
        write_json(config.metrics_dir / "variance_calibration.json", calibrations)
    else:
        if "zero" not in forecasts:
            forecasts = {"zero": np.zeros_like(np.asarray(target, dtype=float)), **forecasts}
        metrics, monthly = evaluate_raw_return_forecasts(
            returns=panel.returns,
            dates=panel.dates,
            forecasts=forecasts,
            test_indices=split.test_indices,
            benchmark_name="zero",
        )
    metrics.to_csv(config.metrics_dir / "metrics.csv", index=False)
    monthly.to_csv(config.metrics_dir / "monthly_loss_differentials.csv", index=False)
    benchmark_name = "baseline" if config.target_type == "log_squared_return" else "zero"
    dm_tables = [
        _raw_dm_table(monthly, "vara_validation_shrunk", comparator)
        for comparator in (benchmark_name, "ar1", "garch11")
        if comparator in forecasts
    ]
    dm = pd.concat(dm_tables, ignore_index=True) if dm_tables else pd.DataFrame()
    dm.to_csv(config.metrics_dir / "diebold_mariano_raw.csv", index=False)
    diagnostics = {
        "validation": correction_diagnostics(
            np.asarray(errors)[split.validation_indices],
            np.asarray(correction)[split.validation_indices],
        ),
        "test": correction_diagnostics(
            np.asarray(errors)[split.test_indices],
            np.asarray(correction)[split.test_indices],
        ),
    }
    write_json(config.metrics_dir / "correction_diagnostics.json", diagnostics)
    _print("Final test metrics")
    print(metrics.to_string(index=False), flush=True)
    return metrics


def run_stage(config: WorkflowConfig, stage: str) -> Any:
    if stage == "audit":
        return stage_audit(config)
    if stage == "prepare":
        return stage_prepare(config)
    if stage == "baseline":
        return stage_baseline(config)
    if stage == "features":
        return stage_features(config)
    if stage == "validate":
        return stage_forecast(config, validation_only=True)
    if stage == "forecast":
        return stage_forecast(config)
    if stage == "evaluate":
        return stage_evaluate(config)
    if stage == "run":
        result = None
        for sequential_stage in ("audit", "prepare", "baseline", "features", "forecast", "evaluate"):
            _print(f"=== Stage: {sequential_stage} ===")
            result = run_stage(config, sequential_stage)
        return result
    raise ValueError(f"Unknown stage: {stage}")


def main() -> None:
    parser = _config_parser()
    args = parser.parse_args()
    config = _load_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_stage(config, args.stage)


if __name__ == "__main__":
    main()
