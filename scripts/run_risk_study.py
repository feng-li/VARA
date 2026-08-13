#!/usr/bin/env python3
"""Run a cached base-plus-VARA study on a prepared risk panel."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

from vara.api import (
    AutoARIMAForecaster,
    AutoETSForecaster,
    CachedForecaster,
    ExecutionConfig,
    ForecastStudy,
    HARRidgeForecaster,
    NBEATSForecaster,
    NeuralModelConfig,
    PanelData,
    PaperEtaConfig,
    PrecomputedForecaster,
    RISK_BENCHMARK_MODELS,
    RISK_PAPER_COMPONENT_COUNT,
    RISK_PAPER_PROFILE,
    RISK_VARA_MODELS,
    RiskBenchmarkArtifacts,
    RiskForecastEvaluator,
    RollingSplit,
    StaticFeatureSet,
    VARAConfig,
    VARAResidualCorrector,
    risk_paper_execution_config,
    risk_paper_split,
    risk_paper_vara_config,
)
from vara.datamine import load_panel


DEFAULT_SOURCE_OUTPUT = Path(
    "outputs/datamine_vara/"
    "ew_log_risk_full_formula_topk500_exact_lambda24"
)
DEFAULT_PAPER_CONTRACT = Path(
    "outputs/datamine_vara/"
    "ew_log_risk_dense_attention_table10/"
    "paper/paper_contract.json"
)
DEFAULT_FEATURE_SNAPSHOT = DEFAULT_SOURCE_OUTPUT / "features" / "snapshot_200712"
PAPER_SPLIT = risk_paper_split()
PAPER_VARA = risk_paper_vara_config()
PAPER_EXECUTION = risk_paper_execution_config()
DEFAULT_MODELS = "paper_five"
PAPER_MODEL_ALIASES = {
    "paper_har_ridge": RISK_BENCHMARK_MODELS[0],
    "pooled_lightgbm": RISK_BENCHMARK_MODELS[1],
    "riskmetrics_ewma": RISK_BENCHMARK_MODELS[2],
    "garch11": RISK_BENCHMARK_MODELS[3],
    "garch11_t": RISK_BENCHMARK_MODELS[4],
}
MODEL_CHOICES = frozenset({
    "paper_five",
    "har_ridge",
    "auto_ets",
    "auto_arima",
    "nbeats",
    *PAPER_MODEL_ALIASES,
})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-output", type=Path, default=DEFAULT_SOURCE_OUTPUT)
    parser.add_argument("--feature-snapshot", type=Path)
    parser.add_argument("--paper-contract", type=Path, default=DEFAULT_PAPER_CONTRACT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/risk_study"))
    parser.add_argument(
        "--models",
        default=DEFAULT_MODELS,
        help=(
            "Comma-separated models: har_ridge, auto_ets, auto_arima, nbeats, "
            "paper_five, paper_har_ridge, pooled_lightgbm, riskmetrics_ewma, "
            "garch11, or garch11_t."
        ),
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=0,
        help=(
            "Deterministic diagnostic subset size; 0 (the default) uses the "
            "paper's complete 21,800-series universe."
        ),
    )
    parser.add_argument("--seed", type=int, default=202407)
    parser.add_argument(
        "--validation-months",
        type=int,
        default=PAPER_SPLIT.validation,
    )
    parser.add_argument("--test-months", type=int, default=PAPER_SPLIT.test)
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument(
        "--error-window-months",
        type=int,
        default=PAPER_VARA.error_window_months,
    )
    parser.add_argument(
        "--refit-frequency-months",
        type=int,
        default=PAPER_VARA.refit_frequency_months,
    )
    parser.add_argument(
        "--lambda-window-months",
        type=int,
        default=PAPER_VARA.lambda_window_months,
    )
    parser.add_argument(
        "--response-standardization",
        choices=("median_mad", "training_zscore"),
        default=PAPER_VARA.response_standardization,
        help=(
            "Componentwise normalization fitted on each rolling VARA residual "
            "window; median_mad matches the Risk application."
        ),
    )
    parser.add_argument(
        "--response-clip",
        type=float,
        default=PAPER_VARA.response_clip,
    )
    parser.add_argument("--vara-ridge", type=float, default=PAPER_VARA.ridge)
    parser.add_argument(
        "--lambda-ridge",
        type=float,
        default=PAPER_VARA.lambda_ridge,
    )
    parser.add_argument("--eta-lower", type=float, default=PAPER_VARA.eta.lower)
    parser.add_argument("--eta-upper", type=float, default=PAPER_VARA.eta.upper)
    parser.add_argument(
        "--eta-grid-size",
        type=int,
        default=PAPER_VARA.eta.grid_size,
    )
    parser.add_argument(
        "--eta-local-tolerance",
        type=float,
        default=PAPER_VARA.eta.local_tolerance,
    )
    parser.add_argument(
        "--attention-mode",
        choices=("dense", "topk"),
        default=PAPER_VARA.attention_mode,
    )
    parser.add_argument(
        "--attention-top-k",
        type=int,
        default=PAPER_VARA.attention_top_k,
    )
    parser.add_argument(
        "--attention-cache-mode",
        choices=("exact", "cluster"),
        default=PAPER_VARA.attention_cache_mode,
    )
    parser.add_argument(
        "--attention-batch-size",
        type=int,
        default=PAPER_VARA.attention_batch_size,
    )
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=PAPER_EXECUTION.ray_num_cpus,
    )
    parser.add_argument(
        "--max-cpus-per-task",
        type=int,
        default=PAPER_EXECUTION.max_cpus_per_task,
    )
    parser.add_argument("--refresh-base-cache", action="store_true")
    parser.add_argument("--nbeats-max-steps", type=int, default=200)
    return parser


def _model_names(raw_names: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in str(raw_names).split(",") if name.strip())
    if not names:
        raise ValueError("--models must contain at least one model.")
    unknown = sorted(set(names).difference(MODEL_CHOICES))
    if unknown:
        raise ValueError(f"unknown models: {', '.join(unknown)}.")
    if len(set(names)) != len(names):
        raise ValueError("--models cannot contain duplicates.")
    return names


def _component_indices(component_count: int, max_series: int, seed: int) -> np.ndarray:
    if int(max_series) < 0:
        raise ValueError("--max-series must be nonnegative.")
    if int(max_series) == 0 or int(max_series) >= int(component_count):
        return np.arange(component_count, dtype=np.int64)
    if int(max_series) < 2:
        raise ValueError("VARA requires at least two selected series.")
    generator = np.random.default_rng(int(seed))
    return np.sort(
        generator.choice(component_count, size=int(max_series), replace=False)
    ).astype(np.int64)


def _resolve_feature_snapshot(source_output: Path, requested: Path | None) -> Path:
    if requested is not None:
        return requested
    candidates = sorted((source_output / "features").glob("snapshot_*"))
    if len(candidates) != 1:
        raise ValueError(
            "--feature-snapshot is required unless source-output contains exactly "
            "one features/snapshot_* directory."
        )
    return candidates[0]


def _load_inputs(
    source_output: Path,
    feature_snapshot: Path,
    indices: np.ndarray,
) -> tuple[PanelData, StaticFeatureSet]:
    prepared = load_panel(source_output / "processed", mmap_mode="r")
    target = np.load(
        source_output / "baseline" / "target.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    feature_values = np.load(
        feature_snapshot / "Z.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    feature_order = pd.read_csv(
        feature_snapshot / "component_order.csv",
        dtype={"component_id": str},
    )
    if target.shape != prepared.returns.shape:
        raise ValueError("the saved target does not match the prepared panel shape.")
    if feature_values.shape[0] != target.shape[1]:
        raise ValueError("the feature snapshot does not match the source panel.")
    if feature_order["component_id"].tolist() != prepared.component_ids:
        raise ValueError("the feature snapshot component order differs from the panel.")

    columns = pd.read_csv(feature_snapshot / "feature_columns.csv")
    if columns["column_index"].tolist() != list(range(feature_values.shape[1])):
        raise ValueError("feature_columns.csv is not in feature-matrix order.")
    provenance_path = feature_snapshot / "feature_provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance = {
        **provenance,
        "source_output": source_output.as_posix(),
        "feature_snapshot": feature_snapshot.as_posix(),
        "selected_source_indices": indices.tolist(),
    }

    select_all = np.array_equal(
        indices,
        np.arange(target.shape[1], dtype=np.int64),
    )
    target_selected = target if select_all else np.asarray(target[:, indices])
    returns_selected = (
        prepared.returns
        if select_all
        else np.asarray(prepared.returns[:, indices])
    )
    features_selected = (
        feature_values
        if select_all
        else np.asarray(feature_values[indices])
    )
    metadata = prepared.metadata.iloc[indices].copy().reset_index(drop=True)
    data = PanelData(
        target=target_selected,
        dates=prepared.dates,
        series_ids=tuple(np.asarray(prepared.component_ids)[indices].tolist()),
        metadata=metadata,
        fields={"returns": returns_selected},
    )
    features = StaticFeatureSet(
        values=features_selected,
        names=tuple(columns["feature_name"].astype(str)),
        provenance=provenance,
    )
    return data, features


def _paper_forecasters(
    contract_path: Path,
    source_output: Path,
    data: PanelData,
    split: RollingSplit,
    component_indices: np.ndarray,
) -> dict[str, Any]:
    artifacts = RiskBenchmarkArtifacts.from_paper_contract(contract_path)
    artifacts.validate_primary_profile()
    contract_source = Path(str(artifacts.contract["source_output"]))
    if contract_source.resolve() != source_output.resolve():
        raise ValueError(
            "the paper contract source_output differs from --source-output; "
            "artifact component order is therefore not guaranteed."
        )
    models = artifacts.forecasters(
        data,
        split,
        component_indices=component_indices,
    )
    return {model.name: model for model in models}


def _build_models(
    names: Sequence[str],
    *,
    args: argparse.Namespace,
    data: PanelData,
    split: RollingSplit,
    component_indices: np.ndarray,
) -> list[Any]:
    selected: list[Any] = []
    paper_requested = "paper_five" in names or any(
        name in PAPER_MODEL_ALIASES for name in names
    )
    paper = (
        _paper_forecasters(
            args.paper_contract,
            args.source_output,
            data,
            split,
            component_indices,
        )
        if paper_requested
        else {}
    )
    for name in names:
        if name == "paper_five":
            selected.extend(paper[model_name] for model_name in RISK_BENCHMARK_MODELS)
        elif name in PAPER_MODEL_ALIASES:
            selected.append(paper[PAPER_MODEL_ALIASES[name]])
        elif name == "har_ridge":
            selected.append(HARRidgeForecaster(
                ridge_candidates=(1.0,),
                minimum_observations=120,
                response_normalization="none",
            ))
        elif name == "auto_ets":
            selected.append(AutoETSForecaster(
                season_length=args.season_length,
                error_history_months=args.error_window_months,
                response_normalization="training_zscore",
                origin_cache_dir=args.output_dir / "origin_cache",
                refresh_origin_cache=args.refresh_base_cache,
            ))
        elif name == "auto_arima":
            selected.append(AutoARIMAForecaster(
                season_length=args.season_length,
                error_history_months=args.error_window_months,
                origin_cache_dir=args.output_dir / "origin_cache",
                refresh_origin_cache=args.refresh_base_cache,
            ))
        elif name == "nbeats":
            selected.append(NBEATSForecaster(
                season_length=args.season_length,
                error_history_months=args.error_window_months,
                neural_config=NeuralModelConfig(max_steps=args.nbeats_max_steps),
                origin_cache_dir=args.output_dir / "origin_cache",
                refresh_origin_cache=args.refresh_base_cache,
            ))

    model_names = [model.name for model in selected]
    if len(model_names) != len(set(model_names)):
        raise ValueError(
            "the requested model set has duplicate fitted names; do not combine "
            "paper_five with a separately requested paper model."
        )
    return [
        model
        if isinstance(model, PrecomputedForecaster)
        else CachedForecaster(
            model=model,
            cache_dir=args.output_dir / "base_cache",
            refresh=args.refresh_base_cache,
        )
        for model in selected
    ]


def _write_run_config(
    path: Path,
    *,
    args: argparse.Namespace,
    model_names: Sequence[str],
    component_indices: np.ndarray,
    vara_config: VARAConfig,
    execution: ExecutionConfig,
) -> None:
    departures = _paper_profile_departures(
        model_names=model_names,
        component_indices=component_indices,
        validation_months=args.validation_months,
        test_months=args.test_months,
        vara_config=vara_config,
        source_output=args.source_output,
        feature_snapshot=args.feature_snapshot,
        paper_contract=args.paper_contract,
        execution=execution,
    )

    payload = {
        "schema": "vara_risk_study_run_v1",
        "application_profile": RISK_PAPER_PROFILE,
        "paper_consistent": not departures,
        "paper_departures": departures,
        "source_output": args.source_output.as_posix(),
        "feature_snapshot": args.feature_snapshot.as_posix(),
        "paper_contract": args.paper_contract.as_posix(),
        "models": list(model_names),
        "selected_component_count": int(len(component_indices)),
        "selected_source_indices": component_indices.tolist(),
        "seed": int(args.seed),
        "validation_months": int(args.validation_months),
        "test_months": int(args.test_months),
        "vara": asdict(vara_config),
        "execution": asdict(execution),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _paper_profile_departures(
    *,
    model_names: Sequence[str],
    component_indices: np.ndarray,
    validation_months: int,
    test_months: int,
    vara_config: VARAConfig,
    source_output: Path = DEFAULT_SOURCE_OUTPUT,
    feature_snapshot: Path = DEFAULT_FEATURE_SNAPSHOT,
    paper_contract: Path = DEFAULT_PAPER_CONTRACT,
    execution: ExecutionConfig = PAPER_EXECUTION,
) -> dict[str, dict[str, Any]]:
    """Return a machine-readable audit of deviations from the paper run."""
    indices = np.asarray(component_indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("component_indices must be one-dimensional.")
    departures: dict[str, dict[str, Any]] = {}

    def compare(name: str, actual: Any, expected: Any) -> None:
        if actual != expected:
            departures[name] = {"actual": actual, "paper": expected}

    compare("models", list(model_names), list(RISK_BENCHMARK_MODELS))
    compare(
        "source_output",
        Path(source_output).resolve().as_posix(),
        DEFAULT_SOURCE_OUTPUT.resolve().as_posix(),
    )
    compare(
        "paper_contract",
        Path(paper_contract).resolve().as_posix(),
        DEFAULT_PAPER_CONTRACT.resolve().as_posix(),
    )
    compare(
        "feature_snapshot",
        Path(feature_snapshot).resolve().as_posix(),
        DEFAULT_FEATURE_SNAPSHOT.resolve().as_posix(),
    )
    compare(
        "selected_component_count",
        int(len(indices)),
        RISK_PAPER_COMPONENT_COUNT,
    )
    expected_full_universe = (
        len(indices) == RISK_PAPER_COMPONENT_COUNT
        and np.array_equal(
            indices,
            np.arange(RISK_PAPER_COMPONENT_COUNT, dtype=np.int64),
        )
    )
    if not expected_full_universe:
        departures["selected_source_indices"] = {
            "actual": "diagnostic_subset",
            "paper": f"0:{RISK_PAPER_COMPONENT_COUNT}",
        }
    compare("validation_months", int(validation_months), PAPER_SPLIT.validation)
    compare("test_months", int(test_months), PAPER_SPLIT.test)
    compare("execution", asdict(execution), asdict(PAPER_EXECUTION))
    expected_vara = asdict(risk_paper_vara_config(len(indices)))
    actual_vara = asdict(vara_config)
    for setting, expected in expected_vara.items():
        compare(f"vara.{setting}", actual_vara[setting], expected)
    return departures


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    names = _model_names(args.models)
    args.feature_snapshot = _resolve_feature_snapshot(
        args.source_output,
        args.feature_snapshot,
    )
    source_panel = load_panel(args.source_output / "processed", mmap_mode="r")
    indices = _component_indices(
        len(source_panel.component_ids),
        args.max_series,
        args.seed,
    )
    data, features = _load_inputs(args.source_output, args.feature_snapshot, indices)
    split = RollingSplit(
        validation=args.validation_months,
        test=args.test_months,
    )
    split.plan(data)
    if data.component_count < 2:
        raise ValueError("VARA requires at least two selected series.")

    attention_top_k = min(args.attention_top_k, data.component_count - 1)
    attention_batch_size = min(args.attention_batch_size, data.component_count)
    vara_config = VARAConfig(
        error_window_months=args.error_window_months,
        response_standardization=args.response_standardization,
        response_clip=args.response_clip,
        refit_frequency_months=args.refit_frequency_months,
        ridge=args.vara_ridge,
        attention_top_k=attention_top_k,
        attention_cache_mode=args.attention_cache_mode,
        attention_batch_size=attention_batch_size,
        attention_mode=args.attention_mode,
        lambda_window_months=args.lambda_window_months,
        lambda_ridge=args.lambda_ridge,
        eta=PaperEtaConfig(
            lower=args.eta_lower,
            upper=args.eta_upper,
            grid_size=args.eta_grid_size,
            local_tolerance=args.eta_local_tolerance,
        ),
    )
    execution = ExecutionConfig(
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
    )
    models = _build_models(
        names,
        args=args,
        data=data,
        split=split,
        component_indices=indices,
    )
    study = ForecastStudy(
        data=data,
        split=split,
        features=features,
        vara=vara_config,
        evaluator=RiskForecastEvaluator(),
        corrector=VARAResidualCorrector(
            vara_config,
            model_names=RISK_VARA_MODELS,
        ),
    )

    print(
        f"Risk study profile={RISK_PAPER_PROFILE}: "
        f"series={data.component_count:,} models={len(models)} "
        f"attention={vara_config.attention_mode} "
        f"normalization={vara_config.response_standardization}",
        flush=True,
    )

    def progress(row: dict[str, Any]) -> None:
        refit = row.get("refit_index", "?")
        print(f"Completed VARA refit at target index {refit}", flush=True)

    result = study.run(
        models,
        execution=execution,
        progress_callback=progress,
    )
    output = result.save(args.output_dir)
    _write_run_config(
        output / "run_config.json",
        args=args,
        model_names=tuple(result.bases),
        component_indices=indices,
        vara_config=vara_config,
        execution=execution,
    )
    paper_columns = [
        "model",
        "log_risk_mspe",
        "absolute_r2",
        "rank_score",
        "overall_rank",
    ]
    print(result.metrics.loc[:, paper_columns].to_string(index=False), flush=True)
    print(f"Saved risk study to {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
