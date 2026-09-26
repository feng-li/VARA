from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import vara.correction as correction_module
import vara.benchmark_models as benchmark_module
from vara.api import (
    AutoARIMAForecaster,
    AutoETSForecaster,
    BaseForecast,
    CachedForecaster,
    ExecutionConfig,
    ForecastStudy,
    HARRidgeForecaster,
    NBEATSForecaster,
    NeuralModelConfig,
    PanelData,
    PointForecastEvaluator,
    PrecomputedForecaster,
    RISK_BENCHMARK_MODELS,
    RISK_PAPER_COMPONENT_COUNT,
    RISK_PAPER_PROFILE,
    RISK_PAPER_TABLE_MODELS,
    RISK_VARA_MODELS,
    RiskBenchmarkArtifacts,
    RiskForecastEvaluator,
    RollingSplit,
    StaticFeatureSet,
    VARAForecast,
    VARAResidualCorrector,
    log_risk_target,
    risk_paper_split,
    risk_paper_vara_config,
    risk_paper_execution_config,
    risk_paper_corrector,
)
from vara.datamine_vara_runner import RollingVARAFamilyResult


def _risk_study_script():
    script_dir = Path(__file__).resolve().parents[1] / "scripts"
    sys.path.insert(0, str(script_dir))
    try:
        specification = importlib.util.spec_from_file_location(
            "_test_run_risk_study",
            script_dir / "run_risk_study.py",
        )
        assert specification is not None and specification.loader is not None
        module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(script_dir))


def _panel(*, periods: int = 8, components: int = 2) -> PanelData:
    values = np.arange(periods * components, dtype=float).reshape(periods, components)
    returns = 0.01 * (values + 1.0)
    return PanelData(
        target=values,
        dates=pd.date_range("2000-01-31", periods=periods, freq="ME"),
        series_ids=tuple(f"s{index}" for index in range(components)),
        metadata=pd.DataFrame({
            "series_id": [f"s{index}" for index in range(components)],
            "formula_type": ["group"] * components,
        }),
        fields={"returns": returns},
    )


def _features(components: int = 2) -> StaticFeatureSet:
    return StaticFeatureSet(
        values=np.arange(components, dtype=float)[:, None],
        names=("feature",),
        provenance={"source": "test"},
    )


def test_panel_data_and_trailing_split_validate_once() -> None:
    data = _panel()
    split = RollingSplit(validation=2, test=2)

    plan = split.plan(data)

    np.testing.assert_array_equal(plan.validation_indices, [4, 5])
    np.testing.assert_array_equal(plan.test_indices, [6, 7])
    assert data.shape == (8, 2)
    assert isinstance(data.fields, MappingProxyType)
    with pytest.raises(TypeError):
        data.fields["new"] = np.zeros(data.shape)  # type: ignore[index]
    with pytest.raises(ValueError, match="too short"):
        RollingSplit(validation=4, test=4).plan(data)


def test_point_evaluator_uses_one_common_mask_and_origin_win_rank() -> None:
    data = _panel()
    split = RollingSplit(validation=2, test=2)
    first = data.target.copy()
    second = data.target.copy()
    first[6:] += 1.0
    second[6] += 2.0
    second[7] += 0.5
    second[7, 1] = np.nan

    result = PointForecastEvaluator().evaluate(
        data,
        split,
        {"first": first, "second": second},
    )

    metrics = result.metrics.set_index("model")
    assert metrics.loc["first", "mse"] == pytest.approx(1.0)
    assert metrics.loc["second", "mse"] == pytest.approx(2.75)
    assert metrics.loc["first", "rank_score"] == pytest.approx(50.0)
    assert metrics.loc["second", "rank_score"] == pytest.approx(50.0)
    assert result.details["common_matched_cells"] == 3
    assert sorted(result.by_origin["observed_cells"].unique()) == [1, 2]


def test_residual_corrector_routes_typed_inputs_to_existing_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _panel()
    split = RollingSplit(validation=2, test=2)
    point = data.target - 1.0
    base = BaseForecast("base", point, data.target - point)
    execution = ExecutionConfig()
    captured: dict[str, object] = {}

    correction = np.full(data.shape, 0.25)
    lambda_weights = np.full(data.shape, 0.5)
    expected = RollingVARAFamilyResult(
        correction=correction,
        refit_index=np.full(data.time_count, -1, dtype=np.int64),
        core_fit_summary=pd.DataFrame({"eta_hat": [0.1]}),
        rolling_lambdas=lambda_weights,
        lambda_weights=lambda_weights,
        lambda_refit_fits=pd.DataFrame({"lambda": [0.5]}),
        unit_forecast=point + correction,
        lambda_forecast=point + lambda_weights * correction,
        correction_execution={"backend": "test"},
    )

    def fake_run(request, config, *, execution, progress_callback=None):
        captured.update({
            "request": request,
            "config": config,
            "execution": execution,
            "progress_callback": progress_callback,
        })
        return expected

    monkeypatch.setattr(correction_module, "run_vara_family", fake_run)

    result = VARAResidualCorrector().forecast(
        data,
        split,
        _features(),
        base,
        execution,
    )

    request = captured["request"]
    assert request.base.error is base.error
    np.testing.assert_array_equal(request.evaluation.evaluation_indices, [4, 5, 6, 7])
    assert captured["execution"] is execution
    assert result.name == "base+lambda*VARA"
    np.testing.assert_allclose(result.combined_forecast, expected.lambda_forecast)
    assert result.execution == {"backend": "test"}


def test_risk_corrector_uses_paper_model_identifiers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _panel()
    split = RollingSplit(validation=2, test=2)
    point = data.target - 1.0
    base = BaseForecast("har_ridge", point, data.target - point)
    expected = RollingVARAFamilyResult(
        correction=np.zeros(data.shape),
        refit_index=np.full(data.time_count, -1, dtype=np.int64),
        core_fit_summary=pd.DataFrame(),
        rolling_lambdas=np.zeros(data.shape),
        lambda_weights=np.zeros(data.shape),
        lambda_refit_fits=pd.DataFrame(),
        unit_forecast=point,
        lambda_forecast=point,
        correction_execution={},
    )
    monkeypatch.setattr(
        correction_module,
        "run_vara_family",
        lambda *args, **kwargs: expected,
    )

    result = VARAResidualCorrector(
        model_names=RISK_VARA_MODELS,
    ).forecast(
        data,
        split,
        _features(),
        base,
        ExecutionConfig(),
    )

    assert result.name == "har_vara_lambda"


class _HalfErrorCorrector:
    def forecast(
        self,
        data,
        split,
        features,
        base,
        execution,
        *,
        progress_callback=None,
    ):
        del split, features, execution, progress_callback
        correction = 0.5 * base.error
        combined = base.point + correction
        return VARAForecast(
            name=f"{base.name}+lambda*VARA",
            base=base,
            correction=correction,
            lambda_weights=np.ones(data.shape),
            unit_forecast=combined,
            combined_forecast=combined,
            eta_refits=pd.DataFrame({"eta_hat": [0.0]}),
            lambda_refits=pd.DataFrame({"lambda": [1.0]}),
            execution={"backend": "fake"},
        )


def test_forecast_study_orchestrates_precomputed_base_and_saves_outputs(tmp_path) -> None:
    data = _panel()
    split = RollingSplit(validation=2, test=2)
    point = data.target - 1.0
    base = BaseForecast("base", point, data.target - point)
    study = ForecastStudy(
        data=data,
        split=split,
        features=_features(),
        corrector=_HalfErrorCorrector(),  # type: ignore[arg-type]
    )

    result = study.run([PrecomputedForecaster(base)])

    metrics = result.metrics.set_index("model")
    assert metrics.loc["base", "mse"] == pytest.approx(1.0)
    assert metrics.loc["base+lambda*VARA", "mse"] == pytest.approx(0.25)
    assert metrics.loc["base+lambda*VARA", "rank_score"] == pytest.approx(100.0)
    output = result.save(tmp_path / "study")
    assert (output / "metrics.csv").is_file()
    assert (output / "forecasts" / "base.npy").is_file()
    assert (output / "vara" / "base_correction.npy").is_file()
    assert (output / "vara" / "base_eta_refits.csv").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "vara_forecast_study_v1"
    assert manifest["base_models"] == ["base"]
    assert manifest["vara_config"]["response_standardization"] == "median_mad"


class _CountingForecaster:
    name = "counting"

    def __init__(self) -> None:
        self.calls = 0

    def rolling_forecast(self, data, split, execution):
        del split, execution
        self.calls += 1
        point = np.asarray(data.target) - float(self.calls)
        return BaseForecast(
            self.name,
            point,
            np.asarray(data.target) - point,
            variance=np.ones(data.shape),
            diagnostics={"calls": self.calls},
        )


def test_cached_forecaster_reuses_matching_complete_base_result(tmp_path) -> None:
    data = _panel()
    split = RollingSplit(validation=2, test=2)
    wrapped = _CountingForecaster()
    cached = CachedForecaster(wrapped, tmp_path / "cache")

    first = cached.rolling_forecast(data, split, ExecutionConfig())
    second = cached.rolling_forecast(data, split, ExecutionConfig())

    assert wrapped.calls == 1
    assert first.diagnostics["cache_status"] == "miss"
    assert second.diagnostics["cache_status"] == "hit"
    np.testing.assert_array_equal(second.point, first.point)
    np.testing.assert_array_equal(second.error, first.error)
    np.testing.assert_array_equal(second.variance, first.variance)


def test_cached_forecaster_invalidates_when_panel_changes(tmp_path) -> None:
    data = _panel()
    changed = _panel()
    changed_target = np.asarray(changed.target).copy()
    changed_target[0, 0] += 0.5
    changed = PanelData(
        target=changed_target,
        dates=changed.dates,
        series_ids=changed.series_ids,
        metadata=changed.metadata,
        fields=changed.fields,
    )
    split = RollingSplit(validation=2, test=2)
    wrapped = _CountingForecaster()
    cached = CachedForecaster(wrapped, tmp_path / "cache")

    cached.rolling_forecast(data, split, ExecutionConfig())
    cached.rolling_forecast(changed, split, ExecutionConfig())

    assert wrapped.calls == 2


def test_risk_evaluator_reuses_validation_calibration_and_risk_metrics() -> None:
    raw = _panel()
    returns = raw.fields["returns"]
    data = PanelData(
        target=log_risk_target(returns),
        dates=raw.dates,
        series_ids=raw.series_ids,
        metadata=raw.metadata,
        fields={"returns": returns},
    )
    first = data.target + 0.1
    second = data.target + 0.2

    result = RiskForecastEvaluator(benchmark_name="first").evaluate(
        data,
        RollingSplit(validation=2, test=2),
        {"first": first, "second": second},
    )

    metrics = result.metrics.set_index("model")
    assert metrics.loc["first", "log_risk_mspe"] == pytest.approx(0.01)
    assert metrics.loc["second", "log_risk_mspe"] == pytest.approx(0.04)
    assert metrics.loc["first", "log_risk_rmse"] == pytest.approx(0.1)
    assert metrics.loc["second", "log_risk_rmse"] == pytest.approx(0.2)
    assert metrics.loc["first", "rank_score"] == pytest.approx(100.0)
    assert set(result.details["variance_calibration"]) == {"first", "second"}
    assert result.details["evaluation_scale"] == "original_log_risk"


def test_risk_paper_profile_is_the_primary_application_specification() -> None:
    split = risk_paper_split()
    config = risk_paper_vara_config()

    assert RISK_PAPER_PROFILE == "accounting_signal_risk_primary_v1"
    assert RISK_PAPER_COMPONENT_COUNT == 21_800
    assert (split.validation, split.test) == (60, 120)
    assert config.error_window_months == 240
    assert config.response_standardization == "median_mad"
    assert config.response_clip == 8.0
    assert config.refit_frequency_months == 12
    assert config.ridge == pytest.approx(0.001)
    assert config.attention_mode == "dense"
    assert config.attention_top_k == 21_799
    assert config.attention_cache_mode == "exact"
    assert config.attention_batch_size == 512
    assert config.pilot_size == 1000
    assert config.pilot_seed == 20260924
    assert config.lambda_window_months == 24
    assert config.lambda_ridge == pytest.approx(0.001)
    assert (config.eta.lower, config.eta.upper) == (-2.0, 2.0)
    assert config.eta.grid_size == 7
    assert config.eta.local_tolerance == pytest.approx(1e-4)
    diagnostic_config = risk_paper_vara_config(10)
    assert diagnostic_config.attention_top_k == 9
    assert diagnostic_config.attention_batch_size == 10
    execution = risk_paper_execution_config()
    assert execution.ray_num_cpus == 15
    assert execution.max_cpus_per_task == 1
    corrector = risk_paper_corrector(10)
    assert corrector.config == diagnostic_config
    assert corrector.model_names == RISK_VARA_MODELS


def test_risk_application_defaults_and_departure_audit_match_paper() -> None:
    runner = _risk_study_script()
    args = runner._parser().parse_args([])
    config = risk_paper_vara_config()
    full_indices = np.arange(RISK_PAPER_COMPONENT_COUNT, dtype=np.int64)

    assert args.models == "paper_five"
    assert args.max_series == 0
    assert (args.validation_months, args.test_months) == (60, 120)
    assert args.response_standardization == "median_mad"
    assert args.attention_mode == "dense"
    assert (args.ray_num_cpus, args.max_cpus_per_task) == (15, 1)
    assert runner._paper_profile_departures(
        model_names=RISK_BENCHMARK_MODELS,
        component_indices=full_indices,
        validation_months=args.validation_months,
        test_months=args.test_months,
        vara_config=config,
    ) == {}

    departures = runner._paper_profile_departures(
        model_names=(RISK_BENCHMARK_MODELS[0],),
        component_indices=np.arange(10, dtype=np.int64),
        validation_months=args.validation_months,
        test_months=args.test_months,
        vara_config=risk_paper_vara_config(10),
    )
    assert set(departures) == {
        "models",
        "selected_component_count",
        "selected_source_indices",
    }


def test_native_har_ridge_adapter_returns_causal_errors() -> None:
    periods = 40
    target = np.column_stack((
        np.linspace(0.0, 1.0, periods),
        np.linspace(1.0, 2.0, periods) + 0.01 * np.sin(np.arange(periods)),
    ))
    data = PanelData(
        target=target,
        dates=pd.date_range("2000-01-31", periods=periods, freq="ME"),
        series_ids=("s0", "s1"),
        metadata=pd.DataFrame({"series_id": ["s0", "s1"]}),
    )
    split = RollingSplit(validation=3, test=3)

    result = HARRidgeForecaster(
        ridge_candidates=(0.01, 1.0),
        minimum_observations=5,
    ).rolling_forecast(data, split, ExecutionConfig())

    evaluation = split.plan(data).evaluation_indices
    assert np.isfinite(result.point[evaluation]).all()
    np.testing.assert_allclose(
        result.error[evaluation],
        data.target[evaluation] - result.point[evaluation],
    )
    assert result.diagnostics["selected_ridge"] in {0.01, 1.0}


def test_five_risk_artifacts_become_standard_base_forecasters(tmp_path) -> None:
    data = _panel()
    split = RollingSplit(validation=2, test=2)
    source = tmp_path / "source"
    har = source / "baseline" / "baseline_forecast.npy"
    lightgbm = tmp_path / "lightgbm" / "forecasts" / "pooled_lightgbm_forecast.npy"
    ewma = (
        tmp_path
        / "ewma"
        / "forecasts"
        / "riskmetrics_ewma_validation_offset_forecast.npy"
    )
    garch = tmp_path / "garch" / "forecasts" / "garch_validation_offset_forecast.npy"
    student_t = (
        tmp_path
        / "student_t"
        / "forecasts"
        / "garch11_t_validation_offset_forecast.npy"
    )
    files = (har, lightgbm, ewma, garch, student_t)
    for position, path in enumerate(files, start=1):
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, data.target - float(position), allow_pickle=False)

    contract_path = tmp_path / "paper" / "paper_contract.json"
    contract_path.parent.mkdir()
    contract_path.write_text(json.dumps({
        "component_count": data.component_count,
        "source_output": source.as_posix(),
        "pooled_lightgbm_vara_output": (tmp_path / "lightgbm").as_posix(),
        "riskmetrics_ewma_vara_output": (tmp_path / "ewma").as_posix(),
        "garch_vara_output": (tmp_path / "garch").as_posix(),
        "student_t_garch_output": (tmp_path / "student_t").as_posix(),
        "split": {
            "validation_start": "2000-05-31",
            "validation_end": "2000-06-30",
            "test_start": "2000-07-31",
            "test_end": "2000-08-31",
        },
    }), encoding="utf-8")

    artifacts = RiskBenchmarkArtifacts.from_paper_contract(contract_path)
    models = artifacts.forecasters(data, split)

    assert tuple(model.name for model in models) == RISK_BENCHMARK_MODELS
    for position, model in enumerate(models, start=1):
        result = model.rolling_forecast(data, split, ExecutionConfig())
        np.testing.assert_allclose(result.error, float(position))
        assert result.diagnostics["source"] == "risk_paper_artifact"
        assert Path(result.diagnostics["artifact"]).is_file()

    study = ForecastStudy(
        data=data,
        split=split,
        features=_features(),
        corrector=_HalfErrorCorrector(),  # type: ignore[arg-type]
    )
    result = study.run(models)
    assert tuple(result.bases) == RISK_BENCHMARK_MODELS
    assert len(result.forecasts) == 2 * len(RISK_BENCHMARK_MODELS)


def test_risk_artifact_contract_validates_primary_paper_profile(tmp_path) -> None:
    paths = {
        model: tmp_path / f"{model}.npy"
        for model in RISK_BENCHMARK_MODELS
    }
    contract = {
        "component_count": 21_800,
        "response_standardization": "median_mad",
        "lightgbm_response_normalization": "none",
        "table_methods": list(RISK_PAPER_TABLE_MODELS),
        "eta_estimation": {
            "attention_support": "full_dense_off_diagonal_scores",
            "bounds": [-2.0, 2.0],
            "grid_size": 7,
            "local_tolerance": 1e-4,
            "parameterization": "paper_eta_log_component_count_v1",
        },
        "standalone_table": {
            "attention_support": "full_dense_off_diagonal_scores",
            "full_dense_attention_comparison": True,
        },
    }
    artifacts = RiskBenchmarkArtifacts(files=paths, contract=contract)

    artifacts.validate_primary_profile()

    invalid = RiskBenchmarkArtifacts(
        files=paths,
        contract={**contract, "response_standardization": "training_zscore"},
    )
    with pytest.raises(ValueError, match="response_standardization"):
        invalid.validate_primary_profile()


def test_five_risk_artifacts_support_aligned_component_subset(tmp_path) -> None:
    source_data = _panel(components=3)
    indices = np.array([0, 2])
    subset = PanelData(
        target=np.asarray(source_data.target)[:, indices],
        dates=source_data.dates,
        series_ids=tuple(np.asarray(source_data.series_ids)[indices]),
        metadata=source_data.metadata.iloc[indices].reset_index(drop=True),
    )
    files = {}
    for position, name in enumerate(RISK_BENCHMARK_MODELS, start=1):
        path = tmp_path / f"{name}.npy"
        np.save(path, source_data.target - float(position), allow_pickle=False)
        files[name] = path
    artifacts = RiskBenchmarkArtifacts(
        files=files,
        contract={"component_count": source_data.component_count},
    )

    models = artifacts.forecasters(
        subset,
        RollingSplit(validation=2, test=2),
        component_indices=indices,
    )

    for position, model in enumerate(models, start=1):
        result = model.rolling_forecast(
            subset,
            RollingSplit(validation=2, test=2),
            ExecutionConfig(),
        )
        np.testing.assert_allclose(result.error, float(position))


def test_risk_artifact_registry_rejects_contract_split_drift(tmp_path) -> None:
    data = _panel()
    files = {}
    for name in RISK_BENCHMARK_MODELS:
        path = tmp_path / f"{name}.npy"
        np.save(path, data.target, allow_pickle=False)
        files[name] = path
    artifacts = RiskBenchmarkArtifacts(
        files=files,
        contract={"split": {"test_start": "1999-01-31"}},
    )

    with pytest.raises(ValueError, match="test_start"):
        artifacts.forecasters(data, RollingSplit(validation=2, test=2))


@pytest.mark.parametrize(
    ("forecaster", "expected_kind", "expected_options"),
    [
        (
            AutoETSForecaster(
                season_length=4,
                error_history_months=3,
                model_specification="ANN",
                damped=False,
                response_normalization="training_zscore",
            ),
            "auto_ets",
            {
                "model_specification": "ANN",
                "damped": False,
                "phi": None,
                "response_normalization": "training_zscore",
            },
        ),
        (
            AutoARIMAForecaster(season_length=4, error_history_months=3),
            "auto_arima",
            {},
        ),
        (
            NBEATSForecaster(
                season_length=2,
                error_history_months=3,
                neural_config=NeuralModelConfig(input_size=3, max_steps=1),
            ),
            "nbeats",
            {
                "input_size": 3,
                "max_steps": 1,
                "hidden_size": 64,
                "num_layers": 2,
                "batch_size": 32,
                "learning_rate": 0.001,
                "scaler_type": "robust",
                "random_seed": 2026,
            },
        ),
    ],
)
def test_existing_benchmark_adapters_build_causal_vara_error_history(
    monkeypatch: pytest.MonkeyPatch,
    forecaster,
    expected_kind: str,
    expected_options: dict[str, object],
) -> None:
    data = _panel(periods=12)
    split = RollingSplit(validation=2, test=2)
    calls: list[tuple[int, str, int, dict[str, object]]] = []

    def fake_fit(target, series_ids, target_index, *, kind, season_length, model_options):
        del target
        calls.append((target_index, kind, season_length, model_options))
        point = np.full((1, len(series_ids)), float(target_index))
        return {
            "forecast": point,
            "forecast_variance": np.ones_like(point),
            "config": {"kind": kind},
        }

    monkeypatch.setattr(benchmark_module, "_fit_one_step", fake_fit)

    result = forecaster.rolling_forecast(data, split, ExecutionConfig())

    # Evaluation begins at 8, and the three-month VARA history adds targets 5--7.
    assert [call[0] for call in calls] == list(range(5, 12))
    assert {call[1] for call in calls} == {expected_kind}
    assert {call[2] for call in calls} == {forecaster.season_length}
    assert all(call[3] == expected_options for call in calls)
    np.testing.assert_allclose(result.point[8], 8.0)
    np.testing.assert_allclose(result.error[8], data.target[8] - 8.0)
    np.testing.assert_allclose(result.variance[8], 1.0)
    assert result.diagnostics["forecast_start_index"] == 5
    assert result.diagnostics["forecast_protocol"].startswith("expanding")


def test_native_benchmark_origin_cache_reuses_completed_origins(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _panel(periods=12)
    split = RollingSplit(validation=2, test=2)
    calls: list[int] = []

    def fake_fit(target, series_ids, target_index, **kwargs):
        del target, kwargs
        calls.append(target_index)
        point = np.full((1, len(series_ids)), float(target_index))
        return {
            "forecast": point,
            "forecast_variance": np.ones_like(point),
            "config": {"test": True},
        }

    monkeypatch.setattr(benchmark_module, "_fit_one_step", fake_fit)
    model = AutoARIMAForecaster(
        season_length=1,
        error_history_months=3,
        origin_cache_dir=tmp_path / "origins",
    )

    first = model.rolling_forecast(data, split, ExecutionConfig())
    second = model.rolling_forecast(data, split, ExecutionConfig())

    assert calls == list(range(5, 12))
    assert first.diagnostics["origin_cache_misses"] == 7
    assert second.diagnostics["origin_cache_hits"] == 7
    assert second.diagnostics["origin_cache_misses"] == 0
    assert second.diagnostics["backend"] == "cache"
    np.testing.assert_array_equal(second.point, first.point)


def test_autoets_and_autoarima_native_adapters_smoke() -> None:
    periods = 28
    time = np.arange(periods, dtype=float)
    target = np.column_stack((
        10.0 + 0.2 * time + np.sin(2.0 * np.pi * time / 4.0),
        20.0 - 0.1 * time + np.cos(2.0 * np.pi * time / 4.0),
    ))
    data = PanelData(
        target=target,
        dates=pd.date_range("2000-01-31", periods=periods, freq="ME"),
        series_ids=("s0", "s1"),
        metadata=pd.DataFrame(index=range(2)),
    )
    split = RollingSplit(validation=1, test=1)
    models = (
        AutoETSForecaster(
            season_length=4,
            error_history_months=1,
            model_specification="ANN",
            damped=False,
        ),
        AutoARIMAForecaster(season_length=1, error_history_months=1),
    )

    for model in models:
        result = model.rolling_forecast(data, split, ExecutionConfig())
        evaluation = split.plan(data).evaluation_indices
        assert np.isfinite(result.point[evaluation]).all()
        np.testing.assert_allclose(
            result.error[evaluation],
            data.target[evaluation] - result.point[evaluation],
        )
