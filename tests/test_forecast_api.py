from __future__ import annotations

import json
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import vara.correction as correction_module
from vara.api import (
    BaseForecast,
    ExecutionConfig,
    ForecastStudy,
    HARRidgeForecaster,
    PanelData,
    PointForecastEvaluator,
    PrecomputedForecaster,
    RiskForecastEvaluator,
    RollingSplit,
    StaticFeatureSet,
    VARAForecast,
    VARAResidualCorrector,
    log_risk_target,
)
from vara.datamine_vara_runner import RollingVARAFamilyResult


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
    assert metrics.loc["first", "log_risk_rmse"] == pytest.approx(0.1)
    assert metrics.loc["second", "log_risk_rmse"] == pytest.approx(0.2)
    assert metrics.loc["first", "rank_score"] == pytest.approx(100.0)
    assert set(result.details["variance_calibration"]) == {"first", "second"}


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
