from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import vara.datamine_vara_runner as runner
from vara.datamine import RollingHARVARAResult, rolling_refit_group_lambdas
from vara.forecast_contracts import (
    BaseForecastResult,
    ExecutionConfig,
    ForecastPanel,
    PaperEtaConfig,
    RollingEvaluationPlan,
    RollingVARAConfig,
    StaticFeatureSet,
    VARARunRequest,
)


def test_combine_base_vara_forecasts_accepts_scalar_and_panel_weights() -> None:
    base = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    correction = np.asarray([[0.5, -0.5], [1.0, -1.0]])

    scalar_unit, scalar_lambda = runner.combine_base_vara_forecasts(
        base_forecast=base,
        correction=correction,
        lambda_weights=0.25,
    )
    np.testing.assert_allclose(scalar_unit, base + correction)
    np.testing.assert_allclose(scalar_lambda, base + 0.25 * correction)

    panel_weights = np.asarray([[0.0, 0.5], [0.25, 1.0]])
    panel_unit, panel_lambda = runner.combine_base_vara_forecasts(
        base_forecast=base,
        correction=correction,
        lambda_weights=panel_weights,
    )
    np.testing.assert_allclose(panel_unit, base + correction)
    np.testing.assert_allclose(
        panel_lambda,
        base + panel_weights * correction,
    )


def test_combine_base_vara_forecasts_rejects_inconsistent_shapes() -> None:
    with pytest.raises(ValueError, match="base_forecast and correction must have the same shape"):
        runner.combine_base_vara_forecasts(
            base_forecast=np.zeros((2, 2)),
            correction=np.zeros((2, 1)),
            lambda_weights=0.5,
        )

    with pytest.raises(ValueError, match="lambda_weights must be scalar or match"):
        runner.combine_base_vara_forecasts(
            base_forecast=np.zeros((2, 2)),
            correction=np.zeros((2, 2)),
            lambda_weights=np.zeros((2, 1)),
        )


def test_combine_rolling_vara_family_matches_direct_lambda_fit() -> None:
    base = np.arange(18, dtype=float).reshape(9, 2) / 10.0
    base_error = np.zeros_like(base)
    base_error[3:] = np.asarray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
        ]
    )
    correction = np.full_like(base, 4.0)
    metadata = pd.DataFrame({"formula_type": ["a", "a"]})
    evaluation = np.arange(3, 9, dtype=np.int64)
    refit_index = np.full(9, -1, dtype=np.int64)
    refit_index[evaluation] = [2, 2, 4, 4, 6, 6]
    fit_summary = pd.DataFrame({"refit_index": [2, 4, 6], "eta_hat": [0.1, 0.2, 0.3]})
    execution = {"backend": "saved", "ray_used": False}

    expected_lambdas, expected_table = rolling_refit_group_lambdas(
        base_error,
        correction,
        metadata,
        evaluation,
        refit_index,
        window_months=2,
        min_history_months=2,
        ridge=0.125,
    )
    expected_weights = np.nan_to_num(expected_lambdas, nan=0.0)

    result = runner.combine_rolling_vara_family(
        base_forecast=base,
        base_error=base_error,
        correction=correction,
        refit_index=refit_index,
        core_fit_summary=fit_summary,
        correction_execution=execution,
        metadata=metadata,
        evaluation_indices=evaluation,
        lambda_window_months=2,
        lambda_ridge=0.125,
    )

    np.testing.assert_allclose(
        result.rolling_lambdas,
        expected_lambdas,
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.lambda_weights,
        expected_weights,
        rtol=0.0,
        atol=0.0,
    )
    pd.testing.assert_frame_equal(result.lambda_refit_fits, expected_table)
    np.testing.assert_allclose(result.unit_forecast, base + correction)
    np.testing.assert_allclose(
        result.lambda_forecast,
        base + expected_weights * correction,
    )
    np.testing.assert_array_equal(result.refit_index, refit_index)
    pd.testing.assert_frame_equal(result.core_fit_summary, fit_summary)
    assert result.correction_execution == execution
    assert result.correction_execution is not execution


def test_fit_rolling_vara_family_routes_sequential_and_composes_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = np.arange(14, dtype=float).reshape(7, 2) / 10.0
    base_error = np.zeros_like(base)
    base_error[3:] = np.asarray(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]]
    )
    correction = np.full_like(base, 0.5)
    evaluation = np.arange(3, 7, dtype=np.int64)
    refit_index = np.full(len(base), -1, dtype=np.int64)
    refit_index[evaluation] = [2, 2, 4, 4]
    fit_summary = pd.DataFrame({"refit_index": [2, 4], "eta_hat": [0.2, 0.4]})
    feature_matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    metadata = pd.DataFrame({"formula_type": ["a", "b"]})
    callback = lambda row: None
    captured: dict[str, object] = {}

    def fake_sequential(**kwargs):
        captured.update(kwargs)
        return RollingHARVARAResult(
            standardized_forecast=np.zeros_like(base),
            correction=correction,
            refit_index=refit_index,
            fit_summary=fit_summary,
        )

    def unexpected_ray(**kwargs):
        raise AssertionError(f"Ray route was unexpectedly called: {sorted(kwargs)}")

    monkeypatch.setattr(runner, "rolling_har_vara_corrections", fake_sequential)
    monkeypatch.setattr(runner, "rolling_har_vara_corrections_ray", unexpected_ray)

    result = runner.fit_rolling_vara_family(
        base_forecast=base,
        base_error=base_error,
        metadata=metadata,
        evaluation_indices=evaluation,
        feature_matrix=feature_matrix,
        error_window_months=37,
        response_standardization="training_zscore",
        response_clip=7.5,
        refit_frequency_months=2,
        ridge=0.012,
        attention_top_k=3,
        attention_cache_mode="exact",
        attention_batch_size=16,
        eta_bounds=(-0.4, 0.7),
        eta_grid_size=9,
        eta_local_tolerance=0.002,
        lambda_window_months=2,
        lambda_ridge=0.125,
        zero_own_lag_coefficients=True,
        ray_num_cpus=1,
        max_cpus_per_task=1,
        progress_callback=callback,
    )

    np.testing.assert_array_equal(captured.pop("baseline_errors"), base_error)
    np.testing.assert_array_equal(captured.pop("evaluation_indices"), evaluation)
    np.testing.assert_array_equal(captured.pop("feature_matrix"), feature_matrix)
    assert captured == {
        "error_window_months": 37,
        "response_standardization": "training_zscore",
        "response_clip": 7.5,
        "refit_frequency_months": 2,
        "ridge": 0.012,
        "attention_top_k": 3,
        "attention_cache_mode": "exact",
        "attention_batch_size": 16,
        "attention_mode": "topk",
        "eta_bounds": (-0.4, 0.7),
        "eta_grid_size": 9,
        "eta_local_tolerance": 0.002,
        "zero_own_lag_coefficients": True,
        "progress_callback": callback,
    }

    expected_lambdas, expected_table = rolling_refit_group_lambdas(
        base_error,
        correction.astype(np.float32),
        metadata,
        evaluation,
        refit_index,
        window_months=2,
        min_history_months=2,
        ridge=0.125,
    )
    expected_weights = np.nan_to_num(expected_lambdas, nan=0.0)
    np.testing.assert_allclose(
        result.correction,
        correction.astype(np.float32),
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        result.rolling_lambdas,
        expected_lambdas,
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(result.lambda_weights, expected_weights)
    pd.testing.assert_frame_equal(result.lambda_refit_fits, expected_table)
    np.testing.assert_allclose(result.unit_forecast, base + correction)
    np.testing.assert_allclose(
        result.lambda_forecast,
        base + expected_weights * correction,
    )
    pd.testing.assert_frame_equal(result.core_fit_summary, fit_summary)
    assert result.correction_execution["backend"] == "sequential"
    assert result.correction_execution["ray_used"] is False
    assert result.correction_execution["task_unit"] == "rolling_refit_block"
    assert result.correction_execution["task_count"] == 2
    assert result.correction_execution["elapsed_seconds"] >= 0.0


def test_typed_run_vara_family_matches_legacy_keyword_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = np.arange(14, dtype=float).reshape(7, 2) / 10.0
    base_error = np.zeros_like(base)
    base_error[3:] = np.asarray(
        [[1.0, 2.0], [2.0, 1.0], [3.0, 4.0], [4.0, 3.0]]
    )
    target = base + base_error
    correction = np.full_like(base, 0.5)
    evaluation = np.arange(3, 7, dtype=np.int64)
    refit_index = np.full(len(base), -1, dtype=np.int64)
    refit_index[evaluation] = [2, 2, 4, 4]
    fit_summary = pd.DataFrame({"refit_index": [2, 4], "eta_hat": [0.2, 0.4]})
    feature_matrix = np.asarray([[1.0, 0.0], [0.0, 1.0]])
    metadata = pd.DataFrame({"formula_type": ["a", "b"]})

    def fake_sequential(**kwargs):
        return RollingHARVARAResult(
            standardized_forecast=np.zeros_like(base),
            correction=correction,
            refit_index=refit_index,
            fit_summary=fit_summary,
        )

    monkeypatch.setattr(runner, "rolling_har_vara_corrections", fake_sequential)

    eta = PaperEtaConfig(
        lower=-0.4,
        upper=0.7,
        grid_size=9,
        local_tolerance=0.002,
    )
    config = RollingVARAConfig(
        error_window_months=37,
        response_standardization="training_zscore",
        response_clip=7.5,
        refit_frequency_months=2,
        ridge=0.012,
        attention_mode="dense",
        attention_top_k=1,
        attention_cache_mode="exact",
        attention_batch_size=16,
        eta=eta,
        lambda_window_months=2,
        lambda_ridge=0.125,
        zero_own_lag_coefficients=True,
    )
    execution = ExecutionConfig(
        ray_num_cpus=1,
        max_cpus_per_task=1,
        ray_debug=False,
    )
    callback = lambda row: None
    legacy = runner.fit_rolling_vara_family(
        base_forecast=base,
        base_error=base_error,
        metadata=metadata,
        evaluation_indices=evaluation,
        feature_matrix=feature_matrix,
        error_window_months=config.error_window_months,
        response_standardization=config.response_standardization,
        response_clip=config.response_clip,
        refit_frequency_months=config.refit_frequency_months,
        ridge=config.ridge,
        attention_mode=config.attention_mode,
        attention_top_k=config.attention_top_k,
        attention_cache_mode=config.attention_cache_mode,
        attention_batch_size=config.attention_batch_size,
        eta_bounds=eta.resolve(len(metadata)).effective_bounds,
        eta_grid_size=eta.grid_size,
        eta_local_tolerance=eta.resolve(len(metadata)).effective_local_tolerance,
        lambda_window_months=config.lambda_window_months,
        lambda_ridge=config.lambda_ridge,
        zero_own_lag_coefficients=config.zero_own_lag_coefficients,
        ray_num_cpus=execution.ray_num_cpus,
        max_cpus_per_task=execution.max_cpus_per_task,
        ray_debug=execution.ray_debug,
        progress_callback=callback,
    )
    request = VARARunRequest(
        panel=ForecastPanel(
            dates=pd.date_range("2000-01-01", periods=len(base), freq="MS"),
            series_ids=("a", "b"),
            target=target,
            metadata=metadata,
        ),
        evaluation=RollingEvaluationPlan(
            validation_indices=evaluation,
            test_indices=np.asarray([], dtype=np.int64),
        ),
        features=StaticFeatureSet(
            values=feature_matrix,
            names=("feature_a", "feature_b"),
            provenance={"source": "test"},
        ),
        base=BaseForecastResult(
            model_name="baseline",
            forecast=base,
            error=base_error,
        ),
    )
    typed = runner.run_vara_family(
        request,
        config,
        execution=execution,
        progress_callback=callback,
    )

    np.testing.assert_allclose(typed.correction, legacy.correction, rtol=0.0, atol=0.0)
    np.testing.assert_array_equal(typed.refit_index, legacy.refit_index)
    pd.testing.assert_frame_equal(typed.core_fit_summary, legacy.core_fit_summary)
    np.testing.assert_allclose(
        typed.rolling_lambdas,
        legacy.rolling_lambdas,
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        typed.lambda_weights,
        legacy.lambda_weights,
        rtol=0.0,
        atol=0.0,
    )
    pd.testing.assert_frame_equal(typed.lambda_refit_fits, legacy.lambda_refit_fits)
    np.testing.assert_allclose(
        typed.unit_forecast,
        legacy.unit_forecast,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        typed.lambda_forecast,
        legacy.lambda_forecast,
        rtol=0.0,
        atol=0.0,
    )
    legacy_execution = {
        key: value
        for key, value in legacy.correction_execution.items()
        if key != "elapsed_seconds"
    }
    typed_execution = {
        key: value
        for key, value in typed.correction_execution.items()
        if key != "elapsed_seconds"
    }
    assert typed_execution == legacy_execution
