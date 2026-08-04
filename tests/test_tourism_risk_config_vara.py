import numpy as np
import pandas as pd
import pytest

import scripts.run_tourism_risk_config_vara as tourism_runner
import vara.datamine_vara_runner as shared_runner
from vara.datamine import RollingHARVARAResult
from scripts.run_tourism_risk_config_vara import (
    EvaluationSplit,
    TourismAutoETSPanel,
    _typed_vara_inputs,
    build_tourism_hierarchy_features,
    calculate_tourism_metrics,
    trailing_evaluation_split,
)


def _tourism_leaf_ids() -> tuple[str, ...]:
    codes = """
        AAA AAB ABA ABB ACA ADA ADB ADC ADD AEA AEB AEC AED AFA
        BAA BAB BAC BBA BCA BCB BCC BDA BDB BDC BDD BDE BDF
        BEA BEB BEC BED BEE BEF BEG BEH
        CAA CAB CAC CBA CBB CBC CBD CCA CCB CCC CDA CDB
        DAA DAB DAC DBA DBB DBC DCA DCB DCC DCD DDA DDB
        EAA EAB EAC EBA ECA
        FAA FBA FBB FCA FCB
        GAA GAB GAC GBA GBB GBC GBD
    """.split()
    return tuple(f"{code}All" for code in codes)


def test_hierarchy_features_have_fixed_shape_norm_and_prefix_similarity() -> None:
    series_ids = _tourism_leaf_ids()

    features, names = build_tourism_hierarchy_features(series_ids)

    assert features.shape == (76, 34)
    assert features.dtype == np.float64
    assert names[:7] == [f"L1_{value}" for value in "ABCDEFG"]
    assert len(names) == len(set(names)) == 34
    np.testing.assert_allclose(np.linalg.norm(features, axis=1), 1.0)
    np.testing.assert_array_equal(np.count_nonzero(features, axis=1), 2)

    row = {series_id: index for index, series_id in enumerate(series_ids)}
    similarities = features @ features.T
    assert np.isclose(similarities[row["AAAAll"], row["AABAll"]], 1.0)
    assert np.isclose(similarities[row["AAAAll"], row["ABAAll"]], 0.5)
    assert np.isclose(similarities[row["AAAAll"], row["BAAAll"]], 0.0)


def test_trailing_split_reserves_60_validation_and_12_test_months() -> None:
    dates = pd.date_range("2000-01-01", periods=204, freq="MS")

    split = trailing_evaluation_split(
        dates,
        validation_months=60,
        test_months=12,
    )

    np.testing.assert_array_equal(
        split.validation_indices,
        np.arange(132, 192, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        split.test_indices,
        np.arange(192, 204, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        split.evaluation_indices,
        np.arange(132, 204, dtype=np.int64),
    )
    assert dates[split.validation_indices[0]] == pd.Timestamp("2011-01-01")
    assert dates[split.validation_indices[-1]] == pd.Timestamp("2015-12-01")
    assert dates[split.test_indices[0]] == pd.Timestamp("2016-01-01")
    assert dates[split.test_indices[-1]] == pd.Timestamp("2016-12-01")


def test_metrics_divide_each_error_by_its_origin_training_scale() -> None:
    actual = np.asarray([
        [100.0, 200.0],
        [10.0, 20.0],
        [30.0, 40.0],
    ])
    training_scale = np.asarray([
        [1.0, 1.0],
        [2.0, 4.0],
        [5.0, 10.0],
    ])
    autoets = actual + np.asarray([
        [1000.0, 1000.0],
        [2.0, 8.0],
        [10.0, 10.0],
    ])
    adjusted = actual + np.asarray([
        [1000.0, 1000.0],
        [1.0, 4.0],
        [5.0, 5.0],
    ])

    metrics, monthly = calculate_tourism_metrics(
        actual=actual,
        training_scale=training_scale,
        forecasts={"AutoETS": autoets, "AutoETS + VARA": adjusted},
        evaluation_indices=np.asarray([1, 2], dtype=np.int64),
    )

    observed = metrics.set_index("method")
    assert observed.loc["AutoETS", "scaled_mse"] == 2.5
    assert np.isclose(observed.loc["AutoETS", "scaled_rmse"], np.sqrt(2.5))
    assert observed.loc["AutoETS", "scaled_mae"] == 1.5
    assert observed.loc["AutoETS", "raw_mse"] == 67.0
    assert np.isclose(observed.loc["AutoETS", "raw_rmse"], np.sqrt(67.0))
    assert observed.loc["AutoETS", "scaled_r2_vs_autoets"] == 0.0

    assert observed.loc["AutoETS + VARA", "scaled_mse"] == 0.625
    assert observed.loc["AutoETS + VARA", "scaled_mae"] == 0.75
    assert observed.loc["AutoETS + VARA", "raw_mse"] == 16.75
    assert observed.loc["AutoETS + VARA", "scaled_r2_vs_autoets"] == 0.75
    assert observed.loc["AutoETS + VARA", "cells"] == 4
    assert observed.loc["AutoETS + VARA", "origins"] == 2
    assert observed.loc["AutoETS + VARA", "series"] == 2
    assert metrics.iloc[0]["method"] == "AutoETS + VARA"

    assert len(monthly) == 4
    assert set(monthly["target_index"]) == {1, 2}
    autoets_monthly = monthly.loc[monthly["method"].eq("AutoETS")]
    np.testing.assert_allclose(autoets_monthly["scaled_mse"], [2.5, 2.5])


def test_typed_tourism_run_matches_legacy_api_with_autoets_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dates = pd.date_range("2014-01-01", periods=36, freq="MS")
    series_ids = ("AAAAll", "BAAAll")
    metadata = pd.DataFrame({
        "series_id": series_ids,
        "group": ["A", "B"],
    })
    base_forecast = np.arange(72, dtype=float).reshape(36, 2) / 10.0 + 10.0
    base_error = np.column_stack((
        np.linspace(-0.7, 1.1, 36),
        np.linspace(0.9, -0.5, 36),
    ))
    actual = base_forecast + base_error
    panel = TourismAutoETSPanel(
        dates=dates,
        series_ids=series_ids,
        metadata=metadata,
        actual=actual,
        base_forecast=base_forecast,
        base_error=base_error,
        training_scale=np.ones_like(actual),
        source_metadata={},
    )
    split = EvaluationSplit(
        validation_indices=np.arange(28, 32, dtype=np.int64),
        test_indices=np.arange(32, 36, dtype=np.int64),
        evaluation_indices=np.arange(28, 36, dtype=np.int64),
    )
    features = np.eye(2, dtype=float)
    settings = {
        "error_window_months": 24,
        "response_standardization": "training_zscore",
        "response_clip": 8.0,
        "refit_frequency_months": 2,
        "ridge": 0.003,
        "attention_top_k": 1,
        "attention_cache_mode": "exact",
        "attention_batch_size": 16,
        "eta_lower": -0.8,
        "eta_upper": 0.6,
        "eta_grid_size": 9,
        "eta_local_tolerance": 0.002,
        "lambda_window_months": 2,
        "lambda_ridge": 0.04,
        "zero_own_lag_coefficients": True,
    }
    request, config, execution = _typed_vara_inputs(
        panel=panel,
        split=split,
        features=features,
        feature_names=["L1_A", "L1_B"],
        settings=settings,
        ray_num_cpus=1,
        max_cpus_per_task=1,
        ray_debug=False,
    )

    assert request.base.error is base_error
    np.testing.assert_array_equal(
        request.evaluation.evaluation_indices,
        split.evaluation_indices,
    )
    assert config.response_standardization == settings["response_standardization"]
    assert config.zero_own_lag_coefficients is True
    assert execution.ray_num_cpus == 1

    correction = np.full_like(base_forecast, np.nan)
    correction[split.evaluation_indices] = np.asarray([0.25, -0.125])
    refit_index = np.full(len(dates), -1, dtype=np.int64)
    fit_rows: list[dict[str, float | int]] = []
    for start in range(0, len(split.evaluation_indices), 2):
        block = split.evaluation_indices[start : start + 2]
        history_end = int(block[0] - 1)
        refit_index[block] = history_end
        fit_rows.append({"refit_index": history_end, "eta_hat": 0.1 + start})

    def fake_corrections(**kwargs):
        np.testing.assert_array_equal(
            kwargs["baseline_errors"],
            base_error,
        )
        return RollingHARVARAResult(
            standardized_forecast=np.zeros_like(base_forecast),
            correction=correction,
            refit_index=refit_index,
            fit_summary=pd.DataFrame(fit_rows),
        )

    monkeypatch.setattr(
        shared_runner,
        "rolling_har_vara_corrections",
        fake_corrections,
    )
    eta_specification = config.eta.resolve(len(series_ids))
    legacy = shared_runner.fit_rolling_vara_family(
        base_forecast=panel.base_forecast,
        base_error=panel.base_error,
        metadata=panel.metadata,
        evaluation_indices=split.evaluation_indices,
        feature_matrix=features,
        error_window_months=int(settings["error_window_months"]),
        response_standardization=str(settings["response_standardization"]),
        response_clip=float(settings["response_clip"]),
        refit_frequency_months=int(settings["refit_frequency_months"]),
        ridge=float(settings["ridge"]),
        attention_top_k=int(settings["attention_top_k"]),
        attention_cache_mode=str(settings["attention_cache_mode"]),
        attention_batch_size=int(settings["attention_batch_size"]),
        eta_bounds=eta_specification.effective_bounds,
        eta_grid_size=int(settings["eta_grid_size"]),
        eta_local_tolerance=eta_specification.effective_local_tolerance,
        lambda_window_months=int(settings["lambda_window_months"]),
        lambda_ridge=float(settings["lambda_ridge"]),
        zero_own_lag_coefficients=bool(settings["zero_own_lag_coefficients"]),
        ray_num_cpus=1,
        max_cpus_per_task=1,
        ray_debug=False,
    )
    typed = tourism_runner.run_vara_family(
        request,
        config,
        execution=execution,
    )

    for field in (
        "correction",
        "refit_index",
        "rolling_lambdas",
        "lambda_weights",
        "unit_forecast",
        "lambda_forecast",
    ):
        np.testing.assert_allclose(
            getattr(typed, field),
            getattr(legacy, field),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
    pd.testing.assert_frame_equal(typed.core_fit_summary, legacy.core_fit_summary)
    pd.testing.assert_frame_equal(typed.lambda_refit_fits, legacy.lambda_refit_fits)
    assert {
        key: value
        for key, value in typed.correction_execution.items()
        if key != "elapsed_seconds"
    } == {
        key: value
        for key, value in legacy.correction_execution.items()
        if key != "elapsed_seconds"
    }
