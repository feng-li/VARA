from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.run_datamine_garch_vara as runner
import vara.datamine_vara_runner as vara_runner
from scripts.run_datamine_garch_vara import (
    EWMA_OFFSET_MODEL,
    EWMA_VARA_LAMBDA_MODEL,
    EWMA_VARA_UNIT_MODEL,
    GARCH_OFFSET_MODEL,
    GARCH_VARA_LAMBDA_MODEL,
    GARCH_VARA_UNIT_MODEL,
    RAW_EWMA_VARA_UNIT_MODEL,
    RAW_GARCH_VARA_UNIT_MODEL,
    RAW_STUDENT_T_GARCH_VARA_UNIT_MODEL,
    STUDENT_T_GARCH_OFFSET_MODEL,
    STUDENT_T_GARCH_VARA_LAMBDA_MODEL,
    STUDENT_T_GARCH_VARA_UNIT_MODEL,
    _save_float32_array,
    _save_int64_array,
    combine_garch_vara_forecasts,
    derive_raw_garch_vara_unit,
    estimate_validation_log_offset,
    load_validated_har_vara_forecast,
)
from vara.forecast_contracts import (
    ExecutionConfig,
    RollingVARAConfig,
    VARARunRequest,
)


def test_validation_log_offset_uses_only_matched_validation_cells() -> None:
    target = np.asarray([[1.0, 2.0], [3.0, 4.0], [1000.0, 1000.0]])
    garch = np.asarray([[3.0, 4.0], [5.0, np.nan], [-1000.0, -1000.0]])
    reference = np.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])

    result = estimate_validation_log_offset(
        target=target,
        garch_log_forecast=garch,
        reference_forecast=reference,
        validation_indices=np.asarray([0, 1]),
    )

    assert result["offset"] == 2.0
    assert result["validation_cells"] == 3
    assert abs(result["adjusted_validation_bias"]) < 1e-12


def test_garch_vara_combinations_include_unit_and_selected_lambda() -> None:
    base = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    correction = np.asarray([[0.5, -0.5], [1.0, -1.0]])

    forecasts = combine_garch_vara_forecasts(
        base_forecast=base,
        correction=correction,
        selected_lambda=0.25,
    )

    np.testing.assert_allclose(forecasts[GARCH_OFFSET_MODEL], base)
    np.testing.assert_allclose(forecasts[GARCH_VARA_UNIT_MODEL], base + correction)
    np.testing.assert_allclose(
        forecasts[GARCH_VARA_LAMBDA_MODEL],
        base + 0.25 * correction,
    )


def test_garch_vara_combinations_accept_origin_specific_lambdas() -> None:
    base = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    correction = np.asarray([[0.5, -0.5], [1.0, -1.0]])
    lambdas = np.asarray([[0.0, 0.0], [0.25, 0.25]])

    forecasts = combine_garch_vara_forecasts(
        base_forecast=base,
        correction=correction,
        selected_lambda=lambdas,
    )

    np.testing.assert_allclose(
        forecasts[GARCH_VARA_LAMBDA_MODEL],
        base + lambdas * correction,
    )


def test_garch_vara_combinations_accept_student_t_model_ids() -> None:
    base = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    correction = np.asarray([[0.5, -0.5], [1.0, -1.0]])

    forecasts = combine_garch_vara_forecasts(
        base_forecast=base,
        correction=correction,
        selected_lambda=0.25,
        offset_model=STUDENT_T_GARCH_OFFSET_MODEL,
        unit_model=STUDENT_T_GARCH_VARA_UNIT_MODEL,
        lambda_model=STUDENT_T_GARCH_VARA_LAMBDA_MODEL,
    )

    assert set(forecasts) == {
        STUDENT_T_GARCH_OFFSET_MODEL,
        STUDENT_T_GARCH_VARA_UNIT_MODEL,
        STUDENT_T_GARCH_VARA_LAMBDA_MODEL,
    }
    np.testing.assert_allclose(forecasts[STUDENT_T_GARCH_OFFSET_MODEL], base)
    np.testing.assert_allclose(
        forecasts[STUDENT_T_GARCH_VARA_UNIT_MODEL],
        base + correction,
    )
    np.testing.assert_allclose(
        forecasts[STUDENT_T_GARCH_VARA_LAMBDA_MODEL],
        base + 0.25 * correction,
    )


def test_garch_vara_combinations_accept_riskmetrics_ewma_model_ids() -> None:
    base = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    correction = np.asarray([[0.5, -0.5], [1.0, -1.0]])

    forecasts = combine_garch_vara_forecasts(
        base_forecast=base,
        correction=correction,
        selected_lambda=0.25,
        offset_model=EWMA_OFFSET_MODEL,
        unit_model=EWMA_VARA_UNIT_MODEL,
        lambda_model=EWMA_VARA_LAMBDA_MODEL,
    )

    assert EWMA_OFFSET_MODEL == "riskmetrics_ewma_validation_offset"
    assert EWMA_VARA_UNIT_MODEL == "riskmetrics_ewma_vara_unit"
    assert EWMA_VARA_LAMBDA_MODEL == "riskmetrics_ewma_vara_lambda"
    assert RAW_EWMA_VARA_UNIT_MODEL == "riskmetrics_ewma_raw_vara_unit"
    assert set(forecasts) == {
        EWMA_OFFSET_MODEL,
        EWMA_VARA_UNIT_MODEL,
        EWMA_VARA_LAMBDA_MODEL,
    }
    np.testing.assert_allclose(forecasts[EWMA_OFFSET_MODEL], base)
    np.testing.assert_allclose(
        forecasts[EWMA_VARA_UNIT_MODEL],
        base + correction,
    )
    np.testing.assert_allclose(
        forecasts[EWMA_VARA_LAMBDA_MODEL],
        base + 0.25 * correction,
    )


def test_raw_garch_unit_branch_is_translation_equivalent_to_offset_branch() -> None:
    raw_base = np.asarray([[3.0, 5.0], [7.0, 9.0]])
    offset_correction = np.asarray([[0.5, -0.5], [1.0, -1.0]])
    offset = 2.0

    raw_correction, raw_forecast = derive_raw_garch_vara_unit(
        raw_base_forecast=raw_base,
        offset_correction=offset_correction,
        offset=offset,
    )

    np.testing.assert_allclose(raw_correction, offset_correction - offset)
    np.testing.assert_allclose(
        raw_forecast,
        (raw_base - offset) + offset_correction,
    )
    assert RAW_GARCH_VARA_UNIT_MODEL == "garch11_raw_vara_unit"


def test_atomic_array_save_supports_same_path_memory_map(tmp_path) -> None:
    path = tmp_path / "correction.npy"
    np.save(path, np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    memory_map = np.load(path, mmap_mode="r")

    _save_float32_array(path, memory_map)

    np.testing.assert_allclose(np.load(path), [[1.0, 2.0], [3.0, 4.0]])


def test_atomic_index_save_supports_same_path_memory_map(tmp_path) -> None:
    path = tmp_path / "refit_index.npy"
    expected = np.arange(100, dtype=np.int64)
    np.save(path, expected)
    memory_map = np.load(path, mmap_mode="r")

    _save_int64_array(path, memory_map)

    np.testing.assert_array_equal(np.load(path), expected)


def test_har_comparator_requires_matching_contract_and_shape(tmp_path) -> None:
    output_dir = tmp_path / "har"
    forecast_dir = output_dir / "forecasts"
    forecast_dir.mkdir(parents=True)
    required = {
        "application_contract_schema": "datamine_vara_paper_eta_v2",
        "source_output": "/source",
        "eta_estimation": {"parameterization": "paper_eta_log_component_count_v1"},
        "forecast_scope": "validation_and_test",
    }
    (forecast_dir / "contract.json").write_text(
        json.dumps(required),
        encoding="utf-8",
    )
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    np.save(forecast_dir / "vara_final_forecast.npy", expected)

    loaded = load_validated_har_vara_forecast(
        output_dir,
        expected_shape=(3, 4),
        required_contract_fields=required,
    )
    np.testing.assert_array_equal(loaded, expected)

    with pytest.raises(ValueError, match="eta_estimation"):
        load_validated_har_vara_forecast(
            output_dir,
            expected_shape=(3, 4),
            required_contract_fields={
                **required,
                "eta_estimation": {"parameterization": "legacy"},
            },
        )
    with pytest.raises(ValueError, match="panel shape"):
        load_validated_har_vara_forecast(
            output_dir,
            expected_shape=(4, 3),
            required_contract_fields=required,
        )


def test_garch_runner_forwards_paper_eta_on_effective_scale(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    for directory in (
        source / "baseline",
        source / "features" / "snapshot_200712",
        source / "forecasts",
        source / "processed",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    time_count, component_count = 36, 4
    shape = (time_count, component_count)
    zeros = np.zeros(shape, dtype=np.float32)
    target = np.tile(np.arange(component_count, dtype=np.float32), (time_count, 1))
    np.save(source / "baseline" / "target.npy", target)
    np.save(source / "baseline" / "baseline_forecast.npy", zeros)
    np.save(source / "baseline" / "garch11_forecast.npy", zeros)
    np.save(source / "baseline" / "ar1_forecast.npy", zeros)
    np.save(source / "features" / "snapshot_200712" / "Z.npy", np.eye(component_count))
    np.save(source / "processed" / "dates.npy", np.arange(time_count))
    (source / "processed" / "component_order.csv").write_text("id\n0\n", encoding="utf-8")
    (source / "forecasts" / "locked_test_config.json").write_text(
        json.dumps({
            "validation_months": 6,
            "test_months": 6,
            "epsilon": 1e-6,
            "variance_floor": 1e-8,
        }),
        encoding="utf-8",
    )

    dates = pd.date_range("2000-01-31", periods=time_count, freq="ME")
    panel = SimpleNamespace(
        dates=dates,
        component_ids=np.arange(component_count),
        metadata=pd.DataFrame({"component": np.arange(component_count)}),
        returns=zeros,
    )
    split = SimpleNamespace(
        validation_indices=np.arange(24, 30, dtype=np.int64),
        test_indices=np.arange(30, 36, dtype=np.int64),
    )
    captured: dict[str, object] = {}
    captured_typed: dict[str, object] = {}
    real_run_vara_family = runner.run_vara_family

    def capture_run_vara_family(request, config, **kwargs):
        captured_typed.update({
            "request": request,
            "config": config,
            "execution": kwargs["execution"],
        })
        return real_run_vara_family(request, config, **kwargs)

    def fake_vara(**kwargs):
        captured.update(kwargs)
        refit_index = np.full(time_count, -1, dtype=np.int64)
        evaluation = np.concatenate((split.validation_indices, split.test_indices))
        refit_index[evaluation] = 23
        lower, upper = kwargs["eta_bounds"]
        return SimpleNamespace(
            correction=zeros.copy(),
            refit_index=refit_index,
            fit_summary=pd.DataFrame({
                "refit_index": [23],
                "eta_hat": [0.8 * np.log(component_count)],
                "eta_on_boundary": [False],
                "eta_grid_best": [0.5 * np.log(component_count)],
                "eta_grid_best_objective": [-2.0],
                "eta_objective": [-1.5],
                "eta_bounds_lower": [lower],
                "eta_bounds_upper": [upper],
                "eta_grid_size": [kwargs["eta_grid_size"]],
                "eta_local_tolerance": [kwargs["eta_local_tolerance"]],
            }),
        )

    def fake_lambdas(*args, **kwargs):
        values = np.zeros(shape, dtype=float)
        table = pd.DataFrame({
            "refit_index": [23],
            "group": ["__global__"],
            "series": ["__global__"],
            "lambda": [0.0],
            "lambda_unbounded": [0.0],
            "ridge": [0.001],
            "fallback_global": [False],
            "lambda_history_sufficient": [False],
        })
        return values, table

    metric_models = [
        "garch11",
        GARCH_OFFSET_MODEL,
        RAW_GARCH_VARA_UNIT_MODEL,
        GARCH_VARA_UNIT_MODEL,
        GARCH_VARA_LAMBDA_MODEL,
    ]
    metrics = pd.DataFrame({
        "model": metric_models,
        "log_risk_rmse": np.ones(len(metric_models)),
        "log_risk_mae": np.ones(len(metric_models)),
        "log_risk_bias": np.zeros(len(metric_models)),
        "absolute_r2": np.zeros(len(metric_models)),
        "gaussian_return_crps": np.ones(len(metric_models)),
        "oos_r2_vs_benchmark": np.zeros(len(metric_models)),
        "rank_score": np.zeros(len(metric_models)),
    })

    monkeypatch.setattr(runner, "load_panel", lambda path: panel)
    monkeypatch.setattr(runner, "relative_panel_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(runner, "run_vara_family", capture_run_vara_family)
    monkeypatch.setattr(vara_runner, "rolling_har_vara_corrections", fake_vara)
    monkeypatch.setattr(vara_runner, "rolling_refit_group_lambdas", fake_lambdas)
    monkeypatch.setattr(
        runner,
        "evaluate_risk_forecasts",
        lambda **kwargs: (metrics, pd.DataFrame(), {}),
    )
    monkeypatch.setattr(runner, "_dm_table", lambda monthly: pd.DataFrame())
    monkeypatch.setattr(
        "sys.argv",
        ["run_datamine_garch_vara.py", str(source), str(output)],
    )

    runner.main()

    typed_request = captured_typed["request"]
    assert isinstance(typed_request, VARARunRequest)
    assert typed_request.base.model_name == GARCH_OFFSET_MODEL
    np.testing.assert_allclose(typed_request.base.forecast, 1.5)
    np.testing.assert_allclose(typed_request.base.error, target - 1.5)
    np.testing.assert_array_equal(
        typed_request.evaluation.evaluation_indices,
        np.concatenate((split.validation_indices, split.test_indices)),
    )
    assert isinstance(captured_typed["config"], RollingVARAConfig)
    assert isinstance(captured_typed["execution"], ExecutionConfig)
    scale = np.log(component_count)
    assert captured["eta_bounds"] == pytest.approx((-2.0 * scale, 2.0 * scale))
    assert captured["eta_local_tolerance"] == pytest.approx(1e-4 * scale)
    assert captured["response_standardization"] == "median_mad"
    saved = pd.read_csv(output / "forecasts" / "fit_summary.csv")
    assert saved.loc[0, "eta_hat"] == pytest.approx(0.8)
    assert saved.loc[0, "eta_effective_slope_hat"] == pytest.approx(0.8 * scale)
    assert np.load(output / "forecasts" / "vara_correction.npy").dtype == np.float32

    contract_path = output / "forecasts" / "contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    assert contract["response_standardization"] == "median_mad"
    contract.pop("garch_variant")
    contract.pop("upstream_garch_output")
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_datamine_garch_vara.py",
            str(source),
            str(output),
            "--reuse-saved-correction",
        ],
    )
    runner.main()

    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["response_standardization"] = "training_zscore"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    with pytest.raises(ValueError, match="response_standardization"):
        runner.main()

    contract["response_standardization"] = "median_mad"
    contract["offset"] = float(contract["offset"]) + 0.1
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_datamine_garch_vara.py",
            str(source),
            str(output),
            "--reuse-saved-correction",
        ],
    )
    with pytest.raises(ValueError, match="different GARCH validation offset"):
        runner.main()


def test_student_t_runner_uses_own_residuals_and_dynamic_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    upstream = tmp_path / "student_t_garch"
    output = tmp_path / "student_t_vara"
    for directory in (
        source / "baseline",
        source / "features" / "snapshot_200712",
        source / "forecasts",
        source / "processed",
        upstream / "forecasts",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    time_count, component_count = 36, 4
    shape = (time_count, component_count)
    dates = pd.date_range("2000-01-31", periods=time_count, freq="ME")
    split = SimpleNamespace(
        validation_indices=np.arange(24, 30, dtype=np.int64),
        test_indices=np.arange(30, 36, dtype=np.int64),
    )
    split_contract = {
        "universe_selection_end": dates[23].date().isoformat(),
        "validation_start": dates[24].date().isoformat(),
        "validation_end": dates[29].date().isoformat(),
        "test_start": dates[30].date().isoformat(),
        "test_end": dates[35].date().isoformat(),
    }
    target = np.tile(np.arange(component_count, dtype=np.float32), (time_count, 1))
    raw_student_t = target + np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    adjusted_student_t = raw_student_t - 2.5
    zeros = np.zeros(shape, dtype=np.float32)
    np.save(source / "baseline" / "target.npy", target)
    np.save(source / "baseline" / "baseline_forecast.npy", zeros)
    np.save(source / "baseline" / "ar1_forecast.npy", zeros)
    np.save(source / "features" / "snapshot_200712" / "Z.npy", np.eye(component_count))
    np.save(source / "processed" / "dates.npy", dates.to_numpy())
    (source / "processed" / "component_order.csv").write_text(
        "component_id\n0\n1\n2\n3\n",
        encoding="utf-8",
    )
    (source / "forecasts" / "locked_test_config.json").write_text(
        json.dumps({
            "validation_months": 6,
            "test_months": 6,
            "epsilon": 1e-6,
            "variance_floor": 1e-8,
        }),
        encoding="utf-8",
    )
    np.save(upstream / "forecasts" / "garch11_t_forecast.npy", raw_student_t)
    np.save(
        upstream / "forecasts" / "garch11_t_validation_offset_forecast.npy",
        adjusted_student_t,
    )
    (upstream / "forecasts" / "contract.json").write_text(
        json.dumps({
            "source_output": source.resolve().as_posix(),
            "component_count": component_count,
            "innovation_distribution": "variance-standardized Student-t",
            "parameters_frozen_during_validation_and_test": True,
            "split": split_contract,
        }),
        encoding="utf-8",
    )

    panel = SimpleNamespace(
        dates=dates,
        component_ids=np.arange(component_count),
        metadata=pd.DataFrame({"component": np.arange(component_count)}),
        returns=zeros,
    )
    correction = np.full(shape, 0.2, dtype=np.float32)
    evaluation = np.concatenate((split.validation_indices, split.test_indices))
    refit_index = np.full(time_count, -1, dtype=np.int64)
    refit_index[evaluation] = 23
    captured_vara: dict[str, object] = {}
    captured_evaluation: dict[str, object] = {}
    captured_typed: dict[str, object] = {}
    real_run_vara_family = runner.run_vara_family

    def capture_run_vara_family(request, config, **kwargs):
        captured_typed.update({"request": request, "config": config})
        return real_run_vara_family(request, config, **kwargs)

    def fake_vara(**kwargs):
        captured_vara.update(kwargs)
        lower, upper = kwargs["eta_bounds"]
        return SimpleNamespace(
            correction=correction,
            refit_index=refit_index,
            fit_summary=pd.DataFrame({
                "refit_index": [23],
                "eta_hat": [0.5 * np.log(component_count)],
                "eta_on_boundary": [False],
                "eta_grid_best": [0.5 * np.log(component_count)],
                "eta_grid_best_objective": [-2.0],
                "eta_objective": [-1.5],
                "eta_bounds_lower": [lower],
                "eta_bounds_upper": [upper],
                "eta_grid_size": [kwargs["eta_grid_size"]],
                "eta_local_tolerance": [kwargs["eta_local_tolerance"]],
            }),
        )

    def fake_lambdas(*args, **kwargs):
        weights = np.zeros(shape, dtype=float)
        weights[evaluation] = 0.25
        table = pd.DataFrame({
            "refit_index": [23],
            "group": ["__global__"],
            "series": [component_count],
            "lambda": [0.25],
            "lambda_unbounded": [0.25],
            "ridge": [0.001],
            "fallback_global": [False],
            "lambda_history_sufficient": [True],
        })
        return weights, table

    def fake_evaluate(**kwargs):
        captured_evaluation.update(kwargs)
        models = list(kwargs["forecasts"])
        metrics = pd.DataFrame({
            "model": models,
            "log_risk_rmse": np.ones(len(models)),
            "log_risk_mae": np.ones(len(models)),
            "log_risk_bias": np.zeros(len(models)),
            "absolute_r2": np.zeros(len(models)),
            "gaussian_return_crps": np.ones(len(models)),
            "oos_r2_vs_benchmark": np.zeros(len(models)),
            "rank_score": np.zeros(len(models)),
        })
        return metrics, pd.DataFrame(), {}

    monkeypatch.setattr(runner, "load_panel", lambda path: panel)
    monkeypatch.setattr(runner, "relative_panel_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(runner, "run_vara_family", capture_run_vara_family)
    monkeypatch.setattr(vara_runner, "rolling_har_vara_corrections", fake_vara)
    monkeypatch.setattr(vara_runner, "rolling_refit_group_lambdas", fake_lambdas)
    monkeypatch.setattr(runner, "evaluate_risk_forecasts", fake_evaluate)
    monkeypatch.setattr(runner, "_dm_table", lambda *args, **kwargs: pd.DataFrame())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_datamine_garch_vara.py",
            str(source),
            str(output),
            "--student-t-garch-output",
            str(upstream),
            "--response-standardization",
            "training_zscore",
        ],
    )

    runner.main()

    expected_error = target - adjusted_student_t
    typed_request = captured_typed["request"]
    assert isinstance(typed_request, VARARunRequest)
    assert typed_request.base.model_name == STUDENT_T_GARCH_OFFSET_MODEL
    np.testing.assert_allclose(typed_request.base.forecast, adjusted_student_t)
    np.testing.assert_allclose(typed_request.base.error, expected_error)
    assert isinstance(captured_typed["config"], RollingVARAConfig)
    np.testing.assert_allclose(captured_vara["baseline_errors"], expected_error)
    assert captured_vara["response_standardization"] == "training_zscore"
    assert captured_evaluation["benchmark_name"] == STUDENT_T_GARCH_OFFSET_MODEL
    evaluated = captured_evaluation["forecasts"]
    assert STUDENT_T_GARCH_OFFSET_MODEL in evaluated
    assert STUDENT_T_GARCH_VARA_UNIT_MODEL in evaluated
    assert STUDENT_T_GARCH_VARA_LAMBDA_MODEL in evaluated
    assert RAW_STUDENT_T_GARCH_VARA_UNIT_MODEL in evaluated

    forecast_dir = output / "forecasts"
    expected_lambda = adjusted_student_t.copy()
    expected_lambda[evaluation] += 0.25 * correction[evaluation]
    np.testing.assert_allclose(
        np.load(forecast_dir / "garch11_t_vara_lambda_forecast.npy"),
        expected_lambda,
    )
    assert (forecast_dir / "garch11_t_vara_unit_forecast.npy").exists()
    assert not (forecast_dir / "garch_vara_lambda_forecast.npy").exists()

    contract = json.loads((forecast_dir / "contract.json").read_text(encoding="utf-8"))
    assert contract["garch_variant"] == "student_t"
    assert contract["response_standardization"] == "training_zscore"
    assert contract["innovation_distribution"] == "variance-standardized Student-t"
    assert contract["upstream_garch_output"] == upstream.resolve().as_posix()
    assert contract["model_ids"]["lambda_vara"] == STUDENT_T_GARCH_VARA_LAMBDA_MODEL
    assert contract["forecast_artifacts"]["lambda_vara"] == (
        "garch11_t_vara_lambda_forecast.npy"
    )
    assert "garch11_t_forecast" in contract["correction_input_fingerprints"]
    assert "student_t_garch_contract" in contract["correction_input_fingerprints"]


def test_riskmetrics_ewma_runner_uses_own_residuals_and_provenance(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    output = tmp_path / "riskmetrics_ewma_vara"
    for directory in (
        source / "baseline",
        source / "features" / "snapshot_200712",
        source / "forecasts",
        source / "processed",
    ):
        directory.mkdir(parents=True, exist_ok=True)

    time_count, component_count = 36, 4
    shape = (time_count, component_count)
    dates = pd.date_range("2000-01-31", periods=time_count, freq="ME")
    split = SimpleNamespace(
        validation_indices=np.arange(24, 30, dtype=np.int64),
        test_indices=np.arange(30, 36, dtype=np.int64),
    )
    evaluation = np.concatenate((split.validation_indices, split.test_indices))
    epsilon = 1e-6
    variance_floor = 1e-8
    target = np.tile(np.arange(component_count, dtype=np.float32), (time_count, 1))
    raw_log_forecast = target + np.asarray(
        [1.0, 2.0, 3.0, 4.0],
        dtype=np.float32,
    )
    ewma_variance = np.exp(raw_log_forecast.astype(float)) - epsilon
    zeros = np.zeros(shape, dtype=np.float32)
    np.save(source / "baseline" / "target.npy", target)
    np.save(source / "baseline" / "baseline_forecast.npy", zeros)
    np.save(source / "baseline" / "ar1_forecast.npy", zeros)
    np.save(source / "features" / "snapshot_200712" / "Z.npy", np.eye(component_count))
    np.save(source / "processed" / "dates.npy", dates.to_numpy())
    np.save(source / "processed" / "returns.npy", zeros)
    (source / "processed" / "component_order.csv").write_text(
        "component_id\n0\n1\n2\n3\n",
        encoding="utf-8",
    )
    (source / "forecasts" / "locked_test_config.json").write_text(
        json.dumps({
            "validation_months": 6,
            "test_months": 6,
            "epsilon": epsilon,
            "variance_floor": variance_floor,
        }),
        encoding="utf-8",
    )

    panel = SimpleNamespace(
        dates=dates,
        component_ids=np.arange(component_count),
        metadata=pd.DataFrame({"component": np.arange(component_count)}),
        returns=zeros,
    )
    correction = np.full(shape, 0.2, dtype=np.float32)
    refit_index = np.full(time_count, -1, dtype=np.int64)
    refit_index[evaluation] = 23
    captured_ewma: dict[str, object] = {}
    captured_vara: dict[str, object] = {}
    captured_lambda: dict[str, object] = {}
    captured_evaluation: dict[str, object] = {}
    captured_dm: dict[str, object] = {}
    captured_typed: dict[str, object] = {}
    real_run_vara_family = runner.run_vara_family

    def capture_run_vara_family(request, config, **kwargs):
        captured_typed.update({"request": request, "config": config})
        return real_run_vara_family(request, config, **kwargs)

    def fake_ewma(returns, *, decay, variance_floor):
        captured_ewma.update({
            "returns": returns,
            "decay": decay,
            "variance_floor": variance_floor,
        })
        return ewma_variance.copy()

    def fake_vara(**kwargs):
        captured_vara.update(kwargs)
        lower, upper = kwargs["eta_bounds"]
        return SimpleNamespace(
            correction=correction,
            refit_index=refit_index,
            fit_summary=pd.DataFrame({
                "refit_index": [23],
                "eta_hat": [0.5 * np.log(component_count)],
                "eta_on_boundary": [False],
                "eta_grid_best": [0.5 * np.log(component_count)],
                "eta_grid_best_objective": [-2.0],
                "eta_objective": [-1.5],
                "eta_bounds_lower": [lower],
                "eta_bounds_upper": [upper],
                "eta_grid_size": [kwargs["eta_grid_size"]],
                "eta_local_tolerance": [kwargs["eta_local_tolerance"]],
            }),
        )

    def fake_lambdas(
        baseline_error,
        vara_correction,
        metadata,
        evaluation_indices,
        assigned_refits,
        **kwargs,
    ):
        captured_lambda.update({
            "baseline_error": baseline_error,
            "correction": vara_correction,
            "metadata": metadata,
            "evaluation_indices": evaluation_indices,
            "refit_index": assigned_refits,
            **kwargs,
        })
        weights = np.zeros(shape, dtype=float)
        weights[evaluation] = 0.25
        table = pd.DataFrame({
            "refit_index": [23],
            "group": ["__global__"],
            "series": [component_count],
            "lambda": [0.25],
            "lambda_unbounded": [0.25],
            "ridge": [0.001],
            "fallback_global": [False],
            "lambda_history_sufficient": [True],
        })
        return weights, table

    def fake_evaluate(**kwargs):
        captured_evaluation.update(kwargs)
        models = list(kwargs["forecasts"])
        metrics = pd.DataFrame({
            "model": models,
            "log_risk_rmse": np.ones(len(models)),
            "log_risk_mae": np.ones(len(models)),
            "log_risk_bias": np.zeros(len(models)),
            "absolute_r2": np.zeros(len(models)),
            "gaussian_return_crps": np.ones(len(models)),
            "oos_r2_vs_benchmark": np.zeros(len(models)),
            "rank_score": np.zeros(len(models)),
        })
        return metrics, pd.DataFrame(), {}

    def fake_dm(monthly, **kwargs):
        captured_dm.update({"monthly": monthly, **kwargs})
        return pd.DataFrame()

    monkeypatch.setattr(runner, "load_panel", lambda path: panel)
    monkeypatch.setattr(runner, "relative_panel_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(runner, "riskmetrics_ewma_variance_forecast", fake_ewma)
    monkeypatch.setattr(runner, "run_vara_family", capture_run_vara_family)
    monkeypatch.setattr(vara_runner, "rolling_har_vara_corrections", fake_vara)
    monkeypatch.setattr(vara_runner, "rolling_refit_group_lambdas", fake_lambdas)
    monkeypatch.setattr(runner, "evaluate_risk_forecasts", fake_evaluate)
    monkeypatch.setattr(runner, "_dm_table", fake_dm)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_datamine_garch_vara.py",
            str(source),
            str(output),
            "--ewma-riskmetrics",
            "--response-standardization",
            "training_zscore",
        ],
    )

    runner.main()

    assert captured_ewma["returns"] is panel.returns
    assert captured_ewma["decay"] == pytest.approx(0.97)
    assert captured_ewma["variance_floor"] == pytest.approx(variance_floor)
    offset = 2.5
    adjusted = raw_log_forecast - offset
    expected_error = target - adjusted
    typed_request = captured_typed["request"]
    assert isinstance(typed_request, VARARunRequest)
    assert typed_request.base.model_name == EWMA_OFFSET_MODEL
    np.testing.assert_allclose(typed_request.base.forecast, adjusted)
    np.testing.assert_allclose(typed_request.base.error, expected_error)
    assert isinstance(captured_typed["config"], RollingVARAConfig)
    np.testing.assert_allclose(captured_vara["baseline_errors"], expected_error)
    assert captured_vara["response_standardization"] == "training_zscore"
    np.testing.assert_allclose(captured_lambda["baseline_error"], expected_error)
    np.testing.assert_allclose(captured_lambda["correction"], correction)
    np.testing.assert_array_equal(captured_lambda["evaluation_indices"], evaluation)
    np.testing.assert_array_equal(captured_lambda["refit_index"], refit_index)
    assert captured_evaluation["benchmark_name"] == EWMA_OFFSET_MODEL
    evaluated = captured_evaluation["forecasts"]
    assert EWMA_OFFSET_MODEL in evaluated
    assert EWMA_VARA_UNIT_MODEL in evaluated
    assert EWMA_VARA_LAMBDA_MODEL in evaluated
    assert RAW_EWMA_VARA_UNIT_MODEL in evaluated
    assert set(captured_dm) == {
        "monthly",
        "offset_model",
        "unit_model",
        "lambda_model",
        "raw_model",
        "raw_unit_model",
    }
    assert captured_dm["offset_model"] == EWMA_OFFSET_MODEL
    assert captured_dm["unit_model"] == EWMA_VARA_UNIT_MODEL
    assert captured_dm["lambda_model"] == EWMA_VARA_LAMBDA_MODEL
    assert captured_dm["raw_model"] == "riskmetrics_ewma_raw"
    assert captured_dm["raw_unit_model"] == RAW_EWMA_VARA_UNIT_MODEL

    forecast_dir = output / "forecasts"
    np.testing.assert_allclose(
        np.load(forecast_dir / "riskmetrics_ewma_variance_forecast.npy"),
        ewma_variance,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.load(forecast_dir / "riskmetrics_ewma_raw_forecast.npy"),
        raw_log_forecast,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.load(forecast_dir / "riskmetrics_ewma_validation_offset_forecast.npy"),
        adjusted,
        rtol=1e-6,
    )
    expected_lambda = adjusted.copy()
    expected_lambda[evaluation] += 0.25 * correction[evaluation]
    np.testing.assert_allclose(
        np.load(forecast_dir / "riskmetrics_ewma_vara_lambda_forecast.npy"),
        expected_lambda,
        rtol=1e-6,
    )

    contract = json.loads((forecast_dir / "contract.json").read_text(encoding="utf-8"))
    assert contract["base_variant"] == "riskmetrics_ewma"
    assert contract["response_standardization"] == "training_zscore"
    assert contract["upstream_garch_output"] is None
    assert contract["model_ids"] == {
        "raw": "riskmetrics_ewma_raw",
        "validation_offset": EWMA_OFFSET_MODEL,
        "unit_vara": EWMA_VARA_UNIT_MODEL,
        "lambda_vara": EWMA_VARA_LAMBDA_MODEL,
        "raw_unit_vara": RAW_EWMA_VARA_UNIT_MODEL,
    }
    assert contract["forecast_artifacts"]["raw"] == (
        "riskmetrics_ewma_raw_forecast.npy"
    )
    assert contract["forecast_artifacts"]["conditional_variance"] == (
        "riskmetrics_ewma_variance_forecast.npy"
    )
    assert contract["forecast_artifacts"]["validation_offset"] == (
        "riskmetrics_ewma_validation_offset_forecast.npy"
    )
    assert contract["forecast_artifacts"]["lambda_vara"] == (
        "riskmetrics_ewma_vara_lambda_forecast.npy"
    )
    assert contract["base_model_artifact"] == (
        forecast_dir / "riskmetrics_ewma_raw_forecast.npy"
    ).resolve().as_posix()
    assert contract["offset"] == pytest.approx(offset)
    assert contract["riskmetrics_ewma"] == {
        "decay": 0.97,
        "observation_frequency": "monthly",
        "conditional_mean": 0.0,
        "initialization": "first_finite_squared_return_per_series",
        "missing_return_update": "carry_previous_variance_state",
        "one_step_ahead": True,
        "decay_tuned": False,
        "variance_floor": variance_floor,
    }
    assert "RiskMetrics-style EWMA" in contract["base_log_risk_forecast"]
    assert "RiskMetrics-style EWMA" in contract["vara_response"]
    assert "ewma_return_history" in contract["correction_input_fingerprints"]
