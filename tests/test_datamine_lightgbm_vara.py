from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.run_datamine_lightgbm_vara as runner


class _FakeObjectRef:
    def __init__(self, *, value=None, error: BaseException | None = None):
        self.value = value
        self.error = error


class _FakeRemoteFunction:
    def __init__(self, ray, function):
        self.ray = ray
        self.function = function

    def remote(self, *args, **kwargs):
        resolved = [
            value.value if isinstance(value, _FakeObjectRef) else value
            for value in args
        ]
        try:
            return _FakeObjectRef(value=self.function(*resolved, **kwargs))
        except BaseException as exc:
            return _FakeObjectRef(error=exc)


class _FakeRay:
    __version__ = "fake-ray"

    def __init__(self):
        self.initialized = False
        self.shutdown_calls = 0

    def is_initialized(self):
        return self.initialized

    def init(self, **kwargs):
        self.initialized = True

    def shutdown(self):
        self.initialized = False
        self.shutdown_calls += 1

    def available_resources(self):
        return {"CPU": 8.0}

    def put(self, value):
        return _FakeObjectRef(value=value)

    def remote(self, **options):
        assert options["num_cpus"] >= 1
        return lambda function: _FakeRemoteFunction(self, function)

    def wait(self, pending, *, num_returns, timeout):
        assert num_returns == 1
        assert timeout == 60.0
        return [pending[-1]], pending[:-1]

    def get(self, reference):
        if reference.error is not None:
            raise reference.error
        return reference.value

    def cancel(self, reference, *, force):
        assert force is True


def _small_parameters() -> dict[str, object]:
    return runner.fixed_lightgbm_parameters(
        seed=17,
        num_threads=1,
        learning_rate=0.1,
        num_leaves=7,
        max_depth=3,
        min_data_in_leaf=2,
        lambda_l2=0.1,
        max_bin=31,
    )


def _small_panel() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(202407)
    target = rng.normal(size=(84, 7)).astype(np.float32)
    target[20, 2] = np.nan
    return target, runner.build_causal_har_features(target)


def test_finite_componentwise_training_normalization_handles_missing_and_constant_series() -> None:
    history = np.asarray(
        [
            [1.0, np.nan, 5.0, np.nan],
            [3.0, 4.0, 5.0, np.nan],
            [np.nan, 6.0, 5.0, np.nan],
        ]
    )

    fitted = runner.fit_finite_componentwise_training_normalization(history)

    np.testing.assert_allclose(fitted["mean"], [2.0, 5.0, 5.0, 0.0])
    np.testing.assert_allclose(fitted["variance"], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(fitted["scale"], [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(fitted["finite_count"], [2, 2, 3, 0])
    np.testing.assert_array_equal(
        fitted["scale_fallback"],
        [False, False, True, True],
    )


def test_training_zscore_is_componentwise_and_predictions_are_inverted() -> None:
    rng = np.random.default_rng(91)
    latent = rng.normal(size=84)
    target = np.column_stack((latent, 10.0 + 3.0 * latent)).astype(np.float32)
    features = runner.build_causal_har_features(target)
    evaluation = np.arange(72, 78)

    result = runner._fit_lightgbm_refit_block(
        target,
        features,
        evaluation,
        block_position=0,
        training_window_months=36,
        num_boost_round=8,
        lightgbm_parameters=_small_parameters(),
        response_normalization="training_zscore",
    )

    expected = runner.fit_finite_componentwise_training_normalization(target[36:72])
    audit = result["normalization_statistics"]
    for name in ("mean", "variance", "scale"):
        np.testing.assert_allclose(audit[name], expected[name], rtol=1e-6)
    np.testing.assert_array_equal(audit["finite_count"], expected["finite_count"])
    np.testing.assert_allclose(
        result["forecast"][:, 1],
        10.0 + 3.0 * result["forecast"][:, 0],
        rtol=1e-5,
        atol=1e-5,
    )
    assert result["fit_row"]["response_normalization"] == "training_zscore"
    assert result["fit_row"]["normalization_statistics_frozen_within_block"] is True
    assert result["fit_row"]["forecast_returned_to_original_log_risk_scale"] is True


def test_causal_har_features_use_only_prior_targets() -> None:
    target = np.arange(30, dtype=float).reshape(15, 2)
    features = runner.build_causal_har_features(target)

    np.testing.assert_allclose(features[12, :, 0], target[11])
    np.testing.assert_allclose(features[12, :, 1], target[9:12].mean(axis=0))
    np.testing.assert_allclose(features[12, :, 2], target[:12].mean(axis=0))

    changed = target.copy()
    changed[12:] += 10_000.0
    changed_features = runner.build_causal_har_features(changed)
    np.testing.assert_allclose(changed_features[12], features[12])


def test_base_coverage_supplies_full_prevalidation_error_window() -> None:
    validation = np.arange(678, 738)
    test = np.arange(738, 858)

    evaluation = runner.lightgbm_base_evaluation_indices(
        validation_indices=validation,
        test_indices=test,
        error_window_months=240,
        refit_frequency_months=12,
    )

    np.testing.assert_array_equal(evaluation, np.arange(438, 858))
    assert len(runner._rolling_refit_blocks(evaluation, 12)) == 35


def test_lightgbm_block_excludes_forecast_labels_and_is_deterministic() -> None:
    target, features = _small_panel()
    evaluation = np.arange(72, 78)
    kwargs = {
        "block_position": 0,
        "training_window_months": 36,
        "num_boost_round": 8,
        "lightgbm_parameters": _small_parameters(),
    }

    first = runner._fit_lightgbm_refit_block(
        target,
        features,
        evaluation,
        **kwargs,
    )
    changed = target.copy()
    changed[evaluation] += 50_000.0
    second = runner._fit_lightgbm_refit_block(
        changed,
        features,
        evaluation,
        **kwargs,
    )

    np.testing.assert_allclose(first["forecast"], second["forecast"], equal_nan=True)
    assert first["fit_row"]["training_end_index"] == 71
    assert first["fit_row"]["training_start_index"] == 36
    assert first["fit_row"]["forecast_start_index"] == 72
    assert first["fit_row"]["response_normalization"] == "none"


def test_fake_ray_lightgbm_matches_serial_and_restores_block_order() -> None:
    target, features = _small_panel()
    evaluation = np.arange(60, 72)
    common = {
        "target": target,
        "causal_features": features,
        "evaluation_indices": evaluation,
        "training_window_months": 36,
        "refit_frequency_months": 4,
        "num_boost_round": 6,
        "lightgbm_parameters": _small_parameters(),
        "max_cpus_per_task": 1,
        "response_normalization": "training_zscore",
    }
    serial = runner.rolling_pooled_lightgbm_forecast(
        **common,
        ray_num_cpus=1,
    )
    fake_ray = _FakeRay()
    parallel = runner.rolling_pooled_lightgbm_forecast(
        **common,
        ray_num_cpus=3,
        ray_module=fake_ray,
    )

    np.testing.assert_allclose(parallel[0], serial[0], equal_nan=True)
    np.testing.assert_array_equal(parallel[1], serial[1])
    pd.testing.assert_frame_equal(
        parallel[2].drop(columns="elapsed_seconds"),
        serial[2].drop(columns="elapsed_seconds"),
    )
    assert parallel[3]["backend"] == "ray"
    assert parallel[3]["task_count"] == 3
    assert parallel[3]["response_normalization"] == "training_zscore"
    for name in (
        "refit_index",
        "mean",
        "variance",
        "scale",
        "finite_count",
        "scale_fallback",
    ):
        np.testing.assert_allclose(parallel[4][name], serial[4][name])
    assert fake_ray.shutdown_calls == 1


def test_lightgbm_response_normalization_rejects_unknown_method() -> None:
    target, features = _small_panel()
    with pytest.raises(ValueError, match="lightgbm_response_normalization"):
        runner.rolling_pooled_lightgbm_forecast(
            target=target,
            causal_features=features,
            evaluation_indices=np.arange(60, 64),
            training_window_months=36,
            refit_frequency_months=4,
            num_boost_round=2,
            lightgbm_parameters=_small_parameters(),
            ray_num_cpus=1,
            max_cpus_per_task=1,
            response_normalization="global_zscore",
        )


def test_lightgbm_vara_combination_uses_requested_weight() -> None:
    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    correction = np.array([[0.5, -0.5], [1.0, -1.0]])

    forecasts = runner.combine_lightgbm_vara_forecasts(
        base_forecast=base,
        correction=correction,
        selected_lambda=0.25,
    )

    np.testing.assert_allclose(forecasts[runner.POOLED_LIGHTGBM_MODEL], base)
    np.testing.assert_allclose(
        forecasts[runner.POOLED_LIGHTGBM_VARA_UNIT_MODEL],
        base + correction,
    )
    np.testing.assert_allclose(
        forecasts[runner.POOLED_LIGHTGBM_VARA_LAMBDA_MODEL],
        base + 0.25 * correction,
    )


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"num_threads": 0}, "num_threads"),
        ({"learning_rate": 0.0}, "learning_rate"),
        ({"num_leaves": 1}, "num_leaves"),
        ({"max_depth": 0}, "max_depth"),
        ({"min_data_in_leaf": 0}, "min_data_in_leaf"),
        ({"lambda_l2": -1.0}, "lambda_l2"),
        ({"max_bin": 2}, "max_bin"),
    ],
)
def test_fixed_lightgbm_parameters_reject_invalid_values(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "seed": 17,
        "num_threads": 1,
        "learning_rate": 0.1,
        "num_leaves": 7,
        "max_depth": 3,
        "min_data_in_leaf": 2,
        "lambda_l2": 0.1,
        "max_bin": 31,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        runner.fixed_lightgbm_parameters(**values)  # type: ignore[arg-type]


def test_lightgbm_main_forwards_typed_vara_request_and_preserves_outputs(
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

    time_count, component_count, feature_count = 60, 4, 3
    shape = (time_count, component_count)
    target = np.ones(shape, dtype=np.float32)
    base_forecast = np.zeros(shape, dtype=np.float32)
    base_error = target.astype(float) - base_forecast.astype(float)
    ar1 = np.full(shape, 0.2, dtype=np.float32)
    har = np.full(shape, 0.3, dtype=np.float32)
    features = np.arange(
        component_count * feature_count,
        dtype=np.float32,
    ).reshape(component_count, feature_count)
    np.save(source / "baseline" / "target.npy", target)
    np.save(source / "baseline" / "baseline_forecast.npy", har)
    np.save(source / "baseline" / "ar1_forecast.npy", ar1)
    np.save(source / "features" / "snapshot_200712" / "Z.npy", features)
    np.save(source / "processed" / "dates.npy", np.arange(time_count))
    (source / "processed" / "component_order.csv").write_text(
        "component\n0\n1\n2\n3\n",
        encoding="utf-8",
    )
    (source / "forecasts" / "locked_test_config.json").write_text(
        json.dumps(
            {
                "validation_months": 6,
                "test_months": 6,
                "epsilon": 1e-6,
                "variance_floor": 1e-8,
            }
        ),
        encoding="utf-8",
    )

    dates = pd.date_range("2000-01-31", periods=time_count, freq="ME")
    metadata = pd.DataFrame({"component": np.arange(component_count)})
    panel = SimpleNamespace(
        dates=dates,
        component_ids=np.arange(component_count),
        metadata=metadata,
        returns=np.zeros(shape, dtype=np.float32),
    )
    split = SimpleNamespace(
        validation_indices=np.arange(48, 54, dtype=np.int64),
        test_indices=np.arange(54, 60, dtype=np.int64),
    )
    base_refit_index = np.full(time_count, -1, dtype=np.int64)
    base_refit_index[24:] = 23
    normalization_audit = {
        "refit_index": np.asarray([23], dtype=np.int64),
        "mean": np.zeros((1, component_count), dtype=np.float32),
        "variance": np.ones((1, component_count), dtype=np.float32),
        "scale": np.ones((1, component_count), dtype=np.float32),
        "finite_count": np.full((1, component_count), 24, dtype=np.int64),
        "scale_fallback": np.zeros((1, component_count), dtype=bool),
    }
    captured: dict[str, object] = {}

    class _MainFakeRay:
        __version__ = "fake-main-ray"

        def __init__(self) -> None:
            self.initialized = False
            self.init_kwargs: dict[str, object] | None = None
            self.shutdown_calls = 0

        def is_initialized(self) -> bool:
            return self.initialized

        def init(self, **kwargs) -> None:
            self.initialized = True
            self.init_kwargs = kwargs

        def shutdown(self) -> None:
            self.initialized = False
            self.shutdown_calls += 1

    fake_ray = _MainFakeRay()

    def fake_base_fit(**kwargs):
        captured["base_kwargs"] = kwargs
        assert kwargs["ray_module"] is fake_ray
        return (
            base_forecast.copy(),
            base_refit_index.copy(),
            pd.DataFrame({"refit_index": [23], "elapsed_seconds": [0.5]}),
            {
                "backend": "ray",
                "ray_used": True,
                "task_unit": "rolling_refit_block",
                "task_count": 1,
                "elapsed_seconds": 0.5,
            },
            normalization_audit,
        )

    correction = np.full(shape, 0.25, dtype=np.float32)
    lambda_weights = np.full(shape, 0.4, dtype=float)
    unit_forecast = base_forecast.astype(float) + correction.astype(float)
    lambda_forecast = base_forecast.astype(float) + 0.4 * correction.astype(float)
    vara_refit_index = np.full(time_count, -1, dtype=np.int64)
    vara_refit_index[np.concatenate((split.validation_indices, split.test_indices))] = 47
    eta_scale = np.log(component_count)
    core_fit_summary = pd.DataFrame(
        {
            "refit_index": [47],
            "eta_hat": [0.4 * eta_scale],
            "eta_on_boundary": [False],
            "eta_grid_best": [0.2 * eta_scale],
            "eta_grid_best_objective": [-2.0],
            "eta_objective": [-1.5],
            "eta_bounds_lower": [-1.5 * eta_scale],
            "eta_bounds_upper": [1.25 * eta_scale],
            "eta_grid_size": [9],
            "eta_local_tolerance": [2e-4 * eta_scale],
        }
    )
    lambda_table = pd.DataFrame(
        {
            "refit_index": [47],
            "group": ["__global__"],
            "series": ["__global__"],
            "lambda": [0.4],
            "lambda_unbounded": [0.4],
            "ridge": [0.007],
            "fallback_global": [False],
            "lambda_history_sufficient": [True],
        }
    )

    def fake_typed_vara(
        request,
        config,
        *,
        execution,
        progress_callback,
        ray_module,
    ):
        captured["request"] = request
        captured["config"] = config
        captured["execution"] = execution
        captured["progress_callback"] = progress_callback
        captured["ray_module"] = ray_module
        assert ray_module is fake_ray
        return SimpleNamespace(
            correction=correction.copy(),
            refit_index=vara_refit_index.copy(),
            core_fit_summary=core_fit_summary.copy(),
            correction_execution={
                "backend": "ray",
                "ray_used": True,
                "task_unit": "rolling_refit_block",
                "task_count": 1,
                "elapsed_seconds": 0.75,
            },
            rolling_lambdas=lambda_weights.copy(),
            lambda_refit_fits=lambda_table.copy(),
            lambda_weights=lambda_weights.copy(),
            unit_forecast=unit_forecast.copy(),
            lambda_forecast=lambda_forecast.copy(),
        )

    metric_models = (
        runner.POOLED_LIGHTGBM_MODEL,
        runner.POOLED_LIGHTGBM_VARA_UNIT_MODEL,
        runner.POOLED_LIGHTGBM_VARA_LAMBDA_MODEL,
        "har_ridge",
        "ar1_ridge",
    )
    metrics = pd.DataFrame(
        {
            "model": metric_models,
            "log_risk_rmse": np.ones(len(metric_models)),
            "log_risk_mae": np.ones(len(metric_models)),
            "gaussian_return_crps": np.ones(len(metric_models)),
            "log_risk_bias": np.zeros(len(metric_models)),
            "absolute_r2": np.zeros(len(metric_models)),
            "rank_score": np.zeros(len(metric_models)),
        }
    )

    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setitem(
        sys.modules,
        "lightgbm",
        SimpleNamespace(__version__="fake-lightgbm"),
    )
    monkeypatch.setattr(runner, "load_panel", lambda path: panel)
    monkeypatch.setattr(runner, "relative_panel_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(runner, "rolling_pooled_lightgbm_forecast", fake_base_fit)
    monkeypatch.setattr(runner, "run_vara_family", fake_typed_vara)
    monkeypatch.setattr(
        runner,
        "evaluate_risk_forecasts",
        lambda **kwargs: (metrics, pd.DataFrame(), {}),
    )
    monkeypatch.setattr(runner, "_dm_table", lambda monthly: pd.DataFrame())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_datamine_lightgbm_vara.py",
            str(source),
            str(output),
            "--error-window-months",
            "24",
            "--response-standardization",
            "training_zscore",
            "--response-clip",
            "6.5",
            "--refit-frequency-months",
            "12",
            "--ridge",
            "0.004",
            "--attention-top-k",
            "3",
            "--attention-cache-mode",
            "cluster",
            "--attention-batch-size",
            "16",
            "--eta-lower",
            "-1.5",
            "--eta-upper",
            "1.25",
            "--eta-grid-size",
            "9",
            "--eta-local-tolerance",
            "0.0002",
            "--lambda-window-months",
            "12",
            "--lambda-ridge",
            "0.007",
            "--ray-num-cpus",
            "2",
            "--max-cpus-per-task",
            "1",
            "--ray-debug",
        ],
    )

    runner.main()

    request = captured["request"]
    assert isinstance(request, runner.VARARunRequest)
    assert isinstance(request.panel, runner.ForecastPanel)
    np.testing.assert_array_equal(request.panel.target, target)
    assert request.panel.series_ids == ("0", "1", "2", "3")
    pd.testing.assert_frame_equal(request.panel.metadata, metadata)
    np.testing.assert_array_equal(
        request.evaluation.validation_indices,
        split.validation_indices,
    )
    np.testing.assert_array_equal(request.evaluation.test_indices, split.test_indices)
    np.testing.assert_array_equal(request.features.values, features)
    assert request.features.names == ("feature_0", "feature_1", "feature_2")
    assert request.features.provenance["source"] == (
        source / "features" / "snapshot_200712" / "Z.npy"
    ).resolve().as_posix()
    assert request.base.model_name == runner.POOLED_LIGHTGBM_MODEL
    np.testing.assert_array_equal(request.base.forecast, base_forecast)
    np.testing.assert_array_equal(request.base.error, base_error)

    config = captured["config"]
    assert isinstance(config, runner.RollingVARAConfig)
    assert config.error_window_months == 24
    assert config.response_standardization == "training_zscore"
    assert config.response_clip == pytest.approx(6.5)
    assert config.refit_frequency_months == 12
    assert config.ridge == pytest.approx(0.004)
    assert config.attention_top_k == 3
    assert config.attention_cache_mode == "cluster"
    assert config.attention_batch_size == 16
    assert config.eta.lower == pytest.approx(-1.5)
    assert config.eta.upper == pytest.approx(1.25)
    assert config.eta.grid_size == 9
    assert config.eta.local_tolerance == pytest.approx(2e-4)
    assert config.lambda_window_months == 12
    assert config.lambda_ridge == pytest.approx(0.007)
    assert config.zero_own_lag_coefficients is False

    execution = captured["execution"]
    assert isinstance(execution, runner.ExecutionConfig)
    assert execution.ray_num_cpus == 2
    assert execution.max_cpus_per_task == 1
    assert execution.ray_debug is True
    assert captured["ray_module"] is fake_ray
    assert callable(captured["progress_callback"])
    assert fake_ray.shutdown_calls == 1

    forecast_dir = output / "forecasts"
    np.testing.assert_array_equal(
        np.load(forecast_dir / "pooled_lightgbm_forecast.npy"),
        base_forecast,
    )
    np.testing.assert_array_equal(
        np.load(forecast_dir / "pooled_lightgbm_error.npy"),
        base_error.astype(np.float32),
    )
    np.testing.assert_array_equal(
        np.load(forecast_dir / "vara_correction.npy"),
        correction,
    )
    np.testing.assert_array_equal(
        np.load(forecast_dir / "pooled_lightgbm_vara_unit_forecast.npy"),
        unit_forecast.astype(np.float32),
    )
    np.testing.assert_array_equal(
        np.load(forecast_dir / "pooled_lightgbm_vara_lambda_forecast.npy"),
        lambda_forecast.astype(np.float32),
    )
    np.testing.assert_array_equal(
        np.load(forecast_dir / "vara_refit_index.npy"),
        vara_refit_index,
    )
    np.testing.assert_array_equal(
        np.load(forecast_dir / "lambda_by_origin.npy"),
        np.full(time_count, 0.4, dtype=np.float32),
    )
    fit_summary = pd.read_csv(forecast_dir / "fit_summary.csv")
    assert fit_summary.loc[0, "eta_hat"] == pytest.approx(0.4)
    assert fit_summary.loc[0, "eta_effective_slope_hat"] == pytest.approx(
        0.4 * eta_scale
    )
    assert fit_summary.loc[0, "lambda_hat"] == pytest.approx(0.4)
    contract = json.loads((forecast_dir / "contract.json").read_text(encoding="utf-8"))
    assert contract["model_ids"] == {
        "base": runner.POOLED_LIGHTGBM_MODEL,
        "unit_vara": runner.POOLED_LIGHTGBM_VARA_UNIT_MODEL,
        "lambda_vara": runner.POOLED_LIGHTGBM_VARA_LAMBDA_MODEL,
    }
    assert contract["response_standardization"] == "training_zscore"
    assert contract["rolling_refit_execution"]["backend"] == "ray"
    assert contract["eta_estimation"]["bounds"] == [-1.5, 1.25]
