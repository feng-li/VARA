from __future__ import annotations

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
