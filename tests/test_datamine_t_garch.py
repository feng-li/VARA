from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.run_datamine_t_garch import (
    _validate_ray_resources,
    estimate_validation_offset,
    fit_student_t_garch11_panel,
)
from vara.datamine_garch import (
    fit_student_t_garch11_block,
    fit_student_t_garch11_series,
)


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
        block_position = int(kwargs.get("block_position", -1))
        if block_position == self.ray.raise_on_submit_block:
            raise RuntimeError(f"submission sentinel block {block_position}")
        try:
            if block_position == self.ray.fail_block:
                raise RuntimeError(f"sentinel block {block_position}")
            return _FakeObjectRef(value=self.function(*resolved, **kwargs))
        except BaseException as exc:
            return _FakeObjectRef(error=exc)


class _FakeRay:
    __version__ = "fake-ray"

    def __init__(
        self,
        *,
        initialized: bool = False,
        fail_block: int = -1,
        raise_on_submit_block: int = -1,
    ):
        self.initialized = initialized
        self.fail_block = fail_block
        self.raise_on_submit_block = raise_on_submit_block
        self.init_calls: list[dict[str, object]] = []
        self.shutdown_calls = 0
        self.cancelled: list[_FakeObjectRef] = []
        self.max_pending = 0

    def is_initialized(self):
        return self.initialized

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        self.initialized = True

    def shutdown(self):
        self.shutdown_calls += 1
        self.initialized = False

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
        self.max_pending = max(self.max_pending, len(pending))
        return [pending[-1]], pending[:-1]

    def get(self, reference):
        if reference.error is not None:
            raise reference.error
        return reference.value

    def cancel(self, reference, *, force):
        assert force is True
        self.cancelled.append(reference)


def _panel_kwargs(returns: np.ndarray) -> dict[str, object]:
    return {
        "returns": returns,
        "estimation_end_index": 69,
        "minimum_observations": 48,
        "return_scale": 100.0,
        "variance_floor": 1e-10,
        "max_iterations": 200,
        "tolerance": 1e-8,
        "component_block_size": 2,
        "max_cpus_per_task": 1,
    }


def test_student_t_garch_is_positive_and_future_perturbation_safe() -> None:
    rng = np.random.default_rng(719)
    returns = 0.02 * rng.standard_t(df=6, size=(120, 3))
    original, parameters = fit_student_t_garch11_block(
        returns,
        estimation_end_index=79,
        minimum_observations=48,
    )
    changed = returns.copy()
    # A very large first post-estimation shock makes even a near-zero fitted
    # ARCH coefficient visibly affect the following forecast.
    changed[95] = 1.0e5
    perturbed, changed_parameters = fit_student_t_garch11_block(
        changed,
        estimation_end_index=79,
        minimum_observations=48,
    )

    assert np.isfinite(original).all()
    assert (original > 0.0).all()
    assert (parameters["degrees_of_freedom"] > 2.0).all()
    assert (parameters["persistence"] <= 1.000001).all()
    pd.testing.assert_frame_equal(parameters, changed_parameters)
    np.testing.assert_allclose(original[:96], perturbed[:96], rtol=0.0, atol=0.0)
    assert not np.allclose(original[96:], perturbed[96:])


def test_student_t_garch_fallback_is_explicit_and_forecasts_after_missing_lag() -> None:
    rng = np.random.default_rng(17)
    returns = np.full(80, np.nan)
    returns[20:43] = rng.normal(scale=0.02, size=23)
    returns[60:] = rng.normal(scale=0.02, size=20)

    fit = fit_student_t_garch11_series(
        returns,
        estimation_end_index=49,
        minimum_observations=24,
    )

    assert fit.parameters["used_fallback"] is True
    assert fit.parameters["fit_status"] == "fallback:insufficient_observations"
    assert fit.parameters["degrees_of_freedom"] == 8.0
    assert np.isfinite(fit.second_moment[43:]).all()
    assert (fit.second_moment[43:] > 0.0).all()


def test_ray_component_blocks_match_serial_and_restore_order() -> None:
    rng = np.random.default_rng(37)
    returns = 0.015 * rng.standard_t(df=7, size=(100, 5))
    kwargs = _panel_kwargs(returns)
    serial_variance, serial_parameters, _ = fit_student_t_garch11_panel(
        **kwargs,
        ray_num_cpus=1,
    )
    fake_ray = _FakeRay()

    ray_variance, ray_parameters, execution = fit_student_t_garch11_panel(
        **kwargs,
        ray_num_cpus=3,
        ray_module=fake_ray,
    )

    np.testing.assert_allclose(
        ray_variance,
        serial_variance,
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    pd.testing.assert_frame_equal(ray_parameters, serial_parameters)
    assert list(ray_parameters["component_position"]) == list(range(5))
    assert execution["backend"] == "ray"
    assert execution["task_unit"] == "component_block"
    assert execution["task_count"] == 3
    assert fake_ray.max_pending <= 3
    assert fake_ray.shutdown_calls == 1


@pytest.mark.parametrize(
    "ray_num_cpus,max_cpus_per_task,message",
    [
        (0, 1, "at least one"),
        (2, 0, "at least one"),
        (2, 3, "cannot exceed"),
    ],
)
def test_student_t_garch_ray_resource_validation(
    ray_num_cpus,
    max_cpus_per_task,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_ray_resources(ray_num_cpus, max_cpus_per_task)


def test_ray_component_failure_cancels_pending_and_shuts_owned_runtime() -> None:
    rng = np.random.default_rng(39)
    returns = 0.02 * rng.standard_t(df=6, size=(100, 8))
    fake_ray = _FakeRay(fail_block=1)

    with pytest.raises(RuntimeError, match="sentinel block 1"):
        fit_student_t_garch11_panel(
            **_panel_kwargs(returns),
            ray_num_cpus=3,
            ray_module=fake_ray,
        )

    assert fake_ray.cancelled
    assert fake_ray.shutdown_calls == 1


def test_ray_submission_failure_preserves_external_runtime() -> None:
    rng = np.random.default_rng(41)
    returns = 0.02 * rng.standard_t(df=6, size=(100, 8))
    fake_ray = _FakeRay(initialized=True, raise_on_submit_block=1)

    with pytest.raises(RuntimeError, match="submission sentinel block 1"):
        fit_student_t_garch11_panel(
            **_panel_kwargs(returns),
            ray_num_cpus=3,
            ray_module=fake_ray,
        )

    assert len(fake_ray.cancelled) == 1
    assert fake_ray.init_calls == []
    assert fake_ray.shutdown_calls == 0


def test_validation_offset_uses_only_matched_validation_cells() -> None:
    target = np.asarray([[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]])
    forecast = np.asarray([[2.0, 4.0], [6.0, np.nan], [-100.0, -100.0]])
    reference = np.zeros_like(target)

    result = estimate_validation_offset(
        target=target,
        forecast=forecast,
        reference_forecast=reference,
        validation_indices=[0, 1],
    )

    assert result["validation_cells"] == 3
    assert result["offset"] == pytest.approx(2.0)
    assert result["adjusted_validation_bias"] == pytest.approx(0.0)
