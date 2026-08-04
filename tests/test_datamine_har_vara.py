from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import scripts.run_datamine_har_vara as runner
import vara.datamine_vara_runner as vara_runner

from scripts.run_datamine_har_vara import (
    HAR_VARA_LAMBDA_MODEL,
    HAR_VARA_UNIT_MODEL,
    _validate_ray_resources,
    combine_har_vara_forecasts,
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
            value = self.function(*resolved, **kwargs)
            return _FakeObjectRef(value=value)
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
        self.init_calls = []
        self.shutdown_calls = 0
        self.cancelled = []

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
        return [pending[-1]], pending[:-1]

    def get(self, reference):
        if reference.error is not None:
            raise reference.error
        return reference.value

    def cancel(self, reference, *, force):
        assert force is True
        self.cancelled.append(reference)


def test_har_vara_combinations_include_unit_and_selected_lambda() -> None:
    base = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    correction = np.asarray([[0.5, -0.5], [1.0, -1.0]])

    forecasts = combine_har_vara_forecasts(
        har_forecast=base,
        correction=correction,
        selected_lambda=0.25,
    )

    np.testing.assert_allclose(forecasts[HAR_VARA_UNIT_MODEL], base + correction)
    np.testing.assert_allclose(
        forecasts[HAR_VARA_LAMBDA_MODEL],
        base + 0.25 * correction,
    )


def test_har_vara_combinations_reject_inconsistent_shapes() -> None:
    with np.testing.assert_raises_regex(ValueError, "same shape"):
        combine_har_vara_forecasts(
            har_forecast=np.zeros((2, 2)),
            correction=np.zeros((2, 1)),
            selected_lambda=0.5,
        )


def test_har_vara_combinations_accept_origin_specific_lambdas() -> None:
    base = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    correction = np.asarray([[0.5, -0.5], [1.0, -1.0]])
    lambdas = np.asarray([[0.0, 0.0], [0.25, 0.25]])

    forecasts = combine_har_vara_forecasts(
        har_forecast=base,
        correction=correction,
        selected_lambda=lambdas,
    )

    np.testing.assert_allclose(
        forecasts[HAR_VARA_LAMBDA_MODEL],
        base + lambdas * correction,
    )


def _small_rolling_kwargs():
    rng = np.random.default_rng(37)
    errors = rng.normal(size=(80, 6))
    features = rng.normal(size=(6, 4))
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return {
        "baseline_errors": errors,
        "evaluation_indices": np.arange(60, 67),
        "feature_matrix": features,
        "error_window_months": 40,
        "response_standardization": "median_mad",
        "response_clip": 8.0,
        "refit_frequency_months": 3,
        "ridge": 0.01,
        "attention_top_k": 2,
        "attention_cache_mode": "exact",
        "attention_batch_size": 8,
        "eta_bounds": (-1.0, 1.0),
        "eta_grid_size": 3,
        "eta_local_tolerance": 0.05,
        "zero_own_lag_coefficients": False,
    }


@pytest.mark.parametrize(
    "response_standardization",
    ["median_mad", "training_zscore"],
)
def test_ray_refit_blocks_match_sequential_and_restore_order(
    response_standardization,
) -> None:
    kwargs = _small_rolling_kwargs()
    kwargs["response_standardization"] = response_standardization
    sequential = runner.rolling_har_vara_corrections(**kwargs)
    callback_refits = []
    fake_ray = _FakeRay()

    parallel, execution = runner.rolling_har_vara_corrections_ray(
        **kwargs,
        ray_num_cpus=3,
        max_cpus_per_task=1,
        progress_callback=lambda row: callback_refits.append(int(row["refit_index"])),
        ray_module=fake_ray,
    )

    np.testing.assert_allclose(
        parallel.standardized_forecast,
        sequential.standardized_forecast,
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_allclose(
        parallel.correction,
        sequential.correction,
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )
    np.testing.assert_array_equal(parallel.refit_index, sequential.refit_index)
    pd.testing.assert_frame_equal(parallel.fit_summary, sequential.fit_summary)
    assert callback_refits == [59, 62, 65]
    assert execution["backend"] == "ray"
    assert execution["task_count"] == 3
    assert fake_ray.shutdown_calls == 1


@pytest.mark.parametrize(
    "ray_num_cpus,max_cpus_per_task,message",
    [
        (0, 1, "at least one"),
        (2, 0, "at least one"),
        (2, 3, "cannot exceed"),
    ],
)
def test_ray_resource_validation(
    ray_num_cpus,
    max_cpus_per_task,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        _validate_ray_resources(ray_num_cpus, max_cpus_per_task)


def test_ray_refit_failure_cancels_pending_and_shuts_down() -> None:
    fake_ray = _FakeRay(fail_block=1)
    with pytest.raises(RuntimeError, match="sentinel block 1"):
        runner.rolling_har_vara_corrections_ray(
            **_small_rolling_kwargs(),
            ray_num_cpus=3,
            max_cpus_per_task=1,
            ray_module=fake_ray,
        )

    assert fake_ray.cancelled
    assert fake_ray.shutdown_calls == 1


def test_ray_submission_failure_cancels_without_shutting_external_runtime() -> None:
    fake_ray = _FakeRay(initialized=True, raise_on_submit_block=1)
    with pytest.raises(RuntimeError, match="submission sentinel block 1"):
        runner.rolling_har_vara_corrections_ray(
            **_small_rolling_kwargs(),
            ray_num_cpus=3,
            max_cpus_per_task=1,
            ray_module=fake_ray,
        )

    assert len(fake_ray.cancelled) == 1
    assert fake_ray.init_calls == []
    assert fake_ray.shutdown_calls == 0


def test_ray_refits_reject_nonexact_attention_cache() -> None:
    kwargs = _small_rolling_kwargs()
    kwargs["attention_cache_mode"] = "cluster"
    fake_ray = _FakeRay()
    with pytest.raises(ValueError, match="require attention_cache_mode='exact'"):
        runner.rolling_har_vara_corrections_ray(
            **kwargs,
            ray_num_cpus=3,
            max_cpus_per_task=1,
            ray_module=fake_ray,
        )
    assert fake_ray.shutdown_calls == 1


def test_har_runner_forwards_paper_eta_on_effective_scale(
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
    ones = np.ones(shape, dtype=np.float32)
    np.save(source / "baseline" / "target.npy", ones)
    np.save(source / "baseline" / "baseline_forecast.npy", zeros)
    np.save(source / "baseline" / "baseline_error.npy", ones)
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
    (source / "baseline" / "baseline_manifest.json").write_text("{}", encoding="utf-8")

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

    def fake_vara(**kwargs):
        captured.update(kwargs)
        refit_index = np.full(time_count, -1, dtype=np.int64)
        refit_index[split.validation_indices] = 23
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

    monkeypatch.setattr(runner, "load_panel", lambda path: panel)
    monkeypatch.setattr(runner, "relative_panel_split", lambda *args, **kwargs: split)
    monkeypatch.setattr(vara_runner, "rolling_har_vara_corrections", fake_vara)
    monkeypatch.setattr(vara_runner, "rolling_refit_group_lambdas", fake_lambdas)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_datamine_har_vara.py",
            str(source),
            str(output),
            "--response-standardization",
            "training_zscore",
        ],
    )

    runner.main()

    scale = np.log(component_count)
    assert captured["eta_bounds"] == pytest.approx((-2.0 * scale, 2.0 * scale))
    assert captured["eta_local_tolerance"] == pytest.approx(1e-4 * scale)
    assert captured["response_standardization"] == "training_zscore"
    saved = pd.read_csv(output / "forecasts" / "fit_summary.csv")
    assert saved.loc[0, "eta_hat"] == pytest.approx(0.8)
    assert saved.loc[0, "eta_effective_slope_hat"] == pytest.approx(0.8 * scale)
    assert np.load(output / "forecasts" / "vara_correction.npy").dtype == np.float32
    initial_contract = json.loads(
        (output / "forecasts" / "contract.json").read_text(encoding="utf-8")
    )
    assert initial_contract["response_standardization"] == "training_zscore"
    assert initial_contract["rolling_refit_execution"]["backend"] == "sequential"

    def unexpected_refit(**kwargs):
        raise AssertionError("Reuse must not refit or reconvert eta.")

    monkeypatch.setattr(vara_runner, "rolling_har_vara_corrections", unexpected_refit)
    monkeypatch.setattr(vara_runner, "rolling_har_vara_corrections_ray", unexpected_refit)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_datamine_har_vara.py",
            str(source),
            str(output),
            "--reuse-saved-correction",
            "--response-standardization",
            "training_zscore",
            "--ray-num-cpus",
            "2",
        ],
    )
    runner.main()
    reused = pd.read_csv(output / "forecasts" / "fit_summary.csv")
    assert reused.loc[0, "eta_hat"] == pytest.approx(0.8)
    assert reused.loc[0, "eta_effective_slope_hat"] == pytest.approx(0.8 * scale)
    reused_contract = json.loads(
        (output / "forecasts" / "contract.json").read_text(encoding="utf-8")
    )
    assert reused_contract["rolling_refit_execution"] == initial_contract[
        "rolling_refit_execution"
    ]
    assert reused_contract["reused_saved_correction"] is True

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_datamine_har_vara.py",
            str(source),
            str(output),
            "--reuse-saved-correction",
        ],
    )
    with pytest.raises(ValueError, match="response_standardization"):
        runner.main()

    monkeypatch.setattr(
        "sys.argv",
        [
            "run_datamine_har_vara.py",
            str(source),
            str(output),
            "--reuse-saved-correction",
            "--response-standardization",
            "training_zscore",
        ],
    )
    np.save(source / "baseline" / "baseline_error.npy", 2.0 * ones)
    with pytest.raises(ValueError, match="correction_input_fingerprints"):
        runner.main()
