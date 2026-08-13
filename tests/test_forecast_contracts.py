from dataclasses import FrozenInstanceError
import math

import numpy as np
import pandas as pd
import pytest

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


def _valid_parts(*, include_test: bool = False):
    dates = pd.date_range("2000-01-31", periods=36, freq="ME")
    target = np.arange(36 * 3, dtype=np.float32).reshape(36, 3) / 10.0
    forecast = target - np.float32(0.25)
    error = np.full_like(target, np.float32(0.25))
    target[25, 1] = np.nan
    error[25, 1] = np.nan
    panel = ForecastPanel(
        dates=dates,
        series_ids=("s1", "s2", "s3"),
        target=target,
        metadata=pd.DataFrame({"series_id": ("s1", "s2", "s3")}),
    )
    evaluation = RollingEvaluationPlan(
        validation_indices=np.array([24, 25], dtype=np.int64),
        test_indices=(
            np.array([26, 27], dtype=np.int64)
            if include_test
            else np.array([], dtype=np.int64)
        ),
    )
    features = StaticFeatureSet(
        values=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]),
        names=("formula", "type"),
        provenance={"snapshot": "1999-12"},
    )
    base = BaseForecastResult(
        model_name="base",
        forecast=forecast,
        error=error,
    )
    return panel, evaluation, features, base


def test_validation_only_request_preserves_authoritative_base_error() -> None:
    panel, evaluation, features, base = _valid_parts()

    request = VARARunRequest(
        panel=panel,
        evaluation=evaluation,
        features=features,
        base=base,
    )

    assert request.base.error is base.error
    np.testing.assert_array_equal(request.evaluation.evaluation_indices, [24, 25])
    assert request.panel.shape == (36, 3)
    assert request.features.feature_count == 2


def test_full_plan_combines_validation_then_test_indices() -> None:
    _, evaluation, _, _ = _valid_parts(include_test=True)

    np.testing.assert_array_equal(
        evaluation.evaluation_indices,
        np.array([24, 25, 26, 27]),
    )
    assert evaluation.origin_count == 4


def test_request_accepts_float32_rounding_but_rejects_changed_error() -> None:
    panel, evaluation, features, base = _valid_parts()
    rounded_error = base.error.copy()
    rounded_error[24, 0] += np.float32(8e-7)
    rounded = BaseForecastResult("base", base.forecast, rounded_error)
    VARARunRequest(panel, evaluation, features, rounded)

    changed_error = base.error.copy()
    changed_error[24, 0] += np.float32(2e-5)
    changed = BaseForecastResult("base", base.forecast, changed_error)
    with pytest.raises(ValueError, match="authoritative base error is inconsistent"):
        VARARunRequest(panel, evaluation, features, changed)


def test_request_rejects_non_authoritative_error_missingness() -> None:
    panel, evaluation, features, base = _valid_parts()
    error = base.error.copy()
    error[24, 0] = np.nan

    with pytest.raises(ValueError, match="finite cells must exactly match"):
        VARARunRequest(
            panel,
            evaluation,
            features,
            BaseForecastResult("base", base.forecast, error),
        )


@pytest.mark.parametrize(
    ("validation", "test", "message"),
    [
        ([], [], "validation_indices must be nonempty"),
        ([0, 1], [], "positive target indices"),
        ([2, 1], [], "strictly increasing"),
        ([1, 2], [2, 3], "must begin after"),
    ],
)
def test_evaluation_plan_rejects_invalid_target_order(
    validation: list[int],
    test: list[int],
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        RollingEvaluationPlan(validation, test)


def test_panel_and_feature_dimensions_are_checked_by_request() -> None:
    panel, evaluation, _, base = _valid_parts()
    wrong_features = StaticFeatureSet(
        values=np.ones((2, 1)),
        names=("x",),
        provenance={},
    )

    with pytest.raises(ValueError, match="feature rows"):
        VARARunRequest(panel, evaluation, wrong_features, base)


def test_panel_rejects_duplicate_series_and_nonmonotone_dates() -> None:
    target = np.ones((2, 2))
    with pytest.raises(ValueError, match="series_ids must be unique"):
        ForecastPanel(
            dates=pd.date_range("2000-01-31", periods=2, freq="ME"),
            series_ids=("same", "same"),
            target=target,
            metadata=pd.DataFrame(index=range(2)),
        )
    with pytest.raises(TypeError, match="metadata must be a pandas DataFrame"):
        ForecastPanel(
            dates=pd.date_range("2000-01-31", periods=2, freq="ME"),
            series_ids=("a", "b"),
            target=target,
            metadata=None,
        )
    with pytest.raises(ValueError, match="dates must be strictly increasing"):
        ForecastPanel(
            dates=pd.to_datetime(["2000-02-29", "2000-01-31"]),
            series_ids=("a", "b"),
            target=target,
            metadata=pd.DataFrame(index=range(2)),
        )


def test_paper_eta_resolves_to_application_effective_slope() -> None:
    eta = PaperEtaConfig(lower=-1.5, upper=1.0, grid_size=5, local_tolerance=2e-4)

    resolved = eta.resolve(76)

    assert resolved.component_count == 76
    assert resolved.effective_bounds == pytest.approx(
        (-1.5 * math.log(76), math.log(76))
    )
    assert resolved.effective_local_tolerance == pytest.approx(2e-4 * math.log(76))
    with pytest.raises(ValueError, match="at least two"):
        eta.resolve(1)


def test_vara_and_execution_configs_validate_current_modes() -> None:
    config = RollingVARAConfig(
        response_standardization="training_zscore",
        eta=PaperEtaConfig(),
    )
    assert config.response_standardization == "training_zscore"
    assert config.attention_mode == "topk"
    assert config.eta.grid_size == 7
    assert RollingVARAConfig(attention_mode="dense").attention_mode == "dense"
    assert ExecutionConfig(ray_num_cpus=15, max_cpus_per_task=1).ray_num_cpus == 15

    with pytest.raises(ValueError, match="at least 24"):
        RollingVARAConfig(error_window_months=23)
    with pytest.raises(ValueError, match="median_mad or training_zscore"):
        RollingVARAConfig(response_standardization="global")
    with pytest.raises(TypeError, match="ridge must be a real number"):
        RollingVARAConfig(ridge=True)
    with pytest.raises(ValueError, match="attention_mode must be topk or dense"):
        RollingVARAConfig(attention_mode="approximate")
    with pytest.raises(TypeError, match="eta lower must be a real number"):
        PaperEtaConfig(lower=False)
    with pytest.raises(ValueError, match="cannot exceed"):
        ExecutionConfig(ray_num_cpus=1, max_cpus_per_task=2)


def test_contracts_are_frozen_and_slotted() -> None:
    config = ExecutionConfig()
    assert not hasattr(config, "__dict__")
    with pytest.raises(FrozenInstanceError):
        config.ray_num_cpus = 2  # type: ignore[misc]
