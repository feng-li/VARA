from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.recombine_tourism_76_reliability_vara import (
    CORE_MODEL_VARIANCE,
    CORE_MODEL,
    DIRECT_AUTO_ETS_VARIANCE,
    MODEL,
    MODEL_VARIANCE,
    RAW_CORRECTION_VARIANCE,
    RAW_OOS_MODEL,
    RAW_OOS_MODEL_VARIANCE,
    STATE_CORRECTION_VARIANCE,
    ReliabilityCombinationConfig,
    apply_reliability_gates,
    attach_plugin_forecast_variances,
    build_reliability_mapping,
    learn_group_season_lambdas,
    learn_reliability_gates,
    learn_reliability_offsets,
    learn_state_residual_variances,
    rolling_recombine,
)


def test_reliability_groups_use_only_rows_through_cutoff() -> None:
    rows = []
    for date, multiplier in (("2000-01-01", 1.0), ("2010-01-01", 1000.0)):
        for index, series_id in enumerate(("AAAAll", "AABAll", "BAAAll", "BABAll")):
            rows.append({
                "date": date,
                "series_id": series_id,
                "actual": float(index + 1),
                "direct_auto_ets": float(index + 1) + multiplier * index,
            })
    forecasts = pd.DataFrame(rows)
    mapping = build_reliability_mapping(
        forecasts,
        history_end="2009-12-01",
        n_bins=2,
        expected_n_series=4,
    ).set_index("series_id")

    assert mapping.loc["AAAAll", "historical_autoets_mae"] == 0.0
    assert mapping.loc["BABAll", "historical_autoets_mae"] == 3.0
    assert mapping["reliability_group"].value_counts().to_dict() == {0: 2, 1: 2}


def test_group_season_lambdas_are_pooled_and_capped() -> None:
    train = pd.DataFrame({
        "group": ["A", "A", "B", "B"],
        "month": [1, 1, 1, 1],
        "state_correction": [1.0, 1.0, 1.0, 1.0],
        "error_target": [2.0, 2.0, -1.0, -1.0],
    })
    lambdas, global_lambda, prior_information = learn_group_season_lambdas(
        train,
        prior_strength=1.0,
        cap=1.25,
    )

    assert global_lambda == 0.5
    assert prior_information == 2.0
    assert lambdas.loc[("A", 1)] == 1.25
    assert lambdas.loc[("B", 1)] == 0.0


def test_reliability_offsets_are_group_medians() -> None:
    train = pd.DataFrame({
        "reliability_group": [0, 0, 1, 1],
        "actual": [2.0, 4.0, 10.0, 14.0],
    })
    offsets, global_offset = learn_reliability_offsets(
        train,
        np.array([1.0, 1.0, 8.0, 8.0]),
        quantile=0.5,
        prior_observations=0.0,
    )

    assert offsets.loc[0] == 2.0
    assert offsets.loc[1] == 4.0
    assert global_offset == 2.5


def test_state_residual_variances_shrink_series_toward_group() -> None:
    rows = []
    residuals = {
        "A": (0, [-1.0, 0.0, 1.0]),
        "B": (0, [-2.0, 0.0, 2.0]),
        "C": (1, [-3.0, 0.0, 3.0]),
        "D": (1, [-4.0, 0.0, 4.0]),
    }
    for series_id, (group, values) in residuals.items():
        for value in values:
            rows.append({
                "series_id": series_id,
                "reliability_group": group,
                "error_target": value,
                "state_correction": 0.0,
            })
    train = pd.DataFrame(rows)
    series_variances, group_variances, global_variance = (
        learn_state_residual_variances(train, prior_observations=2.0)
    )

    expected_global = 60.0 / 11.0
    expected_group_zero = (5.0 * 2.0 + 2.0 * expected_global) / 7.0
    expected_group_one = (5.0 * 10.0 + 2.0 * expected_global) / 7.0
    assert np.isclose(global_variance, expected_global)
    assert np.isclose(group_variances.loc[0], expected_group_zero)
    assert np.isclose(group_variances.loc[1], expected_group_one)
    assert np.isclose(
        series_variances.loc["A"],
        (2.0 * 1.0 + 2.0 * expected_group_zero) / 4.0,
    )
    assert np.isclose(
        series_variances.loc["D"],
        (2.0 * 16.0 + 2.0 * expected_group_one) / 4.0,
    )


def test_plugin_variance_uses_gate_and_squared_lambda() -> None:
    predictions = pd.DataFrame({
        "date": pd.to_datetime(["2016-01-01", "2016-01-01"]),
        "series_id": ["A", "B"],
        "gate": [0.0, 1.0],
        "lambda": [2.0, 0.5],
        STATE_CORRECTION_VARIANCE: [4.0, 4.0],
        RAW_CORRECTION_VARIANCE: [16.0, 25.0],
    })
    baseline_variances = pd.DataFrame({
        "date": pd.to_datetime(["2016-01-01", "2016-01-01"]),
        "series_id": ["A", "B"],
        DIRECT_AUTO_ETS_VARIANCE: [9.0, 9.0],
    })

    result = attach_plugin_forecast_variances(
        predictions,
        baseline_variances,
    )

    np.testing.assert_allclose(result[CORE_MODEL_VARIANCE], [25.0, 10.0])
    np.testing.assert_allclose(result[MODEL_VARIANCE], [9.0, 10.0])
    np.testing.assert_allclose(result[RAW_OOS_MODEL_VARIANCE], [25.0, 34.0])


def _state_rows() -> pd.DataFrame:
    rows = []
    for date_index, date in enumerate(pd.date_range("2015-01-01", periods=4, freq="MS")):
        for series_id, group, reliability_group in (
            ("AAAAll", "A", 0),
            ("BAAAll", "B", 1),
        ):
            rows.append({
                "date": date,
                "series_id": series_id,
                "group": group,
                "month": 1,
                "actual": 10.0 + date_index,
                "direct_auto_ets": 9.0 + date_index,
                "vara_correction": 0.5,
                "state_correction": 1.0,
                "error_target": 1.0,
                "reliability_group": reliability_group,
            })
    return pd.DataFrame(rows)


def test_rolling_recombine_ignores_future_rows() -> None:
    config = ReliabilityCombinationConfig(
        calibration_window_months=2,
        lambda_prior_strength=1.0,
        lambda_cap=1.25,
        reliability_bins=2,
    )
    state = _state_rows()
    target = pd.Timestamp("2015-03-01")
    first, _, _ = rolling_recombine(state, [target], config)

    changed = state.copy()
    changed.loc[changed["date"] > target, ["actual", "error_target"]] = 1_000_000.0
    second, _, _ = rolling_recombine(changed, [target], config)

    np.testing.assert_allclose(first[CORE_MODEL], second[CORE_MODEL])
    np.testing.assert_allclose(first[RAW_OOS_MODEL], second[RAW_OOS_MODEL])
    np.testing.assert_allclose(first[RAW_OOS_MODEL], [11.5, 11.5])
    np.testing.assert_allclose(
        first[STATE_CORRECTION_VARIANCE],
        second[STATE_CORRECTION_VARIANCE],
    )
    np.testing.assert_allclose(
        first[RAW_CORRECTION_VARIANCE],
        second[RAW_CORRECTION_VARIANCE],
    )
    assert (first["calibration_end"] < first["date"]).all()
    assert first["n_calibration_months"].eq(2).all()


def test_rolling_recombine_allows_prespecified_expanding_warmup() -> None:
    config = ReliabilityCombinationConfig(
        calibration_window_months=3,
        calibration_min_months=2,
        lambda_prior_strength=1.0,
        lambda_cap=1.25,
        reliability_bins=2,
    )
    state = _state_rows()

    predictions, _, _ = rolling_recombine(
        state,
        [pd.Timestamp("2015-03-01"), pd.Timestamp("2015-04-01")],
        config,
    )

    observed = (
        predictions.groupby("date", observed=True)["n_calibration_months"]
        .first()
        .to_dict()
    )
    assert observed == {
        pd.Timestamp("2015-03-01"): 2,
        pd.Timestamp("2015-04-01"): 3,
    }


def test_reliability_gate_uses_validation_mae() -> None:
    validation = pd.DataFrame({
        "reliability_group": [0, 0, 1, 1],
        "actual": [10.0, 10.0, 10.0, 10.0],
        "direct_auto_ets": [9.0, 9.0, 6.0, 6.0],
        CORE_MODEL: [7.0, 7.0, 9.0, 9.0],
    })
    gates = learn_reliability_gates(validation)
    assert gates.set_index("reliability_group")["gate"].to_dict() == {
        0: 0.0,
        1: 1.0,
    }

    applied = apply_reliability_gates(validation, gates)
    np.testing.assert_allclose(applied[MODEL], [9.0, 9.0, 9.0, 9.0])
