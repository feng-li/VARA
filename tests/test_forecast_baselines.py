import numpy as np
from statsforecast.models import AutoETS

from vara.forecast_baselines import (
    auto_ets_baseline_panel,
    auto_ets_config,
    make_auto_ets_model,
)


def test_autoets_configuration_is_standard_and_has_no_validation_tuning() -> None:
    config = auto_ets_config(12)
    assert config == {
        "model": "AutoETS",
        "implementation": "statsforecast.models.AutoETS",
        "alias": "auto_ets",
        "model_specification": "ZZZ",
        "season_length": 12,
        "damped": None,
        "phi": None,
        "selection_criterion": "AICc",
        "optimization_criterion": "likelihood",
        "allow_multiplicative_trend": False,
        "freq": 1,
    }

    model = make_auto_ets_model(12)
    assert model.model == "ZZZ"
    assert model.season_length == 12
    assert model.damped is None
    assert model.phi is None
    assert model.alias == "auto_ets"


def test_frozen_autoets_constructor_matches_statsforecast_standard_default() -> None:
    time = np.arange(60, dtype=float)
    history = 100.0 + 0.5 * time + 8.0 * np.sin(2.0 * np.pi * time / 12.0)

    frozen = make_auto_ets_model(12).fit(history)
    standard = AutoETS(season_length=12, alias="auto_ets").fit(history)

    assert frozen.model_["method"] == standard.model_["method"]
    np.testing.assert_allclose(
        frozen.predict(h=1)["mean"],
        standard.predict(h=1)["mean"],
        rtol=0.0,
        atol=1e-12,
    )


def test_fixed_ann_autoets_configuration_is_a_separate_prespecified_family() -> None:
    config = auto_ets_config(
        12,
        model_specification="ann",
        damped=False,
    )
    assert config["model_specification"] == "ANN"
    assert config["damped"] is False
    assert config["phi"] is None
    assert config["selection_criterion"] is None

    model = make_auto_ets_model(
        12,
        model_specification="ANN",
        damped=False,
    )
    assert model.model == "ANN"
    assert model.damped is False


def test_autoets_rejects_damping_for_a_no_trend_family() -> None:
    try:
        auto_ets_config(12, model_specification="ANN", damped=True)
    except ValueError as exc:
        assert "without trend cannot be damped" in str(exc)
    else:
        raise AssertionError("Expected damping with ETS(A,N,N) to be rejected.")


def test_autoets_training_zscore_returns_response_scale_forecasts() -> None:
    time = np.arange(60, dtype=float)
    history = np.column_stack([
        100.0 + 0.5 * time + 8.0 * np.sin(2.0 * np.pi * time / 12.0),
        300.0 + 2.0 * time + 20.0 * np.cos(2.0 * np.pi * time / 12.0),
    ])

    result = auto_ets_baseline_panel(
        history,
        horizon=2,
        seasonal_lag=12,
        series_ids=["A", "B"],
        model_specification="ZZA",
        response_normalization="training_zscore",
    )

    mean = history.mean(axis=0)
    variance = history.var(axis=0)
    scale = np.sqrt(variance)
    np.testing.assert_allclose(result["training_mean"], mean)
    np.testing.assert_allclose(result["training_variance"], variance)
    np.testing.assert_allclose(result["training_scale"], scale)
    np.testing.assert_allclose(
        result["forecast"],
        mean[None, :] + scale[None, :] * result["forecast_normalized"],
    )
    np.testing.assert_allclose(
        result["forecast_variance"],
        variance[None, :] * result["forecast_variance_normalized"],
    )
    np.testing.assert_allclose(
        result["residuals"],
        history - result["fitted"],
    )
    assert result["response_normalization"] == "training_zscore"
