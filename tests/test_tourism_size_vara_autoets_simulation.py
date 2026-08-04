import numpy as np
import pandas as pd
import pytest

from scripts.simulate_tourism_vara_autoets import (
    draw_paper_dgp,
    evaluate_replication,
    learn_pooled_lambda,
    make_origin_schedule,
    rolling_autoets_forecasts,
    rolling_vara_corrections,
    simulate_panel,
    summarize_replications,
)


def test_origin_schedule_separates_history_calibration_and_test() -> None:
    schedule = make_origin_schedule(
        total_time=20,
        history_window=4,
        calibration_origins=3,
        test_origins=2,
    )
    assert schedule.n_autoets_windows == 9
    assert schedule.first_autoets_target == 11
    assert schedule.calibration_positions == (4, 5, 6)
    assert schedule.test_positions == (7, 8)


def test_pooled_lambda_uses_error_on_correction_regression_and_clips() -> None:
    error = np.array([[2.0, 4.0], [6.0, 8.0]])
    correction = np.array([[1.0, 2.0], [3.0, 4.0]])
    fitted, raw = learn_pooled_lambda(error, correction, lambda_cap=1.25)
    assert raw == pytest.approx(2.0)
    assert fitted == pytest.approx(1.25)


def test_summary_pools_squared_rmse_across_equal_sized_replications() -> None:
    metrics = pd.DataFrame({
        "distribution": ["normal", "normal"],
        "rmse_autoets": [1.0, 3.0],
        "rmse_unit": [0.5, 2.5],
        "rmse_tuned": [0.4, 2.0],
        "mae_autoets": [0.8, 2.0],
        "mae_unit": [0.4, 1.8],
        "mae_tuned": [0.3, 1.5],
        "lambda_hat": [0.5, 0.7],
        "oracle_test_lambda": [0.6, 0.8],
        "unit_better": [True, True],
        "tuned_better": [True, True],
        "eta_hat_mean": [1.0, 1.1],
    })
    summary = summarize_replications(metrics).iloc[0]
    assert summary["rmse_autoets"] == pytest.approx(np.sqrt(5.0))
    assert summary["rmse_unit"] == pytest.approx(np.sqrt(3.25))
    assert summary["mean_lambda_hat"] == pytest.approx(0.6)


def test_small_end_to_end_run_uses_autoets_errors_for_vara() -> None:
    p, q = 4, 3
    schedule = make_origin_schedule(
        total_time=32,
        history_window=4,
        calibration_origins=3,
        test_origins=2,
    )
    params, features, attention, _ = draw_paper_dgp(
        seed=123,
        p=p,
        q=q,
        clusters=2,
        screen_features=False,
        feature_max_tries=1,
    )
    panel = simulate_panel(
        seed=321,
        params=params,
        features=features,
        attention=attention,
        total_time=32,
        burn_in=8,
        distribution="normal",
    )
    autoets = rolling_autoets_forecasts(
        panel,
        n_windows=schedule.n_autoets_windows,
        season_length=1,
        n_jobs=1,
        refit_every=0,
    )
    corrections, diagnostics = rolling_vara_corrections(
        autoets["errors"],
        np.asarray(features),
        history_window=schedule.history_window,
        grid_size=3,
        ridge=0.01,
        eta_lower=-2.0,
        eta_upper=2.0,
        stability_cap=None,
    )
    result, predictions = evaluate_replication(
        distribution="normal",
        replication=1,
        autoets=autoets,
        corrections=corrections,
        correction_diagnostics=diagnostics,
        schedule=schedule,
        lambda_cap=1.25,
    )

    assert panel.shape == (32, p)
    assert autoets["errors"].shape == (schedule.n_autoets_windows, p)
    assert np.isnan(corrections[:schedule.history_window]).all()
    assert np.isfinite(corrections[schedule.history_window:]).all()
    assert len(predictions) == schedule.test_origins * p
    assert np.isfinite(result["rmse_autoets"])
    assert np.isfinite(result["rmse_unit"])
    assert np.isfinite(result["rmse_tuned"])
