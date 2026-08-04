from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from scripts.evaluate_tourism_raw_vara import (
    DEFAULT_STRATEGY,
    MODEL,
    add_raw_vara_to_comparison,
    build_raw_vara_predictions,
    depth_three_all_series,
)


def test_depth_three_all_series_selection() -> None:
    columns = [
        "date",
        "TotalAll",
        "AAAll",
        "AAAAll",
        "AABAll",
        "AAAHol",
    ]
    assert depth_three_all_series(columns) == ["AAAAll", "AABAll"]


def test_raw_vara_uses_only_prior_raw_levels_and_no_autoets() -> None:
    dates = pd.date_range("2015-10-01", periods=5, freq="MS")
    raw = pd.DataFrame({
        "date": dates,
        "AAAAll": [1.0, 2.0, 3.0, 4.0, 5.0],
        "AABAll": [10.0, 20.0, 30.0, 40.0, 50.0],
    })
    calls = []

    def fake_task(**kwargs) -> dict:
        history = np.asarray(kwargs["error_history"], dtype=float)
        calls.append((kwargs["target_date"], history.copy()))
        return {
            "ok": True,
            "correction": history[-1] + np.asarray([1.0, -50.0]),
            "history_start": kwargs["history_start"],
            "history_end": kwargs["history_end"],
            "n_train": len(history),
            "eta_hat": 0.0,
            "eta_on_boundary": False,
            "stationarity_proxy": 0.25,
        }

    result = build_raw_vara_predictions(
        raw,
        final_start="2016-01-01",
        final_end="2016-02-01",
        strategy=replace(DEFAULT_STRATEGY, history_window=2),
        series_ids=["AAAAll", "AABAll"],
        task_runner=fake_task,
    )

    assert len(calls) == 2
    np.testing.assert_allclose(calls[0][1], [[2.0, 20.0], [3.0, 30.0]])
    np.testing.assert_allclose(calls[1][1], [[3.0, 30.0], [4.0, 40.0]])
    np.testing.assert_allclose(
        result.pivot(index="date", columns="series_id", values=MODEL).to_numpy(),
        [[4.0, 0.0], [5.0, 0.0]],
    )
    assert not result["autoets_used"].any()
    assert (result["history_end"] < result["date"]).all()


def test_raw_vara_is_added_and_rank_score_is_recomputed() -> None:
    existing = pd.DataFrame({
        "model": ["auto_ets", "worst"],
        "rmse": [1.0, 3.0],
        "mae": [1.0, 3.0],
        "mase": [1.0, 3.0],
        "rmsse": [1.0, 3.0],
    })
    raw_vara = pd.DataFrame({
        "model": [MODEL],
        "rmse": [2.0],
        "mae": [2.0],
        "mase": [2.0],
        "rmsse": [2.0],
    })

    ranked = add_raw_vara_to_comparison(raw_vara, existing).set_index("model")

    assert ranked.loc["auto_ets", "rank_score"] == 100.0
    assert ranked.loc[MODEL, "rank_score"] == 50.0
    assert ranked.loc["worst", "rank_score"] == 0.0
