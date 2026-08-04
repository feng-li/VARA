import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_tourism_unit_vara_window import (
    evaluate_unit_weight_window,
)
from scripts.search_tourism_oos_error_vara import default_strategies


def test_longest_fixed_window_strategy_is_explicit() -> None:
    by_id = {strategy.strategy_id: strategy for strategy in default_strategies()}
    strategy = by_id["predictive_avg_topk10_w192_stable05"]
    assert strategy.history_window == 192
    assert strategy.feature_source == "predictive-average"
    assert strategy.attention_top_k == 10
    assert strategy.stability_cap == 0.5


def test_unit_weight_window_evaluation_uses_direct_plus_correction() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        raw_dates = pd.date_range("2014-01-01", "2015-12-01", freq="MS")
        raw = pd.DataFrame({
            "date": raw_dates,
            "A": np.arange(len(raw_dates), dtype=float) + 1.0,
            "B": np.arange(len(raw_dates), dtype=float) + 2.0,
        })
        raw_path = root / "raw.csv"
        raw.to_csv(raw_path, index=False)

        forecast_panel = pd.DataFrame({
            "date": ["2016-01-01", "2016-01-01", "2016-02-01", "2016-02-01"],
            "series_id": ["A", "B", "A", "B"],
            "actual": [10.0, 20.0, 12.0, 18.0],
            "direct_auto_ets": [8.0, 22.0, 10.0, 20.0],
        })
        forecast_path = root / "forecasts.csv"
        forecast_panel.to_csv(forecast_path, index=False)

        vara_results = pd.DataFrame({
            "strategy_id": ["test_w192"] * 4,
            "date": forecast_panel["date"],
            "history_start": [
                "2000-01-01",
                "2000-01-01",
                "2000-02-01",
                "2000-02-01",
            ],
            "history_end": [
                "2015-12-01",
                "2015-12-01",
                "2016-01-01",
                "2016-01-01",
            ],
            "series_id": forecast_panel["series_id"],
            "vara_correction": [1.0, -3.0, 1.0, -3.0],
            "n_train": [192] * 4,
        })
        vara_path = root / "vara.csv"
        vara_results.to_csv(vara_path, index=False)

        output_dir = root / "output"
        summary = evaluate_unit_weight_window(
            data_path=raw_path,
            forecast_panel_path=forecast_path,
            vara_results_path=vara_path,
            strategy_id="test_w192",
            history_window_months=192,
            evaluation_start="2016-01-01",
            evaluation_end="2016-02-01",
            output_dir=output_dir,
        )

        unit = summary[summary["model"].eq("raw_oos_vara")].iloc[0]
        direct = summary[summary["model"].eq("direct_auto_ets")].iloc[0]
        assert unit["rmse"] == 1.0
        assert direct["rmse"] == 2.0
        predictions = pd.read_csv(output_dir / "unit_weight_predictions.csv")
        np.testing.assert_allclose(predictions["forecast"], [9.0, 19.0, 11.0, 17.0])
