import inspect
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import jax.numpy as jnp
import numpy as np
import pandas as pd

from scripts.evaluate_tourism_purpose_autoets import _forecast_origin_task
from scripts.recombine_tourism_oos_error_vara import (
    build_leakage_audit,
    load_dm_forecast_panel,
    load_strategy_panel,
    run_recombination,
)
from scripts.search_tourism_oos_error_vara import ErrorStrategy, _error_vara_task
from scripts.search_tourism_oos_error_vara import score_results


class _FakeForecaster:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, history, features):
        self.alpha_hat = jnp.zeros(history.shape[1])
        self.beta_hat = jnp.zeros(history.shape[1])
        return self

    def predict(self, h=1):
        return jnp.zeros((h, len(self.alpha_hat)))

    def diagnostics(self):
        return {"eta_hat": 0.0, "eta_on_boundary": False}


class TourismLeakageControlTests(unittest.TestCase):
    def test_autoets_worker_accepts_history_only(self):
        parameter_names = set(inspect.signature(_forecast_origin_task).parameters)
        self.assertNotIn("values", parameter_names)
        self.assertNotIn("dates", parameter_names)

        history = np.arange(24 * 5, dtype=float).reshape(24, 5)
        expected_forecast = np.arange(5, dtype=float)[None, :]

        def fake_baseline(received_history, **kwargs):
            np.testing.assert_array_equal(received_history, history)
            return {"forecast": expected_forecast}

        with patch(
            "scripts.evaluate_tourism_purpose_autoets.auto_ets_baseline_panel",
            side_effect=fake_baseline,
        ):
            result = _forecast_origin_task(
                history,
                ["AAAAll", "AAAHol", "AAAVis", "AAABus", "AAAOth"],
                1,
                24,
                "2016-01-01",
                "2015-12-01",
                12,
                True,
            )
        self.assertNotIn("actual", result)
        self.assertEqual(result["history_end"], "2015-12-01")
        self.assertEqual(result["n_train"], 24)

    def test_autoets_worker_rejects_target_in_history(self):
        with self.assertRaisesRegex(ValueError, "strictly before"):
            _forecast_origin_task(
                np.ones((24, 5)),
                ["AAAAll", "AAAHol", "AAAVis", "AAABus", "AAAOth"],
                1,
                24,
                "2016-01-01",
                "2016-01-01",
                12,
                True,
            )

    def test_vara_worker_accepts_error_history_only(self):
        parameter_names = set(inspect.signature(_error_vara_task).parameters)
        self.assertNotIn("errors", parameter_names)
        self.assertNotIn("dates", parameter_names)
        history = np.arange(24 * 3, dtype=float).reshape(24, 3)
        strategy = ErrorStrategy(
            "test",
            feature_source="predictive-average",
            attention_mode="score-topk",
            attention_top_k=2,
            history_window=24,
            stability_cap=0.5,
        )
        with patch("vara.forecast.VARAForecaster", _FakeForecaster):
            result = _error_vara_task(
                history,
                ["AAAAll", "AABAll", "ABAAll"],
                asdict(strategy),
                24,
                "2016-01-01",
                "2014-01-01",
                "2015-12-01",
            )
        self.assertTrue(result["ok"])
        self.assertEqual(result["history_end"], "2015-12-01")
        self.assertEqual(result["n_train"], 24)

    def test_vara_worker_rejects_target_error_in_history(self):
        strategy = ErrorStrategy("test", history_window=24)
        result = _error_vara_task(
            np.ones((24, 3)),
            ["AAAAll", "AABAll", "ABAAll"],
            asdict(strategy),
            24,
            "2016-01-01",
            "2014-01-01",
            "2016-01-01",
        )
        self.assertFalse(result["ok"])
        self.assertIn("strictly before", result["error"])

    def test_strategy_panel_rejects_noncausal_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            purpose_path = root / "purpose.csv"
            vara_path = root / "vara.csv"
            pd.DataFrame({
                "date": ["2016-01-01"],
                "series_id": ["AAAAll"],
                "actual": [10.0],
                "direct_auto_ets": [9.0],
                "purpose_sum_auto_ets": [9.5],
                "autoets_history_end": ["2015-12-01"],
                "autoets_n_train": [24],
            }).to_csv(purpose_path, index=False)
            pd.DataFrame({
                "date": ["2016-01-01"],
                "series_id": ["AAAAll"],
                "strategy_id": ["test"],
                "vara_correction": [0.25],
                "history_start": ["2014-02-01"],
                "history_end": ["2016-01-01"],
                "n_train": [24],
            }).to_csv(vara_path, index=False)
            with self.assertRaisesRegex(ValueError, "strictly before"):
                load_strategy_panel(purpose_path, vara_path, "test")

    def test_final_year_cannot_overlap_validation(self):
        with self.assertRaisesRegex(ValueError, "strictly before"):
            run_recombination(
                data_path=Path("unused.csv"),
                purpose_forecasts_path=Path("unused.csv"),
                vara_results_path=Path("unused.csv"),
                strategy_id="test",
                output_dir=Path("unused"),
                validation_start_year=2013,
                validation_end_year=2016,
                final_year=2016,
                calibration_window_months=24,
                vara_weight_cap=0.5,
                benchmark_metrics_path=None,
            )

    def test_dm_panel_rejects_benchmark_trained_through_target(self):
        final = pd.DataFrame({
            "date": ["2016-01-01"],
            "series_id": ["AAAAll"],
            "actual": [10.0],
            "rolling_purpose_oos_vara": [9.5],
            "rolling_purpose_blend": [9.4],
            "auto_ets": [9.0],
            "raw_oos_vara": [9.2],
        })
        with tempfile.TemporaryDirectory() as tmp:
            benchmark_path = Path(tmp) / "benchmark.csv"
            pd.DataFrame({
                "date": ["2016-01-01"],
                "train_end": ["2016-01-01"],
                "series_id": ["AAAAll"],
                "seasonal_naive": [8.0],
            }).to_csv(benchmark_path, index=False)

            with self.assertRaisesRegex(ValueError, "strictly before"):
                load_dm_forecast_panel(
                    final,
                    benchmark_path,
                    ["rolling_purpose_oos_vara", "seasonal_naive"],
                )

    def test_machine_readable_audit_requires_strictly_prior_information(self):
        panel = pd.DataFrame({
            "date": pd.to_datetime(["2016-01-01"]),
            "series_id": ["AAAAll"],
            "autoets_history_end": pd.to_datetime(["2015-12-01"]),
            "history_start": pd.to_datetime(["2014-01-01"]),
            "history_end": pd.to_datetime(["2015-12-01"]),
            "n_train": [24],
        })
        coefficients = pd.DataFrame({
            "date": pd.to_datetime(["2016-01-01"]),
            "training_end": pd.to_datetime(["2015-12-01"]),
            "n_training_months": [24],
        })
        audit = build_leakage_audit(
            panel,
            coefficients,
            coefficients,
            validation_end_year=2015,
            final_year=2016,
        )
        self.assertTrue(audit["all_checks_passed"])

        panel.loc[0, "history_end"] = pd.Timestamp("2016-01-01")
        audit = build_leakage_audit(
            panel,
            coefficients,
            coefficients,
            validation_end_year=2015,
            final_year=2016,
        )
        self.assertFalse(audit["all_checks_passed"])

    def test_strategy_ranking_excludes_dates_after_selection_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            purpose_path = root / "purpose.csv"
            output_dir = root / "output"
            output_dir.mkdir()
            dates = pd.date_range("2012-01-01", "2016-12-01", freq="MS")
            purpose_rows = []
            result_rows = []
            for index, date in enumerate(dates):
                purpose_rows.append({
                    "date": date,
                    "series_id": "AAAAll",
                    "year": date.year,
                    "actual": float(index),
                    "direct_auto_ets": float(index) + 1.0,
                    "purpose_sum_auto_ets": float(index) + 0.5,
                })
                result_rows.append({
                    "strategy_id": "test",
                    "target_index": index,
                    "date": date,
                    "series_id": "AAAAll",
                    "vara_correction": 0.25,
                    "eta_hat": 0.0,
                    "eta_on_boundary": False,
                    "stationarity_proxy": 0.5,
                    "n_train": 24,
                })
            pd.DataFrame(purpose_rows).to_csv(purpose_path, index=False)
            score_results(
                pd.DataFrame(result_rows),
                purpose_path,
                [ErrorStrategy("test")],
                output_dir,
                selection_end="2015-12-01",
            )
            predictions = pd.read_csv(output_dir / "nested_predictions.csv")
            self.assertEqual(pd.to_datetime(predictions["date"]).dt.year.max(), 2015)


if __name__ == "__main__":
    unittest.main()
