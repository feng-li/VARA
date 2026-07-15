import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from scripts.run_benchmarks import _parse_forecast_modes, run_forecast_benchmark


class TourismBenchmarkTests(unittest.TestCase):
    def test_parse_forecast_modes_accepts_comma_list(self):
        self.assertEqual(
            _parse_forecast_modes("rolling-origin,closed-loop,recursive,recursive"),
            ["rolling-origin", "closed-loop", "recursive"],
        )

    def test_statsforecast_rolling_benchmark_writes_expected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            output_dir = root / "out"
            data_dir.mkdir()
            dates = pd.date_range("2020-01-01", periods=36, freq="MS")
            pd.DataFrame({
                "date": dates,
                "AAAAll": np.arange(36, dtype=float) + 10.0,
                "AABAll": 2.0 * np.arange(36, dtype=float) + 30.0,
            }).to_csv(data_dir / "tourism.csv", index=False)

            summary = run_forecast_benchmark(
                data_dir=data_dir,
                output_dir=output_dir,
                models="naive,seasonal_naive,drift,statsmodels_ets,seasonal_ets,ar1,ar1_snaive,ar1_seasonal_ets,var1,var1_snaive,var1_seasonal_ets",
                forecast_mode="rolling-origin",
                horizon=2,
                season_length=12,
                series_depth=3,
            )

            self.assertEqual(set(summary["model"]), {"naive", "seasonal_naive", "drift", "statsmodels_ets", "seasonal_ets", "ar1", "ar1_snaive", "ar1_seasonal_ets", "var1", "var1_snaive", "var1_seasonal_ets"})
            self.assertTrue((output_dir / "benchmark_forecasts_long.csv").exists())
            self.assertTrue((output_dir / "benchmark_forecast_variance_long.csv").exists())
            self.assertTrue((output_dir / "forecast_naive.csv").exists())
            self.assertTrue((output_dir / "forecast_variance_naive.csv").exists())
            self.assertTrue((output_dir / "forecast_statsmodels_ets.csv").exists())
            self.assertTrue((output_dir / "forecast_seasonal_ets.csv").exists())
            self.assertTrue((output_dir / "forecast_ar1.csv").exists())
            self.assertTrue((output_dir / "forecast_ar1_seasonal_ets.csv").exists())
            self.assertTrue((output_dir / "forecast_ar1_snaive.csv").exists())
            self.assertTrue((output_dir / "forecast_var1.csv").exists())
            self.assertTrue((output_dir / "forecast_var1_snaive.csv").exists())
            self.assertTrue((output_dir / "forecast_var1_seasonal_ets.csv").exists())
            self.assertTrue((output_dir / "errors_seasonal_naive.csv").exists())
            self.assertTrue((output_dir / "metrics_by_series.csv").exists())
            self.assertTrue((output_dir / "metrics_by_origin.csv").exists())
            self.assertTrue((output_dir / "metrics_summary.csv").exists())
            self.assertTrue((output_dir / "config_benchmark.json").exists())
            self.assertTrue((output_dir / "_SUCCESS").exists())

            long_forecasts = pd.read_csv(output_dir / "benchmark_forecasts_long.csv")
            self.assertEqual(long_forecasts.shape[0], 4)
            self.assertEqual(set(long_forecasts["origin"]), {1, 2})
            self.assertEqual(set(long_forecasts["series_id"]), {"AAAAll", "AABAll"})
            self.assertIn("seasonal_naive", long_forecasts.columns)
            self.assertIn("statsmodels_ets", long_forecasts.columns)
            self.assertIn("seasonal_ets", long_forecasts.columns)
            self.assertIn("ar1", long_forecasts.columns)
            self.assertIn("ar1_seasonal_ets", long_forecasts.columns)
            self.assertIn("ar1_snaive", long_forecasts.columns)
            self.assertIn("var1", long_forecasts.columns)
            self.assertIn("var1_snaive", long_forecasts.columns)
            self.assertIn("var1_seasonal_ets", long_forecasts.columns)
            self.assertFalse(any(col.endswith("__variance") for col in long_forecasts.columns))

            long_variance = pd.read_csv(output_dir / "benchmark_forecast_variance_long.csv")
            self.assertEqual(long_variance.shape[0], 4)
            self.assertIn("naive", long_variance.columns)
            self.assertIn("ar1", long_variance.columns)
            self.assertTrue(np.isfinite(long_variance["naive"]).all())
            self.assertTrue(np.isfinite(long_variance["ar1"]).all())
            self.assertIn("crps", summary.columns)
            self.assertIn("forecast_variance_mean", summary.columns)
            self.assertIn("oos_r2", summary.columns)
            self.assertIn("oos_r2_zero", summary.columns)

            seasonal_wide = pd.read_csv(output_dir / "forecast_seasonal_naive.csv")
            self.assertEqual(list(seasonal_wide["AAAAll"]), [32.0, 33.0])
            self.assertEqual(list(seasonal_wide["AABAll"]), [74.0, 76.0])

    def test_success_marker_skips_completed_resume_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            output_dir = root / "out"
            data_dir.mkdir()
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            pd.DataFrame({
                "date": dates,
                "AAAAll": np.arange(24, dtype=float) + 10.0,
                "AABAll": 2.0 * np.arange(24, dtype=float) + 30.0,
            }).to_csv(data_dir / "tourism.csv", index=False)

            first = run_forecast_benchmark(
                data_dir=data_dir,
                output_dir=output_dir,
                models="naive",
                forecast_mode="recursive",
                horizon=2,
                season_length=12,
                series_depth=3,
            )

            self.assertTrue((output_dir / "_SUCCESS").exists())
            with patch("scripts.run_benchmarks._forecast_by_mode", side_effect=AssertionError("should skip")):
                second = run_forecast_benchmark(
                    data_dir=data_dir,
                    output_dir=output_dir,
                    models="naive",
                    forecast_mode="recursive",
                    horizon=2,
                    season_length=12,
                    series_depth=3,
                )

            pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))

    def test_closed_loop_uses_forecasts_as_pseudo_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "m4_like.csv"
            output_dir = root / "out"
            dates = pd.date_range("2000-01-01", periods=12, freq="YS")
            pd.DataFrame({
                "date": dates,
                "Y1": np.arange(12, dtype=float) + 5.0,
                "Y2": 2.0 * np.arange(12, dtype=float) + 7.0,
            }).to_csv(data_path, index=False)

            summary = run_forecast_benchmark(
                data_dir=root,
                data_path=data_path,
                output_dir=output_dir,
                models="naive",
                forecast_mode="closed-loop",
                horizon=2,
                season_length=1,
                freq="YS",
                series_depth=None,
                series_suffix=None,
            )

            self.assertEqual(set(summary["forecast_mode"]), {"closed-loop"})
            forecast = pd.read_csv(output_dir / "forecast_naive.csv")
            self.assertEqual(list(forecast["Y1"]), [14.0, 14.0])
            self.assertEqual(list(forecast["Y2"]), [25.0, 25.0])

    def test_generic_wide_csv_path_uses_all_non_date_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "m4_like.csv"
            output_dir = root / "out"
            dates = pd.date_range("2000-01-01", periods=12, freq="YS")
            pd.DataFrame({
                "date": dates,
                "Y1": np.arange(12, dtype=float) + 5.0,
                "Y2": 2.0 * np.arange(12, dtype=float) + 7.0,
            }).to_csv(data_path, index=False)

            summary = run_forecast_benchmark(
                data_dir=root,
                data_path=data_path,
                output_dir=output_dir,
                models="naive,drift,ar1",
                forecast_mode="rolling-origin",
                horizon=2,
                season_length=1,
                freq="YS",
                series_depth=None,
                series_suffix=None,
            )

            self.assertEqual(set(summary["model"]), {"naive", "drift", "ar1"})
            self.assertEqual(set(summary["data"]), {str(data_path)})
            long_forecasts = pd.read_csv(output_dir / "benchmark_forecasts_long.csv")
            self.assertEqual(set(long_forecasts["series_id"]), {"Y1", "Y2"})
            self.assertTrue((output_dir / "config_benchmark.json").exists())

    def test_generic_long_csv_path_uses_series_id_value_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "m4_like_long.csv"
            output_dir = root / "out"
            dates = pd.date_range("2000-01-01", periods=12, freq="YS")
            wide = pd.DataFrame({
                "date": dates,
                "Y1": np.arange(12, dtype=float) + 5.0,
                "Y2": 2.0 * np.arange(12, dtype=float) + 7.0,
            })
            wide.melt(id_vars="date", var_name="series_id", value_name="value").to_csv(data_path, index=False)

            summary = run_forecast_benchmark(
                data_dir=root,
                data_path=data_path,
                data_format="long",
                output_dir=output_dir,
                models="naive,drift,ar1",
                forecast_mode="rolling-origin",
                horizon=2,
                season_length=1,
                freq="YS",
                series_depth=None,
                series_suffix=None,
            )

            self.assertEqual(set(summary["model"]), {"naive", "drift", "ar1"})
            self.assertEqual(set(summary["data"]), {str(data_path)})
            long_forecasts = pd.read_csv(output_dir / "benchmark_forecasts_long.csv")
            self.assertEqual(set(long_forecasts["series_id"]), {"Y1", "Y2"})
            config = pd.read_json(output_dir / "config_benchmark.json", typ="series")
            self.assertEqual(config["data_format"], "long")


if __name__ == "__main__":
    unittest.main()
