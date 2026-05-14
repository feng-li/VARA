import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_tourism_benchmark import run_tourism_benchmark


class TourismBenchmarkTests(unittest.TestCase):
    def test_statsforecast_rolling_benchmark_writes_expected_outputs(self):
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

            summary = run_tourism_benchmark(
                data_dir=data_dir,
                output_dir=output_dir,
                models="naive,seasonal_naive,drift,ar1,var1",
                forecast_mode="rolling-origin",
                horizon=2,
                season_length=12,
                series_depth=3,
            )

            self.assertEqual(set(summary["model"]), {"naive", "seasonal_naive", "drift", "ar1", "var1"})
            self.assertTrue((output_dir / "tourism_benchmark_forecasts_long.csv").exists())
            self.assertTrue((output_dir / "tourism_forecast_naive.csv").exists())
            self.assertTrue((output_dir / "tourism_forecast_ar1.csv").exists())
            self.assertTrue((output_dir / "tourism_forecast_var1.csv").exists())
            self.assertTrue((output_dir / "tourism_errors_seasonal_naive.csv").exists())
            self.assertTrue((output_dir / "metrics_by_series.csv").exists())
            self.assertTrue((output_dir / "metrics_by_origin.csv").exists())
            self.assertTrue((output_dir / "metrics_summary.csv").exists())
            self.assertTrue((output_dir / "config_tourism_benchmark.json").exists())

            long_forecasts = pd.read_csv(output_dir / "tourism_benchmark_forecasts_long.csv")
            self.assertEqual(long_forecasts.shape[0], 4)
            self.assertEqual(set(long_forecasts["origin"]), {1, 2})
            self.assertEqual(set(long_forecasts["series_id"]), {"AAAAll", "AABAll"})
            self.assertIn("seasonal_naive", long_forecasts.columns)
            self.assertIn("ar1", long_forecasts.columns)
            self.assertIn("var1", long_forecasts.columns)

            seasonal_wide = pd.read_csv(output_dir / "tourism_forecast_seasonal_naive.csv")
            self.assertEqual(list(seasonal_wide["AAAAll"]), [20.0, 21.0])
            self.assertEqual(list(seasonal_wide["AABAll"]), [50.0, 52.0])


if __name__ == "__main__":
    unittest.main()
