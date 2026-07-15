import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.recombine_vara_forecasts import recombine_vara_forecasts


def _write_wide(path: Path, dates, series_ids, values) -> None:
    frame = pd.DataFrame(values, columns=series_ids)
    frame.insert(0, "date", dates)
    frame.to_csv(path, index=False)


class RecombineVARAForecastsTests(unittest.TestCase):
    def test_recombine_applies_group_weights_and_regenerates_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            dates = pd.date_range("2024-01-01", periods=3, freq="D")
            series_ids = ["s1", "s2", "s3"]
            baseline = np.asarray([
                [10.0, 8.0, 3.0],
                [11.0, 8.0, 3.0],
                [12.0, 8.0, 3.0],
            ])
            raw_residual = np.asarray([
                [2.0, -2.0, -8.0],
                [1.0, 2.0, 4.0],
                [0.0, 4.0, 2.0],
            ])
            source_weight = 0.5
            source_forecast = np.maximum(baseline + source_weight * raw_residual, 0.0)
            test = np.asarray([
                [11.0, 7.0, 0.0],
                [12.0, 10.0, 7.0],
                [13.0, 11.0, 5.0],
            ])
            baseline_variance = np.full_like(baseline, 1.0)
            raw_residual_variance = np.full_like(baseline, 4.0)

            _write_wide(source_dir / "baseline_forecast.csv", dates, series_ids, baseline)
            _write_wide(source_dir / "forecast.csv", dates, series_ids, source_forecast)
            _write_wide(source_dir / "forecast_transformed.csv", dates, series_ids, raw_residual)
            _write_wide(source_dir / "test.csv", dates, series_ids, test)
            _write_wide(
                source_dir / "baseline_forecast_variance.csv",
                dates,
                series_ids,
                baseline_variance,
            )
            _write_wide(
                source_dir / "residual_forecast_variance.csv",
                dates,
                series_ids,
                source_weight ** 2 * raw_residual_variance,
            )
            _write_wide(
                source_dir / "forecast_variance.csv",
                dates,
                series_ids,
                baseline_variance + source_weight ** 2 * raw_residual_variance,
            )
            pd.DataFrame([{
                "residual_shrinkage": source_weight,
                "residual_transform": "center",
                "clip_nonnegative": True,
                "seasonal_lag": 1,
                "forecast_type": "recursive",
            }]).to_csv(source_dir / "metrics.csv", index=False)
            pd.DataFrame([{
                "model": "vara",
                "data": "synthetic.csv",
                "series_depth": -1,
                "season_length": 1,
                "train_start": "2023-01-01",
                "train_end": "2023-12-31",
                "test_start": str(dates[0].date()),
                "test_end": str(dates[-1].date()),
                "forecast_mode": "recursive",
                "horizon": 3,
                "n_series": 3,
                "n_test": 3,
            }]).to_csv(source_dir / "metrics_summary.csv", index=False)
            pd.DataFrame({
                "model": ["vara"] * 3,
                "series_id": series_ids,
                "seasonal_mae_scale": [1.0, 2.0, 1.5],
                "seasonal_mse_scale": [1.0, 4.0, 2.5],
            }).to_csv(source_dir / "metrics_by_series.csv", index=False)

            shrinkage_path = root / "group_weights.csv"
            pd.DataFrame({
                "series_id": ["s3", "s1", "s2"],
                "group_id": ["full", "baseline", "blend"],
                "residual_shrinkage": [1.0, 0.0, 0.5],
            }).to_csv(shrinkage_path, index=False)

            result = recombine_vara_forecasts(
                source_dir,
                shrinkage_path,
                output_dir,
            )

            forecast = pd.read_csv(output_dir / "forecast.csv").iloc[:, 1:].to_numpy(float)
            residual_variance = pd.read_csv(
                output_dir / "residual_forecast_variance.csv"
            ).iloc[:, 1:].to_numpy(float)
            expected = np.maximum(
                baseline + np.asarray([0.0, 0.5, 1.0])[None, :] * raw_residual,
                0.0,
            )
            comparison_exists = (output_dir / "metrics_summary_comparison.csv").exists()
            config_exists = (output_dir / "recombination_config.json").exists()

        np.testing.assert_allclose(forecast, expected)
        np.testing.assert_allclose(
            residual_variance,
            raw_residual_variance * np.asarray([0.0, 0.5, 1.0])[None, :] ** 2,
        )
        summary = result["metrics_summary"]
        self.assertIn("rmse", summary.columns)
        self.assertIn("oos_r2", summary.columns)
        self.assertIn("oos_r2_zero", summary.columns)
        self.assertEqual(len(result["metrics_by_group"]), 3)
        self.assertTrue(comparison_exists)
        self.assertTrue(config_exists)


if __name__ == "__main__":
    unittest.main()
