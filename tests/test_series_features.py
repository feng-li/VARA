import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from tsfeatures import acf_features

from vara.series_features import (
    extract_series_tsfeatures,
    extract_series_tsfeatures_csv,
    select_series_columns,
)


class SeriesFeatureTests(unittest.TestCase):
    def _panel(self) -> pd.DataFrame:
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        return pd.DataFrame({
            "date": dates,
            "AAAAll": np.arange(1, 25, dtype=float),
            "AABAll": np.arange(2, 50, 2, dtype=float),
            "AAAll": np.arange(3, 27, dtype=float),
            "AAll": np.arange(10, 34, dtype=float),
            "XHol": np.arange(5, 29, dtype=float),
        })

    def test_select_series_columns_filters_suffix_and_depth(self):
        cols = select_series_columns(
            self._panel(),
            series_suffix="All",
            series_depth=3,
        )

        self.assertEqual(cols, ["AAAAll", "AABAll"])

    def test_extract_series_tsfeatures_uses_training_period_only(self):
        features = extract_series_tsfeatures(
            self._panel(),
            series_suffix="All",
            series_depth=3,
            holdout=2,
            transform="log1p-seasonal-diff",
            seasonal_lag=4,
            freq=4,
            features="series_length,statistics",
            scale_series=False,
            standardize_output=False,
            threads=1,
        )

        self.assertEqual(features["series_id"].tolist(), ["AAAAll", "AABAll"])
        self.assertEqual(features["n_obs"].tolist(), [18, 18])
        self.assertEqual(features["train_start"].unique().tolist(), ["2020-05-01"])
        self.assertEqual(features["train_end"].unique().tolist(), ["2021-10-01"])
        self.assertTrue(np.isfinite(features.select_dtypes(include="number").to_numpy()).all())
        self.assertIn("series_length", features.columns)
        self.assertIn("mean", features.columns)

    def test_extract_series_tsfeatures_accepts_callable_feature_list(self):
        features = extract_series_tsfeatures(
            self._panel(),
            series_suffix="All",
            series_depth=3,
            holdout=2,
            transform="level",
            freq=12,
            features=[acf_features],
            threads=1,
        )

        self.assertEqual(features["series_id"].tolist(), ["AAAAll", "AABAll"])
        self.assertIn("x_acf1", features.columns)
        self.assertIn("x_acf10", features.columns)
        self.assertNotIn("hurst", features.columns)

    def test_extract_series_tsfeatures_csv_writes_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "panel.csv"
            output_path = Path(tmp) / "features.csv"
            self._panel().to_csv(input_path, index=False)

            features = extract_series_tsfeatures_csv(
                input_path,
                output_path,
                series_suffix="All",
                series_depth=3,
                holdout=2,
                transform="level",
                freq=12,
                features="series_length",
                threads=1,
            )

            self.assertTrue(output_path.exists())
            saved = pd.read_csv(output_path)
            self.assertEqual(saved.shape, features.shape)
            self.assertEqual(saved["series_id"].tolist(), ["AAAAll", "AABAll"])


if __name__ == "__main__":
    unittest.main()
