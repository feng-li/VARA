import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_vara_forecast import run_vara_forecast
from vara.m5_features import make_m5_attention_features, make_m5_attention_features_from_files
from vara.series_features import load_series_feature_matrix


class M5FeatureTests(unittest.TestCase):
    def _m5_tables(self):
        sales = pd.DataFrame({
            "id": ["HOBBIES_1_001_CA_1_evaluation", "FOODS_1_001_TX_1_evaluation"],
            "item_id": ["HOBBIES_1_001", "FOODS_1_001"],
            "dept_id": ["HOBBIES_1", "FOODS_1"],
            "cat_id": ["HOBBIES", "FOODS"],
            "store_id": ["CA_1", "TX_1"],
            "state_id": ["CA", "TX"],
            **{f"d_{i}": [float(i), float(i + 2)] for i in range(1, 11)},
        })
        prices = pd.DataFrame({
            "store_id": ["CA_1", "CA_1", "TX_1", "TX_1"],
            "item_id": ["HOBBIES_1_001", "HOBBIES_1_001", "FOODS_1_001", "FOODS_1_001"],
            "wm_yr_wk": [1, 2, 1, 2],
            "sell_price": [2.0, 4.0, 3.0, 6.0],
        })
        calendar = pd.DataFrame({
            "d": [f"d_{i}" for i in range(1, 11)],
            "date": pd.date_range("2011-01-29", periods=10, freq="D"),
            "wm_yr_wk": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3],
            "event_name_1": [None, "Event", None, None, None, None, None, None, None, None],
            "event_name_2": [None] * 10,
            "event_type_1": [None, "Cultural", None, None, None, None, None, None, None, None],
            "event_type_2": [None] * 10,
            "snap_CA": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "snap_TX": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            "snap_WI": [0] * 10,
        })
        return sales, prices, calendar

    def test_make_m5_attention_features_returns_series_level_rows(self):
        sales, prices, calendar = self._m5_tables()

        features = make_m5_attention_features(sales, prices, calendar, holdout=2)

        self.assertEqual(features.shape[0], 2)
        self.assertEqual(set(features["series_id"]), set(sales["id"]))
        self.assertEqual(features["n_obs"].tolist(), [8, 8])
        self.assertIn("sales_mean", features.columns)
        self.assertIn("price_mean", features.columns)
        self.assertIn("weekend_lift_rel", features.columns)
        self.assertIn("snap_lift_rel", features.columns)
        self.assertIn("store_id_CA_1", features.columns)
        self.assertIn("cat_id_FOODS", features.columns)

    def test_prepare_m5_attention_features_from_files_and_loads_for_attention(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sales, prices, calendar = self._m5_tables()
            sales_path = root / "sales.parquet"
            prices_path = root / "prices.parquet"
            calendar_path = root / "calendar.parquet"
            output_path = root / "features.parquet"
            sales.to_parquet(sales_path, index=False)
            prices.to_parquet(prices_path, index=False)
            calendar.to_parquet(calendar_path, index=False)

            metadata = make_m5_attention_features_from_files(
                sales_path=sales_path,
                prices_path=prices_path,
                calendar_path=calendar_path,
                output_path=output_path,
                holdout=2,
            )

            self.assertEqual(metadata["n_series"], 2)
            self.assertTrue(output_path.exists())
            Z, feature_df, feature_cols = load_series_feature_matrix(output_path, sales["id"].tolist())
            self.assertEqual(Z.shape[0], 2)
            self.assertGreater(Z.shape[1], 5)
            self.assertEqual(list(feature_df["series_id"]), sales["id"].tolist())
            self.assertTrue(np.isfinite(Z).all())
            self.assertIn("sales_mean", feature_cols)

    def test_vara_forecast_accepts_prepared_feature_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "panel.csv"
            feature_path = root / "features.csv"
            output_dir = root / "out"
            dates = pd.date_range("2020-01-01", periods=12, freq="D")
            pd.DataFrame({
                "date": dates,
                "Y1": np.linspace(10.0, 20.0, 12),
                "Y2": np.linspace(20.0, 10.0, 12),
                "Y3": np.linspace(15.0, 21.0, 12),
            }).to_csv(data_path, index=False)
            pd.DataFrame({
                "series_id": ["Y1", "Y2", "Y3"],
                "group": ["a", "b", "a"],
                "level": [1.0, 2.0, 3.0],
            }).to_csv(feature_path, index=False)

            metrics = run_vara_forecast(
                data_dir=root,
                data_path=data_path,
                output_dir=output_dir,
                max_series=0,
                horizon=2,
                grid_size=3,
                y_transform="center",
                seasonal_lag=1,
                ridge=1e-2,
                series_depth=None,
                series_suffix=None,
                feature_path=feature_path,
                attention_mode="topk",
                attention_top_k=2,
            )

            self.assertEqual(metrics["attention_feature_source"], "file")
            self.assertEqual(metrics["attention_feature_input_file"], str(feature_path))
            self.assertGreater(metrics["n_attention_feature_columns"], 1)
            saved_features = pd.read_csv(output_dir / "tourism_attention_features.csv")
            self.assertIn("group_a", saved_features.columns)
            self.assertEqual(list(saved_features["series_id"]), ["Y1", "Y2", "Y3"])


if __name__ == "__main__":
    unittest.main()
