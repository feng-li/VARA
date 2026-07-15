import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_forecast_workflow import (
    _learn_group_residual_shrinkage_from_validation,
    _load_residual_shrinkage_group_mapping,
    _select_residual_shrinkage_score,
)
from scripts.run_vara_forecast import (
    _apply_residual_shrinkage,
    _load_residual_shrinkage_by_series,
    run_vara_forecast,
)


class GroupResidualShrinkageTests(unittest.TestCase):
    def test_load_residual_shrinkage_aligns_weights_by_series_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.csv"
            pd.DataFrame({
                "series_id": ["s1", "s2", "unused"],
                "group_id": ["a", "b", "other"],
                "residual_shrinkage": [0.2, 0.8, 0.5],
            }).to_csv(path, index=False)

            selected = _load_residual_shrinkage_by_series(path, ["s2", "s1"])

        self.assertEqual(list(selected["series_id"]), ["s2", "s1"])
        np.testing.assert_allclose(selected["residual_shrinkage"], [0.8, 0.2])
        self.assertEqual(list(selected["group_id"]), ["b", "a"])

    def test_load_residual_shrinkage_rejects_missing_selected_series(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weights.csv"
            pd.DataFrame({
                "series_id": ["s1"],
                "residual_shrinkage": [0.5],
            }).to_csv(path, index=False)

            with self.assertRaisesRegex(ValueError, "missing selected series"):
                _load_residual_shrinkage_by_series(path, ["s1", "s2"])

    def test_apply_residual_shrinkage_scales_each_series_and_variance(self):
        baseline = np.asarray([[10.0, 20.0], [11.0, 21.0]])
        residual = np.asarray([[2.0, -4.0], [3.0, 6.0]])
        baseline_variance = np.asarray([[1.0, 2.0], [1.5, 2.5]])
        residual_variance = np.asarray([[4.0, 8.0], [5.0, 12.0]])

        forecast, scaled_residual, forecast_variance, scaled_variance = (
            _apply_residual_shrinkage(
                baseline,
                residual,
                baseline_variance,
                residual_variance,
                np.asarray([0.0, 0.5]),
            )
        )

        np.testing.assert_allclose(scaled_residual, [[0.0, -2.0], [0.0, 3.0]])
        np.testing.assert_allclose(forecast, [[10.0, 18.0], [11.0, 24.0]])
        np.testing.assert_allclose(scaled_variance, [[0.0, 2.0], [0.0, 3.0]])
        np.testing.assert_allclose(forecast_variance, [[1.0, 4.0], [1.5, 5.5]])

    def test_group_learning_selects_strong_weak_and_fallback_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            validation_dir = Path(tmp)
            dates = pd.date_range("2024-01-01", periods=4, freq="D")
            series_ids = ["a1", "a2", "b1", "b2", "c1"]
            baseline = np.full((4, 5), 10.0)
            residual = np.full((4, 5), 2.0)
            actual = baseline.copy()
            actual[:, :2] += residual[:, :2]
            actual[:, 4:] += residual[:, 4:]

            for name, values in (
                ("test.csv", actual),
                ("baseline_forecast.csv", baseline),
                ("residual_forecast.csv", residual),
            ):
                frame = pd.DataFrame(values, columns=series_ids)
                frame.insert(0, "date", dates)
                frame.to_csv(validation_dir / name, index=False)

            mapping = pd.DataFrame({
                "series_id": series_ids,
                "group_id": ["strong", "strong", "weak", "weak", "small"],
            })
            result = _learn_group_residual_shrinkage_from_validation(
                validation_dir,
                "rolling-origin",
                mapping,
                clip_nonnegative=True,
                grid=[0.0, 0.5, 1.0],
                min_group_series=2,
                objective="rmse",
                one_standard_error=False,
            )

            groups = result["group_table"].set_index("group_id")
            series = result["series_table"].set_index("series_id")
            self.assertAlmostEqual(groups.loc["strong", "residual_shrinkage"], 1.0)
            self.assertAlmostEqual(groups.loc["weak", "residual_shrinkage"], 0.0)
            self.assertAlmostEqual(
                groups.loc["small", "residual_shrinkage"],
                result["global_residual_shrinkage"],
            )
            self.assertTrue(bool(groups.loc["small", "used_global_fallback"]))
            self.assertAlmostEqual(series.loc["a1", "residual_shrinkage"], 1.0)
            self.assertAlmostEqual(series.loc["b1", "residual_shrinkage"], 0.0)
            self.assertTrue((validation_dir / "group_residual_shrinkage.csv").exists())
            self.assertTrue((validation_dir / "residual_shrinkage_by_series.csv").exists())

    def test_one_standard_error_rule_prefers_smaller_weight(self):
        scores = pd.DataFrame({
            "residual_shrinkage": [0.0, 0.5, 1.0],
            "mse": [1.08, 1.02, 1.00],
            "rmse": np.sqrt([1.08, 1.02, 1.00]),
            "mae": [0.90, 0.85, 0.84],
            "mse_standard_error": [0.10, 0.10, 0.10],
            "mae_standard_error": [0.08, 0.08, 0.08],
            "mse_difference_from_best": [0.08, 0.02, 0.0],
            "mse_difference_standard_error": [0.10, 0.03, 0.0],
            "mae_difference_from_best": [0.06, 0.01, 0.0],
            "mae_difference_standard_error": [0.08, 0.02, 0.0],
        })

        selected = _select_residual_shrinkage_score(
            scores,
            objective="rmse",
            one_standard_error=True,
        )

        self.assertAlmostEqual(float(selected["residual_shrinkage"]), 0.0)

    def test_group_mapping_can_be_built_from_panel_metadata_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "panel.csv"
            pd.DataFrame({
                "date": pd.date_range("2024-01-01", periods=4, freq="D").repeat(2),
                "series_id": ["s1", "s2"] * 4,
                "value": np.arange(8, dtype=float),
                "dept_id": ["FOODS_3", "FOODS_3"] * 4,
                "store_id": ["WI_2", "CA_1"] * 4,
            }).to_csv(path, index=False)
            config = {
                "data_path": path,
                "series_col": "series_id",
                "residual_shrinkage_group_file": None,
                "residual_shrinkage_group_cols": "dept_id,store_id",
            }

            mapping = _load_residual_shrinkage_group_mapping(config).set_index("series_id")

        self.assertEqual(mapping.loc["s1", "group_id"], "dept_id=FOODS_3|store_id=WI_2")
        self.assertEqual(mapping.loc["s2", "group_id"], "dept_id=FOODS_3|store_id=CA_1")

    def test_vara_runner_applies_per_series_shrinkage_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "panel.csv"
            output_dir = root / "output"
            weights_path = root / "weights.csv"
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            pd.DataFrame({
                "date": dates,
                "s1": 10.0 + np.linspace(0.0, 3.0, 24),
                "s2": 20.0 + np.sin(np.arange(24) * 2.0 * np.pi / 12.0),
                "s3": 30.0 + np.linspace(0.0, 5.0, 24),
            }).to_csv(data_path, index=False)
            pd.DataFrame({
                "series_id": ["s3", "s1", "s2"],
                "group_id": ["full", "baseline", "blend"],
                "residual_shrinkage": [1.0, 0.0, 0.5],
            }).to_csv(weights_path, index=False)

            metrics = run_vara_forecast(
                data_dir=root,
                data_path=data_path,
                output_dir=output_dir,
                max_series=0,
                horizon=2,
                grid_size=3,
                y_transform="none",
                seasonal_lag=1,
                ridge=1e-2,
                series_depth=None,
                series_suffix=None,
                feature_freq=1,
                feature_source="seasonal-corr",
                forecast_scenario="seasonal-naive-residual",
                residual_shrinkage_file=weights_path,
            )

            forecast = pd.read_csv(output_dir / "forecast.csv")
            baseline = pd.read_csv(output_dir / "baseline_forecast.csv")
            residual = pd.read_csv(output_dir / "residual_forecast.csv")
            residual_unshrunk = pd.read_csv(
                output_dir / "residual_forecast_unshrunk.csv"
            )
            applied = pd.read_csv(output_dir / "residual_shrinkage_by_series.csv")

        pd.testing.assert_frame_equal(
            forecast.drop(columns="date"),
            baseline.drop(columns="date") + residual.drop(columns="date"),
            check_dtype=False,
            atol=1e-10,
            rtol=1e-10,
        )
        np.testing.assert_allclose(forecast["s1"], baseline["s1"])
        np.testing.assert_allclose(residual["s1"], 0.0)
        self.assertTrue(np.isfinite(residual_unshrunk["s1"]).all())
        self.assertEqual(list(applied["series_id"]), ["s1", "s2", "s3"])
        np.testing.assert_allclose(applied["residual_shrinkage"], [0.0, 0.5, 1.0])
        self.assertEqual(metrics["residual_shrinkage_mode"], "per_series")
        self.assertAlmostEqual(metrics["residual_shrinkage_min"], 0.0)
        self.assertAlmostEqual(metrics["residual_shrinkage_max"], 1.0)


if __name__ == "__main__":
    unittest.main()
