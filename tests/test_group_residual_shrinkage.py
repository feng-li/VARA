import argparse
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.run_forecast_workflow import (
    DATASET_DEFAULTS,
    _learn_group_residual_shrinkage_from_validation,
    _learn_residual_shrinkage_from_validation,
    _load_config_json_defaults,
    _load_residual_shrinkage_group_mapping,
    _prepare_residual_shrinkage_mode,
    _resolve_residual_shrinkage_mode,
    _select_residual_shrinkage_score,
    _write_nested_residual_shrinkage_backtest,
    _write_validation_panels,
)
from scripts.run_vara_forecast import (
    _apply_residual_shrinkage,
    _load_residual_shrinkage_by_series,
    run_vara_forecast,
)


def _write_validation_outputs(
    output_dir: Path,
    dates: pd.DatetimeIndex,
    actual: np.ndarray,
    baseline: np.ndarray,
    residual: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    series_ids = [f"s{index + 1}" for index in range(actual.shape[1])]
    for name, values in (
        ("test.csv", actual),
        ("baseline_forecast.csv", baseline),
        ("residual_forecast.csv", residual),
        ("residual_forecast_unshrunk.csv", residual),
        ("baseline_forecast_variance.csv", np.ones_like(actual)),
        ("residual_forecast_variance_unshrunk.csv", np.ones_like(actual)),
    ):
        frame = pd.DataFrame(values, columns=series_ids)
        frame.insert(0, "date", dates)
        frame.to_csv(output_dir / name, index=False)


class GroupResidualShrinkageTests(unittest.TestCase):
    def test_unified_residual_shrinkage_modes(self):
        defaults = DATASET_DEFAULTS["m5"]
        for mode in ("full", "global", "group"):
            with self.subTest(mode=mode):
                args = argparse.Namespace(
                    residual_shrinkage_mode=mode,
                    residual_shrinkage=None,
                    residual_shrinkage_file=None,
                    learn_residual_shrinkage=None,
                    learn_group_residual_shrinkage=None,
                )
                self.assertEqual(
                    _resolve_residual_shrinkage_mode(args, defaults),
                    mode,
                )

    def test_legacy_learning_config_is_migrated_to_unified_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy_config.json"
            path.write_text(
                json.dumps({
                    "dataset": "tourism",
                    "residual_shrinkage": 0.6,
                    "learn_residual_shrinkage": True,
                    "learn_group_residual_shrinkage": False,
                }),
                encoding="utf-8",
            )

            defaults = _load_config_json_defaults(path)

        self.assertEqual(defaults["residual_shrinkage_mode"], "global")
        self.assertNotIn("learn_residual_shrinkage", defaults)
        self.assertNotIn("learn_group_residual_shrinkage", defaults)

    def test_prepare_group_mode_enables_only_group_learning(self):
        config = {
            "residual_shrinkage_mode": "group",
            "residual_shrinkage": 0.4,
            "residual_shrinkage_file": None,
        }

        learn_global, learn_group = _prepare_residual_shrinkage_mode(config)

        self.assertFalse(learn_global)
        self.assertTrue(learn_group)
        self.assertEqual(config["residual_shrinkage"], 1.0)

    def test_one_standard_error_rule_is_disabled_by_default(self):
        for dataset, defaults in DATASET_DEFAULTS.items():
            with self.subTest(dataset=dataset):
                self.assertFalse(
                    defaults["residual_shrinkage_group_one_standard_error"]
                )

    def test_tourism_group_defaults_use_top_level_hierarchy(self):
        defaults = DATASET_DEFAULTS["tourism"]
        self.assertEqual(
            defaults["residual_shrinkage_group_cols"],
            "hierarchy_level_1",
        )
        self.assertEqual(defaults["residual_shrinkage_min_group_series"], 5)
        self.assertEqual(defaults["residual_shrinkage_validation_origins"], 1)

    def test_multiple_validation_panels_are_chronological_and_nonoverlapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "panel.csv"
            dates = pd.date_range("2018-01-01", periods=72, freq="MS")
            pd.DataFrame({
                "date": dates,
                "s1": np.arange(72, dtype=float),
                "s2": np.arange(72, dtype=float) + 10.0,
            }).to_csv(data_path, index=False)
            config = {
                "data_path": data_path,
                "output_dir": root / "output",
                "date_col": "date",
                "horizon": 12,
            }

            origins = _write_validation_panels(config, 3)

            self.assertEqual(len(origins), 3)
            self.assertEqual(
                [pd.Timestamp(origin["validation_start"]) for origin in origins],
                [dates[24], dates[36], dates[48]],
            )
            self.assertEqual(
                [pd.Timestamp(origin["validation_end"]) for origin in origins],
                [dates[35], dates[47], dates[59]],
            )
            self.assertEqual(
                [origin["feature_holdout"] for origin in origins],
                [48, 36, 24],
            )
            self.assertEqual(
                [len(pd.read_csv(origin["panel_path"])) for origin in origins],
                [36, 48, 60],
            )
            self.assertTrue(
                (root / "output" / "validation" / "validation_origins.csv").exists()
            )

    def test_global_learning_pools_multiple_validation_origins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = np.full((4, 2), 10.0)
            residual = np.full((4, 2), 2.0)
            first = root / "origin_1"
            second = root / "origin_2"
            _write_validation_outputs(
                first,
                pd.date_range("2021-01-01", periods=4, freq="D"),
                baseline,
                baseline,
                residual,
            )
            _write_validation_outputs(
                second,
                pd.date_range("2022-01-01", periods=4, freq="D"),
                baseline + residual,
                baseline,
                residual,
            )
            learning_dir = root / "learning"

            selected = _learn_residual_shrinkage_from_validation(
                [first, second],
                "rolling-origin",
                clip_nonnegative=False,
                grid=[0.0, 1.0],
                output_dir=learning_dir,
            )

            self.assertAlmostEqual(selected, 0.5)
            report = json.loads(
                (learning_dir / "residual_shrinkage.json").read_text()
            )
            self.assertEqual(report["n_validation_origins"], 2)

    def test_nested_group_backtest_uses_only_earlier_origins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = np.full((4, 2), 10.0)
            residual = np.full((4, 2), 2.0)
            first = root / "origin_1"
            second = root / "origin_2"
            final = root / "final"
            _write_validation_outputs(
                first,
                pd.date_range("2020-01-01", periods=4, freq="D"),
                baseline + residual,
                baseline,
                residual,
            )
            _write_validation_outputs(
                second,
                pd.date_range("2021-01-01", periods=4, freq="D"),
                baseline,
                baseline,
                residual,
            )
            _write_validation_outputs(
                final,
                pd.date_range("2022-01-01", periods=4, freq="D"),
                baseline + 0.5 * residual,
                baseline,
                residual,
            )
            mapping = pd.DataFrame({
                "series_id": ["s1", "s2"],
                "group_id": ["g", "g"],
            })

            result = _write_nested_residual_shrinkage_backtest(
                [first, second],
                final,
                "rolling-origin",
                root / "nested",
                clip_nonnegative=False,
                grid=[0.0, 0.5, 1.0],
                objective="rmse",
                group_mapping=mapping,
                min_group_series=1,
                one_standard_error=False,
            )

            lambdas = result["group_lambdas"].set_index("fold_index")
            self.assertAlmostEqual(lambdas.loc[1, "residual_shrinkage"], 1.0)
            self.assertAlmostEqual(lambdas.loc[2, "residual_shrinkage"], 0.5)
            self.assertEqual(lambdas.loc[1, "n_training_origins"], 1)
            self.assertEqual(lambdas.loc[2, "n_training_origins"], 2)
            first_fold = result["fold_metrics"].query("fold_index == 1").set_index(
                "model"
            )
            self.assertAlmostEqual(first_fold.loc["baseline", "rmse"], 0.0)
            self.assertAlmostEqual(
                first_fold.loc["selected_shrinkage", "rmse"],
                2.0,
            )
            self.assertEqual(len(result["fold_metrics"]), 6)
            self.assertTrue((root / "nested" / "summary.csv").exists())

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

    def test_tourism_group_mapping_is_derived_from_top_level_hierarchy(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tourism.csv"
            pd.DataFrame({
                "date": pd.date_range("2020-01-01", periods=2, freq="MS"),
                "AAAAll": [1.0, 2.0],
                "ABAAll": [3.0, 4.0],
                "BAAAll": [5.0, 6.0],
                "AAAll": [7.0, 8.0],
                "AAAHol": [9.0, 10.0],
            }).to_csv(path, index=False)
            config = {
                "dataset": "tourism",
                "data_path": path,
                "data_format": "wide",
                "date_col": "date",
                "series_col": "series_id",
                "series_suffix": "All",
                "series_depth": 3,
                "residual_shrinkage_group_file": None,
                "residual_shrinkage_group_cols": "hierarchy_level_1",
            }

            mapping = _load_residual_shrinkage_group_mapping(config).set_index(
                "series_id"
            )

        self.assertEqual(set(mapping.index), {"AAAAll", "ABAAll", "BAAAll"})
        self.assertEqual(mapping.loc["AAAAll", "group_id"], "hierarchy_level_1=A")
        self.assertEqual(mapping.loc["BAAAll", "group_id"], "hierarchy_level_1=B")

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
