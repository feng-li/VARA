import unittest

import numpy as np
import pandas as pd

from vara.forecast_metrics import (
    add_composite_rank_score,
    add_origin_win_rank_score,
    forecast_metric_tables,
    normal_crps,
    out_of_sample_r2,
    out_of_sample_r2_zero,
)


class ForecastMetricTests(unittest.TestCase):
    def test_origin_win_rank_score_matches_tourism_strict_win_definition(self):
        comparison = pd.DataFrame({"model": ["a", "b", "c"]})
        origin_errors = pd.DataFrame({
            "date": np.repeat(pd.to_datetime(["2020-01-31", "2020-02-29"]), 3),
            "model": ["a", "b", "c", "a", "b", "c"],
            "mae": [1.0, 2.0, 2.0, 3.0, 1.0, 2.0],
        })

        ranked, audit = add_origin_win_rank_score(
            comparison,
            origin_errors,
            error_column="mae",
        )
        indexed = ranked.set_index("model")

        self.assertEqual(float(indexed.loc["a", "rank_score"]), 50.0)
        self.assertEqual(float(indexed.loc["b", "rank_score"]), 50.0)
        self.assertEqual(float(indexed.loc["c", "rank_score"]), 25.0)
        self.assertEqual(int(indexed.loc["a", "n_rank_score_comparators"]), 2)
        self.assertEqual(len(audit), 6)
        tied = audit.loc[
            audit["date"].eq(pd.Timestamp("2020-01-31"))
            & audit["model"].isin(["b", "c"])
        ]
        self.assertTrue(tied["absolute_error_wins"].eq(0).all())

    def test_origin_win_rank_score_uses_literal_unrounded_comparisons(self):
        comparison = pd.DataFrame({"model": ["a", "b"]})
        origin_errors = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "model": ["a", "b"],
            "mae": [1.0, 1.0 + 1e-12],
        })

        ranked, _ = add_origin_win_rank_score(
            comparison,
            origin_errors,
            error_column="mae",
        )
        indexed = ranked.set_index("model")

        self.assertEqual(float(indexed.loc["a", "rank_score"]), 100.0)
        self.assertEqual(float(indexed.loc["b", "rank_score"]), 0.0)

    def test_origin_win_rank_score_rejects_incomplete_method_origin_grid(self):
        comparison = pd.DataFrame({"model": ["a", "b"]})
        origin_errors = pd.DataFrame({
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-02-29"]),
            "model": ["a", "a", "b"],
            "mae": [1.0, 1.0, 2.0],
        })

        with self.assertRaisesRegex(ValueError, "one finite error"):
            add_origin_win_rank_score(
                comparison,
                origin_errors,
                error_column="mae",
            )

    def test_composite_rank_score_supports_loss_gain_and_zero_targets(self):
        comparison = pd.DataFrame({
            "model": ["best", "middle", "worst"],
            "loss": [1.0, 2.0, 3.0],
            "gain": [3.0, 2.0, 1.0],
            "bias": [-0.1, 0.2, 0.3],
        })

        ranked = add_composite_rank_score(
            comparison,
            lower_is_better=("loss",),
            higher_is_better=("gain",),
            closest_to_zero=("bias",),
        ).set_index("model")

        self.assertEqual(float(ranked.loc["best", "rank_score"]), 100.0)
        self.assertEqual(float(ranked.loc["middle", "rank_score"]), 50.0)
        self.assertEqual(float(ranked.loc["worst", "rank_score"]), 0.0)
        self.assertEqual(int(ranked.loc["best", "overall_rank"]), 1)

    def test_normal_crps_broadcasts_scalar_mean(self):
        actual = np.asarray([0.0, 1.0])
        result = normal_crps(0.0, np.ones(2), actual)

        self.assertAlmostEqual(float(result[0]), (np.sqrt(2.0) - 1.0) / np.sqrt(np.pi))
        self.assertTrue(np.isfinite(result).all())

    def test_out_of_sample_r2_uses_test_window_series_means(self):
        test = np.asarray([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        forecast = np.zeros_like(test)
        error = forecast - test

        self.assertAlmostEqual(float(out_of_sample_r2(error, test)), -6.5)
        np.testing.assert_allclose(out_of_sample_r2(error, test, axis=0), np.asarray([-4.0, -9.0]))
        np.testing.assert_allclose(out_of_sample_r2(error, test, axis=1), np.asarray([-1.5, -11.5]))

    def test_out_of_sample_r2_zero_uses_zero_forecast_denominator(self):
        test = np.asarray([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        forecast = np.zeros_like(test)
        error = forecast - test

        self.assertAlmostEqual(float(out_of_sample_r2_zero(error, test)), 0.0)
        np.testing.assert_allclose(out_of_sample_r2_zero(error, test, axis=0), np.zeros(2))
        np.testing.assert_allclose(out_of_sample_r2_zero(error, test, axis=1), np.zeros(2))

    def test_forecast_metric_tables_emit_oos_r2_columns(self):
        train = np.asarray([
            [10.0, 20.0],
            [11.0, 21.0],
            [12.0, 22.0],
        ])
        test = np.asarray([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        forecasts = {"zero": np.zeros_like(test)}

        by_series, by_origin, summary = forecast_metric_tables(
            forecasts,
            test,
            train,
            pd.date_range("2021-01-01", periods=2, freq="D"),
            ["a", "b"],
            season_length=1,
            forecast_mode="recursive",
            horizon=2,
        )

        self.assertIn("oos_r2", by_series.columns)
        self.assertIn("oos_r2", by_origin.columns)
        self.assertIn("oos_r2", summary.columns)
        self.assertIn("oos_r2_zero", by_series.columns)
        self.assertIn("oos_r2_zero", by_origin.columns)
        self.assertIn("oos_r2_zero", summary.columns)
        self.assertAlmostEqual(float(summary.loc[0, "oos_r2"]), -6.5)
        self.assertAlmostEqual(float(summary.loc[0, "oos_r2_zero"]), 0.0)
        np.testing.assert_allclose(by_series["oos_r2"].to_numpy(), np.asarray([-4.0, -9.0]))
        np.testing.assert_allclose(by_origin["oos_r2"].to_numpy(), np.asarray([-1.5, -11.5]))
        np.testing.assert_allclose(by_series["oos_r2_zero"].to_numpy(), np.zeros(2))
        np.testing.assert_allclose(by_origin["oos_r2_zero"].to_numpy(), np.zeros(2))

    def test_forecast_metric_tables_accept_precomputed_seasonal_scales(self):
        train = np.asarray([
            [10.0, 20.0],
            [11.0, 22.0],
            [13.0, 25.0],
        ])
        test = np.asarray([
            [14.0, 27.0],
            [16.0, 30.0],
        ])
        forecasts = {"model": np.asarray([[13.0, 26.0], [15.0, 29.0]])}
        dates = pd.date_range("2021-01-01", periods=2, freq="D")

        expected = forecast_metric_tables(
            forecasts,
            test,
            train,
            dates,
            ["a", "b"],
            season_length=1,
            forecast_mode="recursive",
            horizon=2,
        )
        seasonal_diff = train[1:] - train[:-1]
        actual = forecast_metric_tables(
            forecasts,
            test,
            None,
            dates,
            ["a", "b"],
            season_length=1,
            forecast_mode="recursive",
            horizon=2,
            seasonal_mae_scale=np.mean(np.abs(seasonal_diff), axis=0),
            seasonal_mse_scale=np.mean(seasonal_diff ** 2, axis=0),
        )

        for expected_frame, actual_frame in zip(expected, actual, strict=True):
            pd.testing.assert_frame_equal(expected_frame, actual_frame)


if __name__ == "__main__":
    unittest.main()
