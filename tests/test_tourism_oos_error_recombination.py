import unittest

import numpy as np
import pandas as pd

from scripts.recombine_tourism_oos_error_vara import (
    add_method_rank_scores,
    historical_metric_table,
    rolling_recombine,
)


def _panel(n_months: int = 18, n_series: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2014-01-01", periods=n_months, freq="MS")
    rows = []
    for index, date in enumerate(dates):
        for series in range(n_series):
            baseline = 100.0 + series
            purpose = float(index % 3 - 1)
            vara = float((index + series) % 5 - 2)
            actual = baseline + 0.4 * purpose + 0.2 * vara
            rows.append({
                "date": date,
                "series_id": f"S{series}",
                "year": date.year,
                "month": date.month,
                "actual": actual,
                "direct_auto_ets": baseline,
                "purpose_sum_auto_ets": baseline + purpose,
                "purpose_correction": purpose,
                "vara_correction": vara,
            })
    return pd.DataFrame(rows)


class RollingOosErrorRecombinationTests(unittest.TestCase):
    def test_rank_score_uses_four_core_accuracy_metrics(self):
        methods = pd.DataFrame({
            "model": ["core_best", "core_middle", "core_worst"],
            "rmse": [1.0, 2.0, 3.0],
            "mae": [1.0, 2.0, 3.0],
            "mase": [1.0, 2.0, 3.0],
            "rmsse": [1.0, 2.0, 3.0],
            # Deliberately reverse diagnostics to verify they do not enter the score.
            "mape_pct": [30.0, 20.0, 10.0],
            "smape_pct": [30.0, 20.0, 10.0],
            "bias": [30.0, 20.0, 10.0],
            "oos_r2": [0.1, 0.2, 0.3],
            "oos_r2_zero": [0.1, 0.2, 0.3],
        })

        ranked = add_method_rank_scores(methods).set_index("model")

        self.assertEqual(ranked.loc["core_best", "overall_rank"], 1)
        self.assertEqual(ranked.loc["core_worst", "overall_rank"], 3)
        np.testing.assert_allclose(
            ranked.loc[["core_best", "core_middle", "core_worst"], "rank_score"],
            [100.0, 50.0, 0.0],
        )
        self.assertEqual(ranked.loc["core_best", "rank_mape_pct"], 3.0)
        self.assertEqual(ranked.loc["core_best", "rank_oos_r2"], 3.0)

    def test_rank_score_rejects_duplicate_methods(self):
        methods = pd.DataFrame({
            "model": ["same", "same"],
            "rmse": [1.0, 2.0],
            "mae": [1.0, 2.0],
            "mase": [1.0, 2.0],
            "rmsse": [1.0, 2.0],
            "mape_pct": [1.0, 2.0],
            "smape_pct": [1.0, 2.0],
            "bias": [1.0, 2.0],
            "oos_r2": [0.5, 0.4],
            "oos_r2_zero": [0.9, 0.8],
        })

        with self.assertRaisesRegex(ValueError, "duplicate methods"):
            add_method_rank_scores(methods)

    def test_uses_only_dates_before_target(self):
        panel = _panel()
        target = pd.Timestamp("2015-01-01")
        first, coefficients = rolling_recombine(
            panel,
            [target],
            calibration_window_months=12,
            vara_weight_cap=0.5,
        )
        changed = panel.copy()
        changed.loc[changed["date"] >= target, "actual"] += 10000.0
        second, changed_coefficients = rolling_recombine(
            changed,
            [target],
            calibration_window_months=12,
            vara_weight_cap=0.5,
        )
        np.testing.assert_allclose(
            first["rolling_purpose_oos_vara"],
            second["rolling_purpose_oos_vara"],
        )
        np.testing.assert_allclose(
            coefficients[["full_purpose_weight", "full_vara_weight"]],
            changed_coefficients[["full_purpose_weight", "full_vara_weight"]],
        )
        self.assertLess(coefficients["training_end"].iloc[0], target)

    def test_fits_no_vara_comparator_independently(self):
        panel = _panel()
        _, coefficients = rolling_recombine(
            panel,
            [pd.Timestamp("2015-01-01")],
            calibration_window_months=12,
            vara_weight_cap=0.5,
        )
        row = coefficients.iloc[0]
        self.assertNotAlmostEqual(
            row["purpose_only_weight"], row["full_purpose_weight"], places=8
        )
        self.assertAlmostEqual(row["full_purpose_weight"], 0.4, places=8)
        self.assertAlmostEqual(row["full_vara_weight"], 0.2, places=8)

    def test_shared_lambda_uses_one_weight_for_both_corrections(self):
        panel = _panel()
        combined_correction = panel["purpose_correction"] + panel["vara_correction"]
        panel["actual"] = panel["direct_auto_ets"] + 0.3 * combined_correction

        predictions, coefficients = rolling_recombine(
            panel,
            [pd.Timestamp("2015-01-01")],
            calibration_window_months=12,
            vara_weight_cap=0.5,
            combination_mode="shared-lambda",
            shared_lambda_cap=0.5,
        )

        self.assertEqual(len(coefficients), 1)
        row = coefficients.iloc[0]
        self.assertEqual(row["combination_group"], "all")
        self.assertAlmostEqual(row["shared_lambda"], 0.3, places=8)
        self.assertAlmostEqual(row["full_purpose_weight"], 0.3, places=8)
        self.assertAlmostEqual(row["full_vara_weight"], 0.3, places=8)
        np.testing.assert_allclose(
            predictions["rolling_purpose_oos_vara"],
            predictions["actual"],
        )

        changed = panel.copy()
        changed.loc[changed["date"] >= pd.Timestamp("2015-01-01"), "actual"] += 1000.0
        changed_predictions, changed_coefficients = rolling_recombine(
            changed,
            [pd.Timestamp("2015-01-01")],
            calibration_window_months=12,
            vara_weight_cap=0.5,
            combination_mode="shared-lambda",
            shared_lambda_cap=0.5,
        )
        np.testing.assert_allclose(
            predictions["rolling_purpose_oos_vara"],
            changed_predictions["rolling_purpose_oos_vara"],
        )
        self.assertAlmostEqual(
            coefficients["shared_lambda"].iloc[0],
            changed_coefficients["shared_lambda"].iloc[0],
            places=12,
        )

    def test_group_shared_lambda_is_isolated_by_group(self):
        panel = _panel(n_series=4)
        panel["group"] = np.where(panel["series_id"].isin(["S0", "S1"]), "A", "B")
        group_lambda = panel["group"].map({"A": 0.1, "B": 0.4}).to_numpy(dtype=float)
        combined_correction = (
            panel["purpose_correction"] + panel["vara_correction"]
        ).to_numpy(dtype=float)
        panel["actual"] = (
            panel["direct_auto_ets"].to_numpy(dtype=float)
            + group_lambda * combined_correction
        )
        target = pd.Timestamp("2015-01-01")

        predictions, coefficients = rolling_recombine(
            panel,
            [target],
            calibration_window_months=12,
            vara_weight_cap=0.5,
            combination_mode="shared-lambda",
            combination_group_col="group",
            shared_lambda_cap=0.5,
        )
        by_group = coefficients.set_index("combination_group")
        self.assertAlmostEqual(by_group.loc["A", "shared_lambda"], 0.1, places=8)
        self.assertAlmostEqual(by_group.loc["B", "shared_lambda"], 0.4, places=8)
        np.testing.assert_allclose(
            predictions["rolling_purpose_oos_vara"],
            predictions["actual"],
        )

        changed = panel.copy()
        changed.loc[(changed["group"] == "B") & (changed["date"] < target), "actual"] += 50.0
        _, changed_coefficients = rolling_recombine(
            changed,
            [target],
            calibration_window_months=12,
            vara_weight_cap=0.5,
            combination_mode="shared-lambda",
            combination_group_col="group",
            shared_lambda_cap=0.5,
        )
        changed_by_group = changed_coefficients.set_index("combination_group")
        self.assertAlmostEqual(
            by_group.loc["A", "shared_lambda"],
            changed_by_group.loc["A", "shared_lambda"],
            places=12,
        )

        _, global_coefficients = rolling_recombine(
            panel,
            [target],
            calibration_window_months=12,
            vara_weight_cap=0.5,
            combination_mode="shared-lambda",
            shared_lambda_cap=0.5,
        )
        _, pooled_coefficients = rolling_recombine(
            panel,
            [target],
            calibration_window_months=12,
            vara_weight_cap=0.5,
            combination_mode="shared-lambda",
            combination_group_col="group",
            shared_lambda_cap=0.5,
            group_lambda_global_weight=0.25,
        )
        global_lambda = global_coefficients["shared_lambda"].iloc[0]
        pooled_by_group = pooled_coefficients.set_index("combination_group")
        self.assertAlmostEqual(
            pooled_by_group.loc["A", "shared_lambda"],
            0.25 * global_lambda + 0.75 * 0.1,
            places=8,
        )
        self.assertAlmostEqual(
            pooled_by_group.loc["B", "shared_lambda"],
            0.25 * global_lambda + 0.75 * 0.4,
            places=8,
        )

    def test_rejects_too_short_calibration_window(self):
        with self.assertRaisesRegex(ValueError, "calibration_window_months"):
            rolling_recombine(
                _panel(),
                [pd.Timestamp("2015-01-01")],
                calibration_window_months=11,
                vara_weight_cap=0.5,
            )

    def test_historical_percentages_use_each_periods_baseline(self):
        predictions = pd.DataFrame({
            "year": [2014, 2014, 2015, 2015],
            "actual": [0.0, 0.0, 0.0, 0.0],
            "auto_ets": [1.0, 1.0, 10.0, 10.0],
            "rolling_purpose_blend": [0.8, 0.8, 8.0, 8.0],
            "rolling_purpose_oos_vara": [0.5, 0.5, 5.0, 5.0],
            "raw_oos_vara": [2.0, 2.0, 20.0, 20.0],
        })
        metrics = historical_metric_table(predictions)
        full = metrics[metrics["model"] == "rolling_purpose_oos_vara"]
        np.testing.assert_allclose(
            full["rmse_improvement_vs_auto_ets_pct"],
            [50.0, 50.0, 50.0],
        )
        np.testing.assert_allclose(
            full["rmse_improvement_vs_purpose_blend_pct"],
            [37.5, 37.5, 37.5],
        )


if __name__ == "__main__":
    unittest.main()
