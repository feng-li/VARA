import unittest

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from vara.forecast_significance import (
    diebold_mariano_test,
    holm_adjust,
    panel_diebold_mariano_tests,
)


class ForecastSignificanceTests(unittest.TestCase):
    def test_holm_adjustment(self):
        np.testing.assert_allclose(
            holm_adjust([0.01, 0.04, 0.03]),
            [0.03, 0.06, 0.06],
        )

    def test_one_step_harvey_corrected_statistic(self):
        focal = np.array([1.0, 3.0, 2.0, 0.0])
        comparator = np.full(4, 4.0)
        differential = focal - comparator
        variance = np.mean((differential - differential.mean()) ** 2)
        raw = differential.mean() / np.sqrt(variance / len(differential))
        expected = raw * np.sqrt((len(differential) - 1) / len(differential))

        result = diebold_mariano_test(focal, comparator, forecast_horizon=1)

        self.assertAlmostEqual(result["dm_statistic"], expected)
        self.assertAlmostEqual(
            result["p_value_raw_two_sided"],
            2.0 * norm.sf(abs(raw)),
        )
        self.assertAlmostEqual(
            result["p_value_raw_focal_better"],
            norm.cdf(raw),
        )
        self.assertAlmostEqual(
            result["p_value_two_sided"],
            2.0 * student_t.sf(abs(expected), df=3),
        )
        self.assertEqual(result["hac_lag"], 0)

    def test_identical_losses_return_null_result(self):
        loss = np.arange(1.0, 13.0)
        result = diebold_mariano_test(loss, loss)
        self.assertEqual(result["dm_statistic"], 0.0)
        self.assertEqual(result["p_value_raw_two_sided"], 1.0)
        self.assertEqual(result["p_value_raw_focal_better"], 0.5)
        self.assertEqual(result["p_value_two_sided"], 1.0)
        self.assertEqual(result["p_value_focal_better"], 0.5)

    def test_panel_test_uses_origins_as_observations(self):
        dates = np.arange("2016-01", "2017-01", dtype="datetime64[M]")
        actual = np.zeros((12, 3))
        focal = np.linspace(0.5, 1.0, 12)[:, None] * np.ones((1, 3))
        comparator = np.linspace(1.0, 2.0, 12)[:, None] * np.ones((1, 3))

        results = panel_diebold_mariano_tests(
            {"focal": focal, "comparator": comparator},
            actual,
            dates,
            focal_model="focal",
        )

        self.assertEqual(len(results), 2)
        self.assertTrue((results["n_origins"] == 12).all())
        self.assertTrue((results["n_series"] == 3).all())
        self.assertTrue((results["dm_statistic"] < 0.0).all())
        self.assertTrue(results["significant_improvement_holm"].all())


if __name__ == "__main__":
    unittest.main()
