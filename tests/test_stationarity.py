import unittest

import numpy as np

from vara.stationarity import (
    check_panel_stationarity,
    standardize_panel,
    transform_panel,
    vara_stability_from_params,
)


class StationarityDiagnosticsTests(unittest.TestCase):
    def test_transform_panel_log1p_seasonal_diff(self):
        Y = np.arange(1.0, 31.0).reshape(15, 2)

        out = transform_panel(Y, transform="log1p-seasonal-diff", seasonal_lag=12)
        expected = np.log1p(Y[12:]) - np.log1p(Y[:-12])

        self.assertEqual(out.shape, (3, 2))
        np.testing.assert_allclose(out, expected)

    def test_standardize_panel_centers_and_scales_columns(self):
        Y = np.array([
            [1.0, 2.0],
            [2.0, 2.0],
            [3.0, 2.0],
        ])

        out = standardize_panel(Y)

        np.testing.assert_allclose(out[:, 0].mean(), 0.0, atol=1e-12)
        np.testing.assert_allclose(out[:, 0].std(), 1.0, atol=1e-12)
        np.testing.assert_allclose(out[:, 1], np.zeros(3), atol=1e-12)

    def test_check_panel_stationarity_handles_constant_series(self):
        Y = np.ones((30, 2))

        transformed, rows, summary = check_panel_stationarity(Y, series_ids=["a", "b"])

        self.assertEqual(transformed.shape, (30, 2))
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["n_stationary_5"], 0)
        self.assertFalse(rows[0]["stationary_5"])
        self.assertIn("constant", rows[0]["adf_error"])

    def test_check_panel_stationarity_validates_series_ids(self):
        with self.assertRaises(ValueError):
            check_panel_stationarity(np.ones((30, 2)), series_ids=["a"])

    def test_vara_stability_from_params_known_diagonal_case(self):
        Z = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        alpha = np.array([0.2, -0.3, 0.4])
        beta = np.zeros(3)

        out = vara_stability_from_params(alpha, beta, Z, eta=0.0)

        self.assertTrue(out["vara_stable"])
        np.testing.assert_allclose(out["vara_spectral_radius"], 0.4, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
