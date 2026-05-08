import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from vara.attention import build_attention
from vara.forecast import VARAForecaster


class VARAForecasterTests(unittest.TestCase):
    def test_fit_predict_shapes_and_residual_identity(self):
        key_y, key_z = jax.random.split(jax.random.PRNGKey(0))
        Y = jax.random.normal(key_y, (30, 4))
        Z = jax.random.normal(key_z, (4, 3))

        forecaster = VARAForecaster(grid_size=5)
        returned = forecaster.fit(Y, Z)
        pred = forecaster.predict(h=3)

        self.assertIs(returned, forecaster)
        self.assertEqual(pred.shape, (3, 4))
        self.assertEqual(forecaster.fitted_values().shape, (29, 4))
        self.assertEqual(forecaster.residuals().shape, (29, 4))
        self.assertTrue(bool(jnp.isfinite(pred).all()))
        np.testing.assert_allclose(
            np.asarray(forecaster.fitted_values() + forecaster.residuals()),
            np.asarray(Y[1:]),
            rtol=1e-6,
            atol=1e-6,
        )

        diagnostics = forecaster.diagnostics()
        self.assertEqual(diagnostics["T"], 30)
        self.assertEqual(diagnostics["p"], 4)
        self.assertEqual(diagnostics["q"], 3)
        self.assertIn("stationarity_proxy", diagnostics)

    def test_predict_matches_one_step_model_equation_for_known_params(self):
        Z = jnp.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
        y_last = jnp.array([0.5, -0.2, 0.3])
        eta = 0.4
        alpha = jnp.array([0.2, -0.1, 0.15])
        beta = jnp.array([0.1, 0.25, -0.2])

        forecaster = VARAForecaster(params_scale={"eta_a": 1.0})
        forecaster.eta_hat = jnp.asarray(eta)
        forecaster.alpha_hat = alpha
        forecaster.beta_hat = beta
        forecaster.sigma2_hat = jnp.ones(3)
        forecaster.Z_ = Z
        forecaster.y_last_ = y_last

        pred = forecaster.predict(h=1)[0]
        W = build_attention(Z, eta, 1.0)
        expected = alpha * y_last + beta * (W @ y_last)

        np.testing.assert_allclose(np.asarray(pred), np.asarray(expected), rtol=1e-6, atol=1e-6)

    def test_predict_requires_fit(self):
        with self.assertRaises(RuntimeError):
            VARAForecaster().predict(h=1)

    def test_predict_validates_horizon(self):
        forecaster = VARAForecaster()
        forecaster.eta_hat = jnp.asarray(0.0)
        forecaster.alpha_hat = jnp.ones(2)
        forecaster.beta_hat = jnp.zeros(2)
        forecaster.sigma2_hat = jnp.ones(2)
        forecaster.Z_ = jnp.ones((2, 1))
        forecaster.y_last_ = jnp.ones(2)

        with self.assertRaises(ValueError):
            forecaster.predict(h=0)
        with self.assertRaises(ValueError):
            forecaster.predict(h=1.5)

    def test_fit_validates_shapes(self):
        Y = jnp.zeros((5, 3))
        Z = jnp.zeros((4, 2))

        with self.assertRaises(ValueError):
            VARAForecaster(grid_size=3).fit(Y, Z)

        with self.assertRaises(ValueError):
            VARAForecaster(grid_size=1).fit(jnp.zeros((5, 3)), jnp.zeros((3, 2)))

    def test_topk_predict_matches_topk_attention_matrix_for_known_params(self):
        Z = jax.random.normal(jax.random.PRNGKey(1), (5, 3))
        y_last = jnp.arange(5.0)
        eta = -0.3
        alpha = jnp.linspace(-0.1, 0.1, 5)
        beta = jnp.linspace(0.05, 0.25, 5)

        forecaster = VARAForecaster(
            params_scale={"eta_a": 1.0},
            attention_mode="topk",
            attention_top_k=2,
            batch_size=2,
            dense_threshold=1,
        )
        forecaster.eta_hat = jnp.asarray(eta)
        forecaster.alpha_hat = alpha
        forecaster.beta_hat = beta
        forecaster.sigma2_hat = jnp.ones(5)
        forecaster.Z_ = Z
        forecaster.y_last_ = y_last

        pred = forecaster.predict(h=1)[0]
        W = build_attention(Z, eta, 1.0, attention_mode="topk", top_k=2)
        expected = alpha * y_last + beta * (W @ y_last)

        np.testing.assert_allclose(np.asarray(pred), np.asarray(expected), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
