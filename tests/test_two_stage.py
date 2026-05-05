import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from vara.attention import build_attention
from vara.two_stage import concentrated_eta_loglik, fit_all_components_two_stage


class TwoStageTests(unittest.TestCase):
    def test_fit_all_components_recovers_noise_free_linear_recursion(self):
        p, q, T = 3, 4, 80
        eta = 0.5
        params_scale = {"eta_a": 1.0}
        Z = jax.random.normal(jax.random.PRNGKey(0), (p, q))
        W = build_attention(Z, eta, params_scale["eta_a"])

        alpha_true = jnp.array([0.2, -0.1, 0.15])
        beta_true = jnp.array([0.1, 0.2, -0.15])

        y = jnp.array([1.0, -0.7, 0.4])
        rows = [y]
        for _ in range(1, T):
            y = alpha_true * y + beta_true * (W @ y)
            rows.append(y)
        Y = jnp.stack(rows)

        fit = fit_all_components_two_stage(Y, Z, eta, params_scale, ridge=1e-12)

        np.testing.assert_allclose(np.asarray(fit["alpha_hat"]), np.asarray(alpha_true), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(np.asarray(fit["beta_hat"]), np.asarray(beta_true), rtol=1e-5, atol=1e-5)
        self.assertLess(float(jnp.max(fit["sigma2_hat"])), 1e-10)

    def test_concentrated_eta_loglik_returns_finite_scalar(self):
        p, q, T = 3, 4, 20
        params_scale = {"eta_a": 1.0}
        Z = jax.random.normal(jax.random.PRNGKey(1), (p, q))
        Y = jax.random.normal(jax.random.PRNGKey(2), (T, p))

        value = concentrated_eta_loglik(0.25, Y, Z, params_scale)

        self.assertEqual(value.shape, ())
        self.assertTrue(bool(jnp.isfinite(value)))

    def test_topk_two_stage_path_returns_finite_estimates(self):
        p, q, T = 5, 4, 20
        params_scale = {"eta_a": 1.0}
        Z = jax.random.normal(jax.random.PRNGKey(3), (p, q))
        Y = jax.random.normal(jax.random.PRNGKey(4), (T, p))

        fit = fit_all_components_two_stage(
            Y, Z, 0.25, params_scale,
            attention_mode="topk",
            attention_top_k=2,
            batch_size=2,
            dense_threshold=1,
        )
        value = concentrated_eta_loglik(
            0.25, Y, Z, params_scale,
            attention_mode="topk",
            attention_top_k=2,
            batch_size=2,
            dense_threshold=1,
        )

        self.assertTrue(bool(jnp.isfinite(value)))
        self.assertTrue(bool(jnp.isfinite(fit["alpha_hat"]).all()))
        self.assertTrue(bool(jnp.isfinite(fit["beta_hat"]).all()))


if __name__ == "__main__":
    unittest.main()
