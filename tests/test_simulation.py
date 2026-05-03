import math
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from vara.attention import build_attention
from vara.dgp import draw_Z, draw_good_Z_for_eta, ensure_stationarity
from vara.simulation import (
    PARAMS_SCALE,
    draw_standard_innovations,
    flatten_simulation_params,
    golden_section_maximize,
    kappa4_for_dist,
    p_for_setting,
    reparam_theta_from_params,
    simulate_vara,
)


class SimulationTests(unittest.TestCase):
    def test_p_for_setting_matches_paper_rules(self):
        self.assertEqual(p_for_setting(16, "low"), 4)
        self.assertEqual(p_for_setting(4, "high"), 8)
        with self.assertRaises(ValueError):
            p_for_setting(16, "unknown")

    def test_kappa4_for_supported_distributions(self):
        self.assertEqual(kappa4_for_dist("normal"), 3.0)
        self.assertEqual(kappa4_for_dist("laplace"), 6.0)
        self.assertEqual(kappa4_for_dist("t5"), 9.0)
        with self.assertRaises(ValueError):
            kappa4_for_dist("bad")

    def test_ensure_stationarity_rescales_only_when_needed(self):
        beta = jnp.array([0.8, -0.7])
        gamma = jnp.array([0.6, 0.2])
        beta_new, gamma_new, info = ensure_stationarity(beta, gamma, margin=0.98)

        self.assertTrue(bool(info["rescaled"]))
        self.assertLess(float(jnp.max(jnp.abs(beta_new)) + jnp.max(jnp.abs(gamma_new))), 0.981)

        beta_ok = jnp.array([0.1, -0.2])
        gamma_ok = jnp.array([0.1, 0.2])
        beta_same, gamma_same, info_ok = ensure_stationarity(beta_ok, gamma_ok, margin=0.98)

        self.assertFalse(bool(info_ok["rescaled"]))
        np.testing.assert_allclose(np.asarray(beta_same), np.asarray(beta_ok), rtol=1e-7)
        np.testing.assert_allclose(np.asarray(gamma_same), np.asarray(gamma_ok), rtol=1e-7)

    def test_flatten_and_reparam_theta_use_expected_order(self):
        params = {
            "eta_a": jnp.asarray(0.5),
            "gamma_a": jnp.array([0.1, 0.2]),
            "beta": jnp.array([-0.1, 0.3]),
            "sigma2": jnp.array([0.5, 1.5]),
        }

        flat = flatten_simulation_params(params, p=2)
        reparam = reparam_theta_from_params(params, p=2)

        np.testing.assert_allclose(np.asarray(flat[:5]), np.array([0.5, 0.1, 0.2, -0.1, 0.3]))
        np.testing.assert_allclose(np.asarray(jax.nn.softplus(reparam[-2:])), np.array([0.5, 1.5]), rtol=1e-6)

    def test_draw_standard_innovations_shapes_and_errors(self):
        key = jax.random.PRNGKey(123)
        for dist in ["normal", "laplace", "t5"]:
            draws = draw_standard_innovations(key, (10, 3), dist)
            self.assertEqual(draws.shape, (10, 3))
            self.assertTrue(bool(jnp.isfinite(draws).all()))

        with self.assertRaises(ValueError):
            draw_standard_innovations(key, (10, 3), "bad")

    def test_simulate_vara_returns_finite_series_with_zero_initial_state(self):
        p, q, T = 3, 4, 8
        key = jax.random.PRNGKey(321)
        params = {
            "eta_a": jnp.asarray(0.5),
            "gamma_a": jnp.array([0.15, 0.2, 0.25]),
            "beta": jnp.array([0.1, -0.1, 0.05]),
            "sigma2": jnp.array([0.5, 0.7, 1.0]),
        }
        Z = draw_Z(key, p, q)

        Y = simulate_vara(key, params, Z, T, p, q, "normal", PARAMS_SCALE)

        self.assertEqual(Y.shape, (T, p))
        np.testing.assert_allclose(np.asarray(Y[0]), np.zeros(p), atol=1e-7)
        self.assertTrue(bool(jnp.isfinite(Y).all()))

    def test_simulate_vara_matches_precomputed_attention(self):
        p, q, T = 4, 5, 10
        key = jax.random.PRNGKey(456)
        params = {
            "eta_a": jnp.asarray(0.5),
            "gamma_a": jnp.array([0.15, 0.2, 0.25, 0.18]),
            "beta": jnp.array([0.1, -0.1, 0.05, 0.02]),
            "sigma2": jnp.array([0.5, 0.7, 1.0, 0.8]),
        }
        Z = draw_Z(key, p, q)
        W = build_attention(Z, float(params["eta_a"]), PARAMS_SCALE["eta_a"])

        Y_on_demand = simulate_vara(key, params, Z, T, p, q, "normal", PARAMS_SCALE)
        Y_precomputed = simulate_vara(
            key,
            params,
            Z,
            T,
            p,
            q,
            "normal",
            PARAMS_SCALE,
            precomputed_attention=W,
        )

        np.testing.assert_allclose(np.asarray(Y_precomputed), np.asarray(Y_on_demand), rtol=1e-7, atol=1e-7)

    def test_simulate_vara_rejects_wrong_precomputed_attention_shape(self):
        p, q, T = 3, 4, 8
        key = jax.random.PRNGKey(789)
        params = {
            "eta_a": jnp.asarray(0.5),
            "gamma_a": jnp.array([0.15, 0.2, 0.25]),
            "beta": jnp.array([0.1, -0.1, 0.05]),
            "sigma2": jnp.array([0.5, 0.7, 1.0]),
        }
        Z = draw_Z(key, p, q)

        with self.assertRaises(ValueError):
            simulate_vara(
                key,
                params,
                Z,
                T,
                p,
                q,
                "normal",
                PARAMS_SCALE,
                precomputed_attention=jnp.eye(p + 1),
            )

    def test_draw_good_Z_for_eta_uses_approximate_screen_when_requested(self):
        p, q = 6, 4
        key = jax.random.PRNGKey(987)
        params = {
            "eta_a": jnp.asarray(1.0),
            "gamma_a": jnp.full((p,), 0.2),
            "beta": jnp.zeros((p,)),
            "sigma2": jnp.ones((p,)),
        }

        Z, info = draw_good_Z_for_eta(
            key,
            p,
            q,
            params,
            PARAMS_SCALE,
            max_tries=1,
            approximate_threshold=1,
            approx_rows=3,
            approx_pairs=10,
            verbose=False,
        )

        self.assertEqual(Z.shape, (p, q))
        self.assertTrue(info["approximate"])
        self.assertEqual(info["approx_rows"], 3)
        self.assertEqual(info["approx_pairs"], 10)

    def test_golden_section_maximize_finds_quadratic_maximum(self):
        fn = lambda x: -(x - 1.25) ** 2 + 3.0

        x_hat, obj = golden_section_maximize(fn, lower=-2.0, upper=3.0, tol=1e-6)

        self.assertTrue(math.isclose(float(x_hat), 1.25, abs_tol=1e-4))
        self.assertTrue(math.isclose(obj, 3.0, abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
