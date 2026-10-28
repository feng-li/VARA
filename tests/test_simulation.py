import math
import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from vara.attention import build_attention, build_topk_attention_cache
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
    full_single_replication,
    draw_simulation_params,
)


class SimulationTests(unittest.TestCase):
    def test_profile_search_checks_multiple_peaks_and_true_boundary(self):
        from unittest.mock import patch
        from vara.simulation import estimate_eta_concentrated

        def objective(eta, *args, **kwargs):
            return max(1.0 - (eta - 0.8)**2, 2.0 - 20.0 * (eta - 3.2)**2)

        with patch("vara.simulation.concentrated_eta_loglik", side_effect=objective):
            eta, info = estimate_eta_concentrated(None, None, {}, (0., 5.))
        self.assertAlmostEqual(eta, 3.2, places=5)
        self.assertFalse(info["on_boundary"])
        self.assertGreater(info["profile_evaluations"], 9)
        with patch("vara.simulation.concentrated_eta_loglik", side_effect=lambda x, *a, **k: -x):
            eta, info = estimate_eta_concentrated(None, None, {}, (0., 5.))
        self.assertEqual(eta, 0.)
        self.assertTrue(info["on_boundary"])

    def test_invalid_standard_errors_are_missing_coverage(self):
        from vara.simulation import pack_result_row, summarize_setting
        import pandas as pd

        params = draw_simulation_params(jax.random.PRNGKey(1), 3)
        rows = []
        for rep, se in enumerate((jnp.ones(3), jnp.full(3, jnp.nan))):
            rows.append(pack_result_row(
                setting="high", method="two_stage", rep=rep, T=20, p=3, q=2,
                dist="normal", seed=1, status=0, elapsed_sec=1., params_true=params,
                eta_hat=1., alpha_hat=params["beta"], beta_hat=params["gamma_a"],
                sigma2_hat=params["sigma2"], se_alpha=se, se_beta=se, se_sigma2=se,
                report_gamma_ecp=False,
            ))
        self.assertTrue(math.isnan(rows[1]["alpha_ecp"]))
        summary = summarize_setting(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(summary["n_fit"], 2)
        self.assertEqual(summary["n_se"], 1)
        self.assertEqual(summary["ECP_alpha"], 1.)

    def test_innovation_cases_share_design_but_not_noise(self):
        from unittest.mock import patch
        captured = []

        def fit(**kwargs):
            captured.append(kwargs)
            raise RuntimeError("capture only")

        key = jax.random.PRNGKey(21)
        params = draw_simulation_params(key, 5)
        with patch("vara.simulation.run_two_stage_on_dataset", side_effect=fit):
            for dist in ("normal", "uniform"):
                full_single_replication(
                    0, key, params, draw_Z(key, 5, 3), "high", 12, 5, 3, dist, 21,
                    ["two_stage_subsampled"], .05, "lbfgs", .05, (0., 5.), 9,
                    .02, 0., 3, 2500, params_scale={"eta_a": math.log(5)}, redraw_dgp=True)
        np.testing.assert_array_equal(captured[0]["Z"], captured[1]["Z"])
        for name in params:
            np.testing.assert_array_equal(captured[0]["params_true"][name], captured[1]["params_true"][name])
        self.assertFalse(np.array_equal(captured[0]["Y"], captured[1]["Y"]))

    def test_replication_full_and_nested_pilots(self):
        from unittest.mock import patch
        key = jax.random.PRNGKey(18)
        p, q, T = 5, 3, 12
        params = draw_simulation_params(key, p)
        captured = []

        def fit(**kwargs):
            captured.append(kwargs)
            estimates = {"eta_hat": 1., "se_eta": jnp.nan}
            for name in ("alpha", "beta", "sigma2"):
                estimates[name + "_hat"] = jnp.ones(p)
                estimates["se_" + name] = jnp.ones(p)
            return estimates, 0

        with patch("vara.simulation.run_two_stage_on_dataset", side_effect=fit):
            rows = full_single_replication(
                0, key, params, draw_Z(key, p, q), "high", T, p, q, "uniform", 18,
                ["two_stage"], .05, "lbfgs", .05, (0., 5.), 5, .02, 0., 3, 2500,
                params_scale={"eta_a": math.log(p)}, redraw_dgp=True, pilot_sizes=[2, 5])
        self.assertEqual([r["status"] for r in rows], [0, 0])
        self.assertEqual(captured[0]["Y"].shape, (T + 1, p))
        np.testing.assert_array_equal(captured[0]["Y"], captured[1]["Y"])
        np.testing.assert_array_equal(captured[0]["component_indices"], captured[1]["component_indices"][:2])

    def test_p_for_setting_matches_paper_rules(self):
        self.assertEqual([p_for_setting(t, "high") for t in (200, 400, 800)], [200, 500, 1600])
        self.assertEqual(p_for_setting(16, "low"), 4)
        self.assertEqual(p_for_setting(4, "high"), 8)
        with self.assertRaises(ValueError):
            p_for_setting(16, "unknown")

    def test_kappa4_for_supported_distributions(self):
        self.assertEqual(kappa4_for_dist("uniform"), 1.8)
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
        for dist in ["normal", "uniform", "laplace", "t5"]:
            draws = draw_standard_innovations(key, (10, 3), dist)
            self.assertEqual(draws.shape, (10, 3))
            self.assertTrue(bool(jnp.isfinite(draws).all()))

        with self.assertRaises(ValueError):
            draw_standard_innovations(key, (10, 3), "bad")

    def test_simulate_vara_returns_finite_series_with_innovation_initial_state(self):
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
        expected = draw_standard_innovations(key, (T, p), "normal")[0] * jnp.sqrt(params["sigma2"])
        np.testing.assert_allclose(np.asarray(Y[0]), np.asarray(expected), atol=1e-7)
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

    def test_simulate_vara_topk_cache_matches_uncached(self):
        p, q, T = 5, 4, 10
        key = jax.random.PRNGKey(654)
        params = {
            "eta_a": jnp.asarray(0.5),
            "gamma_a": jnp.array([0.15, 0.2, 0.25, 0.18, 0.16]),
            "beta": jnp.array([0.1, -0.1, 0.05, 0.02, -0.03]),
            "sigma2": jnp.array([0.5, 0.7, 1.0, 0.8, 0.6]),
        }
        Z = draw_Z(key, p, q)
        cache = build_topk_attention_cache(Z, top_k=2, batch_size=2)

        Y_uncached = simulate_vara(
            key,
            params,
            Z,
            T,
            p,
            q,
            "normal",
            PARAMS_SCALE,
            attention_mode="topk",
            attention_top_k=2,
        )
        Y_cached = simulate_vara(
            key,
            params,
            Z,
            T,
            p,
            q,
            "normal",
            PARAMS_SCALE,
            attention_mode="topk",
            attention_top_k=2,
            attention_cache=cache,
        )

        np.testing.assert_allclose(np.asarray(Y_cached), np.asarray(Y_uncached), rtol=1e-7, atol=1e-7)

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

    def test_draw_good_Z_for_eta_records_topk_screening(self):
        p, q = 6, 4
        key = jax.random.PRNGKey(654)
        params = {
            "eta_a": jnp.asarray(1.0),
            "gamma_a": jnp.full((p,), 0.2),
            "beta": jnp.zeros((p,)),
            "sigma2": jnp.ones((p,)),
        }

        _, info = draw_good_Z_for_eta(
            key,
            p,
            q,
            params,
            PARAMS_SCALE,
            max_tries=1,
            approximate_threshold=1,
            approx_rows=3,
            approx_pairs=10,
            attention_mode="topk",
            attention_top_k=2,
            verbose=False,
        )

        self.assertEqual(info["attention_mode"], "topk")
        self.assertEqual(info["attention_top_k"], 2)

    def test_draw_good_Z_for_eta_records_topk_exact_screening(self):
        p, q = 5, 4
        key = jax.random.PRNGKey(321)
        params = {
            "eta_a": jnp.asarray(1.0),
            "gamma_a": jnp.full((p,), 0.2),
            "beta": jnp.zeros((p,)),
            "sigma2": jnp.ones((p,)),
        }

        _, info = draw_good_Z_for_eta(
            key,
            p,
            q,
            params,
            PARAMS_SCALE,
            max_tries=1,
            attention_mode="topk",
            attention_top_k=2,
            verbose=False,
        )

        self.assertFalse(info["approximate"])
        self.assertEqual(info["attention_mode"], "topk")
        self.assertEqual(info["attention_top_k"], 2)

    def test_golden_section_maximize_finds_quadratic_maximum(self):
        fn = lambda x: -(x - 1.25) ** 2 + 3.0

        x_hat, obj = golden_section_maximize(fn, lower=-2.0, upper=3.0, tol=1e-6)

        self.assertTrue(math.isclose(float(x_hat), 1.25, abs_tol=1e-4))
        self.assertTrue(math.isclose(obj, 3.0, abs_tol=1e-6))


if __name__ == "__main__":
    unittest.main()
