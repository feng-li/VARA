import os
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import pandas as pd

from scripts.run_full_simulation import (
    _full_simulation_output_paths,
    _simulation_reporting_output_paths,
    draw_simulation_dgp_params,
    make_simulation_reporting_tables,
    make_simulation_setup_table,
)


class ScriptHelperTests(unittest.TestCase):
    def test_full_simulation_output_paths_include_major_config(self):
        detail, summary, paper = _full_simulation_output_paths(
            Path("data"),
            R=500,
            T_list=[8],
            settings=["low"],
            q=3,
            seed=2026,
            high_max_p=None,
            run_tag="20260430_12.34.56",
        )

        self.assertEqual(
            summary.name,
            "summary_full_simulation_R=500_p=2_q=3_T=8_seed=2026_20260430_12.34.56.csv",
        )
        self.assertTrue(detail.name.startswith("results_full_simulation_detail_R=500_p=2_q=3_T=8_seed=2026_"))
        self.assertTrue(paper.name.startswith("paper_table_full_simulation_R=500_p=2_q=3_T=8_seed=2026_"))

    def test_simulation_reporting_output_paths_use_expected_prefixes(self):
        paths = _simulation_reporting_output_paths(Path("data"), "R=1_p=2_q=3_T=8_seed=2026_tag")

        self.assertEqual(
            paths["publication_wide"].name,
            "table_simulation_publication_wide_R=1_p=2_q=3_T=8_seed=2026_tag.csv",
        )
        self.assertIn("eta_diagnostics", paths)

    def test_draw_simulation_dgp_params_uses_identification_stable_ranges(self):
        params, stat_info = draw_simulation_dgp_params(jax.random.PRNGKey(2026), p=10)

        self.assertAlmostEqual(float(params["eta_a"]), 1.0)
        self.assertTrue(bool(jnp.all(params["gamma_a"] >= 0.15)))
        self.assertTrue(bool(jnp.all(params["gamma_a"] <= 0.30)))
        self.assertTrue(bool(jnp.all(params["beta"] >= -0.20)))
        self.assertTrue(bool(jnp.all(params["beta"] <= 0.20)))
        self.assertTrue(bool(jnp.all(params["sigma2"] >= 0.50)))
        self.assertTrue(bool(jnp.all(params["sigma2"] <= 1.00)))
        self.assertLessEqual(float(stat_info["lhs_after"]), 0.98 + 1e-6)

    def test_simulation_reporting_tables_split_low_and_high_outputs(self):
        setup = make_simulation_setup_table(
            R=500,
            T_list=[100],
            settings=["low", "high"],
            dist_list=["normal"],
            q=50,
            high_max_p=2000,
            screen_z=True,
            precompute_dgp_attention=True,
            require_good_z=True,
            dgp_eta_a=1.0,
            z_max_tries=200,
            z_score_quantile=0.999,
            z_max_abs_score=5.0,
            z_min_eta_signal=0.05,
            z_min_eta_info=0.05,
            z_min_attention_x2_std=0.02,
            z_min_attention_x2_std_adaptive_factor=1.1,
            z_max_attention_effective_neighbors=1000.0,
            z_max_attention_effective_neighbor_fraction=0.25,
            z_approx_threshold=5000,
            z_approx_rows=512,
            z_approx_pairs=200_000,
            attention_mode="dense",
            attention_top_k=None,
            attention_cache_mode="exact",
            attention_cache_clusters=512,
            attention_cache_probes=4,
            attention_cache_candidate_size=None,
            attention_cache_candidate_multiplier=5.0,
            attention_cache_diagnostic_rows=64,
        )
        summary = pd.DataFrame([
            {
                "setting": "low",
                "dist": "normal",
                "T": 100,
                "p": 10,
                "q": 50,
                "method": "qmle",
                "RMSE_gamma": 0.1,
                "RMSE_alpha": 0.2,
                "RMSE_beta": 0.3,
                "RMSE_sigma2": 0.4,
                "ECP_gamma": 0.95,
                "ECP_alpha": 0.94,
                "ECP_beta": 0.93,
                "ECP_sigma2": 0.92,
                "valid_rate": 1.0,
                "eta_boundary_rate": 0.0,
                "Z_accepted": True,
                "Z_approximate": False,
                "Z_approx_rows": float("nan"),
                "Z_approx_pairs": float("nan"),
                "Z_approx_threshold": 5000,
                "Z_eta_signal": 0.2,
                "Z_attention_x2_std_mean": 0.04,
                "Z_attention_x2_std_effective_min": 0.02,
                "Z_attention_effective_neighbors_mean": 500.0,
                "Z_attention_effective_neighbors_limit": 1000.0,
                "DGP_attention_precomputed": True,
                "TopK_attention_cached": False,
                "TopK_cache_recall_pos_mean": float("nan"),
                "TopK_cache_score_loss_pos_mean": float("nan"),
                "attention_mode": "dense",
                "attention_top_k": float("nan"),
                "attention_cache_mode": "exact",
            },
            {
                "setting": "high",
                "dist": "normal",
                "T": 100,
                "p": 1000,
                "q": 50,
                "method": "two_stage",
                "RMSE_gamma": 0.5,
                "RMSE_alpha": 0.6,
                "RMSE_beta": 0.7,
                "RMSE_sigma2": 0.8,
                "ECP_gamma": float("nan"),
                "ECP_alpha": 0.91,
                "ECP_beta": 0.90,
                "ECP_sigma2": 0.89,
                "valid_rate": 1.0,
                "eta_boundary_rate": 0.1,
                "Z_accepted": True,
                "Z_approximate": True,
                "Z_approx_rows": 512,
                "Z_approx_pairs": 200_000,
                "Z_approx_threshold": 5000,
                "Z_eta_signal": 0.3,
                "Z_attention_x2_std_mean": 0.05,
                "Z_attention_x2_std_effective_min": 0.02,
                "Z_attention_effective_neighbors_mean": 600.0,
                "Z_attention_effective_neighbors_limit": 1000.0,
                "DGP_attention_precomputed": True,
                "TopK_attention_cached": False,
                "TopK_cache_recall_pos_mean": float("nan"),
                "TopK_cache_score_loss_pos_mean": float("nan"),
                "attention_mode": "dense",
                "attention_top_k": float("nan"),
                "attention_cache_mode": "exact",
            },
        ])

        tables = make_simulation_reporting_tables(summary, setup)

        self.assertEqual(list(tables["low_rmse"]["method"]), ["qmle"])
        self.assertEqual(list(tables["high_rmse"]["method"]), ["two_stage"])
        self.assertNotIn("ECP_gamma", tables["high_ecp"].columns)
        self.assertIn("eta_boundary_rate", tables["eta_diagnostics"].columns)
        self.assertIn("Z_attention_x2_std_mean", tables["eta_diagnostics"].columns)
        self.assertIn("Z_attention_x2_std_effective_min", tables["eta_diagnostics"].columns)
        self.assertIn("Z_attention_effective_neighbors_limit", tables["eta_diagnostics"].columns)
        self.assertIn("Z_approximate", tables["eta_diagnostics"].columns)
        self.assertIn("Z_approx_rows", tables["eta_diagnostics"].columns)
        self.assertIn("DGP_attention_precomputed", tables["eta_diagnostics"].columns)
        self.assertIn("TopK_attention_cached", tables["eta_diagnostics"].columns)
        self.assertIn("attention_mode", tables["eta_diagnostics"].columns)
        self.assertIn("attention_cache_mode", tables["eta_diagnostics"].columns)
        self.assertIn("TopK_cache_recall_pos_mean", tables["eta_diagnostics"].columns)
        self.assertIn("TopK_cache_score_loss_pos_mean", tables["eta_diagnostics"].columns)


if __name__ == "__main__":
    unittest.main()
