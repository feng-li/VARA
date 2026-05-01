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

        self.assertAlmostEqual(float(params["eta_a"]), 0.5)
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
                "Z_eta_signal": 0.2,
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
                "Z_eta_signal": 0.3,
            },
        ])

        tables = make_simulation_reporting_tables(summary, setup)

        self.assertEqual(list(tables["low_rmse"]["method"]), ["qmle"])
        self.assertEqual(list(tables["high_rmse"]["method"]), ["two_stage"])
        self.assertNotIn("ECP_gamma", tables["high_ecp"].columns)
        self.assertIn("eta_boundary_rate", tables["eta_diagnostics"].columns)


if __name__ == "__main__":
    unittest.main()
