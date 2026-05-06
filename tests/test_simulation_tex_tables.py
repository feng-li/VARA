import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.make_simulation_tex_tables import generate_tex_tables


class SimulationTexTableTests(unittest.TestCase):
    def test_generate_tex_tables_from_simulation_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_dir = root / "results"
            tex_dir = root / "tex"
            results_dir.mkdir()

            tag = "R=2_p=10_q=50_T=100_seed=2026_test"
            pd.DataFrame([
                {
                    "setting": "low",
                    "p_rule": "floor(sqrt(T))",
                    "dist": "normal",
                    "T": 100,
                    "p": 10,
                    "q": 50,
                    "attention_mode": "dense",
                    "attention_top_k": "",
                }
            ]).to_csv(results_dir / f"table_simulation_setup_{tag}.csv", index=False)

            pd.DataFrame([
                {
                    "setting": "low",
                    "dist": "normal",
                    "T": 100,
                    "p": 10,
                    "q": 50,
                    "method": "two_stage",
                    "RMSE_gamma": 0.12345,
                    "RMSE_alpha": 0.2,
                    "RMSE_beta": 0.3,
                    "RMSE_sigma2": 0.4,
                }
            ]).to_csv(results_dir / f"table_simulation_low_rmse_{tag}.csv", index=False)

            pd.DataFrame([
                {
                    "setting": "low",
                    "dist": "normal",
                    "T": 100,
                    "p": 10,
                    "q": 50,
                    "method": "two_stage",
                    "ECP_gamma": 0.95,
                    "ECP_alpha": 0.94,
                    "ECP_beta": 0.93,
                    "ECP_sigma2": 0.92,
                }
            ]).to_csv(results_dir / f"table_simulation_low_ecp_{tag}.csv", index=False)

            pd.DataFrame([
                {
                    "setting": "high",
                    "dist": "normal",
                    "T": 1000,
                    "p": 31622,
                    "q": 50,
                    "method": "two_stage",
                    "RMSE_gamma": 0.017,
                    "RMSE_alpha": 0.031,
                    "RMSE_beta": 0.178,
                    "RMSE_sigma2": 0.034,
                }
            ]).to_csv(results_dir / f"table_simulation_high_rmse_{tag}.csv", index=False)

            pd.DataFrame([
                {
                    "setting": "high",
                    "dist": "normal",
                    "T": 1000,
                    "p": 31622,
                    "q": 50,
                    "method": "two_stage",
                    "ECP_alpha": 0.95,
                    "ECP_beta": 0.94,
                    "ECP_sigma2": 0.93,
                }
            ]).to_csv(results_dir / f"table_simulation_high_ecp_{tag}.csv", index=False)

            pd.DataFrame([
                {
                    "setting": "high",
                    "dist": "normal",
                    "T": 1000,
                    "p": 31622,
                    "q": 50,
                    "method": "two_stage",
                    "RMSE_gamma": 0.017,
                    "eta_boundary_rate": 0.0,
                    "attention_mode": "topk",
                    "attention_top_k": 200,
                    "TopK_cache_recall_pos_mean": 0.3444,
                    "TopK_cache_recall_neg_mean": 0.3485,
                    "TopK_cache_score_loss_pos_abs_mean": 0.1039,
                    "TopK_cache_score_loss_neg_abs_mean": 0.1048,
                }
            ]).to_csv(results_dir / f"table_simulation_publication_wide_{tag}.csv", index=False)

            written = generate_tex_tables(
                results_dir=results_dir,
                tex_dir=tex_dir,
                config_tag=tag,
            )

            combined = written["combined"].read_text(encoding="utf-8")
            xhigh = written["simulation_xhigh_diagnostics"].read_text(encoding="utf-8")

            self.assertIn(r"\label{tab:simulation_low_rmse}", combined)
            self.assertIn("Normal & Two-stage & 100 & 0.1235", combined)
            self.assertIn(r"\label{tab:simulation_xhigh_diagnostics}", xhigh)
            self.assertIn("1000 & 31622 & 200 & 0.017 & 0 & 0.344/0.348 & 0.104/0.105", xhigh)


if __name__ == "__main__":
    unittest.main()
