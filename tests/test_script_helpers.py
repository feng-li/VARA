import os
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from scripts.run_full_simulation import _full_simulation_output_paths


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


if __name__ == "__main__":
    unittest.main()
