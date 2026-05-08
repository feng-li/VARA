import unittest
from argparse import Namespace

from scripts.run_pilot_consistency import _run_kwargs_from_args


class PilotConsistencyConfigTests(unittest.TestCase):
    def test_kappa4_defaults_to_distribution_moment(self):
        args = Namespace(
            seed=2026,
            R=1,
            T=100,
            p=10,
            q=50,
            k_list="0,1",
            pilot_mode="qmle",
            dist="laplace",
            kappa4=None,
            eta_lower=-2.0,
            eta_upper=2.0,
            qmle_algorithm="lbfgs",
            ray_num_cpus=1,
            max_cpus_per_task=1,
            results_dir="outputs/test",
        )

        kwargs = _run_kwargs_from_args(args)

        self.assertEqual(kwargs["kappa4"], 6.0)

    def test_explicit_kappa4_overrides_distribution_default(self):
        args = Namespace(
            seed=2026,
            R=1,
            T=100,
            p=10,
            q=50,
            k_list="0,1",
            pilot_mode="qmle",
            dist="laplace",
            kappa4=7.0,
            eta_lower=-2.0,
            eta_upper=2.0,
            qmle_algorithm="lbfgs",
            ray_num_cpus=1,
            max_cpus_per_task=1,
            results_dir="outputs/test",
        )

        kwargs = _run_kwargs_from_args(args)

        self.assertEqual(kwargs["kappa4"], 7.0)


if __name__ == "__main__":
    unittest.main()
