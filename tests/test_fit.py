import unittest
from unittest.mock import patch

import jax
import jax.numpy as jnp

from vara.fit import estimate


class FitEstimateTests(unittest.TestCase):
    def test_estimate_forwards_kappa4_to_sandwich_variance(self):
        Y = jnp.zeros((3, 1))
        Z = jnp.zeros((1, 2))
        theta_init = jnp.array([0.1, 0.2, 0.3, 0.0])
        params_scale = {"eta_a": 1.0}

        with (
            patch("vara.fit.mle_optimize") as mle_optimize,
            patch("vara.fit.compute_sandwich_variance") as sandwich,
        ):
            mle_optimize.return_value = (
                theta_init,
                None,
                None,
                {"status": 0},
            )
            sandwich.return_value = {
                "theta": jnp.array([0.1, 0.2, 0.3, 0.5]),
                "se": jnp.ones(4),
            }

            estimate(
                Y=Y,
                Z=Z,
                key=jax.random.PRNGKey(0),
                theta_init=theta_init,
                params_scale=params_scale,
                algorithm="lbfgs",
                kappa4=6.0,
            )

        self.assertEqual(sandwich.call_args.kwargs["kappa4"], 6.0)


if __name__ == "__main__":
    unittest.main()
