import unittest

import jax.numpy as jnp

from vara.cov import assemble_H


class CovarianceAssemblyTests(unittest.TestCase):
    def test_non_gaussian_sigma2_h_block_uses_t_scaled_correction(self):
        p = 3
        T = 17
        sigma2 = jnp.array([0.5, 1.0, 2.0])
        Wa = jnp.eye(p)
        Wa_dot = jnp.zeros((p, p))
        gamma_a = jnp.array([0.2, 0.25, 0.3])
        G = jnp.eye(p) * 0.1
        Gamma0 = jnp.eye(p)

        H_full = assemble_H(
            Wa, Wa_dot, gamma_a, G, Gamma0, sigma2, T, kappa4=6.0, full=True
        )
        H_block = assemble_H(
            Wa, Wa_dot, gamma_a, G, Gamma0, sigma2, T, kappa4=6.0, full=False
        )

        expected_sigma = T * 0.5 / (sigma2 ** 2)
        sigma_idx = 1 + 2 * p + jnp.arange(p)

        self.assertTrue(jnp.allclose(H_full[sigma_idx, sigma_idx], expected_sigma))
        self.assertTrue(jnp.allclose(H_block["sigma_sigma"], expected_sigma))


if __name__ == "__main__":
    unittest.main()
