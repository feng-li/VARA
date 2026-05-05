import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np

from vara.attention import (
    attention_matvec,
    attention_rows,
    attention_weighted_lags,
    build_attention,
)


class AttentionTests(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.PRNGKey(123)
        self.Z = jax.random.normal(self.key, (5, 4))
        self.params_scale = {"eta_a": 1.0}

    def test_build_attention_is_row_stochastic_with_masked_diagonal(self):
        W = build_attention(self.Z, eta=0.7, scale=1.0)

        np.testing.assert_allclose(np.asarray(W.sum(axis=1)), np.ones(5), rtol=1e-6)
        self.assertTrue(np.all(np.asarray(jnp.diag(W)) < 1e-5))

    def test_attention_rows_matches_dense_attention_slice(self):
        W = build_attention(self.Z, eta=0.7, scale=1.0)
        rows = attention_rows(self.Z, eta=0.7, scale=1.0, start=1, stop=4)

        np.testing.assert_allclose(np.asarray(rows), np.asarray(W[1:4]), rtol=1e-6, atol=1e-6)

    def test_batched_attention_matvec_matches_dense_path(self):
        v = jnp.arange(5.0)

        dense = attention_matvec(self.Z, 0.7, v, self.params_scale, dense_threshold=10)
        batched = attention_matvec(
            self.Z, 0.7, v, self.params_scale, batch_size=2, dense_threshold=1
        )

        np.testing.assert_allclose(np.asarray(batched), np.asarray(dense), rtol=1e-6, atol=1e-6)

    def test_batched_attention_weighted_lags_matches_dense_path(self):
        Y_lag = jax.random.normal(jax.random.PRNGKey(456), (7, 5))

        dense = attention_weighted_lags(Y_lag, self.Z, 0.7, self.params_scale, dense_threshold=10)
        batched = attention_weighted_lags(
            Y_lag, self.Z, 0.7, self.params_scale, batch_size=2, dense_threshold=1
        )

        np.testing.assert_allclose(np.asarray(batched), np.asarray(dense), rtol=1e-6, atol=1e-6)

    def test_topk_attention_is_sparse_and_row_stochastic(self):
        W = build_attention(self.Z, eta=0.7, scale=1.0, attention_mode="topk", top_k=2)

        np.testing.assert_allclose(np.asarray(W.sum(axis=1)), np.ones(5), rtol=1e-6)
        self.assertTrue(bool(jnp.all(jnp.sum(W > 0, axis=1) <= 2)))
        self.assertTrue(np.all(np.asarray(jnp.diag(W)) < 1e-5))

    def test_topk_matvec_and_weighted_lags_match_topk_matrix(self):
        v = jnp.arange(5.0)
        Y_lag = jax.random.normal(jax.random.PRNGKey(789), (7, 5))
        W = build_attention(self.Z, eta=0.7, scale=1.0, attention_mode="topk", top_k=2)

        mv = attention_matvec(
            self.Z, 0.7, v, self.params_scale,
            batch_size=2,
            dense_threshold=1,
            attention_mode="topk",
            attention_top_k=2,
        )
        X2 = attention_weighted_lags(
            Y_lag, self.Z, 0.7, self.params_scale,
            batch_size=2,
            dense_threshold=1,
            attention_mode="topk",
            attention_top_k=2,
        )

        np.testing.assert_allclose(np.asarray(mv), np.asarray(W @ v), rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(np.asarray(X2), np.asarray(Y_lag @ W.T), rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
