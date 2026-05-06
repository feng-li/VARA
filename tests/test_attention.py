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
    build_topk_attention_cache,
    evaluate_topk_cache_quality,
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

    def test_topk_attention_cache_matches_uncached_positive_and_negative_eta(self):
        v = jnp.arange(5.0)
        Y_lag = jax.random.normal(jax.random.PRNGKey(987), (7, 5))
        cache = build_topk_attention_cache(self.Z, top_k=2, batch_size=2)

        for eta in (0.7, -0.7):
            W_uncached = build_attention(self.Z, eta=eta, scale=1.0, attention_mode="topk", top_k=2)
            W_cached = build_attention(
                self.Z,
                eta=eta,
                scale=1.0,
                attention_mode="topk",
                top_k=2,
                attention_cache=cache,
            )
            mv_cached = attention_matvec(
                self.Z, eta, v, self.params_scale,
                batch_size=2,
                dense_threshold=1,
                attention_mode="topk",
                attention_top_k=2,
                attention_cache=cache,
            )
            X2_cached = attention_weighted_lags(
                Y_lag, self.Z, eta, self.params_scale,
                batch_size=2,
                dense_threshold=1,
                attention_mode="topk",
                attention_top_k=2,
                attention_cache=cache,
            )

            np.testing.assert_allclose(np.asarray(W_cached), np.asarray(W_uncached), rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(np.asarray(mv_cached), np.asarray(W_uncached @ v), rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(np.asarray(X2_cached), np.asarray(Y_lag @ W_uncached.T), rtol=1e-6, atol=1e-6)

    def test_cluster_topk_cache_matches_exact_when_all_clusters_are_probed(self):
        exact = build_topk_attention_cache(self.Z, top_k=2, batch_size=2)
        cluster = build_topk_attention_cache(
            self.Z,
            top_k=2,
            batch_size=2,
            cache_mode="cluster",
            cluster_count=1,
            cluster_probe=1,
            cluster_candidate_size=self.Z.shape[0],
        )

        for eta in (0.7, -0.7):
            W_exact = build_attention(
                self.Z,
                eta=eta,
                scale=1.0,
                attention_mode="topk",
                top_k=2,
                attention_cache=exact,
            )
            W_cluster = build_attention(
                self.Z,
                eta=eta,
                scale=1.0,
                attention_mode="topk",
                top_k=2,
                attention_cache=cluster,
            )

            np.testing.assert_allclose(np.asarray(W_cluster), np.asarray(W_exact), rtol=1e-6, atol=1e-6)

    def test_cluster_topk_cache_records_metadata_and_is_row_stochastic(self):
        cache = build_topk_attention_cache(
            self.Z,
            top_k=2,
            batch_size=2,
            cache_mode="cluster",
            cluster_count=2,
            cluster_probe=1,
            cluster_candidate_size=3,
        )
        W = build_attention(
            self.Z,
            eta=0.7,
            scale=1.0,
            attention_mode="topk",
            top_k=2,
            attention_cache=cache,
        )

        self.assertEqual(cache.mode, "cluster")
        self.assertEqual(cache.cluster_count, 2)
        self.assertEqual(cache.cluster_probe, 1)
        self.assertEqual(cache.cluster_candidate_size, 3)
        np.testing.assert_allclose(np.asarray(W.sum(axis=1)), np.ones(5), rtol=1e-6)
        self.assertTrue(bool(jnp.all(jnp.sum(W > 0, axis=1) <= 2)))

    def test_topk_cache_quality_is_perfect_for_exact_equivalent_cache(self):
        cache = build_topk_attention_cache(
            self.Z,
            top_k=2,
            batch_size=2,
            cache_mode="cluster",
            cluster_count=1,
            cluster_probe=1,
            cluster_candidate_size=self.Z.shape[0],
        )

        quality = evaluate_topk_cache_quality(
            self.Z,
            cache,
            sample_rows=5,
            batch_size=2,
        )

        self.assertEqual(quality["sample_rows"], 5)
        self.assertAlmostEqual(quality["recall_pos_mean"], 1.0, places=6)
        self.assertAlmostEqual(quality["recall_neg_mean"], 1.0, places=6)
        self.assertLess(quality["score_loss_pos_mean"], 1e-7)
        self.assertLess(quality["score_loss_neg_mean"], 1e-7)
        self.assertEqual(quality["selected_valid_count_pos_min"], 2)
        self.assertEqual(quality["selected_valid_count_neg_min"], 2)
        self.assertEqual(quality["selected_duplicate_pair_count_pos_max"], 0)
        self.assertEqual(quality["selected_duplicate_pair_count_neg_max"], 0)

    def test_cluster_topk_cache_does_not_select_duplicate_padding(self):
        Z = jax.random.normal(jax.random.PRNGKey(321), (12, 4))
        cache = build_topk_attention_cache(
            Z,
            top_k=3,
            batch_size=4,
            cache_mode="cluster",
            cluster_count=8,
            cluster_probe=8,
            cluster_candidate_size=6,
        )

        for idx in (np.asarray(cache.indices_pos), np.asarray(cache.indices_neg)):
            for row in idx:
                valid = row[row >= 0]
                self.assertEqual(len(valid), len(set(valid.tolist())))

        quality = evaluate_topk_cache_quality(Z, cache, sample_rows=12, batch_size=4)
        self.assertEqual(quality["selected_valid_count_pos_min"], 3)
        self.assertEqual(quality["selected_valid_count_neg_min"], 3)
        self.assertEqual(quality["selected_duplicate_pair_count_pos_max"], 0)
        self.assertEqual(quality["selected_duplicate_pair_count_neg_max"], 0)


if __name__ == "__main__":
    unittest.main()
