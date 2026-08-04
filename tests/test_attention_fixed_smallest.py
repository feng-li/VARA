import os
import unittest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax.numpy as jnp
import numpy as np

from vara.attention import (
    build_cached_topk_attention_csr,
    build_fixed_smallest_topk_attention_cache,
    cached_topk_attention_weighted_lags,
    attention_weighted_lags,
    topk_attention_cached_indices_weights,
)


class FixedSmallestAttentionCacheTests(unittest.TestCase):
    def test_exact_ties_use_smallest_component_indices(self):
        # Every off-diagonal cosine score is one, so the graph is determined
        # entirely by the documented component-index tie-break.
        features = jnp.ones((6, 1), dtype=jnp.float32)
        cache = build_fixed_smallest_topk_attention_cache(
            features,
            top_k=2,
            batch_size=2,
        )

        expected = np.asarray([
            [1, 2],
            [0, 2],
            [0, 1],
            [0, 1],
            [0, 1],
            [0, 1],
        ], dtype=np.int32)
        np.testing.assert_array_equal(np.asarray(cache.indices_pos), expected)
        np.testing.assert_array_equal(np.asarray(cache.indices_neg), expected)
        np.testing.assert_array_equal(np.asarray(cache.scores_pos), np.ones((6, 2)))
        np.testing.assert_array_equal(np.asarray(cache.scores_neg), np.ones((6, 2)))
        self.assertEqual(cache.mode, "exact-fixed-smallest")

    def test_matches_lexicographic_smallest_cosine_scores(self):
        features = np.asarray([
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
            [-1.0, 0.0],
            [-1.0, 0.0],
        ], dtype=np.float32)
        cache = build_fixed_smallest_topk_attention_cache(
            jnp.asarray(features),
            top_k=3,
            batch_size=4,
        )

        scores = features.astype(np.float64) @ features.astype(np.float64).T
        expected_idx = []
        expected_scores = []
        columns = np.arange(len(features))
        for row in range(len(features)):
            scores[row, row] = np.inf
            order = np.lexsort((columns, scores[row]))[:3]
            expected_idx.append(order)
            expected_scores.append(scores[row, order])

        np.testing.assert_array_equal(
            np.asarray(cache.indices_pos),
            np.asarray(expected_idx, dtype=np.int32),
        )
        np.testing.assert_allclose(
            np.asarray(cache.scores_pos),
            np.asarray(expected_scores),
            rtol=0.0,
            atol=0.0,
        )
        for row, indices in enumerate(np.asarray(cache.indices_pos)):
            self.assertNotIn(row, indices.tolist())

    def test_tied_cache_is_identical_across_batches_and_eta_signs(self):
        features = jnp.asarray([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ], dtype=jnp.float32)
        first = build_fixed_smallest_topk_attention_cache(
            features,
            top_k=4,
            batch_size=1,
        )
        second = build_fixed_smallest_topk_attention_cache(
            features,
            top_k=4,
            batch_size=5,
        )

        np.testing.assert_array_equal(
            np.asarray(first.indices_pos),
            np.asarray(second.indices_pos),
        )
        np.testing.assert_array_equal(
            np.asarray(first.scores_pos),
            np.asarray(second.scores_pos),
        )
        positive_idx, _ = topk_attention_cached_indices_weights(
            first, 0.5, 1.0, 0, len(features)
        )
        negative_idx, _ = topk_attention_cached_indices_weights(
            first, -0.5, 1.0, 0, len(features)
        )
        np.testing.assert_array_equal(np.asarray(positive_idx), np.asarray(negative_idx))

    def test_sparse_operator_matches_cached_jax_attention(self):
        rng = np.random.default_rng(71)
        features = rng.normal(size=(14, 4)).astype(np.float32)
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        lagged = rng.normal(size=(9, 14)).astype(np.float32)
        cache = build_fixed_smallest_topk_attention_cache(
            jnp.asarray(features),
            top_k=5,
            batch_size=4,
        )

        for eta in (-0.8, 0.0, 0.6):
            sparse_result = cached_topk_attention_weighted_lags(
                lagged,
                cache,
                eta,
            )
            jax_result = np.asarray(attention_weighted_lags(
                jnp.asarray(lagged),
                jnp.asarray(features),
                eta,
                {"eta_a": 1.0},
                batch_size=4,
                dense_threshold=0,
                attention_mode="topk",
                attention_top_k=5,
                attention_cache=cache,
            ))
            np.testing.assert_allclose(sparse_result, jax_result, rtol=2e-6, atol=2e-6)
            operator = build_cached_topk_attention_csr(cache, eta)
            np.testing.assert_allclose(
                np.asarray(operator.sum(axis=1)).ravel(),
                1.0,
                rtol=1e-6,
                atol=1e-6,
            )

    def test_rejects_nonfinite_features_and_nonpositive_batch(self):
        with self.assertRaisesRegex(ValueError, "finite numeric"):
            build_fixed_smallest_topk_attention_cache(
                jnp.asarray([[1.0], [jnp.nan]]),
                top_k=1,
            )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            build_fixed_smallest_topk_attention_cache(
                jnp.asarray([[1.0], [-1.0]]),
                top_k=1,
                batch_size=0,
            )


if __name__ == "__main__":
    unittest.main()
