"""Attention matrix utilities for the VARA model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TopKAttentionCache:
    """Fixed top-k neighbor candidates for a given feature matrix ``Z``."""

    indices_pos: jnp.ndarray
    scores_pos: jnp.ndarray
    indices_neg: Optional[jnp.ndarray] = None
    scores_neg: Optional[jnp.ndarray] = None
    mode: str = "exact"
    cluster_count: int = 0
    cluster_probe: int = 0
    cluster_candidate_size: int = 0

    @property
    def top_k(self) -> int:
        return int(self.indices_pos.shape[1])


def _top_k_size(p: int, top_k: Optional[int]) -> int:
    if top_k is None:
        raise ValueError("top_k must be provided when attention_mode='topk'.")
    return int(min(max(1, top_k), max(p - 1, 1)))


def _check_attention_mode(attention_mode: str) -> str:
    if attention_mode not in {"dense", "topk"}:
        raise ValueError("attention_mode must be 'dense' or 'topk'.")
    return attention_mode


def _build_exact_topk_attention_cache(
    Z: jnp.ndarray,
    top_k: int,
    batch_size: int = 512,
    include_negative: bool = True,
) -> TopKAttentionCache:
    """Precompute exact top-k raw score candidates in row batches."""
    p = Z.shape[0]
    k = _top_k_size(p, top_k)
    pos_idx_chunks = []
    pos_score_chunks = []
    neg_idx_chunks = []
    neg_score_chunks = []

    for start in range(0, p, batch_size):
        stop = min(p, start + batch_size)
        Zb = Z[start:stop]
        raw_scores = Zb @ Z.T

        row_ids = jnp.arange(start, stop)
        col_ids = jnp.arange(p)
        diag_mask = row_ids[:, None] == col_ids[None, :]

        pos_scores, pos_idx = jax.lax.top_k(
            jnp.where(diag_mask, -jnp.inf, raw_scores),
            k,
        )
        pos_idx_chunks.append(pos_idx)
        pos_score_chunks.append(pos_scores)

        if include_negative:
            neg_top_scores, neg_idx = jax.lax.top_k(
                jnp.where(diag_mask, -jnp.inf, -raw_scores),
                k,
            )
            neg_idx_chunks.append(neg_idx)
            neg_score_chunks.append(-neg_top_scores)

    indices_pos = jnp.concatenate(pos_idx_chunks, axis=0)
    scores_pos = jnp.concatenate(pos_score_chunks, axis=0)
    if include_negative:
        indices_neg = jnp.concatenate(neg_idx_chunks, axis=0)
        scores_neg = jnp.concatenate(neg_score_chunks, axis=0)
    else:
        indices_neg = None
        scores_neg = None

    return TopKAttentionCache(
        indices_pos=indices_pos,
        scores_pos=scores_pos,
        indices_neg=indices_neg,
        scores_neg=scores_neg,
        mode="exact",
    )


def _anchor_indices(p: int, cluster_count: int) -> jnp.ndarray:
    n_clusters = int(min(max(1, cluster_count), p))
    if n_clusters == p:
        return jnp.arange(p)
    return jnp.floor(jnp.linspace(0, p - 1, n_clusters)).astype(jnp.int32)


def _assign_to_anchors(
    Z: jnp.ndarray,
    anchors: jnp.ndarray,
    batch_size: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    p = Z.shape[0]
    cluster_chunks = []
    score_chunks = []
    for start in range(0, p, batch_size):
        stop = min(p, start + batch_size)
        scores = Z[start:stop] @ anchors.T
        cluster_chunks.append(jnp.argmax(scores, axis=1).astype(jnp.int32))
        score_chunks.append(jnp.max(scores, axis=1))
    return jnp.concatenate(cluster_chunks), jnp.concatenate(score_chunks)


def _build_cluster_member_table(
    cluster_ids: jnp.ndarray,
    cluster_scores: jnp.ndarray,
    cluster_count: int,
    candidate_size: int,
) -> jnp.ndarray:
    ids_np = np.asarray(cluster_ids)
    scores_np = np.asarray(cluster_scores)
    table = np.full((cluster_count, candidate_size), -1, dtype=np.int32)

    for c in range(cluster_count):
        members = np.flatnonzero(ids_np == c).astype(np.int32)
        if members.size == 0:
            continue
        if members.size > candidate_size:
            member_scores = scores_np[members]
            part = np.argpartition(member_scores, -candidate_size)[-candidate_size:]
            part = part[np.argsort(member_scores[part])[::-1]]
            chosen = members[part]
        else:
            chosen = members
        table[c, : min(chosen.size, candidate_size)] = chosen[:candidate_size]
    return jnp.asarray(table)


def _cluster_candidate_topk(
    Zb: jnp.ndarray,
    row_ids: jnp.ndarray,
    Z: jnp.ndarray,
    anchors: jnp.ndarray,
    member_table: jnp.ndarray,
    top_k: int,
    cluster_probe: int,
    *,
    positive: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    anchor_scores = Zb @ anchors.T
    probe_scores = anchor_scores if positive else -anchor_scores
    _, probe_idx = jax.lax.top_k(probe_scores, cluster_probe)

    candidate_idx = member_table[probe_idx].reshape((Zb.shape[0], -1))
    valid_candidate = candidate_idx >= 0
    safe_candidate_idx = jnp.where(valid_candidate, candidate_idx, 0)
    candidate_Z = Z[safe_candidate_idx]
    raw_scores = jnp.sum(Zb[:, None, :] * candidate_Z, axis=2)
    invalid_or_self = (~valid_candidate) | (candidate_idx == row_ids[:, None])

    if positive:
        masked_scores = jnp.where(invalid_or_self, -jnp.inf, raw_scores)
        selected_scores, selected_pos = jax.lax.top_k(masked_scores, top_k)
        selected_idx = jnp.take_along_axis(candidate_idx, selected_pos, axis=1)
        return selected_idx, selected_scores

    masked_scores = jnp.where(invalid_or_self, jnp.inf, raw_scores)
    selected_neg_scores, selected_pos = jax.lax.top_k(-masked_scores, top_k)
    selected_idx = jnp.take_along_axis(candidate_idx, selected_pos, axis=1)
    return selected_idx, -selected_neg_scores


def _build_cluster_topk_attention_cache(
    Z: jnp.ndarray,
    top_k: int,
    batch_size: int = 512,
    include_negative: bool = True,
    cluster_count: int = 512,
    cluster_probe: int = 4,
    cluster_candidate_size: Optional[int] = None,
    cluster_candidate_multiplier: float = 5.0,
) -> TopKAttentionCache:
    """Build an approximate top-k cache by searching nearby anchor clusters."""
    p = Z.shape[0]
    k = _top_k_size(p, top_k)
    anchor_idx = _anchor_indices(p, cluster_count)
    anchors = Z[anchor_idx]
    n_clusters = int(anchors.shape[0])
    n_probe = int(min(max(1, cluster_probe), n_clusters))
    candidate_size = (
        int(cluster_candidate_size)
        if cluster_candidate_size is not None and int(cluster_candidate_size) > 0
        else int(np.ceil(float(cluster_candidate_multiplier) * float(k)))
    )
    candidate_size = int(min(max(k, candidate_size), p))

    cluster_ids, cluster_scores = _assign_to_anchors(Z, anchors, batch_size)
    member_table = _build_cluster_member_table(
        cluster_ids,
        cluster_scores,
        n_clusters,
        candidate_size,
    )

    pos_idx_chunks = []
    pos_score_chunks = []
    neg_idx_chunks = []
    neg_score_chunks = []

    for start in range(0, p, batch_size):
        stop = min(p, start + batch_size)
        row_ids = jnp.arange(start, stop)
        Zb = Z[start:stop]
        pos_idx, pos_scores = _cluster_candidate_topk(
            Zb,
            row_ids,
            Z,
            anchors,
            member_table,
            k,
            n_probe,
            positive=True,
        )
        pos_idx_chunks.append(pos_idx)
        pos_score_chunks.append(pos_scores)

        if include_negative:
            neg_idx, neg_scores = _cluster_candidate_topk(
                Zb,
                row_ids,
                Z,
                anchors,
                member_table,
                k,
                n_probe,
                positive=False,
            )
            neg_idx_chunks.append(neg_idx)
            neg_score_chunks.append(neg_scores)

    indices_pos = jnp.concatenate(pos_idx_chunks, axis=0)
    scores_pos = jnp.concatenate(pos_score_chunks, axis=0)
    if include_negative:
        indices_neg = jnp.concatenate(neg_idx_chunks, axis=0)
        scores_neg = jnp.concatenate(neg_score_chunks, axis=0)
    else:
        indices_neg = None
        scores_neg = None

    return TopKAttentionCache(
        indices_pos=indices_pos,
        scores_pos=scores_pos,
        indices_neg=indices_neg,
        scores_neg=scores_neg,
        mode="cluster",
        cluster_count=n_clusters,
        cluster_probe=n_probe,
        cluster_candidate_size=candidate_size,
    )


def build_topk_attention_cache(
    Z: jnp.ndarray,
    top_k: int,
    batch_size: int = 512,
    include_negative: bool = True,
    cache_mode: str = "exact",
    cluster_count: int = 512,
    cluster_probe: int = 4,
    cluster_candidate_size: Optional[int] = None,
    cluster_candidate_multiplier: float = 5.0,
) -> TopKAttentionCache:
    """Precompute top-k raw score candidates for repeated top-k attention.

    ``cache_mode='exact'`` searches all columns in row batches. ``'cluster'``
    first assigns rows to deterministic anchor clusters, then searches only the
    member table from the nearest probed clusters before keeping final top-k.
    """
    if cache_mode == "exact":
        return _build_exact_topk_attention_cache(
            Z,
            top_k,
            batch_size=batch_size,
            include_negative=include_negative,
        )
    if cache_mode == "cluster":
        return _build_cluster_topk_attention_cache(
            Z,
            top_k,
            batch_size=batch_size,
            include_negative=include_negative,
            cluster_count=cluster_count,
            cluster_probe=cluster_probe,
            cluster_candidate_size=cluster_candidate_size,
            cluster_candidate_multiplier=cluster_candidate_multiplier,
        )
    raise ValueError("cache_mode must be 'exact' or 'cluster'.")


def _exact_topk_for_rows(
    Z: jnp.ndarray,
    row_idx: jnp.ndarray,
    top_k: int,
    batch_size: int,
    *,
    positive: bool,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute exact top-k raw scores for selected rows."""
    p = Z.shape[0]
    k = _top_k_size(p, top_k)
    idx_chunks = []
    score_chunks = []

    for start in range(0, int(row_idx.shape[0]), batch_size):
        stop = min(int(row_idx.shape[0]), start + batch_size)
        rows = row_idx[start:stop]
        raw_scores = Z[rows] @ Z.T
        diag_mask = rows[:, None] == jnp.arange(p)[None, :]
        if positive:
            masked_scores = jnp.where(diag_mask, -jnp.inf, raw_scores)
            scores, idx = jax.lax.top_k(masked_scores, k)
        else:
            masked_scores = jnp.where(diag_mask, jnp.inf, raw_scores)
            neg_scores, idx = jax.lax.top_k(-masked_scores, k)
            scores = -neg_scores
        idx_chunks.append(idx)
        score_chunks.append(scores)

    return jnp.concatenate(idx_chunks, axis=0), jnp.concatenate(score_chunks, axis=0)


def _topk_recall_and_score_loss(
    exact_idx: jnp.ndarray,
    exact_scores: jnp.ndarray,
    cache_idx: jnp.ndarray,
    cache_scores: jnp.ndarray,
    *,
    positive: bool,
) -> Dict[str, float]:
    """Summarize set recall and raw-score loss for one top-k direction."""
    k = exact_idx.shape[1]
    found = jnp.any(exact_idx[:, :, None] == cache_idx[:, None, :], axis=2)
    recall_rows = jnp.mean(found, axis=1)

    exact_sum = jnp.sum(exact_scores, axis=1)
    finite_cache_scores = jnp.where(jnp.isfinite(cache_scores), cache_scores, 0.0)
    cache_sum = jnp.sum(finite_cache_scores, axis=1)
    if positive:
        loss_rows = jnp.maximum(exact_sum - cache_sum, 0.0)
    else:
        loss_rows = jnp.maximum(cache_sum - exact_sum, 0.0)
    relative_loss_rows = loss_rows / jnp.maximum(jnp.abs(exact_sum), 1e-12)

    return {
        "recall_mean": float(jnp.mean(recall_rows)),
        "recall_min": float(jnp.min(recall_rows)),
        "score_loss_mean": float(jnp.mean(relative_loss_rows)),
        "score_loss_max": float(jnp.max(relative_loss_rows)),
        "score_loss_abs_mean": float(jnp.mean(loss_rows / max(k, 1))),
    }


def _topk_selected_stats(cache_idx: jnp.ndarray, cache_scores: jnp.ndarray) -> Dict[str, float]:
    """Report invalid padding and duplicate selected neighbors."""
    valid = (cache_idx >= 0) & jnp.isfinite(cache_scores)
    valid_count = jnp.sum(valid, axis=1)
    k = int(cache_idx.shape[1])
    upper = jnp.triu(jnp.ones((k, k), dtype=bool), k=1)
    duplicate_pairs = (
        (cache_idx[:, :, None] == cache_idx[:, None, :])
        & valid[:, :, None]
        & valid[:, None, :]
        & upper[None, :, :]
    )
    duplicate_count = jnp.sum(duplicate_pairs, axis=(1, 2))

    return {
        "selected_valid_count_mean": float(jnp.mean(valid_count)),
        "selected_valid_count_min": float(jnp.min(valid_count)),
        "selected_duplicate_pair_count_mean": float(jnp.mean(duplicate_count)),
        "selected_duplicate_pair_count_max": float(jnp.max(duplicate_count)),
    }


def evaluate_topk_cache_quality(
    Z: jnp.ndarray,
    cache: TopKAttentionCache,
    sample_rows: int = 64,
    batch_size: int = 32,
    include_negative: bool = True,
) -> Dict[str, float]:
    """Compare cached top-k candidates against exact row-to-all top-k.

    The diagnostic is intended for approximate cache modes. It samples rows
    deterministically, recomputes exact top-k for those rows, and reports recall
    plus relative raw-score loss.
    """
    p = Z.shape[0]
    n_rows = int(min(max(1, sample_rows), p))
    row_idx = jnp.floor(jnp.linspace(0, p - 1, n_rows)).astype(jnp.int32)
    k = cache.top_k
    diag_batch_size = int(min(max(1, batch_size), n_rows))

    exact_pos_idx, exact_pos_scores = _exact_topk_for_rows(
        Z,
        row_idx,
        k,
        diag_batch_size,
        positive=True,
    )
    cache_pos_idx = cache.indices_pos[row_idx]
    cache_pos_scores = cache.scores_pos[row_idx]
    pos = _topk_recall_and_score_loss(
        exact_pos_idx,
        exact_pos_scores,
        cache_pos_idx,
        cache_pos_scores,
        positive=True,
    )
    pos_selected = _topk_selected_stats(cache_pos_idx, cache_pos_scores)

    out = {
        "sample_rows": int(n_rows),
        "recall_pos_mean": pos["recall_mean"],
        "recall_pos_min": pos["recall_min"],
        "score_loss_pos_mean": pos["score_loss_mean"],
        "score_loss_pos_max": pos["score_loss_max"],
        "score_loss_pos_abs_mean": pos["score_loss_abs_mean"],
        "selected_valid_count_pos_mean": pos_selected["selected_valid_count_mean"],
        "selected_valid_count_pos_min": pos_selected["selected_valid_count_min"],
        "selected_duplicate_pair_count_pos_mean": pos_selected["selected_duplicate_pair_count_mean"],
        "selected_duplicate_pair_count_pos_max": pos_selected["selected_duplicate_pair_count_max"],
    }

    if include_negative and cache.indices_neg is not None and cache.scores_neg is not None:
        exact_neg_idx, exact_neg_scores = _exact_topk_for_rows(
            Z,
            row_idx,
            k,
            diag_batch_size,
            positive=False,
        )
        cache_neg_idx = cache.indices_neg[row_idx]
        cache_neg_scores = cache.scores_neg[row_idx]
        neg = _topk_recall_and_score_loss(
            exact_neg_idx,
            exact_neg_scores,
            cache_neg_idx,
            cache_neg_scores,
            positive=False,
        )
        neg_selected = _topk_selected_stats(cache_neg_idx, cache_neg_scores)
        out.update({
            "recall_neg_mean": neg["recall_mean"],
            "recall_neg_min": neg["recall_min"],
            "score_loss_neg_mean": neg["score_loss_mean"],
            "score_loss_neg_max": neg["score_loss_max"],
            "score_loss_neg_abs_mean": neg["score_loss_abs_mean"],
            "selected_valid_count_neg_mean": neg_selected["selected_valid_count_mean"],
            "selected_valid_count_neg_min": neg_selected["selected_valid_count_min"],
            "selected_duplicate_pair_count_neg_mean": neg_selected["selected_duplicate_pair_count_mean"],
            "selected_duplicate_pair_count_neg_max": neg_selected["selected_duplicate_pair_count_max"],
        })

    return out


def topk_attention_cached_indices_weights(
    cache: TopKAttentionCache,
    eta: float,
    scale: float,
    start: int,
    stop: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return cached top-k indices and eta-dependent normalized weights."""
    factor = float(eta) * float(scale)
    if factor < 0.0 and cache.indices_neg is not None and cache.scores_neg is not None:
        idx = cache.indices_neg[start:stop]
        raw_scores = cache.scores_neg[start:stop]
    else:
        idx = cache.indices_pos[start:stop]
        raw_scores = cache.scores_pos[start:stop]
    valid = (idx >= 0) & jnp.isfinite(raw_scores)
    weighted_scores = jnp.where(valid, factor * raw_scores, -jnp.inf)
    has_valid = jnp.any(valid, axis=1, keepdims=True)
    weighted_scores = jnp.where(has_valid, weighted_scores, 0.0)
    weights = jax.nn.softmax(weighted_scores, axis=1)
    weights = jnp.where(valid, weights, 0.0)
    row_sums = jnp.sum(weights, axis=1, keepdims=True)
    weights = weights / jnp.where(row_sums > 0.0, row_sums, 1.0)

    safe_idx = jnp.where(valid, idx, 0)
    fallback_idx = ((jnp.arange(start, stop) + 1) % int(cache.indices_pos.shape[0]))[:, None]
    fallback_idx = jnp.broadcast_to(fallback_idx, safe_idx.shape)
    fallback_weights = jnp.where(jnp.arange(safe_idx.shape[1])[None, :] == 0, 1.0, 0.0)
    safe_idx = jnp.where(has_valid, safe_idx, fallback_idx)
    weights = jnp.where(has_valid, weights, fallback_weights)
    return safe_idx, weights


def topk_attention_indices_weights(
    Z: jnp.ndarray,
    eta: float,
    scale: float,
    start: int,
    stop: int,
    top_k: int,
    attention_cache: Optional[TopKAttentionCache] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return top-k attention column indices and normalized weights for rows."""
    p = Z.shape[0]
    k = _top_k_size(p, top_k)
    if attention_cache is not None:
        if attention_cache.top_k != k:
            raise ValueError("attention_cache top_k does not match attention_top_k.")
        return topk_attention_cached_indices_weights(attention_cache, eta, scale, start, stop)

    Zb = Z[start:stop]
    scores = Zb @ Z.T
    scores = eta * scores * scale

    row_ids = jnp.arange(start, stop)
    col_ids = jnp.arange(p)
    diag_mask = row_ids[:, None] == col_ids[None, :]
    scores = jnp.where(diag_mask, -1e6, scores)

    top_scores, top_idx = jax.lax.top_k(scores, k)
    weights = jax.nn.softmax(top_scores, axis=1)
    return top_idx, weights


def build_attention(
    Z: jnp.ndarray,
    eta: float,
    scale: float,
    attention_mode: str = "dense",
    top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
) -> jnp.ndarray:
    """Construct the row-stochastic attention matrix from feature vectors.

    ``Z`` has shape ``(p, q)``. Off-diagonal scores are proportional to
    ``eta * scale * Z_i' Z_j``; diagonal entries are masked before
    applying a row-wise softmax.
    """
    attention_mode = _check_attention_mode(attention_mode)
    p = Z.shape[0]
    if attention_mode == "topk":
        idx, weights = topk_attention_indices_weights(
            Z, eta, scale, 0, p, _top_k_size(p, top_k),
            attention_cache=attention_cache,
        )
        rows = jnp.arange(p)[:, None]
        W = jnp.zeros((p, p), dtype=Z.dtype)
        return W.at[rows, idx].add(weights)

    scores = Z @ Z.T
    scores = eta * scores
    mask = jnp.eye(p) * -1e6
    scores = scores * scale + mask
    return jax.nn.softmax(scores, axis=1)


def attention_rows(
    Z: jnp.ndarray,
    eta: float,
    scale: float,
    start: int,
    stop: int,
    attention_mode: str = "dense",
    top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
) -> jnp.ndarray:
    """Return attention rows ``W[start:stop, :]`` without forming all of ``W``."""
    attention_mode = _check_attention_mode(attention_mode)
    p = Z.shape[0]
    if attention_mode == "topk":
        idx, weights = topk_attention_indices_weights(
            Z, eta, scale, start, stop, _top_k_size(p, top_k),
            attention_cache=attention_cache,
        )
        rows = jnp.arange(stop - start)[:, None]
        Wb = jnp.zeros((stop - start, p), dtype=Z.dtype)
        return Wb.at[rows, idx].add(weights)

    Zb = Z[start:stop]
    scores = Zb @ Z.T
    scores = eta * scores * scale

    row_ids = jnp.arange(start, stop)
    col_ids = jnp.arange(p)
    diag_mask = row_ids[:, None] == col_ids[None, :]
    scores = jnp.where(diag_mask, -1e6, scores)
    return jax.nn.softmax(scores, axis=1)


def attention_matvec(
    Z: jnp.ndarray,
    eta: float,
    v: jnp.ndarray,
    params_scale: Dict[str, float],
    batch_size: int = 512,
    dense_threshold: int = 2500,
    attention_mode: str = "dense",
    attention_top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
) -> jnp.ndarray:
    """Compute ``W(eta) @ v`` with optional row batching for large ``p``."""
    attention_mode = _check_attention_mode(attention_mode)
    p = Z.shape[0]
    if attention_mode == "dense" and p <= dense_threshold:
        W = build_attention(Z, eta, params_scale["eta_a"])
        return W @ v

    out = jnp.zeros((p,), dtype=Z.dtype)
    for start in range(0, p, batch_size):
        stop = min(p, start + batch_size)
        if attention_mode == "topk":
            idx, weights = topk_attention_indices_weights(
                Z, eta, params_scale["eta_a"], start, stop, _top_k_size(p, attention_top_k),
                attention_cache=attention_cache,
            )
            out = out.at[start:stop].set(jnp.sum(weights * v[idx], axis=1))
        else:
            Wb = attention_rows(Z, eta, params_scale["eta_a"], start, stop)
            out = out.at[start:stop].set(Wb @ v)
    return out


def attention_weighted_lags(
    Y_lag: jnp.ndarray,
    Z: jnp.ndarray,
    eta: float,
    params_scale: Dict[str, float],
    batch_size: int = 512,
    dense_threshold: int = 2500,
    attention_mode: str = "dense",
    attention_top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
) -> jnp.ndarray:
    """Compute ``X2[:, k] = sum_j W[k, j] * Y_lag[:, j]`` for all ``k``."""
    attention_mode = _check_attention_mode(attention_mode)
    T_eff, p = Y_lag.shape
    if attention_mode == "dense" and p <= dense_threshold:
        W = build_attention(Z, eta, params_scale["eta_a"])
        return Y_lag @ W.T

    X2 = jnp.zeros((T_eff, p), dtype=Y_lag.dtype)
    for start in range(0, p, batch_size):
        stop = min(p, start + batch_size)
        if attention_mode == "topk":
            idx, weights = topk_attention_indices_weights(
                Z, eta, params_scale["eta_a"], start, stop, _top_k_size(p, attention_top_k),
                attention_cache=attention_cache,
            )
            X2b = jnp.sum(Y_lag[:, idx] * weights[None, :, :], axis=2)
        else:
            Wb = attention_rows(Z, eta, params_scale["eta_a"], start, stop)
            X2b = Y_lag @ Wb.T
        X2 = X2.at[:, start:stop].set(X2b)
    return X2
