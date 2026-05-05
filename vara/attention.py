"""Attention matrix utilities for the VARA model."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp


def _top_k_size(p: int, top_k: Optional[int]) -> int:
    if top_k is None:
        raise ValueError("top_k must be provided when attention_mode='topk'.")
    return int(min(max(1, top_k), max(p - 1, 1)))


def _check_attention_mode(attention_mode: str) -> str:
    if attention_mode not in {"dense", "topk"}:
        raise ValueError("attention_mode must be 'dense' or 'topk'.")
    return attention_mode


def topk_attention_indices_weights(
    Z: jnp.ndarray,
    eta: float,
    scale: float,
    start: int,
    stop: int,
    top_k: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Return top-k attention column indices and normalized weights for rows."""
    p, q = Z.shape
    k = _top_k_size(p, top_k)
    Zb = Z[start:stop]
    scores = (Zb @ Z.T) / jnp.sqrt(q)
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
) -> jnp.ndarray:
    """Construct the row-stochastic attention matrix from feature vectors.

    ``Z`` has shape ``(p, q)``. Off-diagonal scores are proportional to
    ``eta * scale * Z_i' Z_j / sqrt(q)``; diagonal entries are masked before
    applying a row-wise softmax.
    """
    attention_mode = _check_attention_mode(attention_mode)
    p, q = Z.shape
    if attention_mode == "topk":
        idx, weights = topk_attention_indices_weights(Z, eta, scale, 0, p, _top_k_size(p, top_k))
        rows = jnp.arange(p)[:, None]
        W = jnp.zeros((p, p), dtype=Z.dtype)
        return W.at[rows, idx].set(weights)

    scores = (Z @ Z.T) / jnp.sqrt(q)
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
) -> jnp.ndarray:
    """Return attention rows ``W[start:stop, :]`` without forming all of ``W``."""
    attention_mode = _check_attention_mode(attention_mode)
    p, q = Z.shape
    if attention_mode == "topk":
        idx, weights = topk_attention_indices_weights(
            Z, eta, scale, start, stop, _top_k_size(p, top_k)
        )
        rows = jnp.arange(stop - start)[:, None]
        Wb = jnp.zeros((stop - start, p), dtype=Z.dtype)
        return Wb.at[rows, idx].set(weights)

    Zb = Z[start:stop]
    scores = (Zb @ Z.T) / jnp.sqrt(q)
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
                Z, eta, params_scale["eta_a"], start, stop, _top_k_size(p, attention_top_k)
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
                Z, eta, params_scale["eta_a"], start, stop, _top_k_size(p, attention_top_k)
            )
            X2b = jnp.sum(Y_lag[:, idx] * weights[None, :, :], axis=2)
        else:
            Wb = attention_rows(Z, eta, params_scale["eta_a"], start, stop)
            X2b = Y_lag @ Wb.T
        X2 = X2.at[:, start:stop].set(X2b)
    return X2
