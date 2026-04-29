"""Attention matrix utilities for the VARA model."""

from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp


def build_attention(Z: jnp.ndarray, eta: float, scale: float) -> jnp.ndarray:
    """Construct the row-stochastic attention matrix from feature vectors.

    ``Z`` has shape ``(p, q)``. Off-diagonal scores are proportional to
    ``eta * scale * Z_i' Z_j / sqrt(q)``; diagonal entries are masked before
    applying a row-wise softmax.
    """
    p, q = Z.shape
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
) -> jnp.ndarray:
    """Return attention rows ``W[start:stop, :]`` without forming all of ``W``."""
    p, q = Z.shape
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
) -> jnp.ndarray:
    """Compute ``W(eta) @ v`` with optional row batching for large ``p``."""
    p = Z.shape[0]
    if p <= dense_threshold:
        W = build_attention(Z, eta, params_scale["eta_a"])
        return W @ v

    out = jnp.zeros((p,), dtype=Z.dtype)
    for start in range(0, p, batch_size):
        stop = min(p, start + batch_size)
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
) -> jnp.ndarray:
    """Compute ``X2[:, k] = sum_j W[k, j] * Y_lag[:, j]`` for all ``k``."""
    T_eff, p = Y_lag.shape
    if p <= dense_threshold:
        W = build_attention(Z, eta, params_scale["eta_a"])
        return Y_lag @ W.T

    X2 = jnp.zeros((T_eff, p), dtype=Y_lag.dtype)
    for start in range(0, p, batch_size):
        stop = min(p, start + batch_size)
        Wb = attention_rows(Z, eta, params_scale["eta_a"], start, stop)
        X2b = Y_lag @ Wb.T
        X2 = X2.at[:, start:stop].set(X2b)
    return X2
