"""Two-stage estimation helpers for the VARA simulation code."""

from __future__ import annotations

from typing import Dict, Optional

import jax.numpy as jnp
import jax
import numpy as np

from .attention import TopKAttentionCache, attention_weighted_lags


def fit_all_components_two_stage(
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    eta: float,
    params_scale: Dict[str, float],
    ridge: float = 1e-10,
    batch_size: int = 512,
    dense_threshold: int = 2500,
    attention_mode: str = "dense",
    attention_top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
    component_indices=None,
) -> Dict[str, jnp.ndarray]:
    """Fit componentwise OLS regressions at a fixed eta value.

    For each coordinate ``k``, the regression uses two regressors:
    ``Y_lag[:, k]`` and the attention-weighted lag ``X2[:, k]``. Returned
    names follow paper notation: ``alpha_hat`` estimates code ``beta`` and
    ``beta_hat`` estimates code ``gamma_a``.

    ``component_indices`` selects response equations, not attention neighbors.
    The selected dense rows retain all off-diagonal neighbors and cost
    O(T m p) to apply. None fits all p equations as before.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    T, _ = Y.shape
    T_eff = T - 1
    Y_lag = Y[:-1]
    Y_now = Y[1:]

    X1 = Y_lag
    if component_indices is not None:
        indices = np.asarray(component_indices)
        if (indices.ndim != 1 or not len(indices) or
                not np.issubdtype(indices.dtype, np.integer) or
                np.any(indices < 0) or np.any(indices >= Y.shape[1]) or
                len(np.unique(indices)) != len(indices)):
            raise ValueError("component_indices must be distinct valid component indices")
        if attention_mode != "dense":
            raise ValueError("Component subsampling requires dense attention")
        blocks = []
        for start in range(0, len(indices), batch_size):
            ids = jnp.asarray(indices[start:start + batch_size])
            scores = eta * params_scale["eta_a"] * (Z[ids] @ Z.T)
            scores = scores.at[jnp.arange(len(ids)), ids].set(-jnp.inf)
            blocks.append(Y_lag @ jax.nn.softmax(scores, axis=1).T)
        X2 = jnp.concatenate(blocks, axis=1)
        X1 = Y_lag[:, indices]
        Y_now = Y_now[:, indices]
    else:
        X2 = attention_weighted_lags(
            Y_lag, Z, eta, params_scale,
            batch_size=batch_size,
            dense_threshold=dense_threshold,
            attention_mode=attention_mode,
            attention_top_k=attention_top_k,
            attention_cache=attention_cache,
        )

    xx11 = jnp.sum(X1 * X1, axis=0)
    xx22 = jnp.sum(X2 * X2, axis=0)
    xx12 = jnp.sum(X1 * X2, axis=0)
    xy1 = jnp.sum(X1 * Y_now, axis=0)
    xy2 = jnp.sum(X2 * Y_now, axis=0)

    det = xx11 * xx22 - xx12 ** 2
    det_reg = det + ridge

    alpha_hat = (xy1 * xx22 - xy2 * xx12) / det_reg
    beta_hat = (xx11 * xy2 - xx12 * xy1) / det_reg

    resid = Y_now - X1 * alpha_hat[None, :] - X2 * beta_hat[None, :]
    sigma2_hat = jnp.mean(resid ** 2, axis=0)

    s11 = xx11 / T_eff
    s22 = xx22 / T_eff
    s12 = xx12 / T_eff
    sdet = s11 * s22 - s12 ** 2 + ridge

    invS11 = s22 / sdet
    invS22 = s11 / sdet

    se_alpha = jnp.sqrt(jnp.maximum(sigma2_hat * invS11 / T_eff, 0.0))
    se_beta = jnp.sqrt(jnp.maximum(sigma2_hat * invS22 / T_eff, 0.0))

    return {
        "alpha_hat": alpha_hat,
        "beta_hat": beta_hat,
        "sigma2_hat": sigma2_hat,
        "se_alpha": se_alpha,
        "se_beta": se_beta,
        "resid": resid,
        "s11": s11,
        "s22": s22,
        "s12": s12,
    }


def concentrated_eta_loglik(
    eta: float,
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    params_scale: Dict[str, float],
    ridge: float = 1e-10,
    eps: float = 1e-12,
    batch_size: int = 512,
    dense_threshold: int = 2500,
    attention_mode: str = "dense",
    attention_top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
    component_indices=None,
) -> jnp.ndarray:
    """Evaluate the concentrated Gaussian log-likelihood for ``eta_a``.

    Given eta, the coordinate-specific slope and variance parameters are
    profiled out by ``fit_all_components_two_stage``. Constants that do not
    affect eta maximization are omitted.
    """
    T_eff = Y.shape[0] - 1
    fit = fit_all_components_two_stage(
        Y, Z, eta, params_scale,
        ridge=ridge,
        batch_size=batch_size,
        dense_threshold=dense_threshold,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        attention_cache=attention_cache,
        component_indices=component_indices,
    )
    sigma2_hat = jnp.maximum(fit["sigma2_hat"], eps)
    return -0.5 * T_eff * jnp.sum(jnp.log(sigma2_hat))
