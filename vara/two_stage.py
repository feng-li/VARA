"""Two-stage estimation helpers for the VARA simulation code."""

from __future__ import annotations

from typing import Dict

import jax.numpy as jnp

from .attention import attention_weighted_lags


def fit_all_components_two_stage(
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    eta: float,
    params_scale: Dict[str, float],
    ridge: float = 1e-10,
    batch_size: int = 512,
    dense_threshold: int = 2500,
) -> Dict[str, jnp.ndarray]:
    """Fit componentwise OLS regressions at a fixed eta value.

    For each coordinate ``k``, the regression uses two regressors:
    ``Y_lag[:, k]`` and the attention-weighted lag ``X2[:, k]``. Returned
    names follow paper notation: ``alpha_hat`` estimates code ``beta`` and
    ``beta_hat`` estimates code ``gamma_a``.
    """
    T, _ = Y.shape
    T_eff = T - 1
    Y_lag = Y[:-1]
    Y_now = Y[1:]

    X1 = Y_lag
    X2 = attention_weighted_lags(
        Y_lag, Z, eta, params_scale,
        batch_size=batch_size,
        dense_threshold=dense_threshold,
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
    )
    sigma2_hat = jnp.maximum(fit["sigma2_hat"], eps)
    return -0.5 * T_eff * jnp.sum(jnp.log(sigma2_hat))
