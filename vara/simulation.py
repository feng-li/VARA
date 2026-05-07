"""Simulation helpers for the VARA numerical experiments.

This module contains reusable pieces that are shared by simulation scripts:

* parameter draws for the paper-style DGP,
* memory-aware attention operations for large ``p``,
* VARA data simulation,
* full QMLE and two-stage estimation wrappers, and
* compact replication summaries for RMSE/ECP tables.

Notation follows the repository's code convention. Relative to the paper:
paper gamma = ``eta_a``, paper alpha = ``beta``, paper beta = ``gamma_a``,
and paper sigma squared = ``sigma2``.

All parameter dictionaries use natural-scale ``sigma2`` unless a function
explicitly says it builds optimizer-scale theta.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import pandas as pd

from .attention import TopKAttentionCache, attention_matvec
from .cov import compute_sandwich_variance
from .dgp import ensure_stationarity
from .fit import estimate
from .params import make_block_order
from .two_stage import concentrated_eta_loglik, fit_all_components_two_stage


PARAMS_SCALE = {"eta_a": 1.0}
Z_CRIT_975 = 1.959963984540054


def p_for_setting(T: int, setting: str) -> int:
    """Return the response dimension used by a paper simulation setting.

    ``low`` uses ``floor(sqrt(T))``. ``high`` uses ``floor(T ** 1.5)``.
    """
    if setting == "low":
        return int(math.floor(math.sqrt(T)))
    if setting == "high":
        return int(math.floor(T ** 1.5))
    raise ValueError("setting must be 'low' or 'high'")


def kappa4_for_dist(dist: str) -> float:
    """Return the fourth moment of the standardized innovation distribution."""
    if dist == "normal":
        return 3.0
    if dist == "laplace":
        return 6.0
    if dist == "t5":
        return 9.0
    raise ValueError(f"Unknown dist={dist}")


def flatten_simulation_params(params: Dict[str, Any], p: int) -> jnp.ndarray:
    """Flatten natural-scale parameters in optimizer block order.

    Parameters
    ----------
    params
        Dictionary with ``eta_a`` scalar and length-``p`` arrays
        ``gamma_a``, ``beta``, and ``sigma2``.
    p
        Response dimension. Kept in the signature for symmetry with callers
        that construct block orders, but the array lengths are read from
        ``params``.
    """
    return jnp.concatenate([
        jnp.atleast_1d(params["eta_a"]),
        jnp.asarray(params["gamma_a"]),
        jnp.asarray(params["beta"]),
        jnp.asarray(params["sigma2"]),
    ])


def reparam_theta_from_params(params: Dict[str, Any], p: int) -> jnp.ndarray:
    """Build optimizer-scale theta from natural-scale parameters.

    The optimizer estimates unconstrained variance coordinates, so this uses
    ``softplus^{-1}(sigma2)`` for the last block while leaving all other
    parameters on their natural scale.
    """
    sigma2_tilde = jnp.log(jnp.expm1(params["sigma2"]))
    return jnp.concatenate([
        jnp.atleast_1d(params["eta_a"]),
        jnp.asarray(params["gamma_a"]),
        jnp.asarray(params["beta"]),
        sigma2_tilde,
    ])


def draw_simulation_params(key: jax.Array, p: int) -> Dict[str, Any]:
    """Draw one paper-style DGP parameter vector for a simulation setting.

    The draw uses ``eta_a = 0.5``, independent uniform draws for ``gamma_a``
    and ``beta`` on ``[-0.2, 0.2]``, and ``sigma2`` on ``[0.25, 1.0]``.
    ``beta`` and ``gamma_a`` are rescaled only if needed for stationarity.
    """
    k_beta_paper, k_alpha_paper, k_sig = jax.random.split(key, 3)
    params = {
        "eta_a": jnp.asarray(0.5),
        "gamma_a": jax.random.uniform(k_beta_paper, (p,), minval=-0.2, maxval=0.2),
        "beta": jax.random.uniform(k_alpha_paper, (p,), minval=-0.2, maxval=0.2),
        "sigma2": jax.random.uniform(k_sig, (p,), minval=0.25, maxval=1.0),
    }

    beta, gamma_a, _ = ensure_stationarity(params["beta"], params["gamma_a"], margin=0.98)
    params["beta"] = beta
    params["gamma_a"] = gamma_a
    return params


def draw_standard_innovations(key: jax.Array, shape: Tuple[int, int], dist: str) -> jnp.ndarray:
    """Draw mean-zero, variance-one innovations.

    Supported values are ``normal``, ``laplace``, and ``t5``. The Student-t
    draw is rescaled to unit variance.
    """
    if dist == "normal":
        return jax.random.normal(key, shape)
    if dist == "laplace":
        u = jax.random.uniform(key, shape, minval=-0.5 + 1e-12, maxval=0.5 - 1e-12)
        b = 1.0 / jnp.sqrt(2.0)
        return -b * jnp.sign(u) * jnp.log1p(-2.0 * jnp.abs(u))
    if dist == "t5":
        k1, k2 = jax.random.split(key)
        z = jax.random.normal(k1, shape)
        chi = jax.random.chisquare(k2, df=5, shape=shape)
        return (z / jnp.sqrt(chi / 5.0)) * jnp.sqrt(3.0 / 5.0)
    raise ValueError("dist must be 'normal', 'laplace', or 't5'.")


def simulate_vara(
    key: jax.Array,
    params: Dict[str, Any],
    Z: jnp.ndarray,
    T: int,
    p: int,
    q: int,
    dist: str,
    params_scale: Dict[str, float],
    batch_size: int = 512,
    dense_threshold: int = 2500,
    precomputed_attention: Optional[jnp.ndarray] = None,
    attention_mode: str = "dense",
    attention_top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
) -> jnp.ndarray:
    """Simulate observations from the VARA DGP.

    The recursion is
    ``Y_t = beta * Y_{t-1} + gamma_a * (W(eta_a) @ Y_{t-1}) + eps_t``.
    Innovations are drawn with unit variance and then scaled coordinatewise by
    ``sqrt(sigma2)``. The initial state is zero.

    If ``precomputed_attention`` is supplied, it is reused for every time step.
    This is useful for large-``p`` simulations when memory is available and the
    DGP attention matrix is fixed across replications.
    """
    params = dict(params)
    beta, gamma_a, _ = ensure_stationarity(params["beta"], params["gamma_a"], margin=0.98)
    eta = float(params["eta_a"])
    sigma2 = params["sigma2"]
    W = precomputed_attention
    if W is not None and W.shape != (p, p):
        raise ValueError(f"precomputed_attention must have shape ({p}, {p})")

    eps_raw = draw_standard_innovations(key, (T, p), dist=dist)
    eps = eps_raw * jnp.sqrt(sigma2)[None, :]

    ys = []
    y_prev = jnp.zeros((p,), dtype=Z.dtype)
    ys.append(y_prev)

    for t in range(1, T):
        if W is None:
            Wy = attention_matvec(
                Z, eta, y_prev, params_scale,
                batch_size=batch_size,
                dense_threshold=dense_threshold,
                attention_mode=attention_mode,
                attention_top_k=attention_top_k,
                attention_cache=attention_cache,
            )
        else:
            Wy = W @ y_prev
        y_t = beta * y_prev + gamma_a * Wy + eps[t]
        ys.append(y_t)
        y_prev = y_t

    return jnp.stack(ys, axis=0)


def golden_section_maximize(fn, lower: float, upper: float, tol: float = 1e-4, max_iter: int = 100):
    """Maximize a scalar function on a bounded interval by golden-section search."""
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    a, b = float(lower), float(upper)
    c = b - gr * (b - a)
    d = a + gr * (b - a)
    fc = float(fn(c))
    fd = float(fn(d))

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break
        if fc < fd:
            a, c, fc = c, d, fd
            d = a + gr * (b - a)
            fd = float(fn(d))
        else:
            b, d, fd = d, c, fc
            c = b - gr * (b - a)
            fc = float(fn(c))

    x = 0.5 * (a + b)
    return jnp.asarray(x), float(fn(x))


def estimate_eta_concentrated(
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    params_scale: Dict[str, float],
    eta_bounds: Tuple[float, float] = (-2.0, 2.0),
    grid_size: int = 41,
    ridge: float = 1e-10,
    batch_size: int = 512,
    dense_threshold: int = 2500,
    attention_mode: str = "dense",
    attention_top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
):
    """Estimate ``eta_a`` by a grid search followed by local refinement.

    The returned ``info`` dictionary records the best grid point, objective
    value, and whether the grid optimum occurred at the search boundary.
    """
    lower, upper = eta_bounds
    grid = jnp.linspace(lower, upper, grid_size)

    vals = []
    for eta in grid:
        vals.append(concentrated_eta_loglik(
            float(eta), Y, Z, params_scale,
            ridge=ridge,
            batch_size=batch_size,
            dense_threshold=dense_threshold,
            attention_mode=attention_mode,
            attention_top_k=attention_top_k,
            attention_cache=attention_cache,
        ))
    vals = jnp.array(vals)
    best_idx = int(jnp.argmax(vals))

    left_idx = max(0, best_idx - 1)
    right_idx = min(grid_size - 1, best_idx + 1)
    local_lower = float(grid[left_idx])
    local_upper = float(grid[right_idx])
    if best_idx == 0:
        local_lower, local_upper = float(grid[0]), float(grid[1])
    elif best_idx == grid_size - 1:
        local_lower, local_upper = float(grid[-2]), float(grid[-1])

    fn = lambda x: concentrated_eta_loglik(
        x, Y, Z, params_scale,
        ridge=ridge,
        batch_size=batch_size,
        dense_threshold=dense_threshold,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        attention_cache=attention_cache,
    )
    eta_hat, obj = golden_section_maximize(fn, local_lower, local_upper)
    info = {
        "grid_best_eta": float(grid[best_idx]),
        "grid_best_obj": float(vals[best_idx]),
        "objective": obj,
        "on_boundary": bool(best_idx == 0 or best_idx == grid_size - 1),
    }
    return eta_hat, info


def finite_diff_second_derivative_eta(
    eta_center: float,
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    params_scale: Dict[str, float],
    h: float = 0.02,
    eta_bounds: Tuple[float, float] = (-2.0, 2.0),
    ridge: float = 1e-10,
    batch_size: int = 512,
    dense_threshold: int = 2500,
    attention_mode: str = "dense",
    attention_top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
) -> jnp.ndarray:
    """Approximate the second derivative of the concentrated eta objective."""
    lower, upper = eta_bounds
    eta_center = float(eta_center)
    if eta_center - h < lower or eta_center + h > upper:
        return jnp.nan

    f0 = concentrated_eta_loglik(
        eta_center, Y, Z, params_scale, ridge,
        batch_size=batch_size,
        dense_threshold=dense_threshold,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        attention_cache=attention_cache,
    )
    fm = concentrated_eta_loglik(
        eta_center - h, Y, Z, params_scale, ridge,
        batch_size=batch_size,
        dense_threshold=dense_threshold,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        attention_cache=attention_cache,
    )
    fp = concentrated_eta_loglik(
        eta_center + h, Y, Z, params_scale, ridge,
        batch_size=batch_size,
        dense_threshold=dense_threshold,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        attention_cache=attention_cache,
    )
    return (fp - 2.0 * f0 + fm) / (h ** 2)


def estimate_eta_curvature(
    eta_center: float,
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    p: int,
    params_scale: Dict[str, float],
    h: float = 0.02,
    eta_bounds: Tuple[float, float] = (-2.0, 2.0),
    ridge: float = 1e-10,
    batch_size: int = 512,
    dense_threshold: int = 2500,
    attention_mode: str = "dense",
    attention_top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
):
    """Estimate the normalized curvature used for eta standard errors.

    Returns ``(-l''(eta) / (T_eff * p), l''(eta))`` where ``l`` is the
    concentrated eta log-likelihood.
    """
    T_eff = Y.shape[0] - 1
    l2 = finite_diff_second_derivative_eta(
        eta_center, Y, Z, params_scale, h, eta_bounds, ridge, batch_size,
        dense_threshold, attention_mode, attention_top_k, attention_cache
    )
    return -l2 / (T_eff * p), l2


def pack_result_row(
    *,
    setting: str,
    method: str,
    rep: int,
    T: int,
    p: int,
    q: int,
    dist: str,
    seed: int,
    status: int,
    elapsed_sec: float,
    params_true: Dict[str, Any],
    eta_hat: float = math.nan,
    se_eta: float = math.nan,
    alpha_hat: Optional[jnp.ndarray] = None,
    beta_hat: Optional[jnp.ndarray] = None,
    sigma2_hat: Optional[jnp.ndarray] = None,
    se_alpha: Optional[jnp.ndarray] = None,
    se_beta: Optional[jnp.ndarray] = None,
    se_sigma2: Optional[jnp.ndarray] = None,
    alpha: float = 0.05,
    report_gamma_ecp: bool = True,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create one compact replication-level result row.

    The row stores scalar eta error/coverage and coordinate-averaged MSE, ECP,
    and confidence interval length for paper alpha, beta, and sigma squared.
    Use ``report_gamma_ecp=False`` for high-dimensional runs where eta
    coverage is not reported.
    """
    zcrit = float(jax.scipy.stats.norm.ppf(1.0 - alpha / 2.0))

    eta_true = float(params_true["eta_a"])
    alpha_true = params_true["beta"]
    beta_true = params_true["gamma_a"]
    sigma2_true = params_true["sigma2"]

    row: Dict[str, Any] = {
        "setting": setting,
        "method": method,
        "rep": rep,
        "T": T,
        "p": p,
        "q": q,
        "dist": dist,
        "seed": seed,
        "status": status,
        "elapsed_sec": elapsed_sec,
        "eta_true": eta_true,
        "eta_hat": float(eta_hat) if eta_hat is not None else math.nan,
        "eta_error": math.nan,
        "eta_sqerr": math.nan,
        "se_eta": float(se_eta) if se_eta is not None else math.nan,
        "eta_covered": math.nan,
        "eta_ci_len": math.nan,
    }

    if math.isfinite(row["eta_hat"]):
        row["eta_error"] = row["eta_hat"] - eta_true
        row["eta_sqerr"] = row["eta_error"] ** 2

    if report_gamma_ecp and math.isfinite(row["eta_hat"]) and math.isfinite(row["se_eta"]) and row["se_eta"] > 0:
        lo = row["eta_hat"] - zcrit * row["se_eta"]
        hi = row["eta_hat"] + zcrit * row["se_eta"]
        row["eta_covered"] = float(lo <= eta_true <= hi)
        row["eta_ci_len"] = float(hi - lo)

    def vector_metrics(name: str, hat, true, se):
        if hat is None:
            row[f"{name}_mse"] = math.nan
            row[f"{name}_ecp"] = math.nan
            row[f"{name}_ci_len"] = math.nan
            return

        err = jnp.asarray(hat) - jnp.asarray(true)
        row[f"{name}_mse"] = float(jnp.mean(err ** 2))
        if se is None:
            row[f"{name}_ecp"] = math.nan
            row[f"{name}_ci_len"] = math.nan
        else:
            se_arr = jnp.asarray(se)
            lo = jnp.asarray(hat) - zcrit * se_arr
            hi = jnp.asarray(hat) + zcrit * se_arr
            covered = (jnp.asarray(true) >= lo) & (jnp.asarray(true) <= hi)
            row[f"{name}_ecp"] = float(jnp.mean(covered))
            row[f"{name}_ci_len"] = float(jnp.mean(hi - lo))

    vector_metrics("alpha", alpha_hat, alpha_true, se_alpha)
    vector_metrics("beta", beta_hat, beta_true, se_beta)
    vector_metrics("sigma2", sigma2_hat, sigma2_true, se_sigma2)

    if extra:
        row.update(extra)
    return row


def run_qmle_on_dataset(
    key: jax.Array,
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    params_true: Dict[str, Any],
    params_scale: Dict[str, float],
    alpha: float,
    kappa4: float,
    algorithm: str,
    perturb_scale: float,
) -> Tuple[Dict[str, Any], int]:
    """Run full QMLE for one simulated dataset.

    The optimizer is initialized near the true optimizer-scale theta. Returned
    estimates are converted into the paper-facing names used by summaries:
    ``alpha_hat`` for code ``beta`` and ``beta_hat`` for code ``gamma_a``.
    """
    _, p = Y.shape
    theta_true_reparam = reparam_theta_from_params(params_true, p)
    block_order = make_block_order(p)

    theta_init = theta_true_reparam + perturb_scale * jax.random.normal(
        key, theta_true_reparam.shape
    )

    result, convergence = estimate(
        Y=Y,
        Z=Z,
        key=key,
        theta_init=theta_init,
        params_scale=params_scale,
        algorithm=algorithm,
        update_idx=slice(None),
        block_order=block_order,
        kappa4=kappa4,
    )

    theta_hat = jnp.asarray(result["theta"])

    try:
        var = compute_sandwich_variance(
            Y=Y,
            Z=Z,
            theta_hat=theta_hat,
            params_scale=params_scale,
            kappa4=kappa4,
            alpha=alpha,
        )
        se = jnp.asarray(var["se"])
    except Exception:
        se = jnp.asarray(result.get("se", jnp.full_like(theta_hat, jnp.nan)))

    estimates = {
        "eta_hat": theta_hat[0],
        "se_eta": se[0],
        "alpha_hat": theta_hat[1 + p:1 + 2 * p],
        "beta_hat": theta_hat[1:1 + p],
        "sigma2_hat": theta_hat[1 + 2 * p:1 + 3 * p],
        "se_alpha": se[1 + p:1 + 2 * p],
        "se_beta": se[1:1 + p],
        "se_sigma2": se[1 + 2 * p:1 + 3 * p],
    }
    return estimates, int(convergence.get("status", 1))


def run_two_stage_on_dataset(
    Y: jnp.ndarray,
    Z: jnp.ndarray,
    params_true: Dict[str, Any],
    params_scale: Dict[str, float],
    alpha: float,
    kappa4: float,
    setting: str,
    eta_bounds: Tuple[float, float],
    grid_size: int,
    curvature_step: float,
    ridge: float,
    batch_size: int,
    dense_threshold: int,
    attention_mode: str = "dense",
    attention_top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
) -> Tuple[Dict[str, Any], int]:
    """Run the two-stage estimator on one simulated dataset.

    First estimate ``eta_a`` with the concentrated objective. Then estimate
    coordinate-specific slopes and variances by componentwise OLS at the
    estimated eta. For low-dimensional settings, eta standard errors are based
    on a finite-difference curvature estimate.
    """
    T, p = Y.shape
    T_eff = T - 1

    eta_hat, info = estimate_eta_concentrated(
        Y=Y,
        Z=Z,
        params_scale=params_scale,
        eta_bounds=eta_bounds,
        grid_size=grid_size,
        ridge=ridge,
        batch_size=batch_size,
        dense_threshold=dense_threshold,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        attention_cache=attention_cache,
    )
    eta_hat_f = float(eta_hat)

    fit = fit_all_components_two_stage(
        Y=Y,
        Z=Z,
        eta=eta_hat_f,
        params_scale=params_scale,
        ridge=ridge,
        batch_size=batch_size,
        dense_threshold=dense_threshold,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        attention_cache=attention_cache,
    )

    sigma2_hat = fit["sigma2_hat"]
    se_sigma2 = jnp.sqrt(jnp.maximum((kappa4 - 1.0) * sigma2_hat ** 2 / T_eff, 0.0))

    report_gamma_ecp = setting == "low"
    se_eta = jnp.nan
    omega_hat = jnp.nan
    if report_gamma_ecp:
        omega_hat, _ = estimate_eta_curvature(
            eta_center=eta_hat_f,
            Y=Y,
            Z=Z,
            p=p,
            params_scale=params_scale,
            h=curvature_step,
            eta_bounds=eta_bounds,
            ridge=ridge,
            batch_size=batch_size,
            dense_threshold=dense_threshold,
            attention_mode=attention_mode,
            attention_top_k=attention_top_k,
            attention_cache=attention_cache,
        )
        if jnp.isfinite(omega_hat) and float(omega_hat) > 0:
            se_eta = 1.0 / jnp.sqrt(T_eff * p * omega_hat)

    estimates = {
        "eta_hat": eta_hat,
        "se_eta": se_eta,
        "alpha_hat": fit["alpha_hat"],
        "beta_hat": fit["beta_hat"],
        "sigma2_hat": sigma2_hat,
        "se_alpha": fit["se_alpha"],
        "se_beta": fit["se_beta"],
        "se_sigma2": se_sigma2,
        "extra": {
            "omega_hat": float(omega_hat) if jnp.isfinite(omega_hat) else math.nan,
            "eta_on_boundary": bool(info.get("on_boundary", False)),
            "grid_best_eta": float(info.get("grid_best_eta", math.nan)),
        },
    }
    return estimates, 0


def full_single_replication(
    rep: int,
    key: jax.Array,
    params_true: Dict[str, Any],
    Z: jnp.ndarray,
    setting: str,
    T: int,
    p: int,
    q: int,
    dist: str,
    seed: int,
    methods: Sequence[str],
    alpha: float,
    qmle_algorithm: str,
    qmle_perturb_scale: float,
    eta_bounds: Tuple[float, float],
    grid_size: int,
    curvature_step: float,
    ridge: float,
    batch_size: int,
    dense_threshold: int,
    dgp_attention: Optional[jnp.ndarray] = None,
    attention_mode: str = "dense",
    attention_top_k: Optional[int] = None,
    attention_cache: Optional[TopKAttentionCache] = None,
) -> List[Dict[str, Any]]:
    """Simulate one dataset and run all requested estimation methods.

    Returns one compact result row per method. Exceptions are caught at the
    method level so a failed QMLE run does not prevent the two-stage estimator
    from being reported for the same simulated dataset.
    """
    sim_key, qmle_key = jax.random.split(key)
    kappa4 = kappa4_for_dist(dist)

    Y = simulate_vara(
        key=sim_key,
        params=params_true,
        Z=Z,
        T=T,
        p=p,
        q=q,
        dist=dist,
        params_scale=PARAMS_SCALE,
        batch_size=batch_size,
        dense_threshold=dense_threshold,
        precomputed_attention=dgp_attention,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        attention_cache=attention_cache,
    )

    rows = []
    for method in methods:
        start = time.time()
        try:
            if method == "qmle":
                estimates, status = run_qmle_on_dataset(
                    key=qmle_key,
                    Y=Y,
                    Z=Z,
                    params_true=params_true,
                    params_scale=PARAMS_SCALE,
                    alpha=alpha,
                    kappa4=kappa4,
                    algorithm=qmle_algorithm,
                    perturb_scale=qmle_perturb_scale,
                )
                extra = {}
                report_gamma_ecp = True
            elif method == "two_stage":
                estimates, status = run_two_stage_on_dataset(
                    Y=Y,
                    Z=Z,
                    params_true=params_true,
                    params_scale=PARAMS_SCALE,
                    alpha=alpha,
                    kappa4=kappa4,
                    setting=setting,
                    eta_bounds=eta_bounds,
                    grid_size=grid_size,
                    curvature_step=curvature_step,
                    ridge=ridge,
                    batch_size=batch_size,
                    dense_threshold=dense_threshold,
                    attention_mode=attention_mode,
                    attention_top_k=attention_top_k,
                    attention_cache=attention_cache,
                )
                extra = estimates.pop("extra", {})
                report_gamma_ecp = setting == "low"
            else:
                raise ValueError(f"Unknown method: {method}")

            elapsed = time.time() - start
            rows.append(pack_result_row(
                setting=setting,
                method=method,
                rep=rep,
                T=T,
                p=p,
                q=q,
                dist=dist,
                seed=seed,
                status=status,
                elapsed_sec=elapsed,
                params_true=params_true,
                alpha=alpha,
                report_gamma_ecp=report_gamma_ecp,
                extra=extra,
                **estimates,
            ))
        except Exception as e:
            elapsed = time.time() - start
            rows.append(pack_result_row(
                setting=setting,
                method=method,
                rep=rep,
                T=T,
                p=p,
                q=q,
                dist=dist,
                seed=seed,
                status=-1,
                elapsed_sec=elapsed,
                params_true=params_true,
                alpha=alpha,
                report_gamma_ecp=(setting == "low"),
                extra={"error": repr(e)},
            ))
    return rows


def summarize_setting(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate replication rows into setting-level RMSE/ECP summaries.

    The expected input is the row format produced by ``pack_result_row``.
    Output columns are named with paper parameter labels:
    ``gamma`` for eta, ``alpha`` for code beta, and ``beta`` for code gamma_a.
    """
    records = []
    group_cols = ["setting", "dist", "T", "p", "q", "method"]

    for keys, sub in df.groupby(group_cols, dropna=False):
        setting, dist, T, p, q, method = keys
        valid = sub[sub["status"] == 0].copy()
        rec = {
            "setting": setting,
            "dist": dist,
            "T": int(T),
            "p": int(p),
            "q": int(q),
            "method": method,
            "n_rep": int(len(sub)),
            "valid_rate": float((sub["status"] == 0).mean()) if len(sub) else math.nan,
            "cpu_sec_mean": float(valid["elapsed_sec"].mean()) if len(valid) else math.nan,
            "cpu_sec_sum": float(valid["elapsed_sec"].sum()) if len(valid) else math.nan,
        }
        if len(valid):
            rec.update({
                "RMSE_gamma": math.sqrt(float(valid["eta_sqerr"].mean())),
                "RMSE_alpha": math.sqrt(float(valid["alpha_mse"].mean())),
                "RMSE_beta": math.sqrt(float(valid["beta_mse"].mean())),
                "RMSE_sigma2": math.sqrt(float(valid["sigma2_mse"].mean())),
                "ECP_gamma": float(valid["eta_covered"].mean()),
                "ECP_alpha": float(valid["alpha_ecp"].mean()),
                "ECP_beta": float(valid["beta_ecp"].mean()),
                "ECP_sigma2": float(valid["sigma2_ecp"].mean()),
                "LEN_gamma": float(valid["eta_ci_len"].mean()),
                "LEN_alpha": float(valid["alpha_ci_len"].mean()),
                "LEN_beta": float(valid["beta_ci_len"].mean()),
                "LEN_sigma2": float(valid["sigma2_ci_len"].mean()),
            })
            if "eta_on_boundary" in valid.columns:
                rec["eta_boundary_rate"] = float(valid["eta_on_boundary"].fillna(False).mean())
            else:
                rec["eta_boundary_rate"] = math.nan

            optional_first_cols = [
                "Z_screened",
                "Z_parallel",
                "Z_parallel_workers",
                "Z_parallel_chunks",
                "Z_parallel_accepted_chunks",
                "Z_approximate",
                "Z_approx_rows",
                "Z_approx_pairs",
                "Z_approx_threshold",
                "Z_accepted",
                "Z_attempt",
                "Z_offdiag_std",
                "Z_max_abs_score",
                "Z_score_quantile",
                "Z_abs_score_quantile",
                "Z_eta_signal",
                "Z_eta_info",
                "Z_attention_x2_std_mean",
                "Z_attention_x2_std_min",
                "Z_attention_x2_std_requested_min",
                "Z_attention_x2_std_effective_min",
                "Z_attention_x2_std_adaptive_min",
                "Z_attention_x2_std_adaptive_factor",
                "Z_attention_effective_neighbors_mean",
                "Z_attention_effective_neighbors_p95",
                "Z_attention_effective_neighbors_limit",
                "Z_attention_max_weight_mean",
                "Z_selection_score",
                "DGP_attention_precomputed",
                "TopK_attention_cached",
                "attention_cache_mode",
                "attention_cache_clusters",
                "attention_cache_probes",
                "attention_cache_candidate_size",
                "TopK_cache_quality_rows",
                "TopK_cache_recall_pos_mean",
                "TopK_cache_recall_pos_min",
                "TopK_cache_score_loss_pos_mean",
                "TopK_cache_score_loss_pos_max",
                "TopK_cache_score_loss_pos_abs_mean",
                "TopK_cache_recall_neg_mean",
                "TopK_cache_recall_neg_min",
                "TopK_cache_score_loss_neg_mean",
                "TopK_cache_score_loss_neg_max",
                "TopK_cache_score_loss_neg_abs_mean",
                "attention_mode",
                "attention_top_k",
                "stationarity_rescaled",
                "stationary_lhs_before",
                "stationary_lhs_after",
            ]
            for col in optional_first_cols:
                if col in valid.columns:
                    rec[col] = valid[col].iloc[0]
        else:
            for name in ["gamma", "alpha", "beta", "sigma2"]:
                rec[f"RMSE_{name}"] = math.nan
                rec[f"ECP_{name}"] = math.nan
                rec[f"LEN_{name}"] = math.nan
            rec["eta_boundary_rate"] = math.nan
        records.append(rec)

    return pd.DataFrame(records).sort_values(["setting", "dist", "T", "method"])


def make_paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert the wide summary into a long paper-style table."""
    rows = []
    for _, r in summary.iterrows():
        for metric in ["RMSE", "ECP"]:
            rows.append({
                "setting": r["setting"],
                "dist": r["dist"],
                "T": r["T"],
                "p": r["p"],
                "q": r["q"],
                "method": r["method"],
                "metric": metric,
                "gamma": r.get(f"{metric}_gamma", math.nan),
                "alpha": r.get(f"{metric}_alpha", math.nan),
                "beta": r.get(f"{metric}_beta", math.nan),
                "sigma2": r.get(f"{metric}_sigma2", math.nan),
            })
    return pd.DataFrame(rows)
