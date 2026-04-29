import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["RAY_memory_usage_threshold"] = "0.9"

import gc
import math
from datetime import datetime

import jax
import jax.numpy as jnp
import pandas as pd
import ray
from tqdm import trange


try:
    from ._bootstrap import RESULTS_DIR
except ImportError:
    from _bootstrap import RESULTS_DIR

from vara.dgp import (
    simulate_VAR_attention,
    flatten_params,
    draw_good_Z_for_eta,
    ensure_stationarity,
)
from vara.attention import build_attention
from vara.fit import estimate


# =====================================================================
# 1. Componentwise second-stage estimator for Theorem 2
# =====================================================================

def fit_component_phi_sigma_given_eta(
    Y,
    Z,
    eta_a,
    k,
    params_scale,
    ridge=1e-10,
):
    """
    Compute the Theorem 2 second-stage estimator for one component k.

    Paper notation:
        Y_{t,k} = alpha_k Y_{t-1,k}
                  + beta_k sum_j w_{k,j}(gamma) Y_{t-1,j}
                  + epsilon_{t,k}

    Code notation:
        paper gamma       = eta_a
        paper alpha_k     = beta[k]
        paper beta_k      = gamma_a[k]

    Therefore:
        phi_k = (alpha_k, beta_k)
              = (code beta[k], code gamma_a[k])

    Parameters
    ----------
    Y : array, shape (T, p)
        Simulated or observed time series.
    Z : array, shape (p, q)
        Feature matrix.
    eta_a : float
        Pilot attention parameter, i.e. paper gamma.
    k : int
        Zero-based component index.
    params_scale : dict
        Attention scaling dictionary.
    ridge : float
        Tiny ridge added to X'X for numerical stability.

    Returns
    -------
    dict with:
        phi_hat      : shape (2,), [alpha_hat, beta_hat] in paper notation
        sigma2_hat   : scalar
        X            : shape (T-1, 2)
        y            : shape (T-1,)
        S_hat        : X'X / (T-1)
        H_phi_hat    : S_hat / sigma2_hat
        inv_H_phi    : inverse H_phi_hat
    """
    T, p = Y.shape
    T_eff = T - 1

    W = build_attention(Z, eta_a, params_scale["eta_a"])

    y = Y[1:, k]

    # First regressor: own lag Y_{t-1,k}
    x_own = Y[:-1, k]

    # Second regressor: attention-weighted lag sum_j w_{k,j}(eta) Y_{t-1,j}
    x_attn = Y[:-1] @ W[k, :]

    X = jnp.column_stack([x_own, x_attn])

    XtX = X.T @ X
    XtX_reg = XtX + ridge * jnp.eye(2, dtype=X.dtype)
    Xty = X.T @ y

    phi_hat = jnp.linalg.solve(XtX_reg, Xty)

    resid = y - X @ phi_hat
    sigma2_hat = jnp.mean(resid ** 2)

    S_hat = XtX / T_eff
    H_phi_hat = S_hat / sigma2_hat
    inv_H_phi = jnp.linalg.inv(H_phi_hat + ridge * jnp.eye(2, dtype=X.dtype))

    return {
        "phi_hat": phi_hat,
        "sigma2_hat": sigma2_hat,
        "X": X,
        "y": y,
        "S_hat": S_hat,
        "H_phi_hat": H_phi_hat,
        "inv_H_phi": inv_H_phi,
    }


def fit_all_components_given_eta(
    Y,
    Z,
    eta_a,
    params_scale,
    ridge=1e-10,
):
    """
    Vectorized componentwise OLS for all k.

    Returns
    -------
    alpha_hat : shape (p,)
        Paper alpha estimates, i.e. code beta estimates.
    beta_hat : shape (p,)
        Paper beta estimates, i.e. code gamma_a estimates.
    sigma2_hat : shape (p,)
        Residual variance estimates.
    """
    T, p = Y.shape

    W = build_attention(Z, eta_a, params_scale["eta_a"])

    Y_lag = Y[:-1]        # (T-1, p)
    Y_now = Y[1:]         # (T-1, p)

    X1 = Y_lag            # own lag for every component
    X2 = Y_lag @ W.T      # column k = Y_lag @ W[k, :]

    xx11 = jnp.sum(X1 * X1, axis=0) + ridge
    xx22 = jnp.sum(X2 * X2, axis=0) + ridge
    xx12 = jnp.sum(X1 * X2, axis=0)

    xy1 = jnp.sum(X1 * Y_now, axis=0)
    xy2 = jnp.sum(X2 * Y_now, axis=0)

    det = xx11 * xx22 - xx12 ** 2 + ridge

    alpha_hat = (xy1 * xx22 - xy2 * xx12) / det
    beta_hat = (xx11 * xy2 - xx12 * xy1) / det

    resid = Y_now - X1 * alpha_hat[None, :] - X2 * beta_hat[None, :]
    sigma2_hat = jnp.mean(resid ** 2, axis=0)

    return alpha_hat, beta_hat, sigma2_hat


# =====================================================================
# 2. Concentrated pilot estimator eta_hat_c
# =====================================================================

def concentrated_loglik_eta(
    eta_a,
    Y,
    Z,
    params_scale,
    ridge=1e-10,
    eps=1e-12,
):
    """
    Concentrated log-quasi-likelihood for eta_a.

    Paper:
        l_c(gamma) = - T/2 sum_k log{ sigma2_tilde_k(gamma) }

    Code:
        paper gamma = eta_a

    We ignore constants and use T_eff = T - 1.
    """
    T_eff = Y.shape[0] - 1

    _, _, sigma2_hat = fit_all_components_given_eta(
        Y=Y,
        Z=Z,
        eta_a=eta_a,
        params_scale=params_scale,
        ridge=ridge,
    )

    sigma2_hat = jnp.maximum(sigma2_hat, eps)
    return -0.5 * T_eff * jnp.sum(jnp.log(sigma2_hat))


def golden_section_maximize(
    fn,
    lower=-2.0,
    upper=2.0,
    tol=1e-4,
    max_iter=100,
):
    """
    Scalar derivative-free maximization on a compact interval.

    This is stable for eta_a because it avoids relying on a noisy eta gradient.
    """
    gr = (math.sqrt(5.0) - 1.0) / 2.0

    a = float(lower)
    b = float(upper)

    c = b - gr * (b - a)
    d = a + gr * (b - a)

    fc = float(fn(c))
    fd = float(fn(d))

    for _ in range(max_iter):
        if abs(b - a) < tol:
            break

        if fc < fd:
            a = c
            c = d
            fc = fd
            d = a + gr * (b - a)
            fd = float(fn(d))
        else:
            b = d
            d = c
            fd = fc
            c = b - gr * (b - a)
            fc = float(fn(c))

    eta_hat = 0.5 * (a + b)
    f_hat = float(fn(eta_hat))
    return eta_hat, f_hat


def estimate_eta_concentrated(
    Y,
    Z,
    params_scale,
    eta_bounds=(-2.0, 2.0),
    grid_size=41,
    ridge=1e-10,
):
    """
    Robust concentrated estimator for eta_a.

    Procedure:
      1. grid search to find a good bracket,
      2. golden-section refinement around the best grid cell.

    Returns
    -------
    eta_hat : scalar
    info : dict
    """
    lower, upper = eta_bounds
    grid = jnp.linspace(lower, upper, grid_size)

    vals = []
    for eta in grid:
        vals.append(
            concentrated_loglik_eta(
                eta,
                Y,
                Z,
                params_scale,
                ridge=ridge,
            )
        )
    vals = jnp.array(vals)

    best_idx = int(jnp.argmax(vals))

    # Build local bracket around best grid point
    left_idx = max(0, best_idx - 1)
    right_idx = min(grid_size - 1, best_idx + 1)

    local_lower = float(grid[left_idx])
    local_upper = float(grid[right_idx])

    # If best point is on boundary, refine on the boundary interval
    if best_idx == 0:
        local_lower = float(grid[0])
        local_upper = float(grid[1])
    elif best_idx == grid_size - 1:
        local_lower = float(grid[-2])
        local_upper = float(grid[-1])

    fn = lambda x: concentrated_loglik_eta(
        x,
        Y,
        Z,
        params_scale,
        ridge=ridge,
    )

    eta_hat, obj_hat = golden_section_maximize(
        fn,
        lower=local_lower,
        upper=local_upper,
        tol=1e-4,
        max_iter=100,
    )

    info = {
        "eta_hat": eta_hat,
        "objective": obj_hat,
        "grid_best_eta": float(grid[best_idx]),
        "grid_best_obj": float(vals[best_idx]),
        "on_boundary": bool(best_idx == 0 or best_idx == grid_size - 1),
    }

    return jnp.asarray(eta_hat), info


# =====================================================================
# 3. Optional QMLE pilot eta_a using your existing estimate()
# =====================================================================

def make_reparam_theta_from_params(params_true, p):
    """
    Build theta on the optimization/reparameterized scale.

    Your estimate() expects sigma2 on the softplus-inverse scale.
    """
    block_order = {
        "eta_a": [0],
        "gamma_a": list(range(1, 1 + p)),
        "beta": list(range(1 + p, 1 + 2 * p)),
        "sigma2": list(range(1 + 2 * p, 1 + 3 * p)),
    }

    sigma2_tilde = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = dict(params_true)
    params_reparam["sigma2"] = sigma2_tilde

    theta_reparam, _ = flatten_params(params_reparam, block_order)
    theta_true, _ = flatten_params(params_true, block_order)

    return theta_reparam, theta_true, block_order


def estimate_eta_qmle_pilot(
    key,
    Y,
    Z,
    params_true,
    params_scale,
    algorithm="lbfgs",
    perturb_scale=0.05,
):
    """
    Use the full QMLE estimator as a pilot eta_a estimator.

    This is useful if you want to verify Theorem 2 with the same eta estimator
    used in your Theorem 1 experiments.
    """
    T, p = Y.shape
    theta_reparam, _, block_order = make_reparam_theta_from_params(params_true, p)

    theta_init = theta_reparam + perturb_scale * jax.random.normal(
        key, theta_reparam.shape
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
    )

    theta_hat = jnp.asarray(result["theta"])
    eta_hat = theta_hat[0]

    return eta_hat, {
        "status": int(convergence["status"]),
        "theta_hat": theta_hat,
    }


# =====================================================================
# 4. Single replication for Theorem 2
# =====================================================================

def theorem2_single_replication(
    key,
    params_true,
    Z,
    T,
    p,
    q,
    k_list,
    params_scale,
    dist="normal",
    pilot_mode="oracle",
    kappa4=3.0,
    eta_bounds=(-2.0, 2.0),
    qmle_algorithm="lbfgs",
):
    """
    One Monte Carlo replication for Theorem 2.

    Returns one row per selected component k.
    """
    params_true = dict(params_true)
    beta, gamma_a, _ = ensure_stationarity(
        params_true["beta"], params_true["gamma_a"], margin=0.98
    )
    params_true["beta"] = beta
    params_true["gamma_a"] = gamma_a

    sim_key, pilot_key = jax.random.split(key)

    Y, Z_used, _ = simulate_VAR_attention(
        sim_key,
        params=params_true,
        T=T,
        p=p,
        q=q,
        dist=dist,
        params_scale=params_scale,
        Z=Z,
    )

    eta_true = params_true["eta_a"]

    if pilot_mode == "oracle":
        eta_pilot = eta_true
        pilot_info = {"status": 0, "mode": "oracle"}

    elif pilot_mode == "concentrated":
        eta_pilot, pilot_info = estimate_eta_concentrated(
            Y=Y,
            Z=Z_used,
            params_scale=params_scale,
            eta_bounds=eta_bounds,
        )
        pilot_info["status"] = 0
        pilot_info["mode"] = "concentrated"

    elif pilot_mode == "qmle":
        eta_pilot, pilot_info = estimate_eta_qmle_pilot(
            key=pilot_key,
            Y=Y,
            Z=Z_used,
            params_true=params_true,
            params_scale=params_scale,
            algorithm=qmle_algorithm,
        )
        pilot_info["mode"] = "qmle"

    else:
        raise ValueError("pilot_mode must be 'oracle', 'concentrated', or 'qmle'.")

    T_eff = T - 1

    rows = []

    for k in k_list:
        est = fit_component_phi_sigma_given_eta(
            Y=Y,
            Z=Z_used,
            eta_a=eta_pilot,
            k=k,
            params_scale=params_scale,
        )

        phi_hat = est["phi_hat"]
        sigma2_hat = est["sigma2_hat"]
        H_phi_hat = est["H_phi_hat"]
        inv_H_phi = est["inv_H_phi"]

        # Paper phi_k = (alpha_k, beta_k)
        # Code alpha_k = beta[k], beta_k = gamma_a[k].
        phi_true = jnp.array([
            params_true["beta"][k],
            params_true["gamma_a"][k],
        ])

        sigma2_true = params_true["sigma2"][k]

        phi_err = phi_hat - phi_true
        sigma_err = sigma2_hat - sigma2_true

        # Marginal standardized statistics from Theorem 2(ii)
        z_phi = (
            jnp.sqrt(T_eff)
            * phi_err
            / jnp.sqrt(jnp.maximum(jnp.diag(inv_H_phi), 1e-12))
        )

        z_sigma = (
            jnp.sqrt(T_eff)
            * sigma_err
            / jnp.sqrt(jnp.maximum((kappa4 - 1.0) * sigma2_true ** 2, 1e-12))
        )

        # Optional joint quadratic statistic for phi_k:
        # T * (phi_hat - phi)' H_phi (phi_hat - phi) approximately chi-square_2.
        q_phi = T_eff * (phi_err @ H_phi_hat @ phi_err)

        rows.append({
            "k": int(k),
            "eta_true": float(eta_true),
            "eta_pilot": float(eta_pilot),
            "eta_error": float(eta_pilot - eta_true),
            "pilot_status": int(pilot_info.get("status", 0)),
            "pilot_mode": pilot_mode,

            # Estimates
            "alpha_hat": float(phi_hat[0]),      # paper alpha, code beta
            "beta_hat": float(phi_hat[1]),       # paper beta, code gamma_a
            "sigma2_hat": float(sigma2_hat),

            # Truth
            "alpha_true": float(phi_true[0]),
            "beta_true": float(phi_true[1]),
            "sigma2_true": float(sigma2_true),

            # Errors
            "alpha_error": float(phi_err[0]),
            "beta_error": float(phi_err[1]),
            "phi_l2_error": float(jnp.linalg.norm(phi_err)),
            "sigma2_error": float(sigma_err),

            # Standardized Theorem 2 stats
            "z_alpha": float(z_phi[0]),
            "z_beta": float(z_phi[1]),
            "z_sigma2": float(z_sigma),
            "q_phi": float(q_phi),

            # Diagnostics
            "H_phi_min_eig": float(jnp.min(jnp.linalg.eigvalsh(H_phi_hat))),
            "H_phi_max_eig": float(jnp.max(jnp.linalg.eigvalsh(H_phi_hat))),
        })

    return rows


# =====================================================================
# 5. Ray actor helpers
# =====================================================================

def failed_replication_rows(params_true, k_list, pilot_mode):
    """Return one failed placeholder row per requested component."""
    rows = []
    eta_true = float(params_true.get("eta_a", jnp.nan))

    for k in k_list:
        rows.append({
            "k": int(k),
            "eta_true": eta_true,
            "eta_pilot": float("nan"),
            "eta_error": float("nan"),
            "pilot_status": -1,
            "pilot_mode": pilot_mode,
            "alpha_hat": float("nan"),
            "beta_hat": float("nan"),
            "sigma2_hat": float("nan"),
            "alpha_true": float(params_true["beta"][k]),
            "beta_true": float(params_true["gamma_a"][k]),
            "sigma2_true": float(params_true["sigma2"][k]),
            "alpha_error": float("nan"),
            "beta_error": float("nan"),
            "phi_l2_error": float("nan"),
            "sigma2_error": float("nan"),
            "z_alpha": float("nan"),
            "z_beta": float("nan"),
            "z_sigma2": float("nan"),
            "q_phi": float("nan"),
            "H_phi_min_eig": float("nan"),
            "H_phi_max_eig": float("nan"),
        })

    return rows


@ray.remote
class Theorem2ReplicationActor:
    def __init__(self):
        self._cache = None

    def run(self, *args, **kwargs):
        try:
            result = theorem2_single_replication(*args, **kwargs)
            self._cache = None
            gc.collect()
            return result
        except Exception as e:
            print(f"[Ray actor error] {e}")
            params_true = kwargs.get("params_true", args[1] if len(args) > 1 else {})
            k_list = kwargs.get("k_list", args[6] if len(args) > 6 else ())
            pilot_mode = kwargs.get("pilot_mode", "unknown")
            return failed_replication_rows(params_true, k_list, pilot_mode)


# =====================================================================
# 6. Monte Carlo driver
# =====================================================================

def run_pilot_consistency_verification(
    seed=2026,
    R=500,
    T=1000,
    p=10,
    q=50,
    k_list=(0, 1, 2),
    pilot_mode="oracle",
    dist="normal",
    kappa4=3.0,
    qmle_algorithm="lbfgs",
    USE_RAY=True,
    MAX_CPUS_PER_TASK=1,
):
    """
    Run a Monte Carlo verification of Theorem 2.

    Recommended workflow:
      1. pilot_mode='oracle' first
      2. pilot_mode='concentrated' second
      3. pilot_mode='qmle' only after Theorem 1 eta_a is stable
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 10)

    params_scale = {"eta_a": 1.0}

    # DGP chosen to keep eta_a identifiable.
    params_true = {
        "eta_a": 0.5,
        "gamma_a": jax.random.uniform(
            keys[1], (p,), minval=0.15, maxval=0.30
        ),
        "beta": jax.random.uniform(
            keys[2], (p,), minval=-0.20, maxval=0.20
        ),
        "sigma2": jax.random.uniform(
            keys[3], (p,), minval=0.50, maxval=1.00
        ),
    }
    beta, gamma_a, stat_info = ensure_stationarity(
        params_true["beta"], params_true["gamma_a"], margin=0.98
    )
    params_true["beta"] = beta
    params_true["gamma_a"] = gamma_a

    # Fixed design, screened for eta signal.
    Z, Z_info = draw_good_Z_for_eta(
        key=keys[0],
        p=p,
        q=q,
        params_true=params_true,
        params_scale=params_scale,
        max_tries=200,
        min_offdiag_std=0.15,
        max_abs_score=3.0,
        min_eta_signal=0.05,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("Theorem 2 verification setup")
    print("=" * 70)
    print(f"seed       = {seed}")
    print(f"R          = {R}")
    print(f"T          = {T}")
    print(f"p, q       = {p}, {q}")
    print(f"k_list     = {list(k_list)}")
    print(f"pilot_mode = {pilot_mode}")
    print(f"dist       = {dist}")
    print(f"kappa4     = {kappa4}")
    print(f"USE_RAY    = {USE_RAY}")
    print(f"stationary = {not bool(stat_info['rescaled'])}")
    print(f"Z_info     = {Z_info}")
    print("=" * 70 + "\n")

    rep_keys = jax.random.split(keys[4], R)

    if USE_RAY:
        print("Running with Ray parallelization")
        print("JAX devices:", jax.devices())
        print(f"\nNUM_CPUS_PER_TASK: {MAX_CPUS_PER_TASK}\n")

        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            _node_ip_address="127.0.0.1",
        )

        n_actors = max(1, min(R, int(os.cpu_count() / MAX_CPUS_PER_TASK)))
        print(f"Ray actors using {n_actors} actors...")

        actors = [
            Theorem2ReplicationActor.options(num_cpus=MAX_CPUS_PER_TASK).remote()
            for _ in range(n_actors)
        ]

        futures = [
            actors[r % n_actors].run.remote(
                key=rep_keys[r],
                params_true=params_true,
                Z=Z,
                T=T,
                p=p,
                q=q,
                k_list=k_list,
                params_scale=params_scale,
                dist=dist,
                pilot_mode=pilot_mode,
                kappa4=kappa4,
                qmle_algorithm=qmle_algorithm,
            )
            for r in range(R)
        ]

        results = []
        for r in trange(R, desc="Theorem 2 replications", ncols=100):
            results.append(ray.get(futures[r]))

        ray.shutdown()
    else:
        print("Running sequentially with JAX")
        results = [
            theorem2_single_replication(
                key=rep_keys[r],
                params_true=params_true,
                Z=Z,
                T=T,
                p=p,
                q=q,
                k_list=k_list,
                params_scale=params_scale,
                dist=dist,
                pilot_mode=pilot_mode,
                kappa4=kappa4,
                qmle_algorithm=qmle_algorithm,
            )
            for r in trange(R, desc="Theorem 2 replications", ncols=100)
        ]

    all_rows = []
    for r, rows in enumerate(results):
        for row in rows:
            row["rep"] = r
            row["T"] = T
            row["p"] = p
            row["q"] = q
            row["seed"] = seed
            row["dist"] = dist
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # Keep only successful pilot runs if using QMLE.
    df_valid = df[df["pilot_status"] == 0].copy()

    print("\n" + "=" * 70)
    print("Theorem 2 consistency diagnostics")
    print("=" * 70)
    consistency = (
        df_valid
        .groupby("k")
        .agg(
            eta_rmse=("eta_error", lambda x: math.sqrt((x ** 2).mean())),
            phi_rmse=("phi_l2_error", lambda x: math.sqrt((x ** 2).mean())),
            alpha_rmse=("alpha_error", lambda x: math.sqrt((x ** 2).mean())),
            beta_rmse=("beta_error", lambda x: math.sqrt((x ** 2).mean())),
            sigma2_rmse=("sigma2_error", lambda x: math.sqrt((x ** 2).mean())),
            H_phi_min_eig_mean=("H_phi_min_eig", "mean"),
        )
    )
    print(consistency.round(4))

    print("\n" + "=" * 70)
    print("Theorem 2 asymptotic normality diagnostics")
    print("Target: mean ~ 0, variance ~ 1, rejection ~ 0.05")
    print("=" * 70)

    z_summary = []
    for k in k_list:
        sub = df_valid[df_valid["k"] == k]
        for name in ["z_alpha", "z_beta", "z_sigma2"]:
            z = sub[name]
            z_summary.append({
                "k": k,
                "stat": name,
                "mean": z.mean(),
                "var": z.var(ddof=0),
                "P_abs_gt_1.96": (z.abs() > 1.959963984540054).mean(),
            })

    z_summary = pd.DataFrame(z_summary)
    print(z_summary.round(4).to_string(index=False))

    print("\n" + "=" * 70)
    print("Pilot eta diagnostics")
    print("=" * 70)
    print(
        df_valid[["eta_true", "eta_pilot", "eta_error"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )

    out_path = RESULTS_DIR / (
        f"results_theorem2_{pilot_mode}_"
        f"T{T}_p{p}_q{q}_seed{seed}_"
        f"{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")

    return df, consistency, z_summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run pilot consistency verification.")
    parser.add_argument(
        "--use-ray",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Ray parallelization. Use --no-use-ray for sequential execution.",
    )
    args = parser.parse_args()

    # First run oracle mode.
    # This checks the core statement of Theorem 2 with gamma known.
    run_pilot_consistency_verification(
        seed=2026,
        R=500,
        T=1000,
        p=10,
        q=50,
        k_list=(0, 1, 2),
        pilot_mode="oracle",
        dist="normal",
        kappa4=3.0,
        USE_RAY=args.use_ray,
        MAX_CPUS_PER_TASK=1,
    )

    # Then try:
    # run_theorem2_verification(pilot_mode="concentrated", ...)
    #
    # Finally, if desired:
    # run_theorem2_verification(pilot_mode="qmle", qmle_algorithm="lbfgs", ...)
