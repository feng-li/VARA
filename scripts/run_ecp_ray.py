import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["RAY_memory_usage_threshold"] = "0.9"

import gc
import ray
import jax
import jax.numpy as jnp
from jax.random import uniform
from tqdm import trange
import pandas as pd
from datetime import datetime
import itertools

try:
    from ._bootstrap import RESULTS_DIR
except ImportError:
    from _bootstrap import RESULTS_DIR

from vara.dgp import simulate_VAR_attention, flatten_params, draw_good_Z_for_eta, ensure_stationarity
from vara.fit import estimate
from vara.cov import (
    compute_projected_asymptotic_covariance_from_theta_hat,
    compute_z_stat_from_theta_hat,
    make_A,
)

# ===============================================================
# === Single replication ========================================
# ===============================================================

def single_replication(subkey, params_true, Z, T, p, q, params_scale, dist, alpha=0.05,
                       algorithm="analytical_gradhess", update_idx=slice(None),
                       max_tries=10, kappa4=3.0):
    """
    One Monte Carlo replication under the current 4-block old model:
        theta = (eta_a, gamma_a, beta, sigma2)

    Returns
    -------
    inside : (d,) bool
    ci_length : (d,) float
    theta_hat : (d,) float
    status : int
    z_stats : (4,) float
        standardized stats for eta_a, gamma_a[1], beta[1], sigma2[1]
    """
    # ---- True theta ----
    block_order = {
        "eta_a": [0],
        "gamma_a": list(range(1, 1 + p)),
        "beta": list(range(1 + p, 1 + 2 * p)),
        "sigma2": list(range(1 + 2 * p, 1 + 3 * p)),
    }

    params_true = dict(params_true)
    beta, gamma_a, _ = ensure_stationarity(
        params_true["beta"], params_true["gamma_a"], margin=0.98
    )
    params_true["beta"] = beta
    params_true["gamma_a"] = gamma_a

    # --- Convert σ² → σ̃² (softplus inverse) ---
    sigma2_tilde_true = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = params_true.copy()
    params_reparam["sigma2"] = sigma2_tilde_true
    theta_true_reparam, param_slices = flatten_params(params_reparam, block_order)
    theta_true, _ = flatten_params(params_true, block_order)

    if isinstance(update_idx, slice):
        update_idx_array = jnp.arange(theta_true_reparam.size)
    else:
        update_idx_array = jnp.array(update_idx)

    d = theta_true_reparam.shape[0]

    last_inside = jnp.zeros(d, dtype=bool)
    last_ci_length = jnp.full(d, jnp.nan)
    last_theta_hat = jnp.full(d, jnp.nan)
    last_z_stats = jnp.full((d,), jnp.nan)
    last_status = -1

    for attempt in range(max_tries):
        attempt_key = jax.random.fold_in(subkey, attempt)
        noise_key = jax.random.fold_in(subkey, attempt + max_tries)

        # 1. Simulate data
        Y, Z_used, W_a = simulate_VAR_attention(
            attempt_key, params_true, T=T, p=p, q=q,
            dist=dist, params_scale=params_scale, Z=Z
        )

        # 2. Initial values
        theta_init = theta_true_reparam
        perturb_scales = {
            "eta_a": 0.1,
            "gamma_a": 0.1,
            "beta": 0.1,
            "sigma2": 0.1,
        }

        for name, sl in param_slices.items():
            key, noise_key = jax.random.split(noise_key)
            scale = perturb_scales[name]
            idx_all = jnp.arange(theta_true_reparam.size)[sl]
            idx = jnp.intersect1d(idx_all, update_idx_array)
            if idx.size > 0:
                noise = scale * jax.random.normal(key, shape=(len(idx),))
                theta_init = theta_init.at[idx].add(noise)

        # 3. Estimate
        result, convergence = estimate(
            Y, Z_used, attempt_key, theta_init,
            params_scale=params_scale,
            algorithm=algorithm,
            update_idx=update_idx_array,
            block_order=block_order
        )

        theta_hat = jnp.asarray(result["theta"])
        se = jnp.asarray(result["se"])

        # 4. CI
        z_crit = jax.scipy.stats.norm.ppf(1 - alpha / 2.0)
        lower = theta_hat - z_crit * se
        upper = theta_hat + z_crit * se

        inside = (theta_true >= lower) & (theta_true <= upper)
        ci_length = upper - lower

        status = int(convergence["status"])

        if status == 0:

            z_eta, _ = compute_z_stat_from_theta_hat(
                Y=Y,
                Z=Z_used,
                theta_hat=theta_hat,
                theta_true=theta_true,
                theta_for_J=theta_true,   # use truth for cleaner theorem verification
                A=make_A(p=p, kind="eta_a"),
                params_scale=params_scale,
                kappa4=kappa4
            )

            z_gamma1, _ = compute_z_stat_from_theta_hat(
                Y=Y,
                Z=Z_used,
                theta_hat=theta_hat,
                theta_true=theta_true,
                theta_for_J=theta_true,   # use truth for cleaner theorem verification
                A=make_A(p=p, kind="gamma_a", j=1),
                params_scale=params_scale,
                kappa4=kappa4
            )

            z_beta1, _ = compute_z_stat_from_theta_hat(
                Y=Y,
                Z=Z_used,
                theta_hat=theta_hat,
                theta_true=theta_true,
                theta_for_J=theta_true,   # use truth for cleaner theorem verification
                A=make_A(p=p, kind="beta", j=1),
                params_scale=params_scale,
                kappa4=kappa4
            )

            z_sigma1, _ = compute_z_stat_from_theta_hat(
                Y=Y,
                Z=Z_used,
                theta_hat=theta_hat,
                theta_true=theta_true,
                theta_for_J=theta_true,   # use truth for cleaner theorem verification
                A=make_A(p=p, kind="sigma2", j=1),
                params_scale=params_scale,
                kappa4=kappa4
            )

            z_stats_list = []

            for group_name, sl in param_slices.items():
                start = sl.start
                stop = sl.stop

                if stop - start == 1:
                    A = make_A(p=p, kind=group_name)
                    z_val, _ = compute_z_stat_from_theta_hat(
                        Y=Y,
                        Z=Z_used,
                        theta_hat=theta_hat,
                        theta_true=theta_true,
                        theta_for_J=theta_true,
                        A=A,
                        params_scale=params_scale,
                        kappa4=kappa4
                    )
                    z_stats_list.append(z_val[0])
                else:
                    for j in range(stop - start):
                        A = make_A(p=p, kind=group_name, j=j+1)
                        z_val, _ = compute_z_stat_from_theta_hat(
                            Y=Y,
                            Z=Z_used,
                            theta_hat=theta_hat,
                            theta_true=theta_true,
                            theta_for_J=theta_true,
                            A=A,
                            params_scale=params_scale,
                            kappa4=kappa4
                        )
                        z_stats_list.append(z_val[0])

            z_stats = jnp.array(z_stats_list)
            return inside, ci_length, theta_hat, status, z_stats

    # breakpoint()
    return last_inside, last_ci_length, last_theta_hat, last_status, last_z_stats


# ===============================================================
# === Ray Actor =================================================
# ===============================================================

@ray.remote
class ReplicationActor:
    def __init__(self):
        self._cache = None

    def run(self, *args, **kwargs):
        try:
            result = single_replication(*args, **kwargs)
            self._cache = None
            gc.collect()
            return result
        except Exception as e:
            print(f"[Ray actor error] {e}")
            p_arg = kwargs["p"] if "p" in kwargs else args[4]
            d = 1 + 3 * p_arg
            return (
                jnp.zeros(d, dtype=bool),
                jnp.full((d,), jnp.nan),
                jnp.full((d,), jnp.nan),
                -1,
                jnp.full((d,), jnp.nan),
            )

# ===============================================================
# === Main Monte Carlo driver ===================================
# ===============================================================

def empirical_coverage_rate(key, R, T, p, q, params_dgp, params_scale,
                            dist="normal", alpha=0.05,
                            algorithm="analytical_gradhess",
                            USE_RAY=False, MAX_CPUS_PER_TASK=1,
                            kappa4=3.0):
    """
    Monte Carlo simulation to estimate:
        - empirical coverage
        - average CI length
        - RMSE
        - theorem verification Z statistics
    """

    print("\n" + "=" * 70 + "\nEmpirical Simulation Setup\n" + "=" * 70)
    print(f"Algorithm      : {algorithm}")
    print(f"Distribution   : {dist}")
    print(f"Sample size  T : {T}")
    print(f"Response dim p : {p}")
    print(f"Feature dim  q : {q}")
    print(f"Replications R : {R}")
    print(f"Random key     : {key}")
    print("=" * 70, "\n")

    subkeys = jax.random.split(key, R)
    pkeys = jax.random.split(key, 4)

    params_true = dict(
        eta_a=params_dgp["eta_a"],
        gamma_a=uniform(
            pkeys[1], (p,),
            minval=params_dgp["gamma_a"]["minval"],
            maxval=params_dgp["gamma_a"]["maxval"]
        ),
        beta=uniform(
            pkeys[2], (p,),
            minval=params_dgp["beta"]["minval"],
            maxval=params_dgp["beta"]["maxval"]
        ),
        sigma2=uniform(
            pkeys[3], (p,),
            minval=params_dgp["sigma2"]["minval"],
            maxval=params_dgp["sigma2"]["maxval"]
        ),
    )
    beta, gamma_a, stat_info = ensure_stationarity(
        params_true["beta"], params_true["gamma_a"], margin=0.98
    )
    params_true["beta"] = beta
    params_true["gamma_a"] = gamma_a
    if bool(stat_info["rescaled"]):
        print("Stationarity rescaled true beta/gamma_a for coverage target.")

    # Z = draw_Z(pkeys[0], p, q)
    Z, Z_info = draw_good_Z_for_eta(
        key=pkeys[0],
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


    # update all by default
    update_idx = slice(None)
    block_order = {
        "eta_a": [0],
        "gamma_a": list(range(1, 1 + p)),
        "beta": list(range(1 + p, 1 + 2 * p)),
        "sigma2": list(range(1 + 2 * p, 1 + 3 * p)),
    }

    if USE_RAY:
        print("🔹 Running with Ray parallelization")
        print("JAX devices:", jax.devices())
        print(f"\nNUM_CPUS_PER_TASK: {MAX_CPUS_PER_TASK}\n")

        ray.init(ignore_reinit_error=True, include_dashboard=False)

        N_ACTORS = max(1, int(os.cpu_count() / MAX_CPUS_PER_TASK))
        print(f"Ray actors using {N_ACTORS} actors...")

        actors = [
            ReplicationActor.options(num_cpus=MAX_CPUS_PER_TASK).remote()
            for _ in range(N_ACTORS)
        ]

        futures = [
            actors[r % N_ACTORS].run.remote(
                subkeys[r], params_true, Z, T, p, q,
                params_scale, dist, alpha,
                algorithm, update_idx, 10, kappa4
            )
            for r in range(R)
        ]

        results = []
        for r in trange(R, desc="Monte Carlo replications", ncols=100):
            results.append(ray.get(futures[r]))

        ray.shutdown()
    else:
        print("🔹 Running sequentially with JAX")
        results = [
            single_replication(
                subkeys[r], params_true, Z, T, p, q,
                params_scale, dist, alpha,
                algorithm, update_idx, 10, kappa4
            )
            for r in trange(R, desc="Monte Carlo replications", ncols=100)
        ]

    inside, ci_length, theta_hat_all, status_all, z_stats_all = zip(*results)
    inside = jnp.stack(inside)
    ci_length = jnp.stack(ci_length)
    theta_hat_all = jnp.stack(theta_hat_all)
    status_all = jnp.stack(status_all)
    z_stats_all = jnp.stack(z_stats_all)

    valid_mask = jnp.isfinite(theta_hat_all).all(axis=1)
    inside = inside[valid_mask]
    ci_length = ci_length[valid_mask]
    theta_hat_all = theta_hat_all[valid_mask]
    z_stats_all = z_stats_all[valid_mask]
    status_all = status_all[valid_mask]

    # --- put the diagnostic block here ---
    converged_mask = (status_all == 0) & jnp.isfinite(theta_hat_all).all(axis=1)
    print("convergence rate =", float(jnp.mean(converged_mask)))
    print("eta_a coverage, all runs      =", float(jnp.mean(inside[:, 0])))
    print("eta_a coverage, converged only=", float(jnp.mean(inside[converged_mask, 0])))
    print("eta_a quantiles =",
          jnp.quantile(theta_hat_all[:, 0], jnp.array([0.01, 0.05, 0.5, 0.95, 0.99])))

    theta_true, idx = flatten_params(params_true, block_order)

    coverage = jnp.nanmean(inside, axis=0)
    avg_length = jnp.nanmean(ci_length, axis=0)
    rmse = jnp.sqrt(jnp.nanmean((theta_hat_all - theta_true) ** 2, axis=0))


    theta_true, idx = flatten_params(params_true, block_order)

    coverage = jnp.nanmean(inside, axis=0)
    avg_length = jnp.nanmean(ci_length, axis=0)
    rmse = jnp.sqrt(jnp.nanmean((theta_hat_all - theta_true) ** 2, axis=0))

    coverage_by_group = {name: float(jnp.nanmean(coverage[s])) for name, s in idx.items()}
    rmse_by_group = {name: float(jnp.nanmean(rmse[s])) for name, s in idx.items()}

    print("\n=== Empirical Coverage and RMSE Summary ===")
    print(f"nominal = {(1 - alpha):.2%}")
    print(f"Overall average coverage: {jnp.nanmean(coverage):.4f}\n")
    for name in idx.keys():
        print(f"{name:<8s}: coverage={coverage_by_group[name]:.3f}, RMSE={rmse_by_group[name]:.4f}")

    print("\nAverage CI length (by parameter):")
    print(jnp.round(avg_length, 3))
    print("\nRMSE (by parameter):")
    print(jnp.round(rmse, 4))

    # theorem verification summary
    # theorem verification summary (all parameters)
    param_names = []
    for group_name, sl in idx.items():
        start = sl.start
        stop = sl.stop

        if stop - start == 1:
            param_names.append(group_name)
        else:
            for j in range(stop - start):
                param_names.append(f"{group_name}[{j}]")

    z_mean = jnp.nanmean(z_stats_all, axis=0)
    z_var = jnp.nanvar(z_stats_all, axis=0)
    z_reject_5 = jnp.nanmean(jnp.abs(z_stats_all) > 1.959963984540054, axis=0)

    print("\n=== Theorem 1 verification (standardized Z statistics) ===")
    for i, name in enumerate(param_names):
        print(
            f"{name:<12s}: mean={float(z_mean[i]): .4f}, "
            f"var={float(z_var[i]): .4f}, "
            f"P(|Z|>1.96)={float(z_reject_5[i]): .4f}"
        )

    return (
        coverage,
        avg_length,
        rmse,
        coverage_by_group,
        rmse_by_group,
        z_stats_all,
        z_mean,
        z_var,
        z_reject_5,
    )


# ===============================================================
# === Main ======================================================
# ===============================================================

def _parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_float_list(value):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_str_list(value):
    return [x.strip() for x in value.split(",") if x.strip()]


if __name__ == "__main__":
    import argparse
    import random
    try:
        from .summarize_ecp import summarize_empirical_coverage
    except ImportError:
        from summarize_ecp import summarize_empirical_coverage

    parser = argparse.ArgumentParser(description="Run empirical coverage simulations.")
    parser.add_argument(
        "--use-ray",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Ray parallelization. Use --no-use-ray for sequential execution.",
    )
    parser.add_argument(
        "--R-list",
        dest="R_list",
        type=str,
        default="100",
        help="Comma-separated replication counts to run, e.g. --R-list 100,200.",
    )
    parser.add_argument(
        "--p-list",
        type=str,
        default="10,20",
        help="Comma-separated response dimensions to run, e.g. --p-list 10,20.",
    )
    parser.add_argument(
        "--q-list",
        type=str,
        default="50,70",
        help="Comma-separated feature dimensions to run, e.g. --q-list 50,70.",
    )
    parser.add_argument(
        "--T-list",
        dest="T_list",
        type=str,
        default="500,1000",
        help="Comma-separated sample sizes to run, e.g. --T-list 500,1000.",
    )
    parser.add_argument(
        "--alpha-list",
        type=str,
        default="0.05,0.1",
        help="Comma-separated significance levels to run, e.g. --alpha-list 0.05,0.1.",
    )
    parser.add_argument(
        "--dist-list",
        type=str,
        default="normal",
        help="Comma-separated innovation distributions to run, e.g. --dist-list normal,t5.",
    )
    args = parser.parse_args()

    jnp.set_printoptions(threshold=10_000, linewidth=200, suppress=True, precision=6)

    USE_RAY = args.use_ray
    seed = random.randint(0, 10000)
    # seed = 5682, 1873
    print(f"=== Using a global seed: {seed}")
    key = jax.random.PRNGKey(seed)

    R_list = _parse_int_list(args.R_list)
    p_list = _parse_int_list(args.p_list)
    q_list = _parse_int_list(args.q_list)
    T_list = _parse_int_list(args.T_list)
    alpha_list = _parse_float_list(args.alpha_list)
    dist_list = _parse_str_list(args.dist_list)
    invalid_dist = sorted(set(dist_list) - {"normal", "t5"})
    if invalid_dist:
        raise ValueError(f"Unsupported dist values: {invalid_dist}. Use normal,t5.")

    algorithm = "lbfgs"   # "auto_gradhess", "analytical_gradhess"
    # algorithm = "auto_gradhess"
    # algorithm = "block_newton"

    out_path = RESULTS_DIR / f"results_coverage_@{datetime.now().strftime('%Y-%m-%d_%H.%M.%S')}.csv"

    params_dgp = dict(
        eta_a=0.5,
        gamma_a=dict(minval=0.15, maxval=0.3),
        beta=dict(minval=-0.2, maxval=0.2),
        sigma2=dict(minval=0.5, maxval=1.0),
    )

    params_scale = dict(eta_a=1)

    header_written = False

    for dist, R, T, p, q, alpha in itertools.product(
            dist_list, R_list, T_list, p_list, q_list, alpha_list
    ):
        print("\n" + "=" * 70 + f"\nRunning Monte Carlo: T={T}, nominal={(1-alpha):.2%}\n" + "=" * 70)

        MAX_CPUS_PER_TASK = 1

        start_time = datetime.now()
        start_ts = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"⏱ Started at {start_ts}")

        (
            coverage,
            avg_len,
            rmse,
            cov_group,
            rmse_group,
            z_stats_all,
            z_mean,
            z_var,
            z_reject_5,
        ) = empirical_coverage_rate(
            key,
            R=R,
            T=T,
            p=p,
            q=q,
            params_dgp=params_dgp,
            params_scale=params_scale,
            dist=dist,
            alpha=alpha,
            algorithm=algorithm,
            USE_RAY=USE_RAY,
            MAX_CPUS_PER_TASK=MAX_CPUS_PER_TASK,
        )

        finish_time = datetime.now()
        finish_ts = finish_time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = (finish_time - start_time).total_seconds() / 60
        print(f"✅ Finished at {finish_ts} (elapsed {elapsed:.2f} min)")

        param_names = []
        param_names.append("eta_a")
        param_names.extend([f"gamma_a[{i}]" for i in range(p)])
        param_names.extend([f"beta[{i}]" for i in range(p)])
        param_names.extend([f"sigma2[{i}]" for i in range(p)])

        records = []
        for i, name in enumerate(param_names):
            records.append({
                "R": R,
                "p": p,
                "q": q,
                "dist": dist,
                "seed": seed,
                "T": T,
                "alpha": alpha,
                "nominal": 1 - alpha,
                "param_group": name.split("[")[0],
                "param": name,
                "coverage": float(coverage[i]),
                "rmse": float(rmse[i]),
                "avg_len": float(avg_len[i]),
                "start_time": start_ts,
                "finish_time": finish_ts,
                "elapsed_min": round(elapsed, 2),
            })

        df_results = pd.DataFrame(records)
        df_results.to_csv(out_path, mode="a", index=False, header=not header_written)
        header_written = True

        print(f"✔ Results appended for T={T}, alpha={alpha}")

    print(f"Results written to: {out_path}")
    summarize_empirical_coverage(str(out_path), latex=False)
