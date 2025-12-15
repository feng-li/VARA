# jax_cpu = False
# if jax_cpu:
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # optional, prevent GPU mem alloc

import jax
import jax.numpy as jnp
from jax.random import uniform
from tqdm import trange
import ray
import logging

import pandas as pd

from dgp import simulate_VAR_attention, flatten_params
from fit import estimate
# from mle_jaxopt import mle_bfgs


def single_replication(subkey, params_true, T, p, q, params_scale, dist, alpha=0.05,
                       algorithm="analytical_gradhess", update_idx=slice(None),
                       max_tries=10):
    """Single Monte Carlo replication (runs on a Ray worker).

    Re-runs simulation & estimation up to `max_tries` times until convergence
    status == 0. On complete failure, returns NaNs.
    """

    # ---- Pre-compute true parameters (independent of attempts) ----
    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    theta_true, param_slices = flatten_params(params_true, order)

    # sigma2 reparameterization
    sigma2_true = theta_true[2 + 3 * p : 2 + 4 * p]              # actual σ² values
    sigma2_tilde_true = jnp.log(jnp.expm1(sigma2_true))          # softplus inverse
    theta_true_reparam = jnp.concatenate(
        [theta_true[: 2 + 3 * p], sigma2_tilde_true]
    )

    # Handle update_idx: if slice, use all parameters; otherwise convert to array
    if isinstance(update_idx, slice):
        update_idx_array = jnp.arange(theta_true_reparam.size)
    else:
        update_idx_array = jnp.array(update_idx)

    # ---- Default "failure" outputs ----
    d = theta_true.shape[0]
    last_inside = jnp.zeros(d, dtype=bool)
    last_ci_length = jnp.full(d, jnp.nan)
    last_theta_hat = jnp.full(d, jnp.nan)
    last_status = -1  # non-zero meaning "no success"

    # ---- Try up to max_tries times ----
    for attempt in range(max_tries):
        # Make attempt-specific keys so each try is different but reproducible
        attempt_key = jax.random.fold_in(subkey, attempt)
        noise_key = jax.random.fold_in(subkey, attempt + max_tries)

        # 1. Simulate one dataset
        Y, Z, _, _ = simulate_VAR_attention(
            attempt_key, params_true, T=T, p=p, q=q,
            dist=dist, params_scale=params_scale
        )

        # 2. Build initial values (perturb true values on update_idx)
        theta_init = theta_true_reparam
        perturb_scales = {"eta_a": 0.3,
                          "eta_b": 0.05,
                          "gamma_a": 0.8,
                          "gamma_b": 0.8,
                          "beta": 0.2,
                          "sigma2": 0.10
                          }

        for name, sl in param_slices.items():
            key, noise_key = jax.random.split(noise_key)
            scale = perturb_scales[name]
            idx = jnp.intersect1d(jnp.arange(len(theta_true_reparam))[sl], update_idx_array)
            if idx.size > 0:
                noise = 1.0 + scale * jax.random.normal(key, shape=(len(idx),))
                theta_init = theta_init.at[idx].set(theta_true_reparam[idx] * noise)


        # 3. Estimation
        result, convergence = estimate(
            Y, Z, attempt_key, theta_init,
            params_scale=params_scale,
            algorithm=algorithm,
            update_idx=update_idx_array,
        )

        theta_hat = jnp.asarray(result["theta"])
        se = jnp.asarray(result["se"])  # standard errors from sandwich variance

        # 4. Confidence intervals
        z_crit = jax.scipy.stats.norm.ppf(1 - alpha / 2.0)
        lower = theta_hat - z_crit * se
        upper = theta_hat + z_crit * se

        inside = (theta_true >= lower) & (theta_true <= upper)
        ci_length = upper - lower

        # Extreme results checking. If the value is too far away from the 99% interval, indicating bad DGP
        x_crit = jax.scipy.stats.norm.ppf(1 - 0.01 / 2.0)
        x_idx = jnp.array([0, 1])
        x_lower = theta_hat[x_idx] - x_crit * se[x_idx]
        x_upper = theta_hat[x_idx] + x_crit * se[x_idx]
        x_inside = (theta_true[x_idx] >= x_lower) & (theta_true[x_idx] <= x_upper)

        # Make sure optimization gives converging code.
        status = int(convergence["status"])
        # If converged and good optimization results, return immediately
        if status == 0:  # and x_inside.all():
            print(f"Successful for simulating proper data after {attempt+1} attempts.")
            return inside, ci_length, theta_hat, status
        # else:
            # print(f"{attempt+1} failed attempts to simulate proper data. Retrying.")


    # If all attempts fail to reach status == 0, return last attempt (or NaNs)
    return last_inside, last_ci_length, last_theta_hat, last_status


# JAX JIT wrapper
single_replication_jit = jax.jit(single_replication,
                                 static_argnames=["params_scale", "dist", "algorithm"])

# Ray wrapper (wrap the same function)
# @ray.remote(num_gpus=0.01)
@ray.remote(num_cpus=1)
def single_replication_ray(*args, **kwargs):
    try:
        return single_replication(*args, **kwargs)
    except Exception as e:
        print(f"[Ray worker error] {e}")
        theta_true = kwargs.get("theta_true", None)
        d = len(theta_true) if theta_true is not None else 1
        return jnp.zeros(d, dtype=bool), jnp.full(d, jnp.nan), jnp.full(d, jnp.nan), jnp.zeros(d, jnp.nan)

# ===============================================================
# === Main Monte Carlo driver ===================================
# ===============================================================
def empirical_coverage_rate(key, R, T, p, q, params_dgp, params_scale,
                            dist="normal", alpha=0.05,
                            algorithm="analytical_gradhess",
                            USE_RAY=False, MAX_CPUS_PER_TASK=1):
    """
    Monte Carlo simulation to estimate empirical coverage probability,
    average CI length, and RMSE for QMLE-based confidence intervals.
    Now parallelized with ray or jax.jit.
    """

    # ===============================================================
    # === 1. Print DGP setup summary ================================
    # ===============================================================
    print("\n" + "=" * 70 + "\nEmpirical Simulation Setup\n" + "=" * 70)
    print(f"Algorithm      : {algorithm}")
    print(f"Distribution   : {dist}")
    print(f"Sample size  T : {T}")
    print(f"Response dim p : {p}")
    print(f"Feature dim  q : {q}")
    print(f"Replications R : {R}")
    print(f"Random seed    : {seed}")
    print("=" * 70, "\n")

    # ===============================================================
    # === 2. Parallel execution with Ray or JAX =====================
    # ===============================================================
    subkeys = jax.random.split(key, R)
    pkeys = jax.random.split(key, 5)
    params_true = dict(
        eta_a=params_dgp['eta_a'],
        eta_b=params_dgp['eta_b'],
        gamma_a=uniform(pkeys[1], (p,), minval=params_dgp['gamma_a']['minval'], maxval=params_dgp['gamma_a']['maxval']),
        gamma_b=uniform(pkeys[2], (p,), minval=params_dgp['gamma_b']['minval'], maxval=params_dgp['gamma_b']['maxval']),
        beta=uniform(pkeys[3], (p,), minval=params_dgp['beta']['minval'], maxval=params_dgp['beta']['maxval']),
        sigma2=uniform(pkeys[4], (p,), minval=params_dgp['sigma2']['minval'], maxval=params_dgp['sigma2']['maxval'])
    )

    # Submit tasks to Ray cluster/workers
    if USE_RAY:
        print("🔹 Running with Ray parallelization")
        print("JAX devices:", jax.devices())
        print(f"\nNUM_CPUS_PER_TASK:  {MAX_CPUS_PER_TASK}\n")

        ray.init(ignore_reinit_error=True,
                 include_dashboard=False,
                 # log_to_driver=False,
                 # logging_level=logging.ERROR
                 )

        # MODIFY THIS TO FIT THE SIMULATION INTO MEMORY

        futures = [
            single_replication_ray.options(num_cpus=MAX_CPUS_PER_TASK).remote(
                subkeys[r], params_true, T, p, q, params_scale, dist, alpha, algorithm, update_idx)
            for r in range(R)
        ]

        results = []
        for result in trange(R, desc="Monte Carlo replications", ncols=100):
            results.append(ray.get(futures[result]))

        ray.shutdown()
    else:
        print("🔹 Running locally with JAX")

        # You can also switch this to jax.jit/vmap if memory allows
        results = [single_replication(
            subkeys[r], params_true, T, p, q, params_scale, dist, alpha, algorithm, update_idx)
                   for r in trange(R, desc="Monte Carlo replications", ncols=100)]

    inside, ci_length, theta_hat_all, status_all = zip(*results)
    inside = jnp.stack(inside)
    ci_length = jnp.stack(ci_length)
    theta_hat_all = jnp.stack(theta_hat_all)
    status_all = jnp.stack(status_all)

    # ===============================================================
    # === 3. Summarize results =====================================
    # ===============================================================
    valid_mask = jnp.isfinite(theta_hat_all).all(axis=1)

    inside = inside[valid_mask]
    ci_length = ci_length[valid_mask]
    theta_hat_all = theta_hat_all[valid_mask]

    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    theta_true, idx = flatten_params(params_true, order)

    # take the mean along an axis (e.g., axis=0) while skipping NaNs,
    coverage = jnp.nanmean(inside, axis=0)
    avg_length = jnp.nanmean(ci_length, axis=0)
    rmse = jnp.sqrt(jnp.nanmean((theta_hat_all - theta_true) ** 2, axis=0))

    # Group indices for summary
    # idx = {
    #     "eta_a": slice(0, 1),
    #     "eta_b": slice(1, 2),
    #     "gamma_a": slice(2, 2 + p),
    #     "gamma_b": slice(2 + p, 2 + 2 * p),
    #     "beta": slice(2 + 2 * p, 2 + 3 * p),
    #     "sigma2": slice(2 + 3 * p, 2 + 4 * p),
    # }
    coverage_by_group = {name: float(jnp.nanmean(coverage[s])) for name, s in idx.items()}
    rmse_by_group = {name: float(jnp.nanmean(rmse[s])) for name, s in idx.items()}

    # ===============================================================
    # === 4. Print summary =========================================
    # ===============================================================
    print("\n=== Empirical Coverage and RMSE Summary ===")
    print(f"nominal = {(1-alpha):.2%}")
    print(f"Overall average coverage: {jnp.nanmean(coverage):.4f}\n")
    for name in idx.keys():
        print(f"{name:<8s}: coverage={coverage_by_group[name]:.3f}, RMSE={rmse_by_group[name]:.4f}")

    print("\nAverage CI length (by parameter):")
    print(jnp.round(avg_length, 3))
    print("\nRMSE (by parameter):")
    print(jnp.round(rmse, 4))

    return coverage, avg_length, rmse, coverage_by_group, rmse_by_group


if __name__ == "__main__":
    import itertools
    from datetime import datetime

    from summarize import summarize_empirical_coverage

    # ===============================================================
    # === 0. Parallelization     ====================================
    # ===============================================================
    # jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(threshold=10_000, linewidth=200, suppress=True, precision=6)

    # Settings
    # USE_RAY = False
    USE_RAY = True

    # ===============================================================
    # === 1. Simulation from DGP ====================================
    # ===============================================================
    # seed = random.randint(0, 1000)
    seed = 2025
    key = jax.random.PRNGKey(seed)

    R_list = [500]
    p_list = [300, 400, 500]
    q_list = [50]
    dist_list = ["normal"]

    # T_list = [50, 100, 200, 500, 1000]
    # T_list = [200]
    T_list = [200]
    alpha_list = [0.1]    # for 90% and 95% coverage

    algorithm = "yule-walker"    # bfgs, analytical_gradhess, auto_gradhess, yule-walker
    # algorithm = "auto_gradhess"    # bfgs, analytical_gradhess, auto_gradhess
    # algorithm = "analytical_gradhess"    # bfgs, analytical_gradhess, auto_gradhess

    out_path = f"results_coverage_@{datetime.now().strftime("%Y-%m-%d_%H.%M.%S")}.csv"

    # ===============================================================
    # === 2. True parameter vector from DGP =========================
    # ===============================================================
    params_dgp = dict(
        eta_a=0.2,
        eta_b=0.2,
        gamma_a=dict(minval=-0.2, maxval=0.2),
        gamma_b=dict(minval=-0.2, maxval=0.2),
        beta=dict(minval=-0.2, maxval=0.2),
        sigma2=dict(minval=0.25, maxval=1.0))

    params_scale = dict(
        eta_a=1,
        eta_b=1)


    # ===============================================================
    # === 3. Loop over all simulation combinations ==================
    # ===============================================================
    # Cartesian product of all (T, alpha) combinations
    header_written = False
    for dist, R, T, p, q, alpha in itertools.product(
            dist_list, R_list, T_list, p_list, q_list, alpha_list):
        print("\n" + "=" * 70 + f"\nRunning Monte Carlo:  T={T},  nominal={(1-alpha):.2%}\n"+"=" * 70)

        # seed = random.randint(0, 10000)
        MAX_CPUS_PER_TASK = int(max(1, 2*jnp.sqrt(104 * p / max(p_list)))) # Use this only when heavy Hessian is used.
        # MAX_CPUS_PER_TASK = 2

        update_idx = [0, 1]
        # update_idx = range(0, 3*p+2)
        # update_idx = slice(None)  # Full updating

        # --- Record start time ---
        start_time = datetime.now()
        start_ts = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"⏱  Started at {start_ts}")

        # Ray run simulation
        coverage, avg_len, rmse, cov_group, rmse_group = empirical_coverage_rate(
            key, R=R, T=T, p=p, q=q, params_dgp=params_dgp, params_scale=params_scale,
            dist=dist, alpha=alpha, algorithm=algorithm,
            USE_RAY=USE_RAY, MAX_CPUS_PER_TASK=MAX_CPUS_PER_TASK)

        # --- Record finish time ---
        finish_time = datetime.now()
        finish_ts = finish_time.strftime("%Y-%m-%d %H:%M:%S")
        elapsed = (finish_time - start_time).total_seconds() / 60
        print(f"✅ Finished at {finish_ts} (elapsed {elapsed:.2f} min)")

        # Collect records for this run. Report each parameter element
        param_names = []
        param_names.append("eta_a")
        param_names.append("eta_b")
        param_names.extend([f"gamma_a[{i}]" for i in range(p)])
        param_names.extend([f"gamma_b[{i}]" for i in range(p)])
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
                "elapsed_min": round(elapsed, 2)
            })

        df_results = pd.DataFrame(records)

        # Append to CSV
        df_results.to_csv(out_path, mode="a", index=False, header=not header_written)
        header_written = True

        print(f"✔ Results appended for T={T}, alpha={alpha}")

    # --- Summary by T and nominal level ---
    summarize_empirical_coverage(out_path, latex=False)
