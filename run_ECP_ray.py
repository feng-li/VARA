# jax_cpu = False
# if jax_cpu:
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""   # hide all GPUs
#     os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # optional, prevent GPU mem alloc

import jax
import jax.numpy as jnp
from jax.random import uniform
from tqdm import trange
import ray

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
    theta_true = flatten_params(params_true, order)

    # sigma2 reparameterisation
    sigma2_true = theta_true[2 + 3 * p : 2 + 4 * p]              # actual σ² values
    sigma2_tilde_true = jnp.log(jnp.expm1(sigma2_true))          # softplus inverse
    theta_true_reparam = jnp.concatenate(
        [theta_true[: 2 + 3 * p], sigma2_tilde_true]
    )

    d = theta_true.shape[0]

    # Handle update_idx: if slice, use all parameters; otherwise convert to array
    if isinstance(update_idx, slice):
        update_idx_array = jnp.arange(theta_true_reparam.size)
    else:
        update_idx_array = jnp.array(update_idx)

    # ---- Default "failure" outputs ----
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
        theta_init_update = theta_true_reparam[update_idx_array]
        perturb = 1 + 0.1 * jax.random.normal(noise_key, theta_init_update.shape)
        theta_init = theta_init.at[update_idx_array].set(theta_init_update * perturb)

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

        status = int(convergence["status"])
        # last_inside, last_ci_length, last_theta_hat, last_status = (
        #     inside, ci_length, theta_hat, status
        # )

        # If converged, return immediately
        if status == 0:
            return inside, ci_length, theta_hat, status
        else:
            print(f"{attempt} attempts failed to simulate proper data under this condition. Retrying.")


    # If all attempts fail to reach status == 0, return last attempt (or NaNs)
    return last_inside, last_ci_length, last_theta_hat, last_status


# JAX JIT wrapper
single_replication_jit = jax.jit(single_replication,
                                 static_argnames=["params_scale", "dist", "algorithm"])

# Ray wrapper (wrap the same function)
@ray.remote
def single_replication_ray(*args, **kwargs):
    try:
        return single_replication(*args, **kwargs)
    except Exception as e:
        print(f"[Ray worker error] {e}")
        theta_true = kwargs.get("theta_true", None)
        d = len(theta_true) if theta_true is not None else 1
        return jnp.zeros(d, dtype=bool), jnp.full(d, jnp.nan), jnp.full(d, jnp.nan)

# ===============================================================
# === Main Monte Carlo driver ===================================
# ===============================================================
def empirical_coverage_rate(key, R, T, p, q, params_scale,
                            dist="normal", alpha=0.05,
                            algorithm="analytical_gradhess", USE_RAY=False):
    """
    Monte Carlo simulation to estimate empirical coverage probability,
    average CI length, and RMSE for QMLE-based confidence intervals.
    Now parallelized with ray or jax.jit.
    """

    # ===============================================================
    # === 1. Print DGP setup summary ================================
    # ===============================================================
    print("=" * 70)
    print("QMLE Empirical Coverage Simulation with RMSE (Ray parallelized)")
    print("=" * 70)
    print(f"Algorithm      : {algorithm}")
    print(f"Distribution   : {dist}")
    print(f"Sample size    : {T}")
    print(f"Response dim p : {p}")
    print(f"Feature dim q  : {q}")
    print(f"Replications R : {R}")
    print(f"Random seed    : {seed}")
    print("=" * 70, "\n")

    # ===============================================================
    # === 2. Parallel execution with Ray or JAX =====================
    # ===============================================================
    subkeys = jax.random.split(key, R)

    # Submit tasks to Ray cluster/workers
    if USE_RAY:
        print("🔹 Running with Ray parallelization")
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        futures = [
            single_replication_ray.remote(
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
    theta_true = flatten_params(params_true, order)

    # take the mean along an axis (e.g., axis=0) while skipping NaNs,
    coverage = jnp.nanmean(inside, axis=0)
    avg_length = jnp.nanmean(ci_length, axis=0)
    rmse = jnp.sqrt(jnp.nanmean((theta_hat_all - theta_true) ** 2, axis=0))

    # Group indices for summary
    idx = {
        "eta_a": slice(0, 1),
        "eta_b": slice(1, 2),
        "gamma_a": slice(2, 2 + p),
        "gamma_b": slice(2 + p, 2 + 2 * p),
        "beta": slice(2 + 2 * p, 2 + 3 * p),
        "sigma2": slice(2 + 3 * p, 2 + 4 * p),
    }
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

    # ray.shutdown()  # Clean up Ray workers

    return coverage, avg_length, rmse, coverage_by_group, rmse_by_group


def summarize_empirical_coverage(file_path: str):
    """
    Summarize empirical coverage results across all parameters, sample sizes (T),
    and nominal confidence levels.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing simulation results.
        Expected columns: ["T", "nominal", "param", "coverage", "rmse", "avg_len"]

    Returns
    -------
    dict
        Dictionary with summary tables:
        - summary_T_nominal
        - param_summary
        - grand_summary
    """
    # --- Load results ---
    df = pd.read_csv(file_path)

    # --- Convert numeric columns explicitly ---
    numeric_cols = ["T", "nominal", "coverage", "rmse", "avg_len"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with missing numeric values if any
    df = df.dropna(subset=numeric_cols)

    # --- Summary by T and nominal ---
    summary_T_nominal = (
        df.groupby(["T", "nominal"], as_index=False)
        .agg(
            mean_coverage=("coverage", "mean"),
            std_coverage=("coverage", "std"),
            mean_rmse=("rmse", "mean"),
            mean_len=("avg_len", "mean")
        )
        .sort_values(["T", "nominal"])
    )

    print("\n=== Average Coverage, RMSE, and CI Length by T and Confidence Level ===")
    print(summary_T_nominal.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    # --- Parameter-level coverage table ---
    param_summary = (
        df.pivot_table(
            index="param",
            columns=["T", "nominal"],
            values="coverage",
            aggfunc="mean"
        )
        .round(3)
    )

    print("\n=== Parameter-level Empirical Coverage (per T and Confidence Level) ===")
    print(param_summary)

    # --- Overall coverage by nominal level ---
    grand_summary = (
        df.groupby(["nominal"], as_index=False)
        .agg(overall_mean_coverage=("coverage", "mean"))
    )

    print("\n=== Overall Average Coverage across all T and Parameters ===")
    print(grand_summary.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    return {
        "summary_T_nominal": summary_T_nominal,
        "param_summary": param_summary,
        "grand_summary": grand_summary
    }



if __name__ == "__main__":
    import itertools
    from datetime import datetime
    import random

    # ===============================================================
    # === 0. Parallelization     ====================================
    # ===============================================================
    # jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(threshold=10_000, linewidth=200, suppress=True, precision=6)

    # Settings
    USE_RAY = True

    # ===============================================================
    # === 1. Simulation from DGP ====================================
    # ===============================================================
    seed = random.randint(0, 1000)
    key = jax.random.PRNGKey(seed)

    R = 1000
    p, q = 5, 5
    dist = "normal"

    T_list = [200, 500, 1000]
    alpha_list = [0.10, 0.05]    # for 90% and 95% coverage

    algorithm = "auto_gradhess"    # bfgs, analytical_gradhess, auto_gradhess
    # algorithm = "analytical_gradhess"    # bfgs, analytical_gradhess, auto_gradhess

    out_path = f"results_coverage_{datetime.now().strftime("%Y-%m-%d_%H.%M.%S")}.csv"

    # ===============================================================
    # === 2. True parameter vector from DGP =========================
    # ===============================================================
    keys = jax.random.split(key, 10)
    params_true = dict(
        eta_a=0.5,
        eta_b=1.0,
        gamma_a=uniform(keys[1], (p,), minval=-0.2, maxval=0.2),
        gamma_b=uniform(keys[2], (p,), minval=-0.2, maxval=0.2),
        beta=uniform(keys[3], (p,), minval=-0.2, maxval=0.2),
        sigma2=uniform(keys[4], (p,), minval=0.25, maxval=1.0),
    )

    update_idx = [0, 1]
    # update_idx = slice(None)  # Full updating

    params_scale = dict(
        eta_a=1,
        eta_b=10)


    # ===============================================================
    # === 3. Loop over all simulation combinations ==================
    # ===============================================================
    # Cartesian product of all (T, alpha) combinations
    header_written = False
    for T, alpha in itertools.product(T_list, alpha_list):
        print("\n" + "=" * 70)
        print(f"Running Monte Carlo:  T={T},  nominal={(1-alpha):.2%}")
        print("=" * 70)

        # seed = random.randint(0, 10000)

        # --- Record start time ---
        start_time = datetime.now()
        start_ts = start_time.strftime("%Y-%m-%d %H:%M:%S")
        print(f"⏱  Started at {start_ts}")

        # Ray run simulation
        coverage, avg_len, rmse, cov_group, rmse_group = empirical_coverage_rate(
            key, R=R, T=T, p=p, q=q, params_scale=params_scale, dist=dist, alpha=alpha,
            algorithm=algorithm, USE_RAY=USE_RAY)

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
    summarize_empirical_coverage(out_path)
