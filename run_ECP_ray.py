jax_cpu = True
if jax_cpu:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = ""   # hide all GPUs
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # optional, prevent GPU mem alloc

import jax
import jax.numpy as jnp
from jax.random import uniform
from tqdm import trange
import ray

from dgp import simulate_VAR_attention, flatten_params
from fit import estimate


# ===============================================================
# === Single replication task (Ray remote) ======================
# ===============================================================
@ray.remote(num_cpus=1)
def single_replication(subkey, params_true, theta_true, T, p, q, dist, alpha=0.05):
    """Single Monte Carlo replication (runs on a Ray worker)."""
    try:
        # Simulate one dataset
        Y, Z, _, _ = simulate_VAR_attention(subkey, params_true, T=T, p=p, q=q, dist=dist)

        # Transform and initialize
        sigma2_true = theta_true[2 + 3 * p : 2 + 4 * p]  # actual σ² values
        sigma2_tilde_true = jnp.log(jnp.expm1(sigma2_true))  # softplus inverse
        theta_true_reparam = jnp.concatenate([theta_true[: 2 + 3 * p], sigma2_tilde_true])

        # Estimate parameters and compute SE
        theta_init = theta_true_reparam + theta_true_reparam * 0.05 * jax.random.normal(subkey, theta_true_reparam.shape)
        result = estimate(Y, Z, subkey, theta_init, analytical_gradhess=True)

        theta_hat = jnp.asarray(result["theta"])
        se = jnp.asarray(result["se"])  # standard errors from sandwich variance

        # Compute lower/upper based on chosen alpha
        z_crit = jax.scipy.stats.norm.ppf(1 - alpha / 2.0)
        lower = theta_hat - z_crit * se
        upper = theta_hat + z_crit * se

        inside = (theta_true >= lower) & (theta_true <= upper)
        ci_length = upper - lower
        return inside, ci_length, theta_hat

    except Exception as e:
        print(e)
        # Return NaNs on failure to avoid breaking aggregation
        d = theta_true.shape[0]
        return jnp.zeros(d, dtype=bool), jnp.full(d, jnp.nan), jnp.full(d, jnp.nan)



# ===============================================================
# === Main Monte Carlo driver ===================================
# ===============================================================
def empirical_coverage_rate(R=100, T=500, p=10, q=10, dist="normal", alpha=0.05, seed=0):
    """
    Monte Carlo simulation to estimate empirical coverage probability,
    average CI length, and RMSE for QMLE-based confidence intervals.
    Now parallelized with Ray for multi-core scaling.
    """
    # Initialize Ray (auto-detects cores; safe to call multiple times)
    ray.init(ignore_reinit_error=True, include_dashboard=False)

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 7)

    # ===============================================================
    # === 1. Print DGP setup summary ================================
    # ===============================================================
    print("=" * 60)
    print("QMLE Empirical Coverage Simulation with RMSE (Ray parallelized)")
    print("=" * 60)
    print(f"Distribution   : {dist}")
    print(f"Sample size    : {T}")
    print(f"Response dim p : {p}")
    print(f"Feature dim q  : {q}")
    print(f"Replications R : {R}")
    print(f"Random seed    : {seed}")
    print("=" * 60, "\n")

    # ===============================================================
    # === 2. True parameter vector from DGP =========================
    # ===============================================================
    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    params_true = dict(
        eta_a=0.5,
        eta_b=1.0,
        gamma_a=uniform(keys[1], (p,), minval=-0.2, maxval=0.2),
        gamma_b=uniform(keys[2], (p,), minval=-0.2, maxval=0.2),
        beta=uniform(keys[3], (p,), minval=-0.2, maxval=0.2),
        sigma2=uniform(keys[4], (p,), minval=0.25, maxval=1.0),
    )
    theta_true = flatten_params(params_true, order)
    # d = theta_true.shape[0]

    # ===============================================================
    # === 3. Parallel execution with Ray ============================
    # ===============================================================
    subkeys = jax.random.split(key, R)

    # Submit tasks to Ray cluster/workers
    futures = [
        single_replication.remote(subkeys[r], params_true, theta_true, T, p, q, dist, alpha)
        for r in range(R)
    ]

    # Progress bar while waiting for Ray tasks
    results = []
    for result in trange(R, desc="Monte Carlo replications", ncols=100):
        results.append(ray.get(futures[result]))

    inside, ci_length, theta_hat_all = zip(*results)
    inside = jnp.stack(inside)
    ci_length = jnp.stack(ci_length)
    theta_hat_all = jnp.stack(theta_hat_all)

    # ===============================================================
    # === 4. Summarize results =====================================
    # ===============================================================
    valid_mask = jnp.isfinite(theta_hat_all).all(axis=1)
    inside = inside[valid_mask]
    ci_length = ci_length[valid_mask]
    theta_hat_all = theta_hat_all[valid_mask]

    coverage = inside.mean(axis=0)
    avg_length = ci_length.mean(axis=0)
    rmse = jnp.sqrt(jnp.mean((theta_hat_all - theta_true) ** 2, axis=0))

    # Group indices for summary
    idx = {
        "eta_a": slice(0, 1),
        "eta_b": slice(1, 2),
        "gamma_a": slice(2, 2 + p),
        "gamma_b": slice(2 + p, 2 + 2 * p),
        "beta": slice(2 + 2 * p, 2 + 3 * p),
        "sigma2": slice(2 + 3 * p, 2 + 4 * p),
    }
    coverage_by_group = {name: float(coverage[s].mean()) for name, s in idx.items()}
    rmse_by_group = {name: float(rmse[s].mean()) for name, s in idx.items()}

    # ===============================================================
    # === 5. Print summary =========================================
    # ===============================================================
    print("\n=== Empirical Coverage and RMSE Summary ===")
    print(f"Overall average coverage: {coverage.mean():.3f}\n")
    for name in idx.keys():
        print(f"{name:<8s}: coverage={coverage_by_group[name]:.3f}, RMSE={rmse_by_group[name]:.4f}")

    print("\nAverage CI length (by parameter):")
    print(jnp.round(avg_length, 3))
    print("\nRMSE (by parameter):")
    print(jnp.round(rmse, 4))

    ray.shutdown()  # Clean up Ray workers

    return coverage, avg_length, rmse, coverage_by_group, rmse_by_group


if __name__ == "__main__":
    import random
    import pandas as pd
    import numpy as np

    # Settings
    R = 1000
    p, q = 10, 10
    dist = "normal"
    seed = random.randint(0, 1000)

    T_list = [100, 300, 500, 1000, 5000]
    alpha_list = [0.10, 0.05]    # for 90% and 95% coverage

    out_path = "results_coverage.csv"

    records = []

    for T in T_list:
        for alpha in alpha_list:
            print("\n" + "=" * 70)
            print(f"Running Monte Carlo:  T={T},  nominal={(1-alpha):.2%}")
            print("=" * 70)

            coverage, avg_len, rmse, cov_group, rmse_group = empirical_coverage_rate(
                R=R, T=T, p=p, q=q, dist=dist, seed=seed, alpha=alpha
            )

            # flatten results for each parameter group
            for name in cov_group.keys():
                records.append({
                    "T": T,
                    "alpha": alpha,
                    "nominal": 1 - alpha,
                    "param": name,
                    "coverage": cov_group[name],
                    "rmse": rmse_group[name],
                    "avg_len": float(np.mean(avg_len))  # overall average length
                })

    # Convert to DataFrame and save
    df_results = pd.DataFrame(records)
    df_results.to_csv(out_path, index=False)
    print(f"\nAll results saved to: {out_path}")
    # print(df_results.head())


    # --- Summary by T and nominal level ---
    summary = (
        df_results
        .groupby(["T", "nominal"], as_index=False)
        .agg(
            mean_coverage=("coverage", "mean"),
            mean_rmse=("rmse", "mean"),
            mean_len=("avg_len", "mean")
        )
        .sort_values(["T", "nominal"])
    )

    print("\n=== Average Coverage, RMSE, and CI Length by T and Confidence Level ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:8.3f}"))

    # --- Optional: show parameter-level changes for each T ---
    param_summary = (
        df_results
        .pivot_table(index=["param"], columns=["T", "nominal"], values="coverage")
        .round(3)
    )

    print("\n=== Parameter-Level Coverage by T and Confidence Level ===")
    print(param_summary)
