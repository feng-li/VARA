import jax
import jax.numpy as jnp
from jax.random import uniform
from tqdm import trange

from dgp import simulate_VAR_attention, flatten_params
from fit import estimate


def empirical_coverage_rate(R=100, T=500, p=10, q=10, dist="normal", seed=0):
    """
    Monte Carlo simulation to estimate empirical coverage probability,
    average CI length, and RMSE for QMLE-based confidence intervals.

    Parameters
    ----------
    R : int
        Number of replications.
    T : int
        Sample size per replication.
    p : int
        Dimension of the VAR process.
    q : int
        Dimension of the feature matrix.
    dist : str
        Innovation distribution ("normal", "t5", etc.).
    seed : int
        Random seed.

    Returns
    -------
    coverage : jnp.ndarray
        Coverage probability for each individual parameter.
    avg_length : jnp.ndarray
        Average CI length for each parameter.
    rmse : jnp.ndarray
        RMSE for each parameter.
    coverage_by_group : dict
        Mean coverage summarized by parameter group.
    """
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 7)

    # ===============================================================
    # === 1. Print DGP setup summary ================================
    # ===============================================================
    print("=" * 60)
    print("QMLE Empirical Coverage Simulation with RMSE")
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
    d = theta_true.shape[0]

    # Storage arrays
    inside = jnp.zeros((R, d))
    ci_length = jnp.zeros((R, d))
    theta_hat_all = jnp.zeros((R, d))

    # ===============================================================
    # === 3. Monte Carlo replications ===============================
    # ===============================================================
    for r in trange(R, desc="Monte Carlo replications", ncols=100):
        subkey = jax.random.fold_in(key, r)
        try:
            # Simulate new dataset
            Y, Z, _, _ = simulate_VAR_attention(subkey, params_true, T=T, p=p, q=q, dist=dist)

            # Transform and initialize
            log_sigma2_true = jnp.log(theta_true[2 + 3 * p : 2 + 4 * p])
            theta_true_reparam = jnp.concatenate([theta_true[: 2 + 3 * p], log_sigma2_true])
            theta_init = theta_true_reparam + 0.01 * jax.random.normal(subkey, theta_true_reparam.shape)

            # Estimate parameters and CIs
            result = estimate(Y, Z, subkey, theta_init, verbose=False)
            lower = jnp.asarray(result["lower"])
            upper = jnp.asarray(result["upper"])
            theta_hat = jnp.asarray(result["theta"])

            # Record results
            inside = inside.at[r].set((theta_true >= lower) & (theta_true <= upper))
            ci_length = ci_length.at[r].set(upper - lower)
            theta_hat_all = theta_hat_all.at[r].set(theta_hat)

        except Exception as e:
            print(f"[Replication {r}] Estimation failed: {e}")
            continue

    # ===============================================================
    # === 4. Summarize results =====================================
    # ===============================================================
    coverage = inside.mean(axis=0)
    avg_length = ci_length.mean(axis=0)

    # RMSE for each parameter
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

    print("\nIndividual parameter coverage:")
    print(jnp.round(coverage, 3))
    print("\nAverage CI length (by parameter):")
    print(jnp.round(avg_length, 3))
    print("\nRMSE (by parameter):")
    print(jnp.round(rmse, 4))

    return coverage, avg_length, rmse, coverage_by_group, rmse_by_group


if __name__ == "__main__":
    # Example short run
    empirical_coverage_rate(R=1000, T=300, p=10, q=10, dist="normal", seed=42)

