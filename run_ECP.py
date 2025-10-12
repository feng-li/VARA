import jax
import jax.numpy as jnp
from tqdm import trange

from dgp import simulate_VAR_attention, flatten_params
from fit import estimate

def empirical_coverage_rate(R=100, T=500, p=3, q=2, dist="normal", seed=0):
    """
    Monte Carlo simulation to estimate empirical coverage probability
    of QMLE-based confidence intervals, with a DGP setup summary.

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
    coverage_by_group : dict
        Mean coverage summarized by parameter group.
    """
    key = jax.random.PRNGKey(seed)

    # ===============================================================
    # === 1. Print DGP setup summary ================================
    # ===============================================================
    print("=" * 60)
    print("QMLE Empirical Coverage Simulation")
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
    Y0, params_true, Z, W_a, W_b = simulate_VAR_attention(key, T=T, p=p, q=q, dist=dist)
    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    theta_true = flatten_params(params_true, order)
    log_sigma2_true = jnp.log(theta_true[2 + 3 * p : 2 + 4 * p])
    theta_true_reparam = jnp.concatenate([theta_true[: 2 + 3 * p], log_sigma2_true])
    d = theta_true.shape[0]

    print("True parameter vector (flattened):")
    print(jnp.round(theta_true, 4))
    print()

    # ===============================================================
    # === 3. Storage for results ===================================
    # ===============================================================
    inside = jnp.zeros((R, d))
    ci_length = jnp.zeros((R, d))

    # ===============================================================
    # === 4. Monte Carlo replications ===============================
    # ===============================================================
    for r in trange(R, desc="Monte Carlo replications", ncols=100):
        subkey = jax.random.fold_in(key, r)
        try:
            # Simulate new dataset
            Y, _, Z, _, _ = simulate_VAR_attention(subkey, T=T, p=p, q=q, dist=dist)

            # Slightly perturbed initialization
            theta_init = theta_true_reparam + 0.1 * jax.random.normal(subkey, theta_true_reparam.shape)

            # Estimate parameters and get confidence intervals
            result = estimate(Y, Z, subkey, theta_init, verbose=False)
            lower = jnp.asarray(result["lower"])
            upper = jnp.asarray(result["upper"])

            # Record coverage & CI length
            inside = inside.at[r].set((theta_true >= lower) & (theta_true <= upper))
            ci_length = ci_length.at[r].set(upper - lower)

        except Exception as e:
            print(f"[Replication {r}] Estimation failed: {e}")
            continue

    # ===============================================================
    # === 5. Summarize results =====================================
    # ===============================================================
    coverage = inside.mean(axis=0)
    avg_length = ci_length.mean(axis=0)

    # Parameter index slices for grouping
    idx = {
        "eta_a": slice(0, 1),
        "eta_b": slice(1, 2),
        "gamma_a": slice(2, 2 + p),
        "gamma_b": slice(2 + p, 2 + 2 * p),
        "beta": slice(2 + 2 * p, 2 + 3 * p),
        "sigma2": slice(2 + 3 * p, 2 + 4 * p),
    }
    coverage_by_group = {name: float(coverage[s].mean()) for name, s in idx.items()}

    # ===============================================================
    # === 6. Print summary =========================================
    # ===============================================================
    print("\n=== Empirical Coverage Summary ===")
    print(f"Overall average coverage: {coverage.mean():.3f}\n")
    for name, val in coverage_by_group.items():
        print(f"{name:<8s}: {val:.3f}")
    print("\nIndividual parameter coverage:")
    print(jnp.round(coverage, 3))
    print("\nAverage CI length (by parameter):")
    print(jnp.round(avg_length, 3))

    return coverage, avg_length, coverage_by_group


if __name__ == "__main__":
    # Example short run for testing
    empirical_coverage_rate(R=100, T=300, p=3, q=2, dist="normal", seed=0)
