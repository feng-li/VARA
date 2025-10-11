import jax
import jax.numpy as jnp
from jax.random import multivariate_normal, uniform, normal
import matplotlib.pyplot as plt


from loglik_attn import build_attention


# ---------- Simulation function ----------
def simulate_VAR_attention(key, T=200, p=100, q=10, dist="normal"):
    """
    Simulate data from the VAR attention model.

    key: JAX PRNGKey
    T: sample size
    p: dimension of time series
    q: dimension of feature vector
    dist: 'normal' or 't5' for noise distribution

    Returns: (Y, params, Z, W_a, W_b)
    """
    keys = jax.random.split(key, 7)

    # Step 1: Generate feature vectors Z ~ N(0, Σ_z)
    idx = jnp.arange(q)
    Sigma_z = 0.5 ** jnp.abs(idx[:, None] - idx[None, :])
    Z = multivariate_normal(keys[0], jnp.zeros(q), Sigma_z, (p,))

    # Step 2: Construct W_a, W_b
    W_a = build_attention(Z, eta=0.5)
    W_b = build_attention(Z, eta=1.0)

    # Step 3: Draw parameters
    gamma_a = uniform(keys[1], (p,), minval=-0.2, maxval=0.2)
    gamma_b = uniform(keys[2], (p,), minval=-0.2, maxval=0.2)
    beta = uniform(keys[3], (p,), minval=-0.2, maxval=0.2)
    sigma2 = uniform(keys[4], (p,), minval=0.25, maxval=1.0)

    # Step 4: Random noise
    if dist == "normal":
        eps_raw = normal(keys[5], (T, p))
    elif dist == "t5":
        k1, k2 = jax.random.split(keys[5])
        z = normal(k1, (T, p))
        u = jax.random.chisquare(k2, df=5, shape=(T, p))
        eps_raw = z / jnp.sqrt(u / 5.0)
        eps_raw = eps_raw * jnp.sqrt(3.0 / 5.0)  # normalize variance to 1
    else:
        raise ValueError("dist must be 'normal' or 't5'.")

    # Scale to Σ_e = diag(sigma2)
    eps = eps_raw * jnp.sqrt(sigma2)

    # Step 5: Simulation recursion
    S = jnp.eye(p) - jnp.diag(gamma_b) @ W_b
    G = jnp.diag(beta) + jnp.diag(gamma_a) @ W_a
    S_inv = jnp.linalg.inv(S)

    Y = jnp.zeros((T, p))
    for t in range(1, T):
        Y = Y.at[t].set(S_inv @ (G @ Y[t - 1] + eps[t]))

    params = dict(beta=beta, gamma_a=gamma_a, gamma_b=gamma_b, sigma2=sigma2)
    return Y, params, Z, W_a, W_b


# ---------- Example usage ----------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)

    # Simulate dataset with p >= 100
    Y, params, Z, W_a, W_b = simulate_VAR_attention(
        key, T=200, p=100, q=10, dist="t5"
    )

    print("Simulated Y shape:", Y.shape)
    print("Sample beta:", params["beta"][:5], "...")
    print("Sample gamma_a:", params["gamma_a"][:5], "...")
    print("Sample gamma_b:", params["gamma_b"][:5], "...")
    print("Sample sigma2:", params["sigma2"][:5], "...")

    # ---------- Visualization ----------
    fig, axes = plt.subplots(10, 10, figsize=(20, 20), sharex=True)
    axes = axes.flatten()

    print("Plotting first 100 series")
    for i in range(100):  # plot first 100 series
        axes[i].plot(Y[:, i], lw=1)
        axes[i].set_title(f"Series {i+1}", fontsize=8)
        axes[i].tick_params(axis='both', which='major', labelsize=6)

    plt.tight_layout()
    plt.show()
