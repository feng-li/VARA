import jax
import jax.numpy as jnp
from jax.random import multivariate_normal, normal
import matplotlib.pyplot as plt

from loglik_attn import build_attention


def simulate_VAR_attention(key, params, T=200, p=10, q=10, dist="normal", params_scale=dict(eta_a=1, eta_b=10)):
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

    # Step 1: Generate feature vectors Z ~ N(0, Σ_z) Use less-correlated features Z. Highly correlated Z makes the
    # attention mapping degenerate.
    idx = jnp.arange(q)
    Sigma_z = 0.1 ** jnp.abs(idx[:, None] - idx[None, :])

    Z = multivariate_normal(keys[0], jnp.zeros(q), Sigma_z, (p,))

    # Step 2: Draw parameters
    eta_a = params["eta_a"]
    eta_b = params["eta_b"]
    gamma_a = params["gamma_a"]
    gamma_b = params["gamma_b"]
    beta = params["beta"]
    sigma2 = params["sigma2"]


    # Step 3: Construct W_a, W_b
    W_a = build_attention(Z, eta_a, params_scale['eta_a'])
    W_b = build_attention(Z, eta_b, params_scale['eta_b'])

    # Step 4: Construct S, G
    S = jnp.eye(p) - jnp.diag(gamma_b) @ W_b
    G = jnp.diag(beta) + jnp.diag(gamma_a) @ W_a
    S_inv = jnp.linalg.inv(S)


    # Step 5: Random noise
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

    Y = jnp.zeros((T, p))
    for t in range(1, T):
        Y = Y.at[t].set(S_inv @ (G @ Y[t - 1] + eps[t]))

    params = dict(eta_a=eta_a, eta_b=eta_b, gamma_a=gamma_a, gamma_b=gamma_b, beta=beta, sigma2=sigma2)

    # Add the attention checking.


    return Y, Z, W_a, W_b


def flatten_params(params: dict, order: list[str]):
    """
    Flatten a parameter dictionary into a single 1D JAX array (concatenated in given order)
    and return parameter slices.

    Args:
        params: dict[str, scalar or jnp.ndarray]
            Dictionary of parameters (some scalars, some arrays).
        order: list[str]
            Desired key order for concatenation.

    Returns:
        theta_flat : jnp.ndarray
            1D array with all parameters concatenated in the specified order.
        param_slices : dict[str, slice]
            Dictionary mapping parameter names to their index slices in theta_flat.
    """
    flat_list = []
    param_slices = {}

    start = 0
    for k in order:
        if k not in params:
            continue

        v = params[k]
        v_arr = jnp.atleast_1d(v).ravel()     # scalar → array([scalar])
        flat_list.append(v_arr)

        end = start + v_arr.size
        param_slices[k] = slice(start, end)
        start = end

    theta_flat = jnp.concatenate(flat_list)
    return theta_flat, param_slices


if __name__ == "__main__":

    # Set random number seed
    key = jax.random.PRNGKey(0)

    # Example parameter dictionary (in natural scale)
    p, q, T = 5, 5, 200
    params = dict(
        eta_a=0.5,
        eta_b=1.0,
        gamma_a=jax.random.uniform(key, (p,), minval=-0.2, maxval=0.2),
        gamma_b=jax.random.uniform(key, (p,), minval=-0.2, maxval=0.2),
        beta=jax.random.uniform(key, (p,), minval=-0.2, maxval=0.2),
        sigma2=jnp.exp(jax.random.normal(key, (p,)))  # positive σ²
    )

    params_scale = dict(
        eta_a=1,
        eta_b=10   )

    # Simulate dataset
    Y, Z, W_a, W_b = simulate_VAR_attention(key, params, T=T, p=p, q=q, dist="normal", params_scale=params_scale)
    print("Simulated Y shape:", Y.shape)

    # Convert σ² → σ̃² for reparameterized estimation
    sigma2_tilde = jnp.log(jnp.expm1(params["sigma2"]))
    params_reparam = params.copy()
    params_reparam["sigma2"] = sigma2_tilde

    # Flatten parameter vector θ̃ (for estimation/Hessian functions)
    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    theta_tilde, _ = flatten_params(params_reparam, order)

    print("Parameter vector θ̃ (reparameterized) length:", theta_tilde.shape[0])
    print("First few σ̃²:", sigma2_tilde[:5])

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
