import jax
import jax.numpy as jnp

from dgp import simulate_VAR_attention
from dgp import flatten_params
from fit import estimate

if __name__ == "__main__":
    # --- 1. Define simulation setup ---
    key = jax.random.PRNGKey(0)
    # R = 100              # number of replications
    T = 500                # sample size
    p = 3                  # number of responses
    q = 2                  # dimension of feature matrix

    # DGP
    Y, params, Z, W_a, W_b = simulate_VAR_attention(
        key=key, T=T, p=p, q=q, dist="t5")

    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    theta_true = flatten_params(params, order)

    log_sigma2_true = jnp.log(theta_true[2+3*p:2+4*p])
    theta_true_reparam = jnp.concatenate([theta_true[:2+3*p], log_sigma2_true])

    theta_init = theta_true_reparam + 0.1 * jax.random.normal(key, theta_true_reparam.shape)

    out = estimate(Y, Z, key, theta_init)
    print(out)
