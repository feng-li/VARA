import jax
import jax.numpy as jnp
from jax import grad, hessian
from jax.random import multivariate_normal, uniform, normal
from scipy.optimize import minimize

from loglik_attn import build_attention
from loglik_attn import loglik


def simulate_VAR_attention(key, T=200, p=5, q=5, dist="normal"):
    """Simulation function"""
    keys = jax.random.split(key, 7)

    # Step 1: Feature vectors
    idx = jnp.arange(q)
    Sigma_z = 0.5 ** jnp.abs(idx[:, None] - idx[None, :])
    Z = multivariate_normal(keys[0], jnp.zeros(q), Sigma_z, (p,))

    # Step 2: Attention matrices
    W_a = build_attention(Z, eta=0.5)
    W_b = build_attention(Z, eta=1.0)

    # Step 3: Parameters
    gamma_a = uniform(keys[1], (p,), minval=-0.2, maxval=0.2)
    gamma_b = uniform(keys[2], (p,), minval=-0.2, maxval=0.2)
    beta    = uniform(keys[3], (p,), minval=-0.2, maxval=0.2)
    sigma2  = uniform(keys[4], (p,), minval=0.25, maxval=1.0)

    # Step 4: Noise
    eps_raw = normal(keys[5], (T, p))
    eps = eps_raw * jnp.sqrt(sigma2)

    # Step 5: Dynamics
    S = jnp.eye(p) - jnp.diag(gamma_b) @ W_b
    G = jnp.diag(beta) + jnp.diag(gamma_a) @ W_a
    S_inv = jnp.linalg.inv(S)

    Y = jnp.zeros((T, p))
    for t in range(1, T):
        Y = Y.at[t].set(S_inv @ (G @ Y[t-1] + eps[t]))

    # Collect params
    theta_true = jnp.concatenate([
        jnp.array([0.5, 1.0]),  # fixed eta_a, eta_b in sim
        gamma_a, gamma_b, beta, sigma2
    ])

    return Y, theta_true, Z


# ---------- MLE Estimator ----------
def mle_bfgs(Y, Z, theta_init):
    # Negative loglik for minimizer
    def objective(theta_np):
        theta = jnp.array(theta_np)
        return -loglik(theta, Y, Z)

    # Use scipy minimize
    res = minimize(objective, theta_init, method="BFGS")
    theta_hat = res.x

    # Gradient and Hessian at optimum
    grad_ll = grad(loglik)(theta_hat, Y, Z)
    hess_ll = hessian(loglik)(theta_hat, Y, Z)

    return theta_hat, grad_ll, hess_ll, res


# ---------- Example usage ----------
if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    Y, theta_true, Z = simulate_VAR_attention(key, T=200, p=5, q=5, dist="normal")

    # Initial guess: perturb truth
    theta_init = theta_true + 0.05 * jax.random.normal(key, theta_true.shape)

    theta_hat, grad_ll, hess_ll, res = mle_bfgs(Y, Z, theta_init)

    print("True theta:", theta_true)
    print("Estimated theta:", theta_hat)
    print("Gradient at optimum:", grad_ll)
    print("Hessian shape:", hess_ll.shape)
    print("Converged:", res.success)
