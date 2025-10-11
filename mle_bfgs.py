import jax
import jax.numpy as jnp
from jax import grad, hessian
from scipy.optimize import minimize

from loglik_attn import loglik

def mle_bfgs(Y, Z, theta_init):
    """
    MLE Estimator
    """

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

    from dgp import simulate_VAR_attention

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
