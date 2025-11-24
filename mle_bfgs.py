import jax
import jax.numpy as jnp
from jax import grad, hessian
from scipy.optimize import minimize

from loglik_attn import loglik_reparam


def mle_bfgs(Y, Z, theta_init):
    """
    MLE Estimator
    """

    # Negative loglik for minimizer
    def objective(theta_np):
        theta = jnp.array(theta_np)
        return -loglik_reparam(theta, Y, Z)

    # Use scipy minimize
    res = minimize(objective, theta_init, method="BFGS")
    theta_hat = res.x

    # Gradient and Hessian at optimum
    grad_ll = []
    hess_ll = []

    # grad_ll = grad(loglik)(theta_hat, Y, Z)
    # hess_ll = hessian(loglik)(theta_hat, Y, Z)

    return theta_hat, grad_ll, hess_ll, res


if __name__ == "__main__":

    from dgp import simulate_VAR_attention, flatten_params

    # Set random number seed
    key = jax.random.PRNGKey(0)

    # Example parameter dictionary (in natural scale)
    algorithm = "bfgs"
    # algorithm = "auto_gradhess"

    p, q, T = 10, 10, 200
    params = dict(
        eta_a=0.5,
        eta_b=1.0,
        gamma_a=jax.random.uniform(key, (p,), minval=-0.2, maxval=0.2),
        gamma_b=jax.random.uniform(key, (p,), minval=-0.2, maxval=0.2),
        beta=jax.random.uniform(key, (p,), minval=-0.3, maxval=0.3),
        sigma2=jnp.exp(jax.random.normal(key, (p,)))  # positive σ²
    )
    Y, Z, W_a, W_b = simulate_VAR_attention(key, params, T=T, p=p, q=q, dist="normal")

    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    # Convert σ² → σ̃² for reparameterized estimation
    sigma2_tilde = jnp.log(jnp.expm1(params["sigma2"]))
    params_reparam = params.copy()
    params_reparam["sigma2"] = sigma2_tilde

    # Flatten parameter vector θ̃ (for estimation/Hessian functions)
    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    theta_tilde, _ = flatten_params(params_reparam, order)

    theta_init = theta_tilde + 0.05 * jax.random.normal(key, theta_tilde.shape)

    theta_hat, grad_ll, hess_ll, res = mle_bfgs(Y, Z, theta_init)

    print("True theta:", theta_true)
    print("Estimated theta:", theta_hat)
    print("Gradient at optimum:", grad_ll)
    print("Hessian shape:", hess_ll.shape)
    print("Converged:", res.success)
