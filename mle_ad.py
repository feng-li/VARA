import jax
import jax.numpy as jnp

from loglik_attn import loglik_reparam
from grad import loglik_grad_analytical
from hess import loglik_hess_analytical


def mle_newton(Y, Z, theta_init, analytical_gradhess=False,
               max_iter=100, tol=1e-8, alpha=1.0):
    """
    Newton–Raphson MLE with:
      1. softplus reparameterization for sigma²,
      2. trust-region step-scaling,
      3. gradient clipping,
      4. Armijo–Goldstein line search.

    Parameters
    ----------
    Y : array
        Dependent variable(s)
    Z : array
        Covariates or regressors
    theta_init : array
        Initial parameter guess (on reparameterized scale)
    analytical_gradhess : bool, default=False
        Whether to use analytical gradient/Hessian
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance on parameter step
    alpha : float
        Initial step size for line search

    Returns
    -------
    theta_hat : array
        Estimated parameters (on reparameterized scale)
    grad_ll : array
        Gradient at optimum
    hess_ll : array
        Hessian at optimum
    """

    # choose derivative functions
    if analytical_gradhess:
        loglik_grad = loglik_grad_analytical
        loglik_hess = loglik_hess_analytical
    else:
        loglik_grad = jax.grad(loglik_reparam)
        loglik_hess = jax.hessian(loglik_reparam)

    theta = theta_init
    for it in range(max_iter):
        g = loglik_grad(theta, Y, Z)
        g_norm = jnp.linalg.norm(g)

        # ---- Gradient Clipping ----
        max_grad_norm = 100.0
        if g_norm > max_grad_norm:
            g = g * (max_grad_norm / g_norm)
            g_norm = max_grad_norm

        H = loglik_hess(theta, Y, Z)

        try:
            step = jnp.linalg.solve(H, g)
        except jnp.linalg.LinAlgError:
            print(f"Hessian not invertible at iteration {it}.")
            break

        # ---- Trust-Region / Step Scaling ----
        max_step_norm = 1.0
        step_norm = jnp.linalg.norm(step)
        if step_norm > max_step_norm:
            step = step * (max_step_norm / step_norm)

        # ---- Armijo–Goldstein Line Search ----
        step_size = alpha
        c1 = 1e-4   # sufficient decrease constant
        c2 = 0.9    # curvature condition constant
        old_ll = loglik_reparam(theta, Y, Z)

        while True:
            theta_trial = theta + step_size * step
            new_ll = loglik_reparam(theta_trial, Y, Z)

            # Armijo condition: sufficient increase in log-likelihood
            if (jnp.isfinite(new_ll)
                and new_ll >= old_ll + c1 * step_size * g_norm**2):
                break  # accept
            step_size *= 0.5
            if step_size < 1e-8:
                print("Step size too small, stopping.")
                return theta, g, H

        theta_new = theta + step_size * step

        if jnp.linalg.norm(step_size * step) < tol:
            print(f"Converged in {it+1} iterations.")
            return theta_new, g, H

        theta = theta_new

    print("Max iterations reached.")
    return theta, g, H


# ---------- Example usage ----------
if __name__ == "__main__":

    from dgp import simulate_VAR_attention
    from dgp import flatten_params

    key = jax.random.PRNGKey(0)
    Y, params, Z, W_a, W_b = simulate_VAR_attention(key, T=200, p=5, q=5, dist="normal")


    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    theta_true = flatten_params(params, order)

    # Initial guess: perturb truth; log-transform sigma2 part
    p = Y.shape[1]
    log_sigma2_true = jnp.log(theta_true[2+3*p:2+4*p])
    theta_true_reparam = jnp.concatenate([theta_true[:2+3*p], log_sigma2_true])

    theta_init = theta_true_reparam + 0.01 * jax.random.normal(key, theta_true_reparam.shape)
    theta_hat_reparam, grad_ll, hess_ll = mle_newton(Y, Z, theta_init)

    # Convert back to natural scale (replace log_sigma2 with sigma2)
    log_sigma2_hat = theta_hat_reparam[2+3*p:2+4*p]
    sigma2_hat = jnp.exp(log_sigma2_hat)

    theta_hat = jnp.concatenate([theta_hat_reparam[:2+3*p], sigma2_hat])

    print("True theta:", theta_true)
    print("Estimated theta:", theta_hat)
    print("Gradient norm at optimum:", jnp.linalg.norm(grad_ll))
    print("Hessian shape:", hess_ll.shape)
