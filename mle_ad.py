import jax
import jax.numpy as jnp

from loglik_attn import loglik_reparam
from grad import loglik_grad_analytical
from hess import loglik_hess_analytical


def mle_newton(Y, Z, theta_init, algorithm="analytical_gradhess",
               max_iter=5000, tol=1e-6, alpha=1.0):
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
    if algorithm == "analytical_gradhess":
        loglik_grad = loglik_grad_analytical
        loglik_hess = loglik_hess_analytical
    elif algorithm == "auto_gradhess":
        loglik_grad = jax.grad(loglik_reparam)
        loglik_hess = jax.hessian(loglik_reparam)
    else:
        raise Exception("No such algorithm implemented.")

    theta = theta_init
    for it in range(max_iter):

        g = loglik_grad(theta, Y, Z)
        g_norm = jnp.linalg.norm(g)

        # ---- Gradient Clipping ----
        max_grad_norm = 100.0
        if g_norm > max_grad_norm:
            g = g * (max_grad_norm / g_norm)
            g_norm = max_grad_norm

            # f ratio ≈ 1.0 → gradient is already moderate; no need to worry.
            # If ratio ≪ 1.0 (e.g. 0.1 or 0.01) → gradient explosion being controlled.
            # If clipping occurs frequently → you may want to reduce step size or inspect Hessian scaling.
            print(f"Gradient clipping {max_grad_norm / g_norm} used.")

        H = loglik_hess(theta, Y, Z)

        try:
            print(H)

            step = -jnp.linalg.solve(H, g)
        except jnp.linalg.LinAlgError:
            print(f"Hessian not invertible at iteration {it}.")
            break

        # ---- Trust-Region / Step Scaling ----
        max_step_norm = 0.3 # A smaller value reduces overshooting, so fewer line search failures.
        step_norm = jnp.linalg.norm(step)
        if step_norm > max_step_norm:
            step = step * (max_step_norm / step_norm)
            print(f"Step scaling {max_step_norm / step_norm:.4f} used.")


        # ---- Armijo–Goldstein Line Search ----
        step_size = alpha
        c1 = 1e-6  # A small value makes the Armijo condition easier to satisfy.
        old_ll = loglik_reparam(theta, Y, Z)


        n_backtrack = 0
        while True:
            theta_trial = theta + step_size * step
            new_ll = loglik_reparam(theta_trial, Y, Z)

            # Armijo condition: sufficient increase in log-likelihood
            if (jnp.isfinite(new_ll)
                and new_ll >= old_ll + c1 * step_size * g.T @ step):
                break  # accept
            step_size *= 0.9
            n_backtrack += 1
           if n_backtrack > 30:
               print(f"Too many line search reductions (it={it}, step_size={step_size:.2f})")
               break
            if step_size < 1e-6:
                print("Step size too small, stopping.")
                return theta, g, H

        theta_new = theta + step_size * step

        if jnp.linalg.norm(step_size * step) < tol or g_norm < 1e-6:
            print(f"Converged in {it+1} iterations.")
            # g_new = loglik_grad(theta_new, Y, Z)
            # H_new = loglik_hess(theta_new, Y, Z)
            g_new = []
            H_new = []
            return theta_new, g_new, H_new


        theta = theta_new

    print("Max iterations reached.")
    # g = loglik_grad(theta, Y, Z)
    # H = loglik_hess(theta, Y, Z)
    g = []
    H = []
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
    log1p_sigma2_true = jnp.log1p(theta_true[2+3*p:2+4*p])
    theta_true_reparam = jnp.concatenate([theta_true[:2+3*p], log1p_sigma2_true])

    theta_init = theta_true_reparam + 0.01 * jax.random.normal(key, theta_true_reparam.shape)
    theta_hat_reparam, grad_ll, hess_ll = mle_newton(Y, Z, theta_init)

    # Convert back to natural scale (replace log1p_sigma2 with sigma2)
    log1p_sigma2_hat = theta_hat_reparam[2+3*p:2+4*p]
    sigma2_hat = jnp.expm1(log1p_sigma2_hat)

    theta_hat = jnp.concatenate([theta_hat_reparam[:2+3*p], sigma2_hat])

    print("True theta:", theta_true)
    print("Estimated theta:", theta_hat)
    print("Gradient norm at optimum:", jnp.linalg.norm(grad_ll))
    print("Hessian shape:", hess_ll.shape)
