import jax
import jax.numpy as jnp

from loglik_attn import loglik_reparam

def mle_newton(Y, Z, theta_init, max_iter=50, tol=1e-6, alpha=1.0):
    """Newton-Raphson Estimator"""

    loglik_grad = jax.grad(loglik_reparam)
    loglik_hess = jax.hessian(loglik_reparam)

    theta = theta_init
    for it in range(max_iter):
        g = loglik_grad(theta, Y, Z)
        H = loglik_hess(theta, Y, Z)

        try:
            step = jnp.linalg.solve(H, g)
        except jnp.linalg.LinAlgError:
            print("Hessian not invertible at iteration", it)
            break

        # line search / damping
        new_ll = loglik_reparam(theta + step, Y, Z)
        old_ll = loglik_reparam(theta, Y, Z)
        step_size = alpha
        while (not jnp.isfinite(new_ll)) or (new_ll < old_ll):
            step_size *= 0.5
            if step_size < 1e-6:
                print("Step size too small, stopping.")
                return theta, g, H
            new_ll = loglik_reparam(theta + step_size * step, Y, Z)

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

    key = jax.random.PRNGKey(0)
    Y, theta_true, Z = simulate_VAR_attention(key, T=200, p=5, q=5, dist="normal")

    # Initial guess: perturb truth; log-transform sigma2 part
    p = Y.shape[1]
    log_sigma2_true = jnp.log(theta_true[2+3*p:2+4*p])
    theta_true_reparam = jnp.concatenate([
        theta_true[:2+3*p], log_sigma2_true
    ])
    theta_init = theta_true_reparam + 0.1 * jax.random.normal(key, theta_true_reparam.shape)

    theta_hat_reparam, grad_ll, hess_ll = mle_newton(Y, Z, theta_init)

    # Convert back to natural scale (replace log_sigma2 with sigma2)
    log_sigma2_hat = theta_hat_reparam[2+3*p:2+4*p]
    sigma2_hat = jnp.exp(log_sigma2_hat)
    theta_hat = jnp.concatenate([theta_hat_reparam[:2+3*p], sigma2_hat])

    print("True theta:", theta_true)
    print("Estimated theta:", theta_hat)
    print("Gradient norm at optimum:", jnp.linalg.norm(grad_ll))
    print("Hessian shape:", hess_ll.shape)
