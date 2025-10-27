import jax
import jax.numpy as jnp

from loglik_attn import loglik_reparam
from grad import loglik_grad_analytical
from hess import loglik_hess_analytical


def mle_newton(Y, Z, theta_init, params_scale,
               algorithm="analytical_gradhess", update_idx=slice(None),
               max_iter=10, tol=1e-3, alpha=1.0):
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

    # slice(None) means update all parameters
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_init.size)
    update_idx = jnp.array(update_idx)

    theta = theta_init[update_idx]
    theta_full = theta_init

    # print(params_scale)
    # print(f"theta_init: {theta_init}")

    for it in range(max_iter):

        # This list is the full list if jax used. clipping zero values.
        g_0 = loglik_grad(theta_full, Y, Z, params_scale, update_idx)
        g = g_0[update_idx]
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

        H_0 = loglik_hess(theta_full, Y, Z, params_scale, update_idx)
        H = H_0[jnp.ix_(update_idx, update_idx)]

        try:
            # A robust and minimal change is to damp the Newton step before entering the line search.
            # This works extremely well for non-quadratic or non-concave likelihoods.
            lambda_damp = 0.3
            step = -jnp.linalg.solve(H, g) * lambda_damp
        except jnp.linalg.LinAlgError:
            print(f"Hessian not invertible at iteration {it}.")
            break

        # ---- Trust-Region / Step Scaling ----
        max_step_norm = 0.3  # A smaller value reduces overshooting, so fewer line search failures.
        step_norm = jnp.linalg.norm(step)
        if step_norm > max_step_norm:
            scale_ratio = max_step_norm / step_norm
            step = step * (max_step_norm / step_norm)

            print(f"Step scaling {scale_ratio:.4f} used.")

            if scale_ratio < tol:
                print(f"Restarting from nearby point (it={it}).")
                theta = theta + 0.1 * jax.random.normal(jax.random.PRNGKey(it), theta.shape)
                continue



        # ---- Armijo–Goldstein Line Search ----
        step_size = alpha
        c1 = 1e-4  # A small value makes the Armijo condition easier to satisfy.
        old_ll = loglik_reparam(theta_full, Y, Z, params_scale, update_idx)

        n_backtrack = 0
        while True:
            theta_trial = theta + step_size * step
            theta_trial_full = theta_full
            theta_trial_full = theta_trial_full.at[update_idx].set(theta_trial)
            new_ll = loglik_reparam(theta_trial_full, Y, Z, params_scale, update_idx)

            # Armijo condition: sufficient increase in log-likelihood
            if (jnp.isfinite(new_ll) and new_ll >= old_ll + c1 * step_size * g.T @ step):
                break  # accept

            step_size *= 0.5
            n_backtrack += 1

            # after a few failures, regularize Hessian once
            # if n_backtrack == 5:
            #     print(f"Regularizing Hessian after {n_backtrack} failed trials.")
            #     H_reg = jnp.diag(H) + eps_reg * jnp.eye(H.shape[0])
            #     step = -jnp.linalg.solve(H_reg, g) * lambda_damp

            if step_size < tol and n_backtrack > 5:
                print(f"Too many line search reductions ({n_backtrack}), step_size={step_size:.4f}. Restarting from nearby point.")
                theta = theta + 0.1 * jax.random.normal(jax.random.PRNGKey(it), theta.shape)
                step_size = alpha
                break

        delta_theta = step_size * step
        theta_new = theta + delta_theta

        # Compute convergence diagnostics
        abs_change = jnp.linalg.norm(delta_theta, 1)
        relative_change = abs_change / jnp.maximum(1.0, jnp.linalg.norm(theta))

        if relative_change < tol or abs_change < tol or g_norm < tol:
            print(f"Converged in {it+1} iterations.")
            # print(f"abs_change: {abs_change:6f}, relative_change: {relative_change:6f}, g_norm: {g_norm:6f}")
            # g_new = loglik_grad(theta_new, Y, Z, params_scale)
            # H_new = loglik_hess(theta_new, Y, Z, params_scale)

            # breakpoint()
            g_new = []
            H_new = []
            return theta_new, g_new, H_new


        theta = theta_new
        theta_full = theta_full.at[update_idx].set(theta)

        # print(f"abs_change: {abs_change:6f}, relative_change: {relative_change:6f}, g_norm: {g_norm:6f}")
        # print(f"theta_full: {theta_full}")


    print("Max iterations reached.")
    # g = loglik_grad(theta, Y, Z)
    # H = loglik_hess(theta, Y, Z)
    g = []
    H = []
    return theta, g, H


# ---------- Example usage ----------
if __name__ == "__main__":

    import os
    import math
    import random
    import matplotlib.pyplot as plt

    from dgp import simulate_VAR_attention
    from dgp import flatten_params

    # jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(threshold=10_000, linewidth=200, suppress=True, precision=6)

    # --- Simulation settings ---
    key = jax.random.PRNGKey(random.randint(0, 10000))
    T, p, q = 1000, 3, 3

    # --- True parameters (in natural scale) ---
    params_true = dict(
        eta_a=0.5,
        eta_b=1.5,
        gamma_a=jnp.linspace(-0.15, 0.15, p),
        gamma_b=jnp.linspace(-0.10, 0.10, p),
        beta=jnp.linspace(-0.20, 0.20, p),
        sigma2=jnp.exp(jax.random.normal(key, (p,)))  # positive σ²
    )

    params_scale = dict(
        eta_a=1,
        eta_b=1)

    algorithm = "auto_gradhess"
    update_idx = [1]
    # update_idx = slice(None)

    # --- Simulate data ---
    Y, Z, W_a, W_b = simulate_VAR_attention(key, params_true, T=T, p=p, q=q, dist="normal")
    print("Simulated Y shape:", Y.shape)

    # --- Convert σ² → σ̃² (softplus inverse) ---
    sigma2_tilde_true = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = params_true.copy()
    params_reparam["sigma2"] = sigma2_tilde_true

    # --- Flatten parameters into reparameterized vector θ̃ ---
    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    theta_true_reparam = flatten_params(params_reparam, order)
    print("θ̃ length:", theta_true_reparam.size)

    # Only initialize update_idx related initial values
    update_idx = jnp.array(update_idx)
    theta_init = theta_true_reparam
    theta_init_update = theta_true_reparam[update_idx]
    theta_init = theta_init.at[update_idx].set(theta_init_update * (1 + 0.1 * jax.random.normal(key, theta_init_update.shape)))

    # Estimation
    theta_hat_reparam, grad_ll, hess_ll = mle_newton(Y, Z, theta_init, params_scale, update_idx=update_idx, algorithm=algorithm)

    # Convert back to natural scale (with softplus)
    sigma2_tilde_hat = theta_hat_reparam[2+3*p:2+4*p]
    sigma2_hat = jax.nn.softplus(sigma2_tilde_hat)

    theta_hat = jnp.concatenate([theta_hat_reparam[:2+3*p], sigma2_hat])


    # --- Plot log-likelihood profile for update_idx parameters ---
    plot_width = 2
    n_points = 100

    n_params = len(update_idx)
    n_cols = min(4, n_params)
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    # --- FIX: make axes iterable in all cases ---
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    for ax, idx in zip(axes, update_idx):
        theta_grid = jnp.linspace(theta_true_reparam[idx] - plot_width, theta_true_reparam[idx] + plot_width, n_points)
        ll_values = []
        for val in theta_grid:
            theta_temp = theta_true_reparam.at[idx].set(val)
            ll = loglik_reparam(theta_temp, Y, Z, params_scale, update_idx)
            ll_values.append(float(ll))

        ax.plot(theta_grid, ll_values, linestyle="-", linewidth=2, label="Likelihood")
        ax.axvline(theta_init[idx], color="green", linestyle=":", linewidth=2, label="Initial value")
        ax.axvline(theta_true_reparam[idx], color="red", linestyle="--", linewidth=2, label="True value")
        ax.axvline(theta_hat_reparam[idx], color="blue", linestyle="-.", linewidth=2, label="MLE estimate")
        ax.legend()
        ax.set_xlabel(f"θ̃[{idx}] value")
        ax.set_ylabel("Log-likelihood")
        ax.set_title(f"Likelihood Profile for θ̃[{idx}]")
        ax.grid(True)

    # hide unused subplots
    for ax in axes[n_params:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.expanduser("likelihood_profiles.pdf"), bbox_inches="tight")
    print("✅ Saved all likelihood profiles to 'likelihood_profiles.pdf'")


    print("True theta:", theta_true_reparam)
    print("Estimated theta:", theta_hat)
