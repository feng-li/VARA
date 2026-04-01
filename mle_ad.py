import random
import jax
import jax.numpy as jnp

from loglik_attn import loglik_reparam
from grad import loglik_grad_analytical
from hess import loglik_hess_analytical

import pandas as pd


def mle_newton(Y, Z, theta_init, params_scale,
               algorithm="analytical_gradhess", update_idx=slice(None),
               max_iter=100, tol=1e-8, alpha=1.0):
    """
    Newton–Raphson MLE with:
      1. trust-region Newton step control,
      2. gradient clipping for numerical stability,
      3. Armijo–Goldstein line search as fallback,
      4. detailed convergence diagnostics.

    The trust-region algorithm adaptively adjusts the radius based on
    the ratio between actual and predicted improvement in the
    log-likelihood.  A fallback Armijo line search is used when the
    trust-region step is rejected repeatedly.  A convergence log is
    returned for iteration diagnostics.

    Parameters
    ----------
    Y : array
        Dependent variable(s)
    Z : array
        Covariates or regressors
    theta_init : array
        Initial parameter guess (on reparameterized scale)
    params_scale : array or dict
        Scaling factors used in reparameterization
    algorithm : {"analytical_gradhess", "auto_gradhess"}, default="analytical_gradhess"
        Choice of gradient/Hessian computation
    update_idx : slice or array
        Parameter indices to update
    max_iter : int
        Maximum number of iterations
    tol : float
        Convergence tolerance
    alpha : float
        Initial step size for Armijo fallback

    Returns
    -------
    theta_hat : array
        Estimated parameters (reparameterized scale)
    grad_ll : array
        Gradient at optimum
    hess_ll : array
        Hessian at optimum
    convergence_log : pandas.DataFrame
        Iteration-wise diagnostics: iteration, loglik, ratio,
        trust_radius, grad_norm, step_norm
    """

    # choose derivative functions
    if algorithm == "analytical_gradhess":
        loglik_grad = loglik_grad_analytical

        def loglik_hess(theta, Y, Z, params_scale, update_idx):
            return loglik_hess_analytical(
                theta=theta, Y=Y, Z=Z, params_scale=params_scale,
                kappa4=3, update_idx=update_idx)

    elif algorithm == "auto_gradhess":
        loglik_grad = jax.grad(loglik_reparam)
        loglik_hess = jax.hessian(loglik_reparam)
    else:
        raise Exception("No such algorithm implemented.")

    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_init.size)
    update_idx = jnp.array(update_idx)

    theta = theta_init[update_idx]
    theta_full = theta_init

    T, p = Y.shape

    # trust-region parameters
    trust_radius = 0.3
    trust_radius_max = 5.0
    shrink_factor = 0.25
    grow_factor = 2.0
    accept_lower = 0.25
    accept_upper = 0.75

    # record diagnostics
    records = []

    for it in range(max_iter):
        g_full = loglik_grad(theta_full, Y, Z, params_scale, update_idx)
        g = g_full[update_idx]
        g_norm = jnp.linalg.norm(g)

        # gradient clipping
        max_grad_norm = 100.0
        if g_norm > max_grad_norm:
            g = g * (max_grad_norm / g_norm)
            g_norm = max_grad_norm
            print(f"Gradient clipped to {max_grad_norm}")

        # breakpoint()
        H_full = loglik_hess(theta_full, Y, Z, params_scale, update_idx)
        H = H_full[jnp.ix_(update_idx, update_idx)]

        try:
            lambda_damp = 0.3
            step = -jnp.linalg.solve(H, g) * lambda_damp
        except jnp.linalg.LinAlgError:
            print(f"Hessian not invertible at iteration {it}.")
            break

        # trust-region control
        step_norm = jnp.linalg.norm(step)
        step_ratio = trust_radius / (step_norm + 1e-12)
        step = jax.lax.cond(
            step_norm > trust_radius,
            lambda _: step * step_ratio,
            lambda _: step,
            operand=None)

        step_norm = jax.lax.cond(
            step_norm > trust_radius,
            lambda _: trust_radius,
            lambda _: step_norm,
            operand=None)


        old_ll = loglik_reparam(theta_full, Y, Z, params_scale, update_idx)
        theta_trial = theta + step
        theta_trial_full = theta_full.at[update_idx].set(theta_trial)
        new_ll = loglik_reparam(theta_trial_full, Y, Z, params_scale, update_idx)

        predicted_gain = -(g.T @ step + 0.5 * step.T @ (H @ step))
        actual_gain = new_ll - old_ll

        ratio = jax.lax.cond(
            predicted_gain > 0,
            lambda _: actual_gain / predicted_gain,
            lambda _: -jnp.inf,
            operand=None)


        # print("actual_gain:", float(actual_gain))

        # adjust trust radius
        # compute conditions
        cond_shrink = ratio < accept_lower
        cond_grow = jnp.logical_and(ratio > accept_upper,
                                    jnp.isclose(step_norm, trust_radius))

        # shrink or keep
        trust_radius = jax.lax.cond(
            cond_shrink,
            lambda r: r * shrink_factor,
            lambda r: r,
            trust_radius)

        # grow or keep
        trust_radius = jax.lax.cond(
            cond_grow,
            lambda r: jnp.minimum(grow_factor * r, trust_radius_max),
            lambda r: r,
            trust_radius)


        # accept/reject
        accept = ratio > 0

        theta = jax.lax.cond(
            accept,
            lambda _: theta_trial,
            lambda _: theta,
            operand=None)

        theta_full = jax.lax.cond(
            accept,
            lambda _: theta_trial_full,
            lambda _: theta_full,
            operand=None)

        if float(accept):
            pass
        else:
            # print("Iter", it, ": trust-region step rejected (ρ=", float(ratio), ")")
            trust_radius *= shrink_factor

            # Armijo fallback
            step_size = alpha
            c1 = 1e-2
            n_backtrack = 0
            while True:
                print("Using Armijo fallback.")
                theta_trial = theta + step_size * step
                theta_trial_full = theta_full.at[update_idx].set(theta_trial)
                new_ll = loglik_reparam(theta_trial_full, Y, Z, params_scale, update_idx)

                armijo_cond = jnp.logical_and(
                    jnp.isfinite(new_ll),
                    new_ll >= old_ll + c1 * step_size * (g.T @ step),
                )
                if bool(armijo_cond):
                    theta = theta_trial
                    theta_full = theta_trial_full
                    break

                step_size *= 0.5
                n_backtrack += 1
                if step_size < tol or n_backtrack > 5:
                    print(f"Iter {it}: Armijo failed ({n_backtrack} reductions). Restart nearby.")
                    theta = theta + 0.1 * jax.random.normal(
                        jax.random.PRNGKey(random.randint(0, 100)), theta.shape
                    )
                    break

        # log diagnostics
        records.append({
            "iteration": it,
            "loglik": float(new_ll),
            "ratio": float(ratio),
            "trust_radius": float(trust_radius),
            "grad_norm": float(g_norm),
            "step_norm": float(step_norm),
        })
        # convergence checks
        abs_change = jnp.linalg.norm(step, jnp.inf)  # measures the single largest change in any parameter.
        rel_change = abs_change / jnp.maximum(1.0, jnp.linalg.norm(theta))
        if rel_change < tol or g_norm < tol: #  or abs_change < tol
            print(f"g_norm: {g_norm}, rel_change: {rel_change}, abs_change: {abs_change}, actual_gain: {actual_gain}")
            print(f"Converged in {it+1} iterations.")
            trace = pd.DataFrame(records)
            return theta, g, H, dict(status=0, trace=trace)

    print("Max iterations reached.")
    trace = pd.DataFrame(records)
    return theta, g, H, dict(status=1, trace=trace)


# ---------- Example usage ----------
if __name__ == "__main__":

    import os
    import math
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
        gamma_a=jnp.linspace(-0.15, 0.15, p),
        beta=jnp.linspace(-0.20, 0.20, p),
        sigma2=jnp.exp(jax.random.normal(key, (p,)))  # positive σ²
    )

    params_scale = dict(eta_a=1)

    algorithm = "auto_gradhess"
    # algorithm = "analytical_gradhess"
    update_idx = [0, 1+p]
    # update_idx = slice(None)

    # --- Simulate data ---
    Y, Z, W_a = simulate_VAR_attention(key, params_true, T=T, p=p, q=q, dist="normal")
    print("Simulated Y shape:", Y.shape)

    # --- Convert σ² → σ̃² (softplus inverse) ---
    sigma2_tilde_true = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = params_true.copy()
    params_reparam["sigma2"] = sigma2_tilde_true

    # --- Flatten parameters into reparameterized vector θ̃ ---
    order = ["eta_a", "gamma_a", "beta", "sigma2"]
    theta_true_reparam, _ = flatten_params(params_reparam, order)
    print("θ̃ length:", theta_true_reparam.size)

    # Only initialize update_idx related initial values
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_true_reparam.size)

    update_idx = jnp.array(update_idx)
    theta_init = theta_true_reparam
    theta_init_update = theta_true_reparam[update_idx]
    theta_init = theta_init.at[update_idx].set(theta_init_update * (1 + 1 * jax.random.normal(key, theta_init_update.shape)))

    # Estimation
    theta_hat_reparam_short, grad_ll, hess_ll, convergence = mle_newton(Y, Z, theta_init, params_scale, update_idx=update_idx, algorithm=algorithm)

    theta_hat_reparam = theta_true_reparam
    theta_hat_reparam = theta_hat_reparam.at[update_idx].set(theta_hat_reparam_short)

    # Convert back to natural scale (with softplus)
    sigma2_tilde_hat = theta_hat_reparam[1+2*p:1+3*p]
    sigma2_hat = jax.nn.softplus(sigma2_tilde_hat)

    theta_hat = jnp.concatenate([theta_hat_reparam[:1+2*p], sigma2_hat])


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
        ax.set_xlabel(f"θ[{idx}] value")
        ax.set_ylabel("Log-likelihood")
        ax.set_title(f"Likelihood Profile for θ[{idx}]")
        ax.grid(True)

    # hide unused subplots
    for ax in axes[n_params:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.expanduser("likelihood_profiles.pdf"), bbox_inches="tight")
    print("✅ Saved all likelihood profiles to 'likelihood_profiles.pdf'")


    print("True theta:", theta_true_reparam)
    print("Estimated theta:", theta_hat_reparam)
    # diff = jnp.stack([theta_true_reparam, theta_hat_reparam], axis=0)
    # print("Diff:", diff)
