try:
    from memory_profiler import profile
except ImportError:
    def profile(func):
        return func

import random
import jax
import jax.numpy as jnp
import numpy as np
from .loglik_attn import loglik_reparam
from .grad import loglik_grad_analytical
from .hess import loglik_hess_analytical
from .dgp import plot_likelihood_profiles

import pandas as pd


# @profile
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


def mle_newton_gibbs(
    Y,
    Z,
    theta_init,
    params_scale,
    block_order,
    algorithm="analytical_gradhess",
    max_outer_iter=20,
    inner_max_iter=20,
    tol=1e-8,
    alpha=1.0,
    update_idx=slice(None),
):
    """
    Blockwise Newton wrapper using a Gibbs-style updating scheme.
    """

    if algorithm == "analytical_gradhess":
        loglik_grad = loglik_grad_analytical

        def loglik_hess(theta, Y, Z, params_scale, update_idx):
            return loglik_hess_analytical(
                theta=theta,
                Y=Y,
                Z=Z,
                params_scale=params_scale,
                kappa4=3,
                update_idx=update_idx,
            )

    elif algorithm == "auto_gradhess":
        loglik_grad = jax.grad(loglik_reparam)
        loglik_hess = jax.hessian(loglik_reparam)
    else:
        raise Exception("No such algorithm implemented.")

    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_init.size)
    else:
        update_idx = jnp.array(update_idx)

    active_set = set(map(int, list(update_idx)))

    if isinstance(block_order, dict):
        block_items = list(block_order.items())
    else:
        block_items = [(f"block_{i}", blk) for i, blk in enumerate(block_order)]

    normalized_blocks = []
    for block_id, (block_name, blk) in enumerate(block_items):
        if isinstance(blk, slice):
            blk_idx = jnp.arange(theta_init.size)[blk]
        else:
            blk_idx = jnp.array(blk)

        blk_idx = jnp.array([int(i) for i in blk_idx if int(i) in active_set])

        if blk_idx.size > 0:
            normalized_blocks.append((block_id, block_name, blk_idx))

    if len(normalized_blocks) == 0:
        raise ValueError("No valid blocks found in block_order.")

    theta_full = theta_init.copy()
    records = []

    for outer_it in range(max_outer_iter):
        theta_old = theta_full.copy()
        ll_before_cycle = float(loglik_reparam(theta_full, Y, Z, params_scale, update_idx))

        for block_id, block_name, blk_idx in normalized_blocks:
            theta_block, grad_block, hess_block, conv_block = mle_newton(
                Y=Y,
                Z=Z,
                theta_init=theta_full,
                params_scale=params_scale,
                algorithm=algorithm,
                update_idx=blk_idx,
                max_iter=inner_max_iter,
                tol=tol,
                alpha=alpha,
            )

            theta_full = theta_full.at[blk_idx].set(theta_block)
            ll_after_block = float(loglik_reparam(theta_full, Y, Z, params_scale, update_idx))

            records.append({
                "outer_iteration": outer_it,
                "stage": "block",
                "block_id": block_id,
                "block_name": block_name,
                "block_size": int(blk_idx.size),
                "block_idx": list(map(int, list(blk_idx))),
                "loglik": ll_after_block,
                "block_status": int(conv_block["status"]),
                "block_inner_steps": len(conv_block["trace"]),
            })

        ll_after_cycle = float(loglik_reparam(theta_full, Y, Z, params_scale, update_idx))
        abs_change = float(jnp.linalg.norm(theta_full - theta_old, ord=jnp.inf))
        rel_change = float(abs_change / jnp.maximum(1.0, jnp.linalg.norm(theta_full)))

        g_full = loglik_grad(theta_full, Y, Z, params_scale, update_idx)
        g_active = g_full[update_idx]
        g_norm = float(jnp.linalg.norm(g_active))

        records.append({
            "outer_iteration": outer_it,
            "stage": "cycle_end",
            "block_id": -1,
            "block_name": "cycle_end",
            "block_size": 0,
            "block_idx": "cycle_end",
            "loglik": ll_after_cycle,
            "block_status": 0,
            "block_inner_steps": 0,
            "cycle_gain": ll_after_cycle - ll_before_cycle,
            "abs_change": abs_change,
            "rel_change": rel_change,
            "grad_norm": g_norm,
        })

        if rel_change < tol or g_norm < tol:
            H_full = loglik_hess(theta_full, Y, Z, params_scale, update_idx)
            H_active = H_full[jnp.ix_(update_idx, update_idx)]
            trace = pd.DataFrame(records)
            return theta_full[update_idx], g_active, H_active, dict(status=0, trace=trace)

    H_full = loglik_hess(theta_full, Y, Z, params_scale, update_idx)
    H_active = H_full[jnp.ix_(update_idx, update_idx)]
    g_full = loglik_grad(theta_full, Y, Z, params_scale, update_idx)
    g_active = g_full[update_idx]
    trace = pd.DataFrame(records)

    return theta_full[update_idx], g_active, H_active, dict(status=1, trace=trace)


def mle_newton_hybrid(
    Y,
    Z,
    theta_init,
    params_scale,
    block_order,
    algorithm="analytical_gradhess",
    gibbs_max_outer_iter=10,
    gibbs_inner_max_iter=20,
    full_max_iter=10,
    tol=1e-8,
    alpha=1.0,
    update_idx=slice(None),
):
    """
    Hybrid block-to-full Newton:
      1. Run blockwise Gibbs/Newton to get a stabilized initial value
      2. Run full Newton starting from that value

    Returns same output structure as mle_newton, plus combined trace.
    """

    theta_gibbs_short, grad_gibbs, hess_gibbs, conv_gibbs = mle_newton_gibbs(
        Y=Y,
        Z=Z,
        theta_init=theta_init,
        params_scale=params_scale,
        block_order=block_order,
        algorithm=algorithm,
        max_outer_iter=gibbs_max_outer_iter,
        inner_max_iter=gibbs_inner_max_iter,
        tol=tol,
        alpha=alpha,
        update_idx=update_idx,
    )

    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_init.size)
    else:
        update_idx = jnp.array(update_idx)

    theta_gibbs_full = theta_init.copy()
    theta_gibbs_full = theta_gibbs_full.at[update_idx].set(theta_gibbs_short)

    theta_full_short, grad_full, hess_full, conv_full = mle_newton(
        Y=Y,
        Z=Z,
        theta_init=theta_gibbs_full,
        params_scale=params_scale,
        algorithm=algorithm,
        update_idx=update_idx,
        max_iter=full_max_iter,
        tol=tol,
        alpha=alpha,
    )

    trace_gibbs = conv_gibbs["trace"].copy()
    trace_gibbs["method"] = "hybrid_gibbs"

    trace_full = conv_full["trace"].copy()
    trace_full["method"] = "hybrid_full"
    trace_full["outer_iteration"] = jnp.nan
    trace_full["block_id"] = -1
    trace_full["block_name"] = "full_newton"
    trace_full["block_size"] = len(update_idx)
    trace_full["block_idx"] = "full_update"
    trace_full["block_status"] = conv_full["status"]
    trace_full["block_inner_steps"] = trace_full["iteration"]

    trace = pd.concat([trace_gibbs, trace_full], ignore_index=True)

    status = conv_full["status"]

    return theta_full_short, grad_full, hess_full, dict(
        status=status,
        trace=trace,
        gibbs_stage=conv_gibbs,
        full_stage=conv_full,
    )


def build_param_name_map(order, block_order):
    """
    Build a mapping from flattened parameter index to parameter name
    using order and dict-valued block_order.
    """
    name_map = {}

    for param_name in order:
        if param_name not in block_order:
            raise KeyError(f"{param_name} is in order but not in block_order.")

        block = block_order[param_name]

        if isinstance(block, slice):
            raise ValueError("Please convert slice blocks to explicit index lists first.")

        block = list(block)

        if len(block) == 1:
            name_map[int(block[0])] = param_name
        else:
            for local_pos, global_idx in enumerate(block, start=0):
                name_map[int(global_idx)] = f"{param_name}[{local_pos}]"

    return name_map


# ---------- Example usage ----------
if __name__ == "__main__":

    from .dgp import flatten_params
    from .mle_optimize import mle_optimize

    # jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(threshold=10_000, linewidth=200, suppress=True, precision=6)

    # --- Simulation settings ---
    # Set random number seed
    seed = random.randrange(10**9)
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 10)
    T, p, q = 1000, 5, 3

    # --- True parameters (in natural scale) ---
    params_true = dict(
        eta_a=0.5,
        gamma_a=jnp.linspace(-0.15, 0.15, p),
        beta=jnp.linspace(-0.20, 0.20, p),
        sigma2=jnp.exp(jnp.log(0.5) + 0.2 * jax.random.normal(keys[0], (p,)))  # positive σ²
    )

    params_scale = dict(eta_a=1)

    algorithm = "auto_gradhess"
    # algorithm = "analytical_gradhess"
    # update_idx = [0, 1+p]
    update_idx = slice(None)

    # --- Simulate data ---
    # Y, Z, W_a = simulate_VAR_attention(key, params_true, T=T, p=p, q=q, dist="normal", save=True)

    data_path = f"T={T}_p={p}_q={q}.npz"
    print(f"Using {data_path}")
    with np.load(data_path, allow_pickle=True) as obj:
        data = obj["payload"].item()

    Y = data["Y"]
    Z = data["Z"]
    W_a = data["W_a"]
    params_true = data["params"]
    # breakpoint()

    print("Simulated Y shape:", Y.shape)

    # --- Convert σ² → σ̃² (softplus inverse) ---
    sigma2_tilde_true = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = params_true.copy()
    params_reparam["sigma2"] = sigma2_tilde_true

    # --- Flatten parameters into reparameterized vector θ̃ ---
    block_order = {
        "eta_a": [0],
        "gamma_a": list(range(1, 1 + p)),
        "beta": list(range(1 + p, 1 + 2 * p)),
        "sigma2": list(range(1 + 2 * p, 1 + 3 * p)),
    }
    param_name_map = build_param_name_map(block_order)

    theta_true_reparam, _ = flatten_params(params_reparam, block_order)
    print("θ tilde length:", theta_true_reparam.size)


    # Select algorithms to run
    selected_algorithms = {
        "Full Newton": True,
        "Block Newton": True,
        "Hybrid Block2Full": True,
        "LBFGS": True,
    }

    # Container of estimated parameter vectors for plotting
    theta_hat_dict = {}
    # Only initialize update_idx related initial values
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_true_reparam.size)

    update_idx = jnp.array(update_idx)
    theta_init = theta_true_reparam.copy()
    theta_init_update = theta_true_reparam[update_idx]
    theta_init = theta_init.at[update_idx].add(0.5 * jax.random.normal(key, theta_init_update.shape))

    if selected_algorithms["Full Newton"]:
        theta_hat_reparam_short_full, grad_ll_full, hess_ll_full, convergence_full = mle_newton(
            Y, Z, theta_init, params_scale, update_idx=update_idx, algorithm=algorithm
        )

        theta_hat_reparam_full = theta_true_reparam.at[update_idx].set(theta_hat_reparam_short_full)
        theta_hat_dict["Full Newton"] = theta_hat_reparam_full

    if selected_algorithms["Block Newton"]:
        theta_hat_reparam_short_gibbs, grad_ll_gibbs, hess_ll_gibbs, convergence_gibbs = mle_newton_gibbs(
            Y=Y,
            Z=Z,
            theta_init=theta_init,
            params_scale=params_scale,
            block_order=block_order,
            algorithm=algorithm,
            max_outer_iter=10,
            inner_max_iter=20,
            tol=1e-8,
            alpha=1.0,
            update_idx=update_idx,
        )

        theta_hat_reparam_gibbs = theta_true_reparam.at[update_idx].set(theta_hat_reparam_short_gibbs)
        theta_hat_dict["Block Newton"] = theta_hat_reparam_gibbs

    if selected_algorithms["Hybrid Block2Full"]:
        theta_hat_reparam_short_hybrid, grad_ll_hybrid, hess_ll_hybrid, convergence_hybrid = mle_newton_hybrid(
            Y=Y,
            Z=Z,
            theta_init=theta_init,
            params_scale=params_scale,
            block_order=block_order,
            algorithm=algorithm,
            gibbs_max_outer_iter=5,
            gibbs_inner_max_iter=20,
            full_max_iter=10,
            tol=1e-8,
            alpha=1.0,
            update_idx=update_idx,
        )

        theta_hat_reparam_hybrid = theta_true_reparam.at[update_idx].set(theta_hat_reparam_short_hybrid)
        theta_hat_dict["Hybrid Block2Full"] = theta_hat_reparam_hybrid

    if selected_algorithms["LBFGS"]:
        theta_hat_reparam_short_opt, grad_ll_opt, hess_ll_opt, convergence_opt = mle_optimize(
            Y=Y,
            Z=Z,
            theta_init=theta_init,
            params_scale=params_scale,
            algorithm="lbfgs",
            update_idx=update_idx,
        )

        theta_hat_reparam_opt = theta_true_reparam.at[update_idx].set(theta_hat_reparam_short_opt)
        theta_hat_dict["LBFGS"] = theta_hat_reparam_opt


    # --- Plot log-likelihood profile for update_idx parameters ---
    plot_likelihood_profiles(
        Y=Y,
        Z=Z,
        params_true=params_true,
        params_scale=params_scale,
        loglik_fn=loglik_reparam,
        update_idx=update_idx,
        theta_init=theta_init,
        theta_hat_dict=theta_hat_dict,
        plot_natural_scale=True,
        plot_width=2.0,
        n_points=100,
        save_path="likelihood_profiles_compare.pdf",
    )
