import jax
import jax.numpy as jnp
import optax

from loglik_attn import loglik_reparam


def mle_optimize(Y, Z, theta_init, params_scale, algorithm, update_idx=slice(None)):
    """
    MLE estimator using Optax LBFGS (line-search version).
    """

    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_init.size)
    update_idx = jnp.array(update_idx)

    theta_full_init = theta_init
    theta = theta_full_init[update_idx]

    # Negative log-likelihood on active block only
    def objective(theta_short):
        theta_full = theta_full_init.at[update_idx].set(theta_short)
        return -loglik_reparam(theta_full, Y, Z, params_scale, update_idx)

    if algorithm == "lbfgs":
        opt_init, opt_update = optax.lbfgs()
    else:
        raise Exception("Not implemented.")

    state = opt_init(theta)

    maxiter = 100
    trace = []

    for _ in range(maxiter):
        loss, grad = jax.value_and_grad(objective)(theta)

        updates, state = opt_update(
            grad, state, theta,
            value=loss,
            grad=grad,
            value_fn=objective
        )

        theta = optax.apply_updates(theta, updates)

        grad_norm = optax.tree.norm(grad)
        update_norm = optax.tree.norm(updates)
        trace.append({
            "loss": float(loss),
            "grad_norm": grad_norm,
            "update_norm": update_norm,
        })

        # print(f"Parameter norm: {update_norm}, grad norm: {grad_norm}")
        converged = (grad_norm < 1e-4) or (update_norm < 1e-4)
        if converged:
            conv_state = dict(status=0, trace=trace)
            theta_hat = theta
            return theta_hat, [], [], conv_state

    theta_hat = theta
    conv_state = dict(status=1, trace=trace)
    return theta_hat, [], [], conv_state


if __name__ == "__main__":
    from dgp import simulate_VAR_attention, flatten_params

    # Set random number seed
    key = jax.random.PRNGKey(0)

    # Example parameter dictionary (in natural scale)
    algorithm = "lbfgs"
    # algorithm = "auto_gradhess"

    p, q, T = 3, 2, 100
    params = dict(
        eta_a=0.5,
        gamma_a=jax.random.uniform(key, (p,), minval=-0.2, maxval=0.2),
        beta=jax.random.uniform(key, (p,), minval=-0.3, maxval=0.3),
        sigma2=jnp.exp(jax.random.normal(key, (p,)))  # positive σ²
    )

    params_scale = dict(eta_a=1)

    Y, Z, W_a = simulate_VAR_attention(key, params, T=T, p=p, q=q,
                                       params_scale=params_scale, dist="normal")

    # Convert σ² → σ̃² for reparameterized estimation
    sigma2_tilde = jnp.log(jnp.expm1(params["sigma2"]))
    params_reparam = params.copy()
    params_reparam["sigma2"] = sigma2_tilde

    # --- Flatten parameters into reparameterized vector θ̃ ---
    block_order = {
        "eta_a": [0],
        "gamma_a": list(range(1, 1 + p)),
        "beta": list(range(1 + p, 1 + 2 * p)),
        "sigma2": list(range(1 + 2 * p, 1 + 3 * p)),
    }

    theta_tilde, _ = flatten_params(params_reparam, block_order)

    theta_init = theta_tilde + 0.05 * jax.random.normal(key, theta_tilde.shape)

    theta_hat, grad_ll, hess_ll, state = mle_optimize(Y, Z, theta_init, params_scale, algorithm)

    print("True theta:", theta_tilde)
    print("Initial theta:", theta_init)
    print("Estimated theta:", theta_hat)
    # print("Gradient at optimum:", grad_ll)
    print(f"Converge state: {state}")
