import jax
import jax.numpy as jnp
import optax

from loglik_attn import loglik_reparam


def mle_optimize(Y, Z, theta_init, params_scale, algorithm):
    """
    MLE estimator using Optax LBFGS (line-search version).
    """

    # Negative log-likelihood
    def objective(theta):
        return -loglik_reparam(theta, Y, Z, params_scale)

    if algorithm == "lbfgs":
        opt_init, opt_update = optax.lbfgs()
    else:
        raise Exception("Not implemented.")

    theta = theta_init
    state = opt_init(theta)

    maxiter = 50

    for _ in range(maxiter):
        loss, grad = jax.value_and_grad(objective)(theta)

        # ★★★ correct required signature ★★★
        updates, state = opt_update(
            grad, state, theta,
            value=loss,
            grad=grad,
            value_fn=objective
        )

        theta = optax.apply_updates(theta, updates)

        if jnp.linalg.norm(updates) < 1e-6:
            break

    theta_hat = theta

    return theta_hat, [], [], state



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

    order = ["eta_a", "gamma_a", "beta", "sigma2"]
    # Convert σ² → σ̃² for reparameterized estimation
    sigma2_tilde = jnp.log(jnp.expm1(params["sigma2"]))
    params_reparam = params.copy()
    params_reparam["sigma2"] = sigma2_tilde

    # Flatten parameter vector θ̃ (for estimation/Hessian functions)
    theta_tilde, _ = flatten_params(params_reparam, order)

    theta_init = theta_tilde + 0.05 * jax.random.normal(key, theta_tilde.shape)

    theta_hat, grad_ll, hess_ll, state = mle_optimize(Y, Z, theta_init, params_scale, algorithm)

    print("True theta:", theta_tilde)
    print("Initial theta:", theta_init)
    print("Estimated theta:", theta_hat)
    # print("Gradient at optimum:", grad_ll)
    print(f"Converge state: {state}")
