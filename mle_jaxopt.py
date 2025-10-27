import jax
import jax.numpy as jnp

from jaxopt import LBFGS

from loglik_attn import loglik_reparam


def mle_bfgs(Y, Z, theta_init, params_scale, algorithm):
    """
    MLE estimator using JAXopt BFGS optimizer.
    Minimizes the negative log-likelihood.
    """

    # Define objective: negative log-likelihood
    def objective(theta):
        return -loglik_reparam(theta, Y, Z, params_scale)

    # Initialize BFGS solver

    if algorithm == "bfgs":
        solver = LBFGS(fun=objective, maxiter=50, tol=1e-6, verbose=False)
    else:
        raise Exception("Not implemented.")

    # Run optimization
    res = solver.run(theta_init)

    # Gradient and Hessian (optional)
    theta_hat = res.params
    state = res.state
    # grad_ll = jax.grad(loglik_reparam)(theta_hat, Y, Z)
    # hess_ll = jax.hessian(loglik_reparam)(theta_hat, Y, Z)
    grad_ll = []
    hess_ll = []

    return theta_hat, grad_ll, hess_ll, state


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

    params_scale = dict(
        eta_a=1,
        eta_b=10)

    Y, Z, W_a, W_b = simulate_VAR_attention(key, params, T=T, p=p, q=q,
                                            params_scale=params_scale, dist="normal")

    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    # Convert σ² → σ̃² for reparameterized estimation
    sigma2_tilde = jnp.log(jnp.expm1(params["sigma2"]))
    params_reparam = params.copy()
    params_reparam["sigma2"] = sigma2_tilde

    # Flatten parameter vector θ̃ (for estimation/Hessian functions)
    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    theta_tilde = flatten_params(params_reparam, order)

    theta_init = theta_tilde + 0.05 * jax.random.normal(key, theta_tilde.shape)


    theta_hat, grad_ll, hess_ll, state = mle_bfgs(Y, Z, theta_init, algorithm)

    print("True theta:", theta_tilde)
    print("Initial theta:",  theta_init)
    print("Estimated theta:", theta_hat)
    # print("Gradient at optimum:", grad_ll)
    print("Converged:", state.error < 1e-6)
