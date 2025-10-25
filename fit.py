import jax
import jax.numpy as jnp
from cov import compute_sandwich_variance
from mle_ad import mle_newton
from mle_jaxopt import mle_bfgs


def estimate(Y, Z, key, theta_init, algorithm, update_idx=slice(None)):
    """
    Perform Quasi–Maximum Likelihood Estimation (QMLE) under the reparameterized model.

    This routine estimates model parameters θ = (ηₐ, η_b, γₐ, γ_b, β, σ²)
    using a Newton-based optimizer operating in an unconstrained parameter space.
    The variance parameters σ² are reparameterized via the softplus transformation:
        σ² = softplus(σ̃²) = log(1 + exp(σ̃²))
    to ensure positivity during optimization.

    Workflow:
        1. Calls `mle_newton` to obtain θ̂ in reparameterized form (`theta_hat_reparam`).
        2. Extracts unconstrained variance parameters σ̃² and maps them back to
           the natural scale σ² using `jax.nn.softplus`.
        3. Concatenates all parameters into θ̂ (natural scale).
        4. Computes the sandwich variance estimator of θ̂ via `compute_sandwich_variance`.

    Args:
        Y (jax.Array): Observed time series matrix of shape (T, p).
        Z (jax.Array): Feature matrix used to build attention weights, shape (p, q).
        key (jax.random.PRNGKey): Random key for reproducibility (not directly used here).
        theta_init (jax.Array): Initial parameter vector in reparameterized form.
        verbose (bool): Whether to print intermediate optimization details.

    Returns:
        var_theta (jax.Array):
            Estimated asymptotic covariance matrix of θ̂ on the natural scale,
            computed using the sandwich (robust) variance estimator.

    Notes:
        - The function assumes that both `mle_newton` and `compute_sandwich_variance`
          are compatible with JAX and operate on the given data structures.
        - The output variance is computed at θ̂ after transforming σ̃² → σ².
    """
    T, p = Y.shape

    # Estimation (theta_hat_reparam contains unconstrained parameters)

    if algorithm == "analytical_gradhess" or algorithm == "auto_gradhess":
        theta_hat_reparam_update, grad_ll, hess_ll = mle_newton(Y, Z, theta_init, algorithm, update_idx=update_idx)

    elif algorithm == "bfgs":
        theta_hat_reparam_update, grad_ll, hess_ll, res = mle_bfgs(Y, Z, theta_init, algorithm="bfgs")

    else:
        raise Exception("No such algorithm implemented")

    # breakpoint()

    # slice(None) means update all parameters
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_init.size)
    update_idx = jnp.array(update_idx)

    theta_hat_reparam = theta_init
    theta_hat_reparam = theta_hat_reparam.at[update_idx].set(theta_hat_reparam_update)

    # Extract unconstrained variance parameter (σ̃²)
    sigma2_tilde = theta_hat_reparam[2 + 3 * p : 2 + 4 * p]

    # Convert back to natural scale σ² = softplus(σ̃²)
    sigma2_hat = jax.nn.softplus(sigma2_tilde)

    # Replace σ̃² with σ² in final θ̂
    theta_hat = jnp.concatenate([theta_hat_reparam[: 2 + 3 * p], sigma2_hat])

    # Compute sandwich variance at θ̂ (natural scale)
    var_theta = compute_sandwich_variance(Y, Z, theta_hat)

    return var_theta
