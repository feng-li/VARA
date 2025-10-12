import jax.numpy as jnp
from cov import compute_sandwich_variance
from mle_ad import mle_newton


def estimate(Y, Z, key, theta_init):
    """Estimation routine using QMLE functions"""

    T, p = Y.shape

    # Estimation
    theta_hat_reparam, grad_ll, hess_ll = mle_newton(Y, Z, theta_init)

    # Convert back to natural scale (replace log_sigma2 with sigma2)
    log_sigma2_hat = theta_hat_reparam[2+3*p:2+4*p]
    sigma2_hat = jnp.exp(log_sigma2_hat)
    theta_hat = jnp.concatenate([theta_hat_reparam[:2+3*p], sigma2_hat])

    # Variance, confidence interval
    var_theta = compute_sandwich_variance(Y, Z, theta_hat)

    return var_theta
