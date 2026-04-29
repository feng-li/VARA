import jax
import jax.numpy as jnp
from .loglik_attn import loglik_t_natural, loglik_natural_from_t
from .loglik_attn import loglik_t_reparam_consistent, loglik_reparam_from_t_consistent

def compute_scores_t_auto_natural(theta_hat, Y, Z, params_scale):
    """
    Per-time score contributions on the natural scale.
    Returns array of shape (T-1, d).
    """
    T, _ = Y.shape
    grad_t = jax.grad(loglik_t_natural)

    scores = []
    for t in range(1, T):
        s_t = grad_t(theta_hat, Y[t], Y[t-1], Z, params_scale, T)
        scores.append(s_t)

    return jnp.stack(scores, axis=0)


def compute_V_auto_natural(theta_hat, Y, Z, params_scale):
    """
    Sample analog of V on the natural scale:
        V_auto = sum_t s_t s_t^T
    """
    scores = compute_scores_t_auto_natural(theta_hat, Y, Z, params_scale)
    V_auto = scores.T @ scores
    return V_auto


def compute_H_auto_natural(theta_hat, Y, Z, params_scale):
    """
    Autodiff Hessian on the natural scale:
        H_auto = - Hessian of full natural-scale loglik
    """
    hess_fn = jax.hessian(loglik_natural_from_t)
    H_auto = -hess_fn(theta_hat, Y, Z, params_scale)
    return H_auto


def compute_scores_t_auto_reparam(theta_hat_reparam, Y, Z, params_scale):
    """
    Per-time score contributions on the reparameterized scale.
    Useful for checking optimization-scale derivatives, but not directly
    comparable to analytical V/H in cov.py.
    """
    T, _ = Y.shape
    grad_t = jax.grad(loglik_t_reparam_consistent)

    scores = []
    for t in range(1, T):
        s_t = grad_t(theta_hat_reparam, Y[t], Y[t-1], Z, params_scale, T)
        scores.append(s_t)

    return jnp.stack(scores, axis=0)


def compute_V_auto_reparam(theta_hat_reparam, Y, Z, params_scale):
    """
    Form the autodiff score covariance on the reparameterized scale.

    Parameters
    ----------
    theta_hat_reparam : jax.Array
        Parameter vector on the optimization scale.
    Y, Z : jax.Array
        Observed data and feature matrix.
    params_scale : dict
        Scaling constants for the attention map.

    Returns
    -------
    jax.Array
        Matrix ``sum_t s_t s_t^T`` where ``s_t`` is the reparameterized
        per-time score contribution.
    """
    scores = compute_scores_t_auto_reparam(theta_hat_reparam, Y, Z, params_scale)
    V_auto = scores.T @ scores
    return V_auto


def compute_H_auto_reparam(theta_hat_reparam, Y, Z, params_scale):
    """
    Compute the autodiff Hessian on the reparameterized scale.

    Parameters
    ----------
    theta_hat_reparam : jax.Array
        Parameter vector on the optimization scale.
    Y, Z : jax.Array
        Observed data and feature matrix.
    params_scale : dict
        Scaling constants for the attention map.

    Returns
    -------
    jax.Array
        Negative Hessian of the reparameterized total log-likelihood.
    """
    hess_fn = jax.hessian(loglik_reparam_from_t_consistent)
    H_auto = -hess_fn(theta_hat_reparam, Y, Z, params_scale)
    return H_auto
