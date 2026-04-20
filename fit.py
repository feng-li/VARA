import jax
import jax.numpy as jnp
# from cov import compute_sandwich_variance
from mle_ad import mle_newton
from yule_walker_ad import yule_walker
# from mle_jaxopt import mle_bfgs # not maintained


import jax
import jax.numpy as jnp

from cov import compute_sandwich_variance
from mle_ad import mle_newton, mle_newton_gibbs, mle_newton_hybrid, mle_optimize
from yule_walker_ad import yule_walker
# from mle_jaxopt import mle_bfgs   # not maintained


def estimate(
    Y,
    Z,
    key,
    theta_init,
    params_scale,
    algorithm,
    update_idx=slice(None),
    block_order=None,
):
    """
    Perform estimation under the current 4-block old model:
        theta = (eta_a, gamma_a, beta, sigma2)

    Parameters
    ----------
    Y : jax.Array
        Observed response matrix of shape (T, p).
    Z : jax.Array
        Feature matrix of shape (p, q).
    key : jax.random.PRNGKey
        Reserved for compatibility; not directly used here.
    theta_init : jax.Array
        Initial parameter vector on the reparameterized scale.
    params_scale : dict
        Scaling dictionary used in attention reparameterization.
    algorithm : str
        One of:
            "analytical_gradhess"
            "auto_gradhess"
            "yule-walker"
            "block_newton"
            "hybrid_block2full"
    update_idx : slice or array-like
        Parameter indices to update.
    block_order : dict or None
        Required for "block_newton" and "hybrid_block2full".
        Example for current model:
            {
                "eta_a": [0],
                "gamma_a": list(range(1, 1+p)),
                "beta": list(range(1+p, 1+2*p)),
                "sigma2": list(range(1+2*p, 1+3*p)),
            }

    Returns
    -------
    var_theta : dict
        Output from compute_sandwich_variance(...), augmented with theta.
    convergence : dict
        Optimizer convergence information.
    """
    T, p = Y.shape

    # -----------------------------------------------------------
    # 1. Run optimization on reparameterized scale
    # -----------------------------------------------------------
    if algorithm in {"analytical_gradhess", "auto_gradhess"}:
        theta_hat_reparam_update, grad_ll, hess_ll, convergence = mle_newton(
            Y=Y,
            Z=Z,
            theta_init=theta_init,
            params_scale=params_scale,
            algorithm=algorithm,
            update_idx=update_idx,
        )

    elif algorithm == "yule-walker":
        theta_hat_reparam_update, grad_ll, hess_ll, convergence = yule_walker(
            Y=Y,
            Z=Z,
            theta_init=theta_init,
            params_scale=params_scale,
            update_idx=update_idx,
        )

    elif algorithm == "block_newton":
        if block_order is None:
            raise ValueError("block_order must be provided for algorithm='block_newton'.")

        theta_hat_reparam_update, grad_ll, hess_ll, convergence = mle_newton_gibbs(
            Y=Y,
            Z=Z,
            theta_init=theta_init,
            params_scale=params_scale,
            block_order=block_order,
            algorithm="analytical_gradhess",
            update_idx=update_idx,
        )

    elif algorithm == "hybrid_block2full":
        if block_order is None:
            raise ValueError("block_order must be provided for algorithm='hybrid_block2full'.")

        theta_hat_reparam_update, grad_ll, hess_ll, convergence = mle_newton_hybrid(
            Y=Y,
            Z=Z,
            theta_init=theta_init,
            params_scale=params_scale,
            block_order=block_order,
            algorithm="analytical_gradhess",
            update_idx=update_idx,
        )

    elif algorithm == "lbfgs":
        theta_hat_reparam_short_opt, grad_ll_opt, hess_ll_opt, convergence_opt = mle_optimize(
            Y=Y,
            Z=Z,
            theta_init=theta_init,
            params_scale=params_scale,
            algorithm="lbfgs",
            update_idx=update_idx,
        )


    else:
        raise ValueError(f"No such algorithm implemented: {algorithm}")

    # -----------------------------------------------------------
    # 2. Put updated entries back into full theta vector
    # -----------------------------------------------------------
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_init.size)
    update_idx = jnp.array(update_idx)

    theta_hat_reparam = theta_init
    theta_hat_reparam = theta_hat_reparam.at[update_idx].set(theta_hat_reparam_update)

    # -----------------------------------------------------------
    # 3. Convert sigma2 from reparameterized scale to natural scale
    # -----------------------------------------------------------
    sigma2_tilde = theta_hat_reparam[1 + 2 * p : 1 + 3 * p]
    sigma2_hat = jax.nn.softplus(sigma2_tilde)

    theta_hat = jnp.concatenate([
        theta_hat_reparam[: 1 + 2 * p],
        sigma2_hat,
    ])

    # -----------------------------------------------------------
    # 4. Compute sandwich variance on natural scale
    # -----------------------------------------------------------
    var_theta = compute_sandwich_variance(
        Y=Y,
        Z=Z,
        theta_hat=theta_hat,
        params_scale=params_scale,
    )

    return var_theta, convergence
