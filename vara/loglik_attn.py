import jax
import jax.numpy as jnp
from jax import grad, hessian


def build_attention(Z, eta, scale):
    """
    Construct an attention weight matrix W from feature vectors Z and scale eta.
    Z: (p, q) feature matrix, each row Z[k] is the feature vector of series k
    eta: scalar temperature parameter
    scale Larger scale → sharper changes in W when η changes → easier for MLE to find the right η.
    Returns W: (p, p) attention weight matrix with rows summing to 1
    """
    p, q = Z.shape
    # Similarity scores
    scores = (Z @ Z.T) / jnp.sqrt(q)     # shape (p, p)
    scores = eta * scores

    # Mask diagonal (no or less self-weighting) by replacing with -inf. This makes sure diag of W is zero.
    mask = jnp.eye(p) * -1e6

    scores = scores * scale + mask

    # Softmax row-wise
    W = jax.nn.softmax(scores, axis=1)

    return W


def loglik(theta, Y, Z, params_scale):
    """
    Log-quasi-likelihood (avg) function with attention-based W_a, W_b.

    Y: (T, p) time series data
    Z: (p, q) feature embeddings
    theta: parameter vector
    """
    T, p = Y.shape

    # S, G, sigma2 = construct_matrices(theta, Z, p)
    eta_a = theta[0]
    # eta_max = 2
    # eta_a = jnp.clip(theta[0], -eta_max, eta_max)
    gamma_a = theta[1:1+p]
    beta = theta[1+p:1+2*p]
    sigma2 = theta[1+2*p:1+3*p]

    # Build attention weights
    W_a = build_attention(Z, eta_a, params_scale['eta_a'])
    # W_b = build_attention(Z, eta_b, params_scale['eta_b'])

    G = (beta[:, None] * jnp.eye(p)) + (gamma_a[:, None] * W_a)
    # S = jnp.eye(p) - (gamma_b[:, None] * W_b)

    # Transformed errors
    # E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)   # shape (T-1, p)
    E = Y[1:] - (Y[:-1] @ G.T)   # shape (T-1, p)

    # Quadratic form
    # Covariance matrix of errors
    # sigma2_inv = jnp.diag(1.0 / sigma2)
    # quad = jnp.sum((E @ sigma2_inv) * E)
    sigma2_inv = 1.0 / sigma2    # shape (p, ) == (1, p) implicitly

    quad = jnp.sum(E ** 2 * sigma2_inv)/(T-1)

    # Log-likelihood
    val = (
        - p / 2 * jnp.log(2 * jnp.pi)
        - 1 / 2 * jnp.sum(jnp.log(sigma2))
        - 1/2 * quad
    )

    return val


def loglik_reparam(theta, Y, Z, params_scale, update_idx=slice(None)):
    """
    Compute log-likelihood (avg) with only for selected parameters.

    Args:
        theta: full parameter vector (1D array)
        Y, Z: data matrices
        update_idx: indices (int, list, or slice) of parameters to differentiate w.r.t.
    Returns:
        Scalar log-likelihood, differentiable only w.r.t. theta[update_idx].
    """

    # slice(None) means update all parameters
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta.size)
    update_idx = jnp.array(update_idx)

    # extract the subset to be optimized
    theta_sub = theta[update_idx]

    # start from the current theta (structure preserved)
    theta = jax.lax.stop_gradient(theta)
    # reinsert differentiable subset back
    theta = theta.at[update_idx].set(theta_sub)

    T, p = Y.shape
    eta_a = theta[0]
    # eta_max = 2
    # eta_a = jnp.clip(theta[0], -eta_max, eta_max)
    # eta_b = theta[1]
    # gamma_a = theta[2:2+p]
    gamma_a = theta[1:1+p]
    # gamma_b = theta[2+p:2+2*p]
    # beta = theta[2+2*p:2+3*p]
    beta = theta[1+p:1+2*p]
    # sigma2 = theta[2+3*p:2+4*p]
    sigma2_tilde = theta[1+2*p:1+3*p]

    sigma2 = jax.nn.softplus(sigma2_tilde)   # Replace σ̃² with σ²
    log_sigma2 = jnp.log(sigma2)

    W_a = build_attention(Z, eta_a, params_scale['eta_a'])
    # W_b = build_attention(Z, eta_b, params_scale['eta_b'])

    G = (beta[:, None] * jnp.eye(p)) + (gamma_a[:, None] * W_a)
    # S = jnp.eye(p) - (gamma_b[:, None] * W_b)

    # sigma2_inv = jnp.diag(1.0 / sigma2)
    # E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)
    E = Y[1:] - (Y[:-1] @ G.T)

    # quad = jnp.sum((E @ sigma2_inv) * E)
    quad = jnp.sum(E **2 * (1.0 / sigma2))/(T-1)

    val = (
        - p / 2 * jnp.log(2 * jnp.pi)
        - 1 / 2 * jnp.sum(log_sigma2)
        - 1/2 * quad
    )

    return val


# ===============================================================
# === Used for Autodiff-based V and H for validation ============
# ===============================================================

def loglik_t_natural(theta, y_t, y_tm1, Z, params_scale, T):
    """
    Single-time contribution consistent with current loglik(theta, Y, Z, params_scale)
    in loglik_attn.py.

    Natural-scale parameterization:
        theta = (eta_a, gamma_a, beta, sigma2)

    Summing this from t=1,...,T-1 reproduces the current total log-likelihood.
    """
    p = y_t.shape[0]

    eta_a = theta[0]
    # eta_max = 2
    # eta_a = jnp.clip(theta[0], -eta_max, eta_max)
    gamma_a = theta[1:1+p]
    beta = theta[1+p:1+2*p]
    sigma2 = theta[1+2*p:1+3*p]

    W_a = build_attention(Z, eta_a, params_scale["eta_a"])
    G = (beta[:, None] * jnp.eye(p)) + (gamma_a[:, None] * W_a)

    e_t = y_t - G @ y_tm1

    ll_t = (
        - (p / 2.0) * jnp.log(2.0 * jnp.pi) / (T - 1)
        - 0.5 * jnp.sum(jnp.log(sigma2)) / (T - 1)
        - 0.5 * jnp.sum((e_t ** 2) / sigma2) / (T - 1)
    )
    return ll_t


def loglik_natural_from_t(theta, Y, Z, params_scale):
    """
    Rebuild the natural-scale total log-likelihood from per-time terms.

    This wrapper is mainly used for autodiff validation, so that gradients
    and Hessians can be compared against analytical expressions built from
    the single-time contribution :func:`loglik_t_natural`.

    Parameters
    ----------
    theta : jax.Array
        Natural-scale parameter vector.
    Y, Z : jax.Array
        Observed data and feature matrix.
    params_scale : dict
        Scaling constants for the attention map.

    Returns
    -------
    jax.Array
        Scalar total log-likelihood obtained by summing over ``t = 1, ..., T-1``.
    """
    T, _ = Y.shape
    vals = []
    for t in range(1, T):
        vals.append(loglik_t_natural(theta, Y[t], Y[t-1], Z, params_scale, T))
    return jnp.sum(jnp.stack(vals))


def loglik_t_reparam_consistent(theta, y_t, y_tm1, Z, params_scale, T):
    """
    Single-time contribution consistent with current loglik_reparam(...)
    in loglik_attn.py.

    Reparameterized parameterization:
        theta = (eta_a, gamma_a, beta, sigma2_tilde)
    """
    p = y_t.shape[0]

    eta_a = theta[0]
    # eta_max = 2
    # eta_a = jnp.clip(theta[0], -eta_max, eta_max)
    gamma_a = theta[1:1+p]
    beta = theta[1+p:1+2*p]
    sigma2_tilde = theta[1+2*p:1+3*p]

    sigma2 = jax.nn.softplus(sigma2_tilde)
    log_sigma2 = jnp.log(sigma2)

    W_a = build_attention(Z, eta_a, params_scale["eta_a"])
    G = (beta[:, None] * jnp.eye(p)) + (gamma_a[:, None] * W_a)

    e_t = y_t - G @ y_tm1

    ll_t = (
        - (p / 2.0) * jnp.log(2.0 * jnp.pi) / (T - 1)
        - 0.5 * jnp.sum(log_sigma2) / (T - 1)
        - 0.5 * jnp.sum((e_t ** 2) / sigma2) / (T - 1)
    )
    return ll_t


def loglik_reparam_from_t_consistent(theta, Y, Z, params_scale):
    """
    Rebuild the reparameterized total log-likelihood from per-time terms.

    Parameters
    ----------
    theta : jax.Array
        Optimization-scale parameter vector.
    Y, Z : jax.Array
        Observed data and feature matrix.
    params_scale : dict
        Scaling constants for the attention map.

    Returns
    -------
    jax.Array
        Scalar total log-likelihood consistent with
        :func:`loglik_t_reparam_consistent`.
    """
    T, _ = Y.shape
    vals = []
    for t in range(1, T):
        vals.append(loglik_t_reparam_consistent(theta, Y[t], Y[t-1], Z, params_scale, T))
    return jnp.sum(jnp.stack(vals))

# --------------------------------------------------------
# Example usage
# --------------------------------------------------------
if __name__ == "__main__":
    # Example dimensions
    # breakpoint()
    T, p, q = 10000, 20, 10
    Y = jnp.ones((T, p))  # dummy time series data
    Z = jax.random.normal(jax.random.PRNGKey(0), (p, q))  # feature embeddings

    # Parameter vector: (eta_a, eta_b, gamma_a, gamma_b, beta, sigma2)
    theta0 = jnp.concatenate([
        jnp.array([2]),        # eta_a, eta_b
        jnp.ones(p) * 0.1,            # gamma_a
        jnp.ones(p) * 0.5,            # beta
        jnp.ones(p) * 1.0             # sigma2
    ])
    params_scale = dict(
        eta_a=1)

    # Compute log-likelihood
    ll_val = loglik(theta0, Y, Z, params_scale)
    print("Log-likelihood:", ll_val)

    grad_ll = grad(loglik)(theta0, Y, Z, params_scale)
    print("Gradient shape:", grad_ll.shape)
    print(grad_ll)

    hess_ll = hessian(loglik)(theta0, Y, Z, params_scale)
    # print("Hessian shape:", hess_ll.shape)
    # print(hess_ll)


    # Compute log-likelihood
    ll_rep = loglik_reparam(theta0, Y, Z, params_scale)
    print("Log-likelihood:", ll_rep)

    grad_rep = grad(loglik_reparam)(theta0, Y, Z, params_scale)
    print("Gradient shape:", grad_rep.shape)
    print(grad_rep)

    hess_rep = hessian(loglik_reparam)(theta0, Y, Z, params_scale)
