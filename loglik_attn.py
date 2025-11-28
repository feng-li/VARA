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

    # Mask diagonal (no self-weighting) by replacing with -inf. This makes sure diag of W is zero.
    mask = jnp.eye(p) * -1e12

    scores = scores * scale + mask

    # Softmax row-wise
    W = jax.nn.softmax(scores, axis=1)

    return W


def loglik(theta, Y, Z, params_scale):
    """
    Log-quasi-likelihood function with attention-based W_a, W_b.

    Y: (T, p) time series data
    Z: (p, q) feature embeddings
    theta: parameter vector
    """
    T, p = Y.shape

    # S, G, sigma2 = construct_matrices(theta, Z, p)
    eta_a = theta[0]
    eta_b = theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta = theta[2+2*p:2+3*p]
    sigma2 = theta[2+3*p:2+4*p]

    # Build attention weights
    W_a = build_attention(Z, eta_a, params_scale['eta_a'])
    W_b = build_attention(Z, eta_b, params_scale['eta_b'])

    # Diagonal matrices
    D_beta = jnp.diag(beta)
    D_ga = jnp.diag(gamma_a)
    D_gb = jnp.diag(gamma_b)

    # Model matrices
    G = D_beta + D_ga @ W_a
    S = jnp.eye(p) - D_gb @ W_b


    # Covariance matrix of errors
    inv_Sigma_e = jnp.diag(1.0 / sigma2)

    # Transformed errors
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)   # shape (T-1, p)

    # Quadratic form
    quad = jnp.sum((E @ inv_Sigma_e) * E)

    # Log-likelihood
    val = (
        -(T-1) * p / 2 * jnp.log(2 * jnp.pi)
        - (T-1) / 2 * jnp.sum(jnp.log(sigma2))
        + (T-1) * jnp.linalg.slogdet(S)[1]   # log|S|
        - 0.5 * quad
    )
    return val


def loglik_reparam(theta, Y, Z, params_scale, update_idx=slice(None)):
    """
    Compute log-likelihood with gradient only for selected parameters.

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
    eta_b = theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]
    sigma2_tilde = theta[2+3*p:2+4*p]
    sigma2 = jax.nn.softplus(sigma2_tilde)   # Replace σ̃² with σ²
    log_sigma2 = jnp.log(sigma2)

    W_a = build_attention(Z, eta_a, params_scale['eta_a'])
    W_b = build_attention(Z, eta_b, params_scale['eta_b'])

    D_beta = jnp.diag(beta)
    D_ga = jnp.diag(gamma_a)
    D_gb = jnp.diag(gamma_b)

    G = D_beta + D_ga @ W_a
    S = jnp.eye(p) - D_gb @ W_b

    inv_Sigma_e = jnp.diag(1.0 / sigma2)
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)
    quad = jnp.sum((E @ inv_Sigma_e) * E)

    signS, logdetS = jnp.linalg.slogdet(S)

    val = (
        -(T-1) * p / 2 * jnp.log(2 * jnp.pi)
        - (T-1) / 2 * jnp.sum(log_sigma2)
        + (T-1) * logdetS
        - 0.5 * quad
    )
    return val / (T-1)


# --------------------------------------------------------
# Example usage
# --------------------------------------------------------
if __name__ == "__main__":
    # Example dimensions
    T, p, q = 10000, 5000, 50
    Y = jnp.ones((T, p))  # dummy time series data
    Z = jax.random.normal(jax.random.PRNGKey(0), (p, q))  # feature embeddings

    # Parameter vector: (eta_a, eta_b, gamma_a, gamma_b, beta, sigma2)
    theta0 = jnp.concatenate([
        jnp.array([2, 2]),        # eta_a, eta_b
        jnp.ones(p) * 0.1,            # gamma_a
        jnp.ones(p) * 0.1,            # gamma_b
        jnp.ones(p) * 0.5,            # beta
        jnp.ones(p) * 1.0             # sigma2
    ])
    params_scale = dict(
        eta_a=1,
        eta_b=10   )

    # Compute log-likelihood
    ll_val = loglik(theta0, Y, Z, params_scale)
    print("Log-likelihood:", ll_val)

    grad_ll = grad(loglik)(theta0, Y, Z, params_scale)
    print("Gradient shape:", grad_ll.shape)
    print(grad_ll)

    hess_ll = hessian(loglik)(theta0, Y, Z, params_scale)
    # print("Hessian shape:", hess_ll.shape)
    # print(hess_ll)
