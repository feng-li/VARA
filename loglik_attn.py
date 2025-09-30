import jax
import jax.numpy as jnp
from jax import grad, hessian

# --------------------------------------------------------
# Attention weight construction
# --------------------------------------------------------

def build_attention(Z, eta):
    """
    Construct an attention weight matrix W from feature vectors Z and scale eta.
    Z: (p, q) feature matrix, each row Z[k] is the feature vector of series k
    eta: scalar temperature parameter
    Returns W: (p, p) attention weight matrix with rows summing to 1
    """
    p, q = Z.shape
    # Similarity scores
    scores = (Z @ Z.T) / jnp.sqrt(q)     # shape (p, p)
    scores = eta * scores

    # Mask diagonal (no self-weighting) by replacing with -inf
    mask = jnp.eye(p) * -1e9
    scores = scores + mask

    # Softmax row-wise
    W = jax.nn.softmax(scores, axis=1)
    return W

# --------------------------------------------------------
#  Model components
# --------------------------------------------------------

def construct_matrices(theta, Z, p):
    """
    Build S and G matrices given theta and feature embeddings Z.
    theta order: (eta_a, eta_b, gamma_a, gamma_b, beta, sigma2)
    """
    eta_a = theta[0]
    eta_b = theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]
    sigma2  = theta[2+3*p:2+4*p]

    # Build attention weights
    W_a = build_attention(Z, eta_a)
    W_b = build_attention(Z, eta_b)

    # Diagonal matrices
    D_beta   = jnp.diag(beta)
    D_ga     = jnp.diag(gamma_a)
    D_gb     = jnp.diag(gamma_b)

    # Model matrices
    G = D_beta + D_ga @ W_a
    S = jnp.eye(p) - D_gb @ W_b

    return S, G, sigma2


def loglik(theta, Y, Z):
    """
    Log-quasi-likelihood function with attention-based W_a, W_b.
    Y: (T, p) time series data
    Z: (p, q) feature embeddings
    theta: parameter vector
    """
    T, p = Y.shape
    S, G, sigma2 = construct_matrices(theta, Z, p)

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

# --------------------------------------------------------
# Example usage
# --------------------------------------------------------
if __name__ == "__main__":
    # Example dimensions
    T, p, q = 50, 4, 8
    Y = jnp.ones((T, p))  # dummy time series data
    Z = jax.random.normal(jax.random.PRNGKey(0), (p, q))  # feature embeddings

    # Parameter vector: (eta_a, eta_b, gamma_a, gamma_b, beta, sigma2)
    theta0 = jnp.concatenate([
        jnp.array([0.5, 0.5]),        # eta_a, eta_b
        jnp.ones(p) * 0.1,            # gamma_a
        jnp.ones(p) * 0.1,            # gamma_b
        jnp.ones(p) * 0.5,            # beta
        jnp.ones(p) * 1.0             # sigma2
    ])

    # Compute log-likelihood
    ll_val = loglik(theta0, Y, Z)
    print("Log-likelihood:", ll_val)

    # Gradient and Hessian
    grad_ll = grad(loglik)(theta0, Y, Z)
    hess_ll = hessian(loglik)(theta0, Y, Z)

    print("Gradient shape:", grad_ll.shape)
    print(grad_ll)
    print("Hessian shape:", hess_ll.shape)
    print(hess_ll)
