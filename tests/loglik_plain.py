import jax.numpy as jnp
from jax import grad, hessian

from .loglik_attn import construct_matrices


def loglik_plain(theta, Y, W_a, W_b):
    """
    Log-quasi-likelihood function.
    Y: array shape (T, p)
    W_a, W_b: weight matrices
    theta: parameter vector
    """
    T, p = Y.shape
    S, G, sigma2 = construct_matrices(theta, W_a, W_b, p)

    # Covariance matrix of errors
    # Sigma_e = jnp.diag(sigma2)
    inv_Sigma_e = jnp.diag(1.0 / sigma2)

    # Transformed errors
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)   # shape (T-1, p)

    # Quadratic form
    quad = jnp.sum((E @ inv_Sigma_e) * E)   # sum over all rows

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
    T, p = 100, 5
    Y = jnp.ones((T, p))  # dummy data
    W_a = jnp.eye(p)      # placeholder
    W_b = jnp.eye(p)      # placeholder

    # Parameter vector (eta_a, eta_b, gamma_a, gamma_b, beta, sigma2)
    theta0 = jnp.concatenate([
        jnp.array([0.5, 0.5]),        # eta_a, eta_b
        jnp.ones(p) * 0.1,            # gamma_a
        jnp.ones(p) * 0.1,            # gamma_b
        jnp.ones(p) * 0.5,            # beta
        jnp.ones(p) * 1.0             # sigma2
    ])

    # Compute log-likelihood
    ll_val = loglik_plain(theta0, Y, W_a, W_b)
    print("Log-likelihood:", ll_val)

    # Gradient and Hessian
    grad_ll = grad(loglik_plain)(theta0, Y, W_a, W_b)
    hess_ll = hessian(loglik_plain)(theta0, Y, W_a, W_b)

    print("Gradient shape:", grad_ll.shape)
    print(grad_ll)
    print("Hessian shape:", hess_ll.shape)
    print(hess_ll)
