import jax.numpy as jnp

from loglik_attn import build_attention


def loglik_grad_analytical(theta, Y, Z):
    """
    Analytical gradient of log-likelihood ℓ(θ).
    θ = (eta_a, eta_b, gamma_a, gamma_b, beta, sigma2)
    """
    T, p = Y.shape
    eta_a = theta[0]
    eta_b = theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]
    sigma2  = theta[2+3*p:2+4*p]

    # Build weights
    W_a = build_attention(Z, eta_a)
    W_b = build_attention(Z, eta_b)

    D_beta = jnp.diag(beta)
    D_ga = jnp.diag(gamma_a)
    D_gb = jnp.diag(gamma_b)

    G = D_beta + D_ga @ W_a
    S = jnp.eye(p) - D_gb @ W_b
    S_inv = jnp.linalg.inv(S)

    # Inverse Sigma_e
    Sigma_inv = jnp.diag(1.0 / sigma2)

    # Errors
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)

    grad = []

    # ∂ℓ/∂η_a
    Wa_tilde = W_a * (Z @ Z.T / jnp.sqrt(Z.shape[1]))  # similarity scaling
    tmp = jnp.einsum("tp,pk->tk", E @ Sigma_inv, jnp.diag(gamma_a) @ Wa_tilde @ Y[:-1].T)
    grad_eta_a = jnp.sum(tmp)
    grad.append(grad_eta_a)

    # ∂ℓ/∂η_b
    Wb_tilde = W_b * (Z @ Z.T / jnp.sqrt(Z.shape[1]))
    term1 = -T * jnp.trace(S_inv @ (jnp.diag(gamma_b) @ Wb_tilde))
    term2 = jnp.sum((E @ Sigma_inv) * (jnp.diag(gamma_b) @ Wb_tilde @ Y[1:].T).T)
    grad.append(term1 + term2)

    # ∂ℓ/∂γ_a (vector length p)
    grad_gamma_a = jnp.sum((E @ Sigma_inv)[:, None, :] * (W_a @ Y[:-1].T).T[:, :, None], axis=0).diagonal()
    grad.extend(grad_gamma_a)

    # ∂ℓ/∂γ_b
    grad_gamma_b = -T * jnp.diag(W_b @ S_inv) + jnp.sum((E @ Sigma_inv)[:, None, :] * (W_b @ Y[1:].T).T[:, :, None], axis=0).diagonal()
    grad.extend(grad_gamma_b)

    # ∂ℓ/∂β
    grad_beta = jnp.sum((E @ Sigma_inv) * Y[:-1], axis=0)
    grad.extend(grad_beta)

    # ∂ℓ/∂σ²
    grad_sigma2 = -0.5 * T * (1.0 / sigma2) + 0.5 * (1.0 / sigma2**2) * jnp.sum(E**2, axis=0)
    grad.extend(grad_sigma2)

    return jnp.array(grad)
