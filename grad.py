import jax
import jax.numpy as jnp
from loglik_attn import build_attention


def loglik_grad_analytical(theta, Y, Z, params_scale=dict(eta_a=1, eta_b=10), update_idx=slice(None)):
    """
    Compute the analytical gradient of the log-likelihood ℓ(θ̃)
    for the attention-based QMLE model with softplus-reparameterized variances.

    Parameterization
    ----------------
    The input parameter vector θ̃ is structured as:
        θ̃ = (ηₐ, η_b, γₐ, γ_b, β, σ̃²)
    where:
        - ηₐ, η_b : scalar attention temperature parameters
        - γₐ, γ_b : length-p autoregressive coefficient vectors
        - β       : length-p baseline (exogenous) coefficient vector
        - σ̃²      : length-p unconstrained variance parameters

    The actual (positive) variances used in the likelihood are given by:
        σ² = softplus(σ̃²) = log(1 + exp(σ̃²))

    Model
    -----
        Y_t S' - Y_{t-1} G' = E_t,
        where
            G = diag(β) + diag(γₐ) Wₐ,
            S = I - diag(γ_b) W_b,
            Wₐ = softmax(ηₐ Z Zᵀ / √q),
            W_b = softmax(η_b Z Zᵀ / √q).

    The log-likelihood is:
        ℓ(θ̃) = -½ ∑_t [ log|Σ_e| + E_t' Σ_e⁻¹ E_t ],
        with Σ_e = diag(σ²).

    The gradient ∂ℓ/∂θ̃ is computed analytically with respect to all
    parameters, including the chain rule correction for σ̃²:
        ∂ℓ/∂σ̃² = ∂ℓ/∂σ² × (1 - exp(-σ²)).

    Args
    ----
    theta : jax.Array
        Parameter vector (1D) in reparameterized form (ηₐ, η_b, γₐ, γ_b, β, σ̃²).
    Y : jax.Array
        Observed data matrix of shape (T, p).
    Z : jax.Array
        Feature matrix of shape (p, q) used to construct attention weights.

    Returns
    -------
    grad : jax.Array
        Analytical gradient vector ∂ℓ/∂θ̃ of length 2 + 4p.

    Notes
    -----
    - The function internally computes Wₐ and W_b from `build_attention(Z, η)`.
    - The derivatives of Wₐ and W_b with respect to η follow the softmax rule:
          ∂W/∂η = W ⊙ (S_mat - row_sum(W ⊙ S_mat)),
      where S_mat = Z Zᵀ / √q and ⊙ denotes elementwise multiplication.
    - The resulting gradient is fully compatible with the Hessian
      and sandwich variance estimators used in subsequent QMLE inference.
    """

    T, p = Y.shape
    eta_a = theta[0]
    eta_b = theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]

    sigma2_tilde  = theta[2+3*p:2+4*p]  # Replace σ̃² with σ²
    sigma2 = jax.nn.softplus(sigma2_tilde)

    # Build weights
    W_a = build_attention(Z, eta_a, params_scale['eta_a'])
    W_b = build_attention(Z, eta_b, params_scale['eta_b'])

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

    grad = jnp.zeros(2 + 4*p)

    # Common similarity matrix
    S_mat = (Z @ Z.T) / jnp.sqrt(Z.shape[1])

    # === ∂ℓ/∂η_a ===
    row_means_a = jnp.sum(W_a * S_mat, axis=1, keepdims=True)
    Wa_dot = W_a * (S_mat - row_means_a)
    grad_eta_a = jnp.sum(
        (E @ Sigma_inv) * (Y[:-1] @ (D_ga @ Wa_dot).T)
    )
    grad = grad.at[0].set(grad_eta_a)

    # === ∂ℓ/∂η_b ===
    row_means_b = jnp.sum(W_b * S_mat, axis=1, keepdims=True)
    Wb_dot = W_b * (S_mat - row_means_b)
    term1 = -T * jnp.trace(S_inv @ (D_gb @ Wb_dot))
    term2 = jnp.sum(
        (E @ Sigma_inv) * (Y[1:] @ (D_gb @ Wb_dot).T)
    )
    grad = grad.at[1].set(term1 + term2)

    # === ∂ℓ/∂γ_a ===
    grad_gamma_a = jnp.sum((E @ Sigma_inv) * (W_a @ Y[:-1].T).T, axis=0)
    grad = grad.at[2:2+p].set(grad_gamma_a)

    # === ∂ℓ/∂γ_b ===
    grad_gamma_b = -T * jnp.diag(W_b @ S_inv) \
        + jnp.sum((E @ Sigma_inv) * (W_b @ Y[1:].T).T, axis=0)
    grad = grad.at[2+p:2+2*p].set(grad_gamma_b)

    # === ∂ℓ/∂β ===
    grad_beta = jnp.sum((E @ Sigma_inv) * Y[:-1], axis=0)
    grad = grad.at[2+2*p:2+3*p].set(grad_beta)

    # === ∂ℓ/∂σ² (softplus reparameterization) ===
    grad_sigma2 = -0.5 * (T-1) * (1.0 / sigma2) \
        + 0.5 * (1.0 / sigma2**2) * jnp.sum(E**2, axis=0)
    grad_sigma2_tilde = grad_sigma2 * jax.nn.sigmoid(sigma2_tilde)  # chain rule correction
    grad = grad.at[2+3*p:2+4*p].set(grad_sigma2_tilde)

    # slice(None) means update all parameters
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta.size)
    update_idx = jnp.array(update_idx)

    return jnp.array(grad)[update_idx] / (T-1)


if __name__ == "__main__":
    import random
    from dgp import simulate_VAR_attention, flatten_params
    from loglik_attn import loglik_reparam


    # --- Simulation settings ---
    key = jax.random.PRNGKey(random.randint(0, 100))
    T, p, q = 200, 3, 3

    # --- True parameters (in natural scale) ---
    params_true = dict(
        eta_a=0.5,
        eta_b=1.0,
        gamma_a=jnp.linspace(-0.15, 0.15, p),
        gamma_b=jnp.linspace(-0.10, 0.10, p),
        beta=jnp.linspace(-0.20, 0.20, p),
        sigma2=jnp.exp(jax.random.normal(key, (p,)))  # positive σ²
    )

    params_scale = dict(
        eta_a=1,
        eta_b=10)

    update_idx = slice(None)

    # --- Simulate data ---
    Y, Z, W_a, W_b = simulate_VAR_attention(key, params_true, T=T, p=p, q=q,
                                            dist="normal", params_scale=params_scale)
    print("Simulated Y shape:", Y.shape)

    # --- Convert σ² → σ̃² (softplus inverse) ---
    sigma2_tilde_true = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = params_true.copy()
    params_reparam["sigma2"] = sigma2_tilde_true

    # --- Flatten parameters into reparameterized vector θ̃ ---
    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    theta_true_reparam = flatten_params(params_reparam, order)
    print("θ̃ length:", theta_true_reparam.size)

    # =====================================================
    # Gradient test
    # =====================================================
    grad_analytical = loglik_grad_analytical(theta_true_reparam, Y, Z,
                                             params_scale=params_scale, update_idx=update_idx)
    grad_auto = jax.grad(loglik_reparam)(theta_true_reparam, Y, Z,
                                         params_scale=params_scale, update_idx=update_idx)

    diff_grad = jnp.max(jnp.abs(grad_analytical - grad_auto))
    print(f"\nMax abs diff (gradient, reparameterized): {diff_grad:.3e}")

    # Optional: print blockwise gradient comparisons
    print("\nAnalytical vs autodiff gradient elements:")

    names = []
    names.append("eta_a")
    names.append("eta_b")
    names.extend([f"gamma_a[{i}]" for i in range(p)])
    names.extend([f"gamma_b[{i}]" for i in range(p)])
    names.extend([f"beta[{i}]" for i in range(p)])
    names.extend([f"sigma2[{i}]" for i in range(p)])

    for i, name in enumerate(names):
        diff = grad_analytical[i] - grad_auto[i]
        print(f"{i:02d}: {name:<10s} | "
              f"analytical={grad_analytical[i]: .10f}, "
              f"auto={grad_auto[i]: .10f}, "
              f"diff={diff: .10f}")
