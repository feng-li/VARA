import jax
import jax.numpy as jnp
from loglik_attn import build_attention


def loglik_grad_analytical(theta, Y, Z, params_scale=dict(eta_a=1), update_idx=slice(None)):
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

    print(params_scale)
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[1:1+p]
    beta = theta[1+p:1+2*p]

    sigma2_tilde = theta[1+2*p:1+3*p]  # Replace σ̃² with σ²
    sigma2 = jax.nn.softplus(sigma2_tilde)

    # Build weights
    W_a = build_attention(Z, eta_a, params_scale['eta_a'])
    # W_b = build_attention(Z, eta_b, params_scale['eta_b'])

    D_beta = jnp.diag(beta)
    D_ga = jnp.diag(gamma_a)

    G = D_beta + D_ga @ W_a

    # Inverse Sigma_e
    Sigma_inv = jnp.diag(1.0 / sigma2)

    # Errors
    E = Y[1:] - Y[:-1] @ G.T

    grad = jnp.zeros(1 + 3*p)

    # Common similarity matrix
    S_mat = (Z @ Z.T) / jnp.sqrt(Z.shape[1])

    # === ∂ℓ/∂η_a ===
    # row_means_a = jnp.sum(W_a * S_mat * params_scale['eta_a'], axis=1, keepdims=True)
    row_means_a = jnp.sum(W_a * S_mat, axis=1, keepdims=True)
    Wa_dot = W_a * (S_mat - row_means_a) * params_scale['eta_a']
    grad_eta_a = jnp.sum(
        (E @ Sigma_inv) * (Y[:-1] @ (D_ga @ Wa_dot).T)
    )
    grad = grad.at[0].set(grad_eta_a)

    # === ∂ℓ/∂γ_a ===
    grad_gamma_a = jnp.sum((E @ Sigma_inv) * (W_a @ Y[:-1].T).T, axis=0)
    grad = grad.at[1:1+p].set(grad_gamma_a)

    # === ∂ℓ/∂β ===
    grad_beta = jnp.sum((E @ Sigma_inv) * Y[:-1], axis=0)
    grad = grad.at[1+p:1+2*p].set(grad_beta)

    # === ∂ℓ/∂σ² (softplus reparameterization) ===
    grad_sigma2 = (-0.5 * (T-1) * (1.0 / sigma2)
                   + 0.5 * (1.0 / sigma2**2) * jnp.sum(E**2, axis=0)
                   )
    grad_sigma2_tilde = grad_sigma2 * jax.nn.sigmoid(sigma2_tilde)  # chain rule correction
    grad = grad.at[1+2*p:1+3*p].set(grad_sigma2_tilde)

    # slice(None) means update all parameters
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta.size)
    update_idx = jnp.array(update_idx)

    return jnp.array(grad)[update_idx] / (T-1) # wrt likelihood loss (log likelihood/N)


if __name__ == "__main__":
    import random
    from dgp import simulate_VAR_attention, flatten_params
    from loglik_attn import loglik_reparam


    # --- Simulation settings ---
    key = jax.random.PRNGKey(random.randint(0, 100))
    keys = jax.random.split(key,10)

    T, p, q = 200, 3, 3

    # --- True parameters (in natural scale) ---
    params_true = dict(
        eta_a=0.5,
        gamma_a=jnp.linspace(-0.15, 0.15, p),
        beta=jnp.linspace(-0.20, 0.20, p),
        sigma2=jnp.exp(jnp.log(0.5) + 0.2 * jax.random.normal(keys[0], (p,)))
    )

    params_scale = dict(eta_a=1)

    update_idx = slice(None)

    # --- Simulate data ---
    Y, Z, W_a = simulate_VAR_attention(key, params_true, T=T, p=p, q=q,
                                       dist="normal", params_scale=params_scale)
    print("Simulated Y shape:", Y.shape)

    # --- Convert σ² → σ̃² (softplus inverse) ---
    sigma2_tilde_true = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = params_true.copy()
    params_reparam["sigma2"] = sigma2_tilde_true

    # --- Flatten parameters into reparameterized vector θ̃ ---
    order = ["eta_a", "gamma_a", "beta", "sigma2"]
    theta_true_reparam, _ = flatten_params(params_reparam, order)
    print("θ tilde length:", theta_true_reparam.size)

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
    names.extend([f"gamma_a[{i}]" for i in range(p)])
    names.extend([f"beta[{i}]" for i in range(p)])
    names.extend([f"sigma2[{i}]" for i in range(p)])

    for i, name in enumerate(names):
        diff = grad_analytical[i] - grad_auto[i]
        print(f"{i:02d}: {name:<10s} | "
              f"analytical={grad_analytical[i]: .10f}, "
              f"auto={grad_auto[i]: .10f}, "
              f"diff={diff: .10f}")
