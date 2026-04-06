import jax
import jax.numpy as jnp
from loglik_attn import build_attention

# =====================================================
# Helpers
# =====================================================

def Sigma_inv_from_sigma2(sigma2):
    """Compute inverse diagonal covariance matrix."""
    return jnp.diag(1.0 / sigma2)


def softplus_sigma2(theta, p):
    """Extract σ² = softplus(σ̃²) and its derivatives wrt σ̃²."""
    sigma2_tilde = theta[1 + 2 * p : 1 + 3 * p]
    sigma2 = jax.nn.softplus(sigma2_tilde)
    d_sigma2 = jax.nn.sigmoid(sigma2_tilde)          # ∂σ²/∂σ̃²
    dd_sigma2 = d_sigma2 * (1.0 - d_sigma2)          # ∂²σ²/∂σ̃²²
    return sigma2, d_sigma2, dd_sigma2


def Sigma_inv_from_tilde(theta, p):
    """Convenience helper: compute Σ⁻¹ given θ with σ̃²."""
    sigma2_tilde = theta[1 + 2 * p : 1 + 3 * p]
    sigma2 = jax.nn.softplus(sigma2_tilde)
    return jnp.diag(1.0 / sigma2)


def compute_Wa_Wb_dots(Z, eta, scale):
    """Return (W, Ẇ, Ẅ) for the softmax attention as in Appendix B.2."""
    p, q = Z.shape
    S_mat = (Z @ Z.T) / jnp.sqrt(q)
    # attention weights
    W = build_attention(Z, eta, scale)

    # first derivative Ẇ = W ⊙ (S_mat - row_means)
    row_means = jnp.sum(W * S_mat, axis=1, keepdims=True)
    W_dot = W * (S_mat - row_means) * scale

    # second derivative Ẅ = W ⊙ (S_mat - row_means)**2 − W ⊙ row_var
    row_var = jnp.sum(W * (S_mat - row_means) ** 2, axis=1, keepdims=True)
    W_ddot = W * ((S_mat - row_means) ** 2 - row_var) * scale**2
    return W, W_dot, W_ddot


# =====================================================
# σ²-related Hessian blocks (with softplus chain rule)
# =====================================================

def hess_block_sigma2(theta, Y, Z, params_scale):
    """
    ∂²ℓ / ∂σ̃² ∂σ̃²ᵀ (diagonal, softplus-reparameterized)
    — gradient-consistent version using (T-1) residuals.

    Analytical form (natural scale):
        ∂²ℓ/∂σ²∂(σ²)ᵀ
          = 0.5*(T-1)*diag(σ⁻⁴)
            − diag(σ⁻⁶ ⊙ Σ_t E_t⊙E_t)

    Chain rule to σ̃²:
        ∂²ℓ/∂σ̃²² = d²·H_σ² + dd·g_σ²
    """
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[1:1+p]
    beta    = theta[1+p:1+2*p]

    # softplus transform and its derivatives
    sigma2, d_sigma2, dd_sigma2 = softplus_sigma2(theta, p)

    # build residuals E_t = Y_t S' - Y_{t-1} G'
    W_a = build_attention(Z, eta_a, params_scale['eta_a'])
    D_beta, D_ga = jnp.diag(beta), jnp.diag(gamma_a)
    G = D_beta + D_ga @ W_a

    E = Y[1:] - Y[:-1] @ G.T
    T_eff = T - 1  # number of residuals

    # === Hessian wrt σ² (natural scale) ===
    H_sigma2_diag = 0.5 * T_eff * (sigma2 ** -2) - jnp.sum(E**2, axis=0) * (sigma2 ** -3)

    # === Gradient wrt σ² (natural scale) ===
    grad_sigma2 = -0.5 * T_eff / sigma2 + 0.5 * jnp.sum(E**2, axis=0) / (sigma2 ** 2)

    # === Chain rule to σ̃² ===
    block_diag = d_sigma2**2 * H_sigma2_diag + dd_sigma2 * grad_sigma2
    return jnp.diag(block_diag)



def hess_block_eta_a_gamma_a(theta, Y, Z, params_scale):
    """∂²ℓ / ∂η_a ∂γ_aᵀ — interaction between attention strength η_a and γ_a."""
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[1:1+p]
    beta = theta[1+p:1+2*p]
    sigma2 = jax.nn.softplus(theta[1+2*p:1+3*p])

    W_a, Wa_dot, _ = compute_Wa_Wb_dots(Z, eta_a, params_scale['eta_a'])

    D_ga = jnp.diag(gamma_a)
    D_beta = jnp.diag(beta)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    G = D_beta + D_ga @ W_a



    E = Y[1:] - Y[:-1] @ G.T
    # E = (Y[1:] - Y[:-1] @ G.T)

    block = jnp.zeros((1, p))
    for t in range(1, T):
        y_prev = Y[t-1]
        term1 = jnp.diag(Sigma_inv @ (D_ga @ (Wa_dot @ y_prev))) @ (W_a @ y_prev)
        term2 = jnp.diag(Sigma_inv @ E[t-1]) @ (Wa_dot @ y_prev)
        block -= (term1 - term2)[None, :]

    return block




def hess_block_eta_a_beta(theta, Y, Z, params_scale):
    """∂²ℓ / ∂η_a ∂βᵀ — interaction between attention strength (η_a)
    and baseline coefficients (β). Appendix B.2 Eq. (∂²ℓ/∂β∂η_a)."""
    T, p = Y.shape
    eta_a   = theta[0]
    gamma_a = theta[1:1+p]
    beta    = theta[1+p:1+2*p]
    sigma2  = jax.nn.softplus(theta[1+2*p:1+3*p])
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    # compute attention derivatives
    W_a, Wa_dot, _ = compute_Wa_Wb_dots(Z, eta_a, params_scale['eta_a'])
    D_ga = jnp.diag(gamma_a)

    block = jnp.zeros((1, p))
    for t in range(1, T):
        y_prev = Y[t-1]
        term = jnp.diag(Sigma_inv @ (D_ga @ (Wa_dot @ y_prev))) @ y_prev
        block -= term[None, :]
    return block


def hess_block_eta_a_sigma2(theta, Y, Z, params_scale):
    """∂²ℓ / ∂η_a ∂σ̃²ᵀ : interaction of attention strength (a) and variances."""
    T, p = Y.shape
    eta_a   = theta[0]
    gamma_a = theta[1:1+p]

    sigma2, d_sigma2, _ = softplus_sigma2(theta, p)

    # attention terms
    W_a, Wa_dot, _ = compute_Wa_Wb_dots(Z, eta_a, params_scale['eta_a'])

    D_ga = jnp.diag(gamma_a)

    # residuals
    D_beta = jnp.diag(theta[1+p:1+2*p])
    G = D_beta + D_ga @ W_a

    E = Y[1:] - Y[:-1] @ G.T        # (T-1)×p

    # ∂ℓ/∂η_a depends on E_t' Σ⁻¹ (∂E_t/∂η_a)
    Xa = Y[:-1] @ (D_ga @ Wa_dot).T        # (T-1)×p

    # derivative of ∂ℓ/∂σ²_i wrt η_a  (natural scale)
    dL_dsig2_eta = jnp.sum(E * (-Xa), axis=0) / (sigma2**2)

    # chain rule to σ̃²
    return dL_dsig2_eta[None, :] * d_sigma2[None, :]



def hess_block_gamma_a_beta(theta, Y, Z, params_scale):
    """∂²ℓ / ∂β ∂γ_aᵀ — Appendix B.2, Eq. (∂²ℓ/∂β∂γ_a)."""
    T, p = Y.shape
    eta_a = theta[0]
    sigma2 = jax.nn.softplus(theta[1+2*p:1+3*p])
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)
    W_a = build_attention(Z, eta_a, params_scale['eta_a'])

    block = jnp.zeros((p, p))
    for t in range(1, T):
        WaY_prev = W_a @ Y[t-1]
        block -= Sigma_inv @ jnp.diag(WaY_prev * Y[t-1])
    return block


def hess_block_gamma_a_sigma2(theta, Y, Z, params_scale):
    """∂²ℓ / ∂σ̃² ∂γ_aᵀ — Appendix B.2, Eq. (∂²ℓ/∂σ²∂γ_a) with σ̃ reparameterization.

    TODO: check
    """
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[1:1+p]

    beta    = theta[1+p:1+2*p]
    sigma2, d_sigma2, _ = softplus_sigma2(theta, p)  # reparameterized σ̃

    W_a = build_attention(Z, eta_a, params_scale["eta_a"])
    D_ga = jnp.diag(gamma_a)

    D_beta = jnp.diag(beta)
    G = D_beta + D_ga @ W_a

    E = Y[1:] - Y[:-1] @ G.T         # ε_t


    WaY_prev = Y[:-1] @ W_a.T                  # W_a Y_{t-1}
    nat_diag = -jnp.sum(E * WaY_prev, axis=0) / (sigma2**2)  # σ⁻⁴ ∘ Σ_t(ε_t∘W_aY_{t-1})

    # Chain rule σ̃² → σ²
    block_diag = nat_diag * d_sigma2

    return jnp.diag(block_diag)


def hess_block_beta_sigma2(theta, Y, Z, params_scale):
    """∂²ℓ / ∂β ∂σ̃²ᵀ : effect of baseline coefficients on variances."""
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[1:1+p]

    beta    = theta[1+p:1+2*p]
    sigma2, d_sigma2, _ = softplus_sigma2(theta, p)

    # model pieces
    W_a = build_attention(Z, eta_a, params_scale['eta_a'])


    D_ga = jnp.diag(gamma_a)
    D_beta = jnp.diag(beta)
    G = D_beta + D_ga @ W_a

    E = Y[1:] - Y[:-1] @ G.T

    # ∂E/∂β = −Y_{t−1}
    Xb = Y[:-1]                              # (T-1)×p
    dL_dsig2_beta = -jnp.sum(E * Xb, axis=0) / (sigma2**2)

    block_diag = dL_dsig2_beta * d_sigma2

    return jnp.diag(block_diag)



# =====================================================
# Remaining blocks (using σ² = softplus(σ̃²))
# =====================================================

def hess_block_eta_a(theta, Y, Z, params_scale):
    """∂²ℓ / ∂η_a² (scalar). Exact second derivative from the verified gradient."""
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[1:1+p]
    beta    = theta[1+p:1+2*p]
    sigma2  = jax.nn.softplus(theta[1+2*p:1+3*p])

    # Attention derivatives and fixed matrices
    W_a, Wa_dot, Wa_ddot = compute_Wa_Wb_dots(Z, eta_a, params_scale['eta_a'])
    D_ga     = jnp.diag(gamma_a)
    Sigma_inv_diag = 1.0 / sigma2                    # use diagonal structure

    # Model pieces to form E_t
    D_beta = jnp.diag(beta)
    G = D_beta + D_ga @ W_a

    E = Y[1:] - Y[:-1] @ G.T               # (T-1) x p

    Xa   = Y[:-1] @ (D_ga @ Wa_dot).T
    Xdda = Y[:-1] @ (D_ga @ Wa_ddot).T

    term_quad = -jnp.sum((Xa * Xa) * Sigma_inv_diag)

    term_curv = jnp.sum((Xdda * E) * Sigma_inv_diag)

    return term_quad + term_curv


def hess_block_gamma_a(theta, Y, Z, params_scale):
    """∂²ℓ / ∂γ_a ∂γ_aᵀ ∈ R^{p×p}."""
    T, p = Y.shape
    eta_a = theta[0]
    sigma2 = jax.nn.softplus(theta[1+2*p:1+3*p])
    W_a = build_attention(Z, eta_a, params_scale['eta_a'])
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    block = jnp.zeros((p, p))
    for t in range(1, T):
        WaY = W_a @ Y[t-1]
        block -= Sigma_inv @ jnp.diag(WaY * WaY)
    return block


def hess_block_beta(theta, Y, Z):
    """∂²ℓ / ∂β ∂βᵀ ∈ R^{p×p}."""
    T, p = Y.shape
    sigma2 = jax.nn.softplus(theta[1+2*p:1+3*p])
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    block = jnp.zeros((p, p))
    for t in range(1, T):
        block -= Sigma_inv @ jnp.diag(Y[t-1] * Y[t-1])
    return block



# =====================================================
# Master Hessian assembly
# =====================================================

def loglik_hess_analytical(theta, Y, Z, params_scale, kappa4, update_idx=slice(None)):
    """Assemble full analytical Hessian H(θ̃)."""
    T, p = Y.shape
    d2 = len(theta)
    H = jnp.zeros((d2, d2))

    # Diagonal blocks
    H = H.at[0, 0].set(hess_block_eta_a(theta, Y, Z, params_scale))
    H = H.at[1:1+p, 1:1+p].set(hess_block_gamma_a(theta, Y, Z, params_scale))
    H = H.at[1+p:1+2*p, 1+p:1+2*p].set(hess_block_beta(theta, Y, Z))
    H = H.at[1+2*p:1+3*p, 1+2*p:1+3*p].set(hess_block_sigma2(theta, Y, Z, params_scale))

    # Off diagonal blocks
    H = H.at[0:1, 1+p:1+2*p].set(hess_block_eta_a_beta(theta, Y, Z, params_scale))
    H = H.at[1+p:1+2*p, 0:1].set(hess_block_eta_a_beta(theta, Y, Z, params_scale).T)

    H = H.at[0:1, 1:1+p].set(hess_block_eta_a_gamma_a(theta, Y, Z, params_scale))
    H = H.at[1:1+p, 0:1].set(hess_block_eta_a_gamma_a(theta, Y, Z, params_scale).T)

    H = H.at[0:1, 1+2*p:1+3*p].set(hess_block_eta_a_sigma2(theta, Y, Z, params_scale))
    H = H.at[1+2*p:1+3*p, 0:1].set(hess_block_eta_a_sigma2(theta, Y, Z, params_scale).T)

    H = H.at[1:1+p, 1+p:1+2*p].set(hess_block_gamma_a_beta(theta, Y, Z, params_scale))
    H = H.at[1+p:1+2*p, 1:1+p].set(hess_block_gamma_a_beta(theta, Y, Z, params_scale).T)

    H = H.at[1:1+p, 1+2*p:1+3*p].set(hess_block_gamma_a_sigma2(theta, Y, Z, params_scale))
    H = H.at[1+2*p:1+3*p, 1:1+p].set(hess_block_gamma_a_sigma2(theta, Y, Z, params_scale).T)

    H = H.at[1+p:1+2*p, 1+2*p:1+3*p].set(hess_block_beta_sigma2(theta, Y, Z, params_scale))
    H = H.at[1+2*p:1+3*p, 1+p:1+2*p].set(hess_block_beta_sigma2(theta, Y, Z, params_scale).T)

    # slice(None) means update all parameters
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta.size)
    update_idx = jnp.array(update_idx)

    H_partial = H[jnp.ix_(update_idx, update_idx)]

    return H_partial/(T-1)  # wrt likelihood loss (log likelihood/N)

# =====================================================
# Test harness
# =====================================================

def get_block(H, p, row_block, col_block):
    """Extract a named Hessian block."""
    def idx(name):
        if name == "eta_a": return jnp.array([0])
        elif name == "gamma_a": return jnp.arange(1, 1+p)
        elif name == "beta": return jnp.arange(1+p, 1+2*p)
        elif name == "sigma2": return jnp.arange(1+2*p, 1+3*p)
        else: raise ValueError("Unknown block name")


    ridx, cidx = idx(row_block), idx(col_block)
    return H[jnp.ix_(ridx, cidx)]


if __name__ == "__main__":

    import random
    from dgp import simulate_VAR_attention, flatten_params
    from loglik_attn import loglik_reparam

    jnp.set_printoptions(threshold=10_000, linewidth=1000, suppress=True, precision=10)

    # --- Simulation settings ---
    key = jax.random.PRNGKey(random.randint(0, 100))
    T, p, q = 100, 3, 3

    # --- True parameters (in natural scale) ---
    params_true = dict(
        eta_a=0.5,
        gamma_a=jnp.linspace(-0.15, 0.15, p),
        beta=jnp.linspace(-0.20, 0.20, p),
        sigma2=jnp.exp(jax.random.normal(key, (p,)))  # positive σ²
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

    # --- Compute analytical and automatic Hessians ---
    H_analytical = loglik_hess_analytical(theta_true_reparam, Y, Z, params_scale,
                                          kappa4=3, update_idx=update_idx)
    H_auto = jax.hessian(loglik_reparam)(theta_true_reparam, Y, Z, params_scale, update_idx=update_idx)

    # --- Full-matrix comparison ---
    diff_full = jnp.max(jnp.abs(H_analytical - H_auto))
    print(f"\nMax abs diff (full Hessian, reparameterized): {diff_full:.4f}")

    print("\nHessian (Analytical, reparameterized):")
    print(H_analytical)

    print("\nHessian (JAX autodiff, reparameterized):")
    print(H_auto)

    print("\nHessian diff (Analytical - Autodiff, reparameterized):")
    print(H_analytical - H_auto)

    # --- Diagonal block-by-block comparison ---
    diag_blocks = ["eta_a", "gamma_a", "beta", "sigma2"]
    for block in diag_blocks:
        Ha = get_block(H_analytical, p, block, block)
        Hu = get_block(H_auto, p, block, block)
        diff = jnp.max(jnp.abs(Ha - Hu))
        print(f"\nBlock {block:<10s} max abs diff = {diff:.4f}")
        print(Ha)
        print(Hu)


    # --- Cross-blocks ---
    cross_blocks = [
        ("eta_a", "gamma_a"),
        ("eta_a", "beta"),
        ("eta_a", "sigma2"),
        ("gamma_a", "beta"),
        ("gamma_a", "sigma2"),
        ("beta", "sigma2"),
    ]

    print("\nCross-block differences:")
    for rb, cb in cross_blocks:
        Ha = get_block(H_analytical, p, rb, cb)
        Hu = get_block(H_auto, p, rb, cb)

        diff = jnp.max(jnp.abs(Ha - Hu))
        print(f"\n({rb}, {cb}) max abs diff = {diff:.4f}")
        print(Ha)
        print(Hu)
