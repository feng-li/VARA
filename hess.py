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
    sigma2_tilde = theta[2 + 3 * p : 2 + 4 * p]
    sigma2 = jax.nn.softplus(sigma2_tilde)
    d_sigma2 = jax.nn.sigmoid(sigma2_tilde)          # ∂σ²/∂σ̃²
    dd_sigma2 = d_sigma2 * (1.0 - d_sigma2)          # ∂²σ²/∂σ̃²²
    return sigma2, d_sigma2, dd_sigma2


def Sigma_inv_from_tilde(theta, p):
    """Convenience helper: compute Σ⁻¹ given θ with σ̃²."""
    sigma2_tilde = theta[2 + 3 * p : 2 + 4 * p]
    sigma2 = jax.nn.softplus(sigma2_tilde)
    return jnp.diag(1.0 / sigma2)


def compute_Wa_Wb_dots(Z, eta):
    """Return (W, Ẇ, Ẅ) for the softmax attention as in Appendix B.2."""
    p, q = Z.shape
    S_mat = (Z @ Z.T) / jnp.sqrt(q)
    # attention weights
    W = build_attention(Z, eta)

    # first derivative Ẇ = W ⊙ (S_mat - row_means)
    row_means = jnp.sum(W * S_mat, axis=1, keepdims=True)
    W_dot = W * (S_mat - row_means)

    # second derivative Ẅ = W ⊙ (S_mat - row_means)**2 − W ⊙ row_var
    row_var = jnp.sum(W * (S_mat - row_means) ** 2, axis=1, keepdims=True)
    W_ddot = W * ((S_mat - row_means) ** 2 - row_var)
    return W, W_dot, W_ddot


# =====================================================
# σ²-related Hessian blocks (with softplus chain rule)
# =====================================================

def hess_block_sigma2(theta, Y, Z):
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
    eta_a, eta_b = theta[0], theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]

    # softplus transform and its derivatives
    sigma2, d_sigma2, dd_sigma2 = softplus_sigma2(theta, p)

    # build residuals E_t = Y_t S' - Y_{t-1} G'
    W_a = build_attention(Z, eta_a)
    W_b = build_attention(Z, eta_b)
    D_beta, D_ga, D_gb = jnp.diag(beta), jnp.diag(gamma_a), jnp.diag(gamma_b)
    G = D_beta + D_ga @ W_a
    S = jnp.eye(p) - D_gb @ W_b
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)
    T_eff = T - 1  # number of residuals

    # === Hessian wrt σ² (natural scale) ===
    H_sigma2_diag = 0.5 * T_eff * (sigma2 ** -2) - jnp.sum(E**2, axis=0) * (sigma2 ** -3)

    # === Gradient wrt σ² (natural scale) ===
    grad_sigma2 = -0.5 * T_eff / sigma2 + 0.5 * jnp.sum(E**2, axis=0) / (sigma2 ** 2)

    # === Chain rule to σ̃² ===
    block_diag = d_sigma2**2 * H_sigma2_diag + dd_sigma2 * grad_sigma2
    return jnp.diag(block_diag)

def hess_block_eta_a_gamma_a(theta, Y, Z):
    """∂²ℓ / ∂η_a ∂γ_aᵀ — interaction between attention strength η_a and γ_a."""
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[2:2+p]
    beta = theta[2+2*p:2+3*p]
    sigma2 = jax.nn.softplus(theta[2+3*p:2+4*p])

    W_a, Ẇ_a, _ = compute_Wa_Wb_dots(Z, eta_a)
    D_ga = jnp.diag(gamma_a)
    D_beta = jnp.diag(beta)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    # residuals E_t = Y_t - Y_{t-1} G'   (S = I for η_a block)
    G = D_beta + D_ga @ W_a
    E = (Y[1:] - Y[:-1] @ G.T)

    block = jnp.zeros((1, p))
    for t in range(1, T):
        y_prev = Y[t-1]
        term1 = jnp.diag(Sigma_inv @ (D_ga @ (Ẇ_a @ y_prev))) @ (W_a @ y_prev)
        term2 = jnp.diag(Sigma_inv @ E[t-1]) @ (Ẇ_a @ y_prev)
        block -= (term1 - term2)[None, :]
    return block


def hess_block_eta_a_gamma_b(theta, Y, Z):
    """
    ∂²ℓ / ∂η_a ∂γ_bᵀ  (returns shape 1×p)
    Appendix B.2 line “∂²ℓ/∂γ_b ∂η_a” (transpose here):
      − Σ_t diag{ Σ⁻¹ diag(γ_a) Ẇ_a Y_{t−1} } (W_b Y_t)
    """
    T, p = Y.shape
    eta_a, eta_b = theta[0], theta[1]
    gamma_a = theta[2:2+p]
    sigma2  = jax.nn.softplus(theta[2+3*p:2+4*p])

    # pieces
    W_a, Wa_dot, _ = compute_Wa_Wb_dots(Z, eta_a)
    W_b            = build_attention(Z, eta_b)
    D_ga           = jnp.diag(gamma_a)
    Sigma_inv_diag = 1.0 / sigma2

    # Xa(t) = diag(γ_a) Ẇ_a Y_{t-1} ; Xb(t) = W_b Y_t
    Xa = Y[:-1] @ (D_ga @ Wa_dot).T        # (T-1)×p
    Xb = Y[1:]  @ W_b.T                    # (T-1)×p

    row = -jnp.sum((Xa * Xb) * Sigma_inv_diag, axis=0)  # (p,)
    return row[None, :]

def hess_block_eta_b_gamma_b(theta, Y, Z):
    """
    ∂²ℓ / ∂η_b ∂γ_bᵀ  (returns shape 1×p)
    Appendix B.2 line “∂²ℓ/∂γ_b ∂η_b”:
      −T · diag{ Ẇ_b S⁻¹ + W_b S⁻¹ diag(γ_b) Ẇ_b S⁻¹ }
      − Σ_t [ (Σ⁻¹ diag(γ_b) Ẇ_b Y_t) ∘ (W_b Y_t)  −  (Σ⁻¹ E_t) ∘ (Ẇ_b Y_t) ]
    """
    T, p = Y.shape
    eta_a, eta_b = theta[0], theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]
    sigma2  = jax.nn.softplus(theta[2+3*p:2+4*p])

    # attentions & derivatives
    W_a = build_attention(Z, eta_a)
    W_b, Wb_dot, _ = compute_Wa_Wb_dots(Z, eta_b)
    D_ga, D_gb = jnp.diag(gamma_a), jnp.diag(gamma_b)
    Sigma_inv_diag = 1.0 / sigma2

    # model pieces
    G = jnp.diag(beta) + D_ga @ W_a
    S = jnp.eye(p) - D_gb @ W_b
    S_inv = jnp.linalg.inv(S)
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)        # (T-1)×p

    # helper: diag(A @ B) = rowsum(A ∘ Bᵀ)
    def diag_AB(A, B):
        return jnp.sum(A * B.T, axis=1)

    # trace (vector) term: −T * diag{ Ẇ_b S⁻¹ + W_b S⁻¹ diag(γ_b) Ẇ_b S⁻¹ }
    term_trace1 = diag_AB(Wb_dot, S_inv)                           # diag(Ẇ_b S⁻¹)
    term_trace2 = diag_AB(W_b @ S_inv, D_gb @ Wb_dot @ S_inv)      # diag(W_b S⁻¹ D_gb Ẇ_b S⁻¹)
    trace_vec   = -T * (term_trace1 + term_trace2)                 # (p,)

    # data terms
    Xb     = Y[1:] @ W_b.T                 # (T-1)×p = W_b Y_t
    Xbdot  = Y[1:] @ Wb_dot.T              # (T-1)×p = Ẇ_b Y_t
    Xgbdot = Y[1:] @ (D_gb @ Wb_dot).T     # (T-1)×p = diag(γ_b) Ẇ_b Y_t

    termA = jnp.sum((Xgbdot * Xb) * Sigma_inv_diag, axis=0)        # Σ_t (Σ⁻¹ diag(γ_b)Ẇ_b Y_t) ∘ (W_b Y_t)
    termB = jnp.sum((E * Xbdot) * Sigma_inv_diag, axis=0)          # Σ_t (Σ⁻¹ E_t) ∘ (Ẇ_b Y_t)

    row = trace_vec - termA + termB
    return row[None, :]


def hess_block_eta_a_beta(theta, Y, Z):
    """∂²ℓ / ∂η_a ∂βᵀ — interaction between attention strength (η_a)
    and baseline coefficients (β). Appendix B.2 Eq. (∂²ℓ/∂β∂η_a)."""
    T, p = Y.shape
    eta_a   = theta[0]
    gamma_a = theta[2:2+p]
    beta    = theta[2+2*p:2+3*p]
    sigma2  = jax.nn.softplus(theta[2+3*p:2+4*p])
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    # compute attention derivatives
    W_a, Ẇ_a, _ = compute_Wa_Wb_dots(Z, eta_a)
    D_ga = jnp.diag(gamma_a)

    block = jnp.zeros((1, p))
    for t in range(1, T):
        y_prev = Y[t-1]
        term = jnp.diag(Sigma_inv @ (D_ga @ (Ẇ_a @ y_prev))) @ y_prev
        block -= term[None, :]
    return block


def hess_block_eta_a_sigma2(theta, Y, Z):
    """∂²ℓ / ∂η_a ∂σ̃²ᵀ : interaction of attention strength (a) and variances."""
    T, p = Y.shape
    eta_a   = theta[0]
    gamma_a = theta[2:2+p]
    sigma2, d_sigma2, _ = softplus_sigma2(theta, p)

    # attention terms
    W_a, Wa_dot, _ = compute_Wa_Wb_dots(Z, eta_a)
    D_ga = jnp.diag(gamma_a)

    # residuals
    D_beta = jnp.diag(theta[2+2*p:2+3*p])
    G = D_beta + D_ga @ W_a
    E = (Y[1:] - Y[:-1] @ G.T)

    # ∂ℓ/∂η_a depends on E_t' Σ⁻¹ (∂E_t/∂η_a)
    Xa = Y[:-1] @ (D_ga @ Wa_dot).T        # (T-1)×p

    # derivative of ∂ℓ/∂σ²_i wrt η_a  (natural scale)
    dL_dsig2_eta = jnp.sum(E * (-Xa), axis=0) / (sigma2**2)

    # chain rule to σ̃²
    return dL_dsig2_eta[None, :] * d_sigma2[None, :]


def hess_block_eta_b_gamma_a(theta, Y, Z):
    """∂²ℓ / ∂η_b ∂γ_aᵀ — should return shape (1, p).
    Appendix B.2: −∑_t diag{ Σ⁻¹ diag(γ_b) Ẇ_b Y_t } (W_a Y_{t−1})
    """
    T, p = Y.shape
    eta_a, eta_b = theta[0], theta[1]
    gamma_b = theta[2+p:2+2*p]
    sigma2  = jax.nn.softplus(theta[2+3*p:2+4*p])
    Sigma_inv_diag = 1.0 / sigma2

    # attentions and derivatives
    W_a = build_attention(Z, eta_a)
    W_b, Wb_dot, _ = compute_Wa_Wb_dots(Z, eta_b)
    D_gb = jnp.diag(gamma_b)

    # compute Xa(t) = diag(γ_b) Ẇ_b Y_t ; Xprev(t) = W_a Y_{t-1}
    Xa = Y[1:] @ (D_gb @ Wb_dot).T     # (T-1)×p
    Xprev = Y[:-1] @ W_a.T             # (T-1)×p

    # elementwise contraction → (p,)
    row = -jnp.sum((Xa * Xprev) * Sigma_inv_diag, axis=0)
    return row[None, :]                # shape (1, p)



def hess_block_eta_b_beta(theta, Y, Z):
    """∂²ℓ / ∂η_b ∂βᵀ — interaction between η_b and β."""
    T, p = Y.shape
    eta_b = theta[1]
    gamma_b = theta[2+p:2+2*p]
    sigma2 = jax.nn.softplus(theta[2+3*p:2+4*p])
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    W_b, Wb_dot, _ = compute_Wa_Wb_dots(Z, eta_b)
    D_gb = jnp.diag(gamma_b)

    block = jnp.zeros((1, p))
    for t in range(1, T):
        y_prev = Y[t-1]
        y_curr = Y[t]
        term = jnp.diag(Sigma_inv @ (D_gb @ (Wb_dot @ y_curr))) @ y_prev
        block -= term[None, :]
    return block


def hess_block_eta_b_sigma2(theta, Y, Z):
    """∂²ℓ / ∂η_b ∂σ̃²ᵀ — Appendix B.2, Eq. (∂²ℓ/∂σ²∂η_b)."""
    T, p = Y.shape
    eta_b = theta[1]
    gamma_b = theta[2+p:2+2*p]
    sigma2, d_sigma2, _ = softplus_sigma2(theta, p)

    W_b, Wb_dot, _ = compute_Wa_Wb_dots(Z, eta_b)
    D_gb = jnp.diag(gamma_b)

    # residuals
    eta_a = theta[0]
    gamma_a = theta[2:2+p]
    beta = theta[2+2*p:2+3*p]
    W_a = build_attention(Z, eta_a)
    G = jnp.diag(beta) + jnp.diag(gamma_a) @ W_a
    S = jnp.eye(p) - D_gb @ W_b
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)

    Xb = Y[1:] @ (D_gb @ Wb_dot).T
    dL_dsig2_eta = jnp.sum(E * (-Xb), axis=0) / (sigma2**2)
    return dL_dsig2_eta[None, :] * d_sigma2[None, :]


def hess_block_gamma_a_gamma_b(theta, Y, Z):
    """∂²ℓ / ∂γ_b ∂γ_aᵀ — Appendix B.2, Eq. (∂²ℓ/∂γ_b∂γ_a)."""
    T, p = Y.shape
    eta_a, eta_b = theta[0], theta[1]
    sigma2 = jax.nn.softplus(theta[2+3*p:2+4*p])
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)
    W_a = build_attention(Z, eta_a)
    W_b = build_attention(Z, eta_b)

    block = jnp.zeros((p, p))
    for t in range(1, T):
        WaY_prev = W_a @ Y[t-1]
        WbY_curr = W_b @ Y[t]
        block -= Sigma_inv @ jnp.diag(WaY_prev * WbY_curr)
    return block


def hess_block_gamma_a_beta(theta, Y, Z):
    """∂²ℓ / ∂β ∂γ_aᵀ — Appendix B.2, Eq. (∂²ℓ/∂β∂γ_a)."""
    T, p = Y.shape
    eta_a = theta[0]
    sigma2 = jax.nn.softplus(theta[2+3*p:2+4*p])
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)
    W_a = build_attention(Z, eta_a)

    block = jnp.zeros((p, p))
    for t in range(1, T):
        WaY_prev = W_a @ Y[t-1]
        block -= Sigma_inv @ jnp.diag(WaY_prev * Y[t-1])
    return block


def hess_block_gamma_a_sigma2(theta, Y, Z):
    """∂²ℓ / ∂σ̃² ∂γ_aᵀ — Appendix B.2, Eq. (∂²ℓ/∂σ²∂γ_a)."""
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[2:2+p]
    sigma2, d_sigma2, _ = softplus_sigma2(theta, p)

    W_a = build_attention(Z, eta_a)
    D_ga = jnp.diag(gamma_a)
    G = D_ga @ W_a
    E = (Y[1:] - Y[:-1] @ (jnp.diag(theta[2+2*p:2+3*p]) + G).T)

    WaY_prev = Y[:-1] @ W_a.T
    nat_diag = -jnp.sum(E * WaY_prev, axis=0) / (sigma2**2)
    return jnp.diag(nat_diag * d_sigma2)


def hess_block_gamma_b_sigma2(theta, Y, Z):
    """
    ∂²ℓ / ∂γ_b ∂σ̃²ᵀ — interaction between γ_b and variances.
    Appendix B.2:  −diag(σ⁻⁴ ∘ Σ_t E_t ∘ (W_b Y_t)).
    """
    T, p = Y.shape
    eta_a, eta_b = theta[0], theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]
    sigma2, d_sigma2, _ = softplus_sigma2(theta, p)

    W_a, W_b = build_attention(Z, eta_a), build_attention(Z, eta_b)
    D_ga, D_gb = jnp.diag(gamma_a), jnp.diag(gamma_b)
    D_beta = jnp.diag(beta)
    G = D_beta + D_ga @ W_a
    S = jnp.eye(p) - D_gb @ W_b
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)

    # Xb_t = W_b Y_t
    Xb = Y[1:] @ W_b.T

    # Natural-scale block: −diag(σ⁻⁴ ∘ Σ_t E_t ∘ Xb_t)
    nat_diag = -jnp.sum(E * Xb, axis=0) / (sigma2**2) # Treat sigma2 as a whole

    # Chain rule σ̃² → σ²
    block_diag = nat_diag * d_sigma2
    return jnp.diag(block_diag)




def hess_block_beta_sigma2(theta, Y, Z):
    """∂²ℓ / ∂β ∂σ̃²ᵀ : effect of baseline coefficients on variances."""
    T, p = Y.shape
    eta_a, eta_b = theta[0], theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]
    sigma2, d_sigma2, _ = softplus_sigma2(theta, p)

    # model pieces
    W_a, W_b = build_attention(Z, eta_a), build_attention(Z, eta_b)
    D_ga, D_gb = jnp.diag(gamma_a), jnp.diag(gamma_b)
    D_beta = jnp.diag(beta)
    G = D_beta + D_ga @ W_a
    S = jnp.eye(p) - D_gb @ W_b
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)

    # ∂E/∂β = −Y_{t−1}
    Xb = Y[:-1]                              # (T-1)×p
    dL_dsig2_beta = -jnp.sum(E * Xb, axis=0) / (sigma2**2)

    block_diag = dL_dsig2_beta * d_sigma2

    return jnp.diag(block_diag)



# =====================================================
# Remaining blocks (using σ² = softplus(σ̃²))
# =====================================================

def hess_block_eta_a(theta, Y, Z):
    """∂²ℓ / ∂η_a² (scalar). Exact second derivative from the verified gradient."""
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[2:2+p]
    beta    = theta[2+2*p:2+3*p]
    sigma2  = jax.nn.softplus(theta[2+3*p:2+4*p])

    # Attention derivatives and fixed matrices
    W_a, Wa_dot, Wa_ddot = compute_Wa_Wb_dots(Z, eta_a)
    D_ga     = jnp.diag(gamma_a)
    Sigma_inv_diag = 1.0 / sigma2                    # use diagonal structure

    # Model pieces to form E_t
    D_beta = jnp.diag(beta)
    G = D_beta + D_ga @ W_a
    # S does not depend on eta_a; only G does
    S = jnp.eye(p)  # placeholder for clarity; we only need S^T = I here
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)               # (T-1) x p

    # x_a(t) = D_ga Wa_dot Y_{t-1}  -> (T-1) x p  via right-multiply by transposed map
    Xa   = Y[:-1] @ (D_ga @ Wa_dot).T
    # ẍ_a(t) = D_ga Wa_ddot Y_{t-1}
    Xdda = Y[:-1] @ (D_ga @ Wa_ddot).T

    # term_quad = -Σ_t x_a' Σ^{-1} x_a  = -Σ_{t,i} x_a[t,i]^2 / σ_i^2
    term_quad = -jnp.sum((Xa * Xa) * Sigma_inv_diag)

    # term_curv =  Σ_t ẍ_a' Σ^{-1} E_t =  Σ_{t,i} Xdda[t,i] * E[t,i] / σ_i^2
    term_curv =  jnp.sum((Xdda * E) * Sigma_inv_diag)

    return term_quad + term_curv


def hess_block_eta_b(theta, Y, Z):
    """∂²ℓ / ∂η_b² (scalar). Exact second derivative incl. S⁻¹ trace term."""
    T, p = Y.shape
    eta_b   = theta[1]
    gamma_a = theta[2:2+p]              # only for building G (E_t needs it)
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]
    sigma2  = jax.nn.softplus(theta[2+3*p:2+4*p])

    # Attention derivatives and fixed matrices
    W_b, Wb_dot, Wb_ddot = compute_Wa_Wb_dots(Z, eta_b)
    D_gb     = jnp.diag(gamma_b)
    Sigma_inv_diag = 1.0 / sigma2

    # Build G and S for E_t
    from loglik_attn import build_attention  # already imported at top-level
    W_a = build_attention(Z, theta[0])
    D_ga = jnp.diag(gamma_a)
    D_beta = jnp.diag(beta)
    G = D_beta + D_ga @ W_a

    S = jnp.eye(p) - D_gb @ W_b
    S_inv = jnp.linalg.inv(S)

    # Residuals
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)    # (T-1) x p

    # x_b(t)   = D_gb Wb_dot Y_t,   ẍ_b(t) = D_gb Wb_ddot Y_t
    Xb   = Y[1:] @ (D_gb @ Wb_dot).T
    Xddb = Y[1:] @ (D_gb @ Wb_ddot).T

    # Trace term from -T * tr(S^{-1} D_gb Ẇ_b):
    # d/dη_b -> -T * [ tr(S^{-1} D_gb Ẇ_b S^{-1} D_gb Ẇ_b) + tr(S^{-1} D_gb Ẅ_b) ]
    term_trace = -T * (
        jnp.trace(S_inv @ D_gb @ Wb_dot @ S_inv @ D_gb @ Wb_dot)
        + jnp.trace(S_inv @ D_gb @ Wb_ddot)
    )

    # Data terms: Σ_t [ ẍ_b' Σ^{-1} E_t  -  x_b' Σ^{-1} x_b ]
    term_quad = -jnp.sum((Xb * Xb) * Sigma_inv_diag)
    term_curv =  jnp.sum((Xddb * E) * Sigma_inv_diag)

    return term_trace + term_quad + term_curv



def hess_block_gamma_a(theta, Y, Z):
    """∂²ℓ / ∂γ_a ∂γ_aᵀ ∈ R^{p×p}."""
    T, p = Y.shape
    eta_a = theta[0]
    sigma2 = jax.nn.softplus(theta[2+3*p:2+4*p])
    W_a = build_attention(Z, eta_a)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    block = jnp.zeros((p, p))
    for t in range(1, T):
        WaY = W_a @ Y[t-1]
        block -= Sigma_inv @ jnp.diag(WaY * WaY)
    return block


def hess_block_gamma_b(theta, Y, Z, kappa4: float = 3.0):
    """
    ∂²ℓ / ∂γ_b ∂γ_bᵀ — includes trace and residual curvature terms,
    consistent with Appendix B.2–B.3.

    Parameters
    ----------
    theta : array
        Reparameterized parameter vector (η_a, η_b, γ_a, γ_b, β, σ̃²).
    Y : array (T × p)
        Time series.
    Z : array (p × q)
        Feature matrix.
    kappa4 : float, optional
        Fourth standardized moment of ε̃_t components.
        Use 3.0 for Gaussian errors; >3.0 for heavy tails.

    Returns
    -------
    (p × p) Hessian block for γ_b.
    """
    T, p = Y.shape
    eta_a, eta_b = theta[0], theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]
    sigma2  = jax.nn.softplus(theta[2+3*p:2+4*p])

    # model matrices
    W_a, W_b = build_attention(Z, eta_a), build_attention(Z, eta_b)
    D_ga, D_gb = jnp.diag(gamma_a), jnp.diag(gamma_b)
    D_beta = jnp.diag(beta)
    Sigma_inv_diag = 1.0 / sigma2

    S = jnp.eye(p) - D_gb @ W_b
    S_inv = jnp.linalg.inv(S)
    G = D_beta + D_ga @ W_a
    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)

    # --- (1) Trace term: -T * [(W_b S^{-1}) ∘ (W_b S^{-1})ᵀ]
    term_trace = -T * ((W_b @ S_inv) * (W_b @ S_inv).T)

    # --- (2) Residual term: -Σ_t (∂E_t/∂γ_b)' Σ⁻¹ (∂E_t/∂γ_b)
    Xb = Y[1:] @ W_b.T
    term_resid = -jnp.diag(jnp.sum((Xb * Xb) * Sigma_inv_diag, axis=0))

    # --- (3) Optional heavy-tail correction Δ_{γ_bγ_b}
    if kappa4 != 3.0:
        delta = (kappa4 - 3.0) * (W_b @ S_inv) * (W_b @ S_inv)
        term_trace = term_trace - T * (delta * jnp.eye(p))

    return term_trace + term_resid

def hess_block_gamma_b_beta(theta, Y, Z):
    """∂²ℓ / ∂γ_b ∂βᵀ — coupling between lag-b and lag-1 coefficients."""
    T, p = Y.shape
    eta_a, eta_b = theta[0], theta[1]
    sigma2 = jax.nn.softplus(theta[2+3*p:2+4*p])
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    # attention matrix for γ_b
    W_b = build_attention(Z, eta_b)

    # --- Theoretical form:  -Σ_t Σ⁻¹ diag(W_b Y_t ∘ Y_{t-1})
    # Loop version (explicit for clarity)
    block = jnp.zeros((p, p))
    for t in range(1, T):
        block -= Sigma_inv @ jnp.diag(W_b @ Y[t] * Y[t-1])
    return block


def hess_block_beta(theta, Y, Z):
    """∂²ℓ / ∂β ∂βᵀ ∈ R^{p×p}."""
    T, p = Y.shape
    sigma2 = jax.nn.softplus(theta[2+3*p:2+4*p])
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    block = jnp.zeros((p, p))
    for t in range(1, T):
        block -= Sigma_inv @ jnp.diag(Y[t-1] * Y[t-1])
    return block


# =====================================================
# Cross-blocks (σ² = softplus(σ̃²))
# =====================================================

def hess_block_eta_a_eta_b(theta, Y, Z):
    """∂²ℓ / ∂η_a ∂η_b."""
    T, p = Y.shape
    eta_a, eta_b = theta[0], theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    sigma2 = jax.nn.softplus(theta[2+3*p:2+4*p])

    W_a, W_b = build_attention(Z, eta_a), build_attention(Z, eta_b)
    D_ga, D_gb = jnp.diag(gamma_a), jnp.diag(gamma_b)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    val = 0.0
    for t in range(1, T):
        Xa = D_ga @ W_a @ Y[t-1]
        Xb = D_gb @ W_b @ Y[t]
        val -= Xa.T @ Sigma_inv @ Xb
    return val


# =====================================================
# Master Hessian assembly
# =====================================================

def loglik_hess_analytical(theta, Y, Z, update_idx=slice(None)):
    """Assemble full analytical Hessian H(θ̃)."""
    T, p = Y.shape
    d2 = 4*p + 2
    H = jnp.zeros((d2, d2))

    # Diagonal blocks
    H = H.at[0, 0].set(hess_block_eta_a(theta, Y, Z))
    H = H.at[1, 1].set(hess_block_eta_b(theta, Y, Z))
    H = H.at[2:2+p, 2:2+p].set(hess_block_gamma_a(theta, Y, Z))
    H = H.at[2+p:2+2*p, 2+p:2+2*p].set(hess_block_gamma_b(theta, Y, Z))
    H = H.at[2+2*p:2+3*p, 2+2*p:2+3*p].set(hess_block_beta(theta, Y, Z))
    H = H.at[2+3*p:2+4*p, 2+3*p:2+4*p].set(hess_block_sigma2(theta, Y, Z))

    # Cross-blocks
    H = H.at[0, 1].set(hess_block_eta_a_eta_b(theta, Y, Z))
    H = H.at[1, 0].set(hess_block_eta_a_eta_b(theta, Y, Z))

    H = H.at[0:1, 2+2*p:2+3*p].set(hess_block_eta_a_beta(theta, Y, Z))
    H = H.at[2+2*p:2+3*p, 0:1].set(hess_block_eta_a_beta(theta, Y, Z).T)

    H = H.at[0:1, 2+p:2+2*p].set(hess_block_eta_a_gamma_b(theta, Y, Z))
    H = H.at[2+p:2+2*p, 0:1].set(hess_block_eta_a_gamma_b(theta, Y, Z).T)

    H = H.at[0:1, 2:2+p].set(hess_block_eta_a_gamma_a(theta, Y, Z))
    H = H.at[2:2+p, 0:1].set(hess_block_eta_a_gamma_a(theta, Y, Z).T)

    H = H.at[1:2, 2+p:2+2*p].set(hess_block_eta_b_gamma_b(theta, Y, Z))
    H = H.at[2+p:2+2*p, 1:2].set(hess_block_eta_b_gamma_b(theta, Y, Z).T)

    H = H.at[0:1, 2+3*p:2+4*p].set(hess_block_eta_a_sigma2(theta, Y, Z))
    H = H.at[2+3*p:2+4*p, 0:1].set(hess_block_eta_a_sigma2(theta, Y, Z).T)

    H = H.at[1:2, 2:2+p].set(hess_block_eta_b_gamma_a(theta, Y, Z))
    H = H.at[2:2+p, 1:2].set(hess_block_eta_b_gamma_a(theta, Y, Z).T)

    H = H.at[1:2, 2+p:2+2*p].set(hess_block_eta_b_gamma_b(theta, Y, Z))
    H = H.at[2+p:2+2*p, 1:2].set(hess_block_eta_b_gamma_b(theta, Y, Z).T)

    H = H.at[1:2, 2+2*p:2+3*p].set(hess_block_eta_b_beta(theta, Y, Z))
    H = H.at[2+2*p:2+3*p, 1:2].set(hess_block_eta_b_beta(theta, Y, Z).T)

    H = H.at[1:2, 2+3*p:2+4*p].set(hess_block_eta_b_sigma2(theta, Y, Z))
    H = H.at[2+3*p:2+4*p, 1:2].set(hess_block_eta_b_sigma2(theta, Y, Z).T)

    H = H.at[2:2+p, 2+p:2+2*p].set(hess_block_gamma_a_gamma_b(theta, Y, Z))
    H = H.at[2+p:2+2*p, 2:2+p].set(hess_block_gamma_a_gamma_b(theta, Y, Z).T)

    H = H.at[2:2+p, 2+2*p:2+3*p].set(hess_block_gamma_a_beta(theta, Y, Z))
    H = H.at[2+2*p:2+3*p, 2:2+p].set(hess_block_gamma_a_beta(theta, Y, Z).T)

    H = H.at[2:2+p, 2+3*p:2+4*p].set(hess_block_gamma_a_sigma2(theta, Y, Z))
    H = H.at[2+3*p:2+4*p, 2:2+p].set(hess_block_gamma_a_sigma2(theta, Y, Z).T)

    H = H.at[2+p:2+2*p, 2+2*p:2+3*p].set(hess_block_gamma_b_beta(theta, Y, Z))
    H = H.at[2+2*p:2+3*p, 2+p:2+2*p].set(hess_block_gamma_b_beta(theta, Y, Z).T)

    H = H.at[2+p:2+2*p, 2+3*p:2+4*p].set(hess_block_gamma_b_sigma2(theta, Y, Z))
    H = H.at[2+3*p:2+4*p, 2+p:2+2*p].set(hess_block_gamma_b_sigma2(theta, Y, Z).T)

    H = H.at[2+2*p:2+3*p, 2+3*p:2+4*p].set(hess_block_beta_sigma2(theta, Y, Z))
    H = H.at[2+3*p:2+4*p, 2+2*p:2+3*p].set(hess_block_beta_sigma2(theta, Y, Z).T)

    # slice(None) means update all parameters
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta.size)
    update_idx = jnp.array(update_idx)

    H_partial = H[jnp.ix_(update_idx, update_idx)]

    return H_partial  / (T-1)

# =====================================================
# Test harness
# =====================================================

def get_block(H, p, row_block, col_block):
    """Extract a named Hessian block."""
    def idx(name):
        if name == "eta_a": return jnp.array([0])
        elif name == "eta_b": return jnp.array([1])
        elif name == "gamma_a": return jnp.arange(2, 2+p)
        elif name == "gamma_b": return jnp.arange(2+p, 2+2*p)
        elif name == "beta": return jnp.arange(2+2*p, 2+3*p)
        elif name == "sigma2": return jnp.arange(2+3*p, 2+4*p)
        else: raise ValueError("Unknown block name")
    ridx, cidx = idx(row_block), idx(col_block)
    return H[jnp.ix_(ridx, cidx)]


if __name__ == "__main__":

    import random
    import matplotlib.pyplot as plt
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

    # --- Simulate data ---
    Y, Z, W_a, W_b = simulate_VAR_attention(key, params_true, T=T, p=p, q=q, dist="normal")
    print("Simulated Y shape:", Y.shape)

    # --- Convert σ² → σ̃² (softplus inverse) ---
    sigma2_tilde_true = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = params_true.copy()
    params_reparam["sigma2"] = sigma2_tilde_true

    # --- Flatten parameters into reparameterized vector θ̃ ---
    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    theta_true_reparam = flatten_params(params_reparam, order)
    print("θ̃ length:", theta_true_reparam.size)

    # --- Compute analytical and automatic Hessians ---
    H_analytical = loglik_hess_analytical(theta_true_reparam, Y, Z)
    H_auto = jax.hessian(loglik_reparam)(theta_true_reparam, Y, Z)

    # --- Full-matrix comparison ---
    diff_full = jnp.max(jnp.abs(H_analytical - H_auto))
    print(f"\nMax abs diff (full Hessian, reparameterized): {diff_full:.4f}")

    # --- Diagonal block-by-block comparison ---
    diag_blocks = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    for block in diag_blocks:
        Ha = get_block(H_analytical, p, block, block)
        Hu = get_block(H_auto, p, block, block)
        diff = jnp.max(jnp.abs(Ha - Hu))
        print(f"Block {block:<10s} max abs diff = {diff:.4f}")

    # --- Cross-blocks ---
    cross_blocks = [
        ("eta_a", "eta_b"),
        ("eta_a", "gamma_a"),
        ("eta_a", "gamma_b"),
        ("eta_a", "beta"),
        ("eta_a", "sigma2"),
        ("gamma_b", "beta"),
        ("gamma_b", "sigma2"),
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
