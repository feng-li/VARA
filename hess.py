import jax
import jax.numpy as jnp


from loglik_attn import build_attention

# --- Helpers ---
def Sigma_inv_from_sigma2(sigma2):
    return jnp.diag(1.0 / sigma2)

# =====================================================
# all Hessian block functions:

# Diagonal:

# hess_block_eta_a

# hess_block_eta_b

# hess_block_gamma_a

# hess_block_gamma_b

# hess_block_beta

# hess_block_sigma2

# Cross:

# hess_block_eta_a_eta_b

# hess_block_eta_a_gamma_a

# hess_block_eta_a_gamma_b

# hess_block_eta_a_beta

# hess_block_eta_a_sigma2

# hess_block_gamma_b_beta

# hess_block_gamma_b_sigma2

# hess_block_beta_sigma2

# =====================================================

def hess_block_eta_a(theta, Y, Z):
    """
    ∂²ℓ / ∂η_a² (scalar).
    """
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[2:2+p]
    sigma2 = theta[2+3*p:2+4*p]

    W_a = build_attention(Z, eta_a)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    val = 0.0
    for t in range(1, T):
        Xa = jnp.diag(gamma_a) @ (W_a @ Y[t-1])
        val -= Xa.T @ Sigma_inv @ Xa
    return val


def hess_block_eta_b(theta, Y, Z):
    """
    ∂²ℓ / ∂η_b² (scalar).
    """
    T, p = Y.shape
    eta_b = theta[1]
    gamma_b = theta[2+p:2+2*p]
    sigma2 = theta[2+3*p:2+4*p]

    W_b = build_attention(Z, eta_b)
    D_gb = jnp.diag(gamma_b)
    S = jnp.eye(p) - D_gb @ W_b
    S_inv = jnp.linalg.inv(S)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    term_trace = -T * jnp.trace(S_inv @ D_gb @ W_b @ S_inv @ D_gb @ W_b)

    val = term_trace
    for t in range(1, T):
        Xb = D_gb @ W_b @ Y[t]
        val -= Xb.T @ Sigma_inv @ Xb
    return val


def hess_block_gamma_a(theta, Y, Z):
    """
    ∂²ℓ / ∂γ_a ∂γ_aᵀ ∈ R^{p×p}.
    """
    T, p = Y.shape
    eta_a = theta[0]
    sigma2 = theta[2+3*p:2+4*p]
    W_a = build_attention(Z, eta_a)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    block = jnp.zeros((p, p))
    for t in range(1, T):
        WaY = W_a @ Y[t-1]
        block -= Sigma_inv @ jnp.diag(WaY * WaY)
    return block


def hess_block_gamma_b(theta, Y, Z):
    """
    ∂²ℓ / ∂γ_b ∂γ_bᵀ ∈ R^{p×p}.
    """
    T, p = Y.shape
    eta_b = theta[1]
    gamma_b = theta[2+p:2+2*p]
    sigma2 = theta[2+3*p:2+4*p]

    W_b = build_attention(Z, eta_b)
    D_gb = jnp.diag(gamma_b)
    S = jnp.eye(p) - D_gb @ W_b
    S_inv = jnp.linalg.inv(S)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    block = -T * (W_b @ S_inv) * (W_b @ S_inv)
    for t in range(1, T):
        WbY = W_b @ Y[t]
        block -= Sigma_inv @ jnp.diag(WbY * WbY)
    return block


def hess_block_beta(theta, Y, Z):
    """
    ∂²ℓ / ∂β ∂βᵀ ∈ R^{p×p}.
    """
    T, p = Y.shape
    sigma2 = theta[2+3*p:2+4*p]
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    block = jnp.zeros((p, p))
    for t in range(1, T):
        block -= Sigma_inv @ jnp.diag(Y[t-1] * Y[t-1])
    return block


def hess_block_sigma2(theta, Y, Z):
    """
    ∂²ℓ / ∂σ² ∂σ²ᵀ ∈ R^{p×p} (diagonal).
    """
    T, p = Y.shape
    sigma2 = theta[2+3*p:2+4*p]
    eta_a = theta[0]
    eta_b = theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]

    W_a = build_attention(Z, eta_a)
    W_b = build_attention(Z, eta_b)

    D_beta = jnp.diag(beta)
    D_ga = jnp.diag(gamma_a)
    D_gb = jnp.diag(gamma_b)

    G = D_beta + D_ga @ W_a
    S = jnp.eye(p) - D_gb @ W_b

    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)

    block = (T/2) * jnp.diag(sigma2**-4)
    block -= jnp.diag((sigma2**-6) * jnp.sum(E * E, axis=0))
    return block

# =====================================================
# Off-Diagonal Blocks
# =====================================================

def hess_block_eta_a_eta_b(theta, Y, Z):
    """
    ∂²ℓ / ∂η_a ∂η_b
    """
    T, p = Y.shape
    eta_a, eta_b = theta[0], theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    sigma2 = theta[2+3*p:2+4*p]

    W_a = build_attention(Z, eta_a)
    W_b = build_attention(Z, eta_b)
    D_ga = jnp.diag(gamma_a)
    D_gb = jnp.diag(gamma_b)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    val = 0.0
    for t in range(1, T):
        Xa = D_ga @ W_a @ Y[t-1]
        Xb = D_gb @ W_b @ Y[t]
        val -= Xa.T @ Sigma_inv @ Xb
    return val


def hess_block_eta_a_gamma_a(theta, Y, Z):
    """
    ∂²ℓ / ∂η_a ∂γ_aᵀ
    """
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[2:2+p]
    sigma2 = theta[2+3*p:2+4*p]

    W_a = build_attention(Z, eta_a)
    D_ga = jnp.diag(gamma_a)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    block = jnp.zeros((1, p))
    for t in range(1, T):
        WaY = W_a @ Y[t-1]
        block -= (WaY * (Sigma_inv @ (D_ga @ WaY)))[None, :]
    return block


def hess_block_eta_a_gamma_b(theta, Y, Z):
    """
    ∂²ℓ / ∂η_a ∂γ_bᵀ
    """
    T, p = Y.shape
    eta_a, eta_b = theta[0], theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    sigma2 = theta[2+3*p:2+4*p]

    W_a = build_attention(Z, eta_a)
    W_b = build_attention(Z, eta_b)
    D_ga = jnp.diag(gamma_a)
    D_gb = jnp.diag(gamma_b)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    block = jnp.zeros((1, p))
    for t in range(1, T):
        Xa = D_ga @ W_a @ Y[t-1]
        Xb = D_gb @ W_b @ Y[t]
        block -= (Xa * (Sigma_inv @ Xb))[None, :]
    return block


def hess_block_eta_a_beta(theta, Y, Z):
    """
    ∂²ℓ / ∂η_a ∂βᵀ
    """
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[2:2+p]
    sigma2 = theta[2+3*p:2+4*p]

    W_a = build_attention(Z, eta_a)
    D_ga = jnp.diag(gamma_a)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    block = jnp.zeros((1, p))
    for t in range(1, T):
        WaY = W_a @ Y[t-1]
        block -= (WaY * (Sigma_inv @ Y[t-1]))[None, :]
    return block


def hess_block_eta_a_sigma2(theta, Y, Z):
    """
    ∂²ℓ / ∂η_a ∂σ²ᵀ
    """
    T, p = Y.shape
    eta_a = theta[0]
    gamma_a = theta[2:2+p]
    sigma2 = theta[2+3*p:2+4*p]

    W_a = build_attention(Z, eta_a)
    D_ga = jnp.diag(gamma_a)

    block = jnp.zeros((1, p))
    for t in range(1, T):
        WaY = D_ga @ W_a @ Y[t-1]
        block -= (sigma2**-4 * (WaY * WaY))[None, :]
    return block


def hess_block_gamma_b_beta(theta, Y, Z):
    """
    ∂²ℓ / ∂γ_b ∂βᵀ ∈ R^{p×p}.
    """
    T, p = Y.shape
    eta_b = theta[1]
    sigma2 = theta[2+3*p:2+4*p]

    W_b = build_attention(Z, eta_b)
    Sigma_inv = Sigma_inv_from_sigma2(sigma2)

    block = jnp.zeros((p, p))
    for t in range(1, T):
        block -= Sigma_inv @ jnp.diag(W_b @ Y[t] * Y[t-1])
    return block


def hess_block_gamma_b_sigma2(theta, Y, Z):
    """
    ∂²ℓ / ∂γ_b ∂σ²ᵀ ∈ R^{p×p} (diagonal).
    """
    T, p = Y.shape
    eta_a = theta[0]
    eta_b = theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]
    sigma2  = theta[2+3*p:2+4*p]

    W_a = build_attention(Z, eta_a)
    W_b = build_attention(Z, eta_b)

    D_beta = jnp.diag(beta)
    D_ga = jnp.diag(gamma_a)
    D_gb = jnp.diag(gamma_b)

    G = D_beta + D_ga @ W_a
    S = jnp.eye(p) - D_gb @ W_b

    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)

    block = -jnp.diag(sigma2**-4 * jnp.sum(E * (W_b @ Y[1:].T).T, axis=0))
    return block


def hess_block_beta_sigma2(theta, Y, Z):
    """
    ∂²ℓ / ∂β ∂σ²ᵀ ∈ R^{p×p} (diagonal).
    """
    T, p = Y.shape
    eta_a = theta[0]
    eta_b = theta[1]
    gamma_a = theta[2:2+p]
    gamma_b = theta[2+p:2+2*p]
    beta    = theta[2+2*p:2+3*p]
    sigma2  = theta[2+3*p:2+4*p]

    W_a = build_attention(Z, eta_a)
    W_b = build_attention(Z, eta_b)

    D_beta = jnp.diag(beta)
    D_ga = jnp.diag(gamma_a)
    D_gb = jnp.diag(gamma_b)

    G = D_beta + D_ga @ W_a
    S = jnp.eye(p) - D_gb @ W_b

    E = (Y[1:] @ S.T) - (Y[:-1] @ G.T)

    block = -jnp.diag(sigma2**-4 * jnp.sum(E * Y[:-1], axis=0))
    return block


# =====================================================
# The final master block
# =====================================================

def loglik_hess_analytical(theta, Y, Z):
    """
    Assemble full analytical Hessian H(θ) ∈ R^{(4p+2) × (4p+2)}.
    Uses all block functions defined previously.
    """
    T, p = Y.shape
    d2 = 4*p + 2
    H = jnp.zeros((d2, d2))

    # --- Diagonal blocks ---
    H = H.at[0,0].set(hess_block_eta_a(theta, Y, Z))
    H = H.at[1,1].set(hess_block_eta_b(theta, Y, Z))

    H = H.at[2:2+p, 2:2+p].set(hess_block_gamma_a(theta, Y, Z))
    H = H.at[2+p:2+2*p, 2+p:2+2*p].set(hess_block_gamma_b(theta, Y, Z))
    H = H.at[2+2*p:2+3*p, 2+2*p:2+3*p].set(hess_block_beta(theta, Y, Z))
    H = H.at[2+3*p:2+4*p, 2+3*p:2+4*p].set(hess_block_sigma2(theta, Y, Z))

    # --- Off-diagonal blocks ---
    # ηa–ηb
    val = hess_block_eta_a_eta_b(theta, Y, Z)
    H = H.at[0,1].set(val)
    H = H.at[1,0].set(val)

    # ηa–γa
    block = hess_block_eta_a_gamma_a(theta, Y, Z)
    H = H.at[0, 2:2+p].set(block)
    H = H.at[2:2+p, 0].set(block.T)

    # ηa–γb
    block = hess_block_eta_a_gamma_b(theta, Y, Z)
    H = H.at[0, 2+p:2+2*p].set(block)
    H = H.at[2+p:2+2*p, 0].set(block.T)

    # ηa–β
    block = hess_block_eta_a_beta(theta, Y, Z)
    H = H.at[0, 2+2*p:2+3*p].set(block)
    H = H.at[2+2*p:2+3*p, 0].set(block.T)

    # ηa–σ²
    block = hess_block_eta_a_sigma2(theta, Y, Z)
    H = H.at[0, 2+3*p:2+4*p].set(block)
    H = H.at[2+3*p:2+4*p, 0].set(block.T)

    # γb–β
    block = hess_block_gamma_b_beta(theta, Y, Z)
    H = H.at[2+p:2+2*p, 2+2*p:2+3*p].set(block)
    H = H.at[2+2*p:2+3*p, 2+p:2+2*p].set(block.T)

    # γb–σ²
    block = hess_block_gamma_b_sigma2(theta, Y, Z)
    H = H.at[2+p:2+2*p, 2+3*p:2+4*p].set(block)
    H = H.at[2+3*p:2+4*p, 2+p:2+2*p].set(block.T)

    # β–σ²
    block = hess_block_beta_sigma2(theta, Y, Z)
    H = H.at[2+2*p:2+3*p, 2+3*p:2+4*p].set(block)
    H = H.at[2+3*p:2+4*p, 2+2*p:2+3*p].set(block.T)

    # --- Symmetrize for safety ---
    H = 0.5 * (H + H.T)

    return H


# =====================================================
# We have already have defined:
# - build_attention
# - simulate_VAR_attention
# - loglik
# - all hess_block_* functions
# - loglik_hess_analytical
# =====================================================

def get_block(H, p, row_block, col_block):
    """Extract (row_block, col_block) from Hessian matrix H."""
    def idx(name):
        if name == "eta_a":
            return jnp.array([0])
        elif name == "eta_b":
            return jnp.array([1])
        elif name == "gamma_a":
            return jnp.arange(2, 2+p)
        elif name == "gamma_b":
            return jnp.arange(2+p, 2+2*p)
        elif name == "beta":
            return jnp.arange(2+2*p, 2+3*p)
        elif name == "sigma2":
            return jnp.arange(2+3*p, 2+4*p)
        else:
            raise ValueError("Unknown block name")
    ridx, cidx = idx(row_block), idx(col_block)
    return H[jnp.ix_(ridx, cidx)]


if __name__ == "__main__":
    # --- Simulate dataset ---
    from dgp import simulate_VAR_attention
    from loglik_attn import loglik

    key = jax.random.PRNGKey(0)
    Y, theta_true, Z = simulate_VAR_attention(key, T=20, p=3, q=3, dist="normal")

    # --- Compute Hessians ---
    H_analytical = loglik_hess_analytical(theta_true, Y, Z)
    H_auto = jax.hessian(loglik)(theta_true, Y, Z)

    # --- Full matrix comparison ---
    diff = jnp.max(jnp.abs(H_analytical - H_auto))
    print("Max abs diff (full Hessian):", diff)

    # --- Diagonal blocks ---
    diag_blocks = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    for block in diag_blocks:
        Ha = get_block(H_analytical, Y.shape[1], block, block)
        Hu = get_block(H_auto, Y.shape[1], block, block)
        print(f"Block {block}: max abs diff = {jnp.max(jnp.abs(Ha - Hu))}")

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

    for rb, cb in cross_blocks:
        Ha = get_block(H_analytical, Y.shape[1], rb, cb)
        Hu = get_block(H_auto, Y.shape[1], rb, cb)
        diff = jnp.max(jnp.abs(Ha - Hu))
        print(f"Block ({rb}, {cb}): max abs diff = {diff}")
