import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.sparse.linalg import cg
from jax.scipy.linalg import cho_solve

from hess import compute_Wa_Wb_dots

# ===============================================================
# === Utility functions =========================================
# ===============================================================
def vec(X):
    """Vectorize matrix X (column-major flatten)."""
    return X.reshape(-1, 1)


def unvec(v, shape):
    """Inverse of vec() — reshape vector to given shape."""
    return v.reshape(shape)


# ===============================================================
# === Stationary covariance =====================================
# ===============================================================
def Gamma0_stationary(S, G, sigma2):
    """
    Solve for the stationary covariance matrix Γ₀ of Yₜ in:
        Yₜ = S⁻¹ G Yₜ₋₁ + S⁻¹ εₜ ,   εₜ ∼ (0, Σₑ).

    Note that this function is very memory consuming.
    """
    p = S.shape[0]
    Sinv = jnp.linalg.inv(S)
    A = Sinv @ G
    kron_term = jnp.kron(A, A)
    I = jnp.eye(p * p)
    Sigma_e = jnp.diag(sigma2)
    rhs = vec(Sinv @ Sigma_e @ Sinv.T)
    vecGamma = jnp.linalg.solve(I - kron_term, rhs)

    return unvec(vecGamma, (p, p))

def lyapunov_doubling(A, Q, max_iter=30, tol=1e-6):
    """
    Solve Γ = A Γ Aᵀ + Q using the doubling (Smith) iteration.
    JAX-compatible, O(p²) memory.
    """
    Ak = A
    Qk = Q

    for _ in range(max_iter):
        Q_next = Qk + Ak @ Qk @ Ak.T
        Ak = Ak @ Ak

        if jnp.linalg.norm(Ak) < tol:
            Qk = Q_next
            break

        Qk = Q_next

    return Qk


def Gamma0_stationary_with_lyapunov_doubling(
    S,
    G,
    sigma2,
    max_iter=30,
    tol=1e-6,
):
    """
    Compute stationary covariance Γ₀ using Lyapunov doubling.

    Model:
        Y_t = S^{-1} G Y_{t-1} + S^{-1} ε_t,
        ε_t ~ (0, Σ_e)

    Γ₀ solves:
        Γ₀ = A Γ₀ Aᵀ + Q
    where:
        A = S^{-1} G
        Q = S^{-1} Σ_e S^{-T}

    Parameters
    ----------
    S : (p, p) array
        Structural matrix (I - diag(gamma_b) @ Wb)
    G : (p, p) array
        Autoregressive matrix
    Sigma_e : (p, p) array
        Innovation covariance (typically diag(sigma2))
    max_iter : int
        Maximum doubling iterations
    tol : float
        Convergence tolerance for ||A_k||

    Returns
    -------
    Gamma0 : (p, p) array
        Stationary covariance matrix
    """
    A = jnp.linalg.solve(S, G)
    # Q = jnp.linalg.solve(S, jnp.diag(sigma2))
    Q = rsolve_S(S, jnp.eye(S.shape[0])) * sigma2[None, :]
    Q = jnp.linalg.solve(S, Q.T).T   # left + right solve => S^{-1} Σ_e S^{-T}

    # Solve Γ₀ = A Γ₀ Aᵀ + Q
    Gamma0 = lyapunov_doubling(A, Q, max_iter=max_iter, tol=tol)

    return Gamma0


def ldiag_mul(gamma, X):
    """
    Compute diag(gamma) @ X without forming diag(gamma).

    Parameters
    ----------
    gamma : (p,) array
        Diagonal entries.
    X : (p, q) array
        Matrix to be left-multiplied.

    Returns
    -------
    (p, q) array
        diag(gamma) @ X
    """
    return gamma[:, None] * X

def rsolve_S(S, M):
    """
    Compute M @ S^{-1} without forming inverse.
    """
    return jnp.linalg.solve(S.T, M.T).T

# ===============================================================
# === Assemble analytical V =====================================
# ===============================================================
def assemble_V(Wa, Wb, Wa_dot, Wb_dot,
                    gamma_a, gamma_b,
                    G, Gamma0,
                    sigma2, T, kappa4=3.0, full=True):
    """
    Assemble analytical covariance matrix V(θ) = E[s_t s_tᵀ].
    """
    p = Wa.shape[0]
    d = 4 * p + 2

    S = jnp.eye(Wb.shape[0]) - ldiag_mul(gamma_b, Wb)
    # Sinv = jnp.linalg.inv(S)

    gamma_b_x_Wb_dot = ldiag_mul(gamma_b, Wb_dot)
    gamma_a_x_Wa_dot = ldiag_mul(gamma_a, Wa_dot)


    # Sigma_inv = jnp.diag(sigma2_inv)  # explicit diag for clarity
    sigma2_inv = 1.0 / sigma2

    # --- Diagonal blocks ---
    # V_eta_a_eta_a0 = T * jnp.trace(
    #     gamma_a_x_Wa_dot.T @ Sigma_inv @
    #     gamma_a_x_Wa_dot @ Gamma0
    # )

    SA = ldiag_mul(sigma2_inv, gamma_a_x_Wa_dot)     # Σ⁻¹ A
    AG = gamma_a_x_Wa_dot @ Gamma0                      # A Γ₀
    V_eta_a_eta_a = T * jnp.sum(SA * AG)  # vector 1

    WbS   = rsolve_S(S, Wb)
    WbS_d = rsolve_S(S, Wb_dot)
    GbS   = rsolve_S(S, gamma_b_x_Wb_dot)   # or X

    # V_eta_b_eta_b0 = T * jnp.trace(gamma_b_x_Wb_dot.T @ Sigma_inv @ gamma_b_x_Wb_dot @ Gamma0 +
    #                               (rsolve_S(S, gamma_b_x_Wb_dot)) @ (rsolve_S(S, gamma_b_x_Wb_dot)) +
    #                               (kappa4 - 3.0) * (rsolve_S(S, gamma_b_x_Wb_dot)) * (rsolve_S(S, gamma_b_x_Wb_dot)))

    V_eta_b_eta_b = T * (
        # term1 = tr(gamma_b_x_Wb_dotᵀ Σ⁻¹ gamma_b_x_Wb_dot Γ0)
        jnp.einsum(
            "i,ik,kj,ij->",
            sigma2_inv,              # Σ^{-1} diagonal
            gamma_b_x_Wb_dot,
            Gamma0,
            gamma_b_x_Wb_dot
        ) +
        # term2 = tr(X @ X) = sum_{i,j} X_ij * X_ji
        jnp.einsum("ij,ji->", GbS, GbS) +
        # term3 = (kappa4-3) * tr(X * X) = (kappa4-3) * sum_i X_ii^2
        (kappa4 - 3.0) * jnp.einsum("ii->", GbS * GbS)
    )  # vector 1

    # V_gamma_a_gamma_a = T * (jnp.diag(sigma2_inv) * (Wa @ Gamma0 @ Wa.T))
    diag_M = jnp.sum(Wa * (Wa @ Gamma0), axis=1)
    V_gamma_a_gamma_a = T * sigma2_inv * diag_M  # vector p

    diag_term = jnp.sum(Wb * (Gamma0 @ Wb.T).T, axis=1)
    V_gamma_b_gamma_b = T * (jnp.diag(sigma2_inv * diag_term) + WbS * (WbS.T) +
                             (kappa4 - 3.0) * jnp.diag(jnp.diag(WbS * WbS)))  # matrix p^2

    # V_beta_beta = T * (jnp.diag(sigma2_inv) * Gamma0)
    V_beta_beta = T * sigma2_inv * jnp.diag(Gamma0)  # vector p

    V_sigma_sigma = T * (kappa4 - 1.0) / 4.0 * sigma2_inv**2  # vector p

    # --- Cross-covariance examples ---
    # V_eta_b_eta_a0 = T * jnp.trace(
    #     Wa_dot.T @ jnp.diag(gamma_a) @ Sigma_inv @
    #     rsolve_S(S, gamma_b_x_Wb_dot) @ G @ Gamma0
    # )

    V_eta_b_eta_a = T * jnp.einsum(
        "i,ij,ij->",
        gamma_a / sigma2,
        Wa_dot,
        GbS @ G @ Gamma0)  # 1x1

    diag_term = jnp.sum(Wa * (Gamma0 @ Wa_dot.T).T, axis=1)
    V_gamma_a_eta_a = T * (gamma_a * sigma2_inv * diag_term)  # vector p


    # V_gamma_b_eta_a = T * (gamma_a * (sigma2_inv) *
    #                        jnp.diag((WbS @ G) @ Gamma0 @ Wa_dot.T))
    M = (WbS @ G) @ Gamma0
    diag_term = jnp.sum(M * Wa_dot, axis=1)
    V_gamma_b_eta_a = T * gamma_a * sigma2_inv * diag_term  # vector p

    # V_beta_eta_a = T * (gamma_a * (sigma2_inv) * jnp.diag(Gamma0 @ Wa_dot.T))
    diag_term = jnp.sum(Gamma0 * Wa_dot, axis=1)
    V_beta_eta_a = T * gamma_a * sigma2_inv * diag_term  # vector p


    # V_sigma_eta_a = jnp.zeros((p, ))
    V_sigma_eta_a = jnp.zeros(p)


    # V_gamma_a_eta_b0 = T * (gamma_b * (sigma2_inv) *
    #                        jnp.diag(Wa_dot @ Gamma0 @ (WbS_d @ G).T))
    M = Wa_dot @ Gamma0
    diag_term = jnp.sum(M * (WbS_d @ G), axis=1)
    V_gamma_a_eta_b = T * gamma_b * sigma2_inv * diag_term  # vector p


    V_gamma_b_eta_b = T * (
        gamma_b * sigma2_inv * jnp.diag(Wb @ Gamma0 @ Wb_dot) +
        jnp.diag(WbS @ rsolve_S(S, gamma_b_x_Wb_dot)) +
        (kappa4 - 3.0) * gamma_b * jnp.diag(WbS) * jnp.diag(WbS_d))  # vector p

    A = WbS_d @ G  # p×p
    diag_term = jnp.sum(Gamma0 * A, axis=1)
    V_beta_eta_b = T * (gamma_b * sigma2_inv * diag_term)  # vector p

    V_sigma_eta_b = T * (kappa4 - 1.0) / 2.0 * (gamma_b * sigma2_inv * jnp.diag(WbS_d))  # vector p

    M = WbS @ G                      # = W_b S^{-1} G
    diag_term = jnp.sum((M @ Gamma0) * Wa, axis=1)
    V_gamma_b_gamma_a = T * (sigma2_inv * diag_term)  # vector p

    diag_term = jnp.sum(Gamma0 * Wa, axis=1)
    V_beta_gamma_a = T * (sigma2_inv * diag_term)  # vector p

    V_sigma_gamma_a = jnp.zeros(p)

    A = WbS @ G                          # = W_b S^{-1} G
    diag_term = jnp.sum(Gamma0 * A, axis=1)
    V_beta_gamma_b = T * (sigma2_inv * diag_term)

    diag_term = jnp.diag(WbS)
    V_sigma_gamma_b = (
        T * (kappa4 - 1.0) / 2.0
        * (sigma2_inv * diag_term)
    )

    # V_sigma_beta    = jnp.zeros((p, p))
    V_sigma_beta = jnp.zeros(p)

    # --- Assemble selected blocks ---
    if full:
        # --- Diag-block filling ---
        V = jnp.zeros((d, d))
        V = V.at[0, 0].set(V_eta_a_eta_a)
        V = V.at[1, 1].set(V_eta_b_eta_b)

        V = V.at[2:2+p, 2:2+p].set(jnp.diag(V_gamma_a_gamma_a))

        V = V.at[2+p:2+2*p, 2+p:2+2*p].set(V_gamma_b_gamma_b)  # matrix p^2

        V = V.at[2+2*p:2+3*p, 2+2*p:2+3*p].set(jnp.diag(V_beta_beta))

        V = V.at[2+3*p:2+4*p, 2+3*p:2+4*p].set(jnp.diag(V_sigma_sigma))

        # --- Cross-block for vector symmetric filling ---
        V = V.at[1, 0].set(V_eta_b_eta_a)
        V = V.at[0, 1].set(V_eta_b_eta_a)

        V = V.at[2:2+p, 0].set(V_gamma_a_eta_a)
        V = V.at[0, 2:2+p].set(V_gamma_a_eta_a)

        V = V.at[2+p:2+2*p, 0].set(V_gamma_b_eta_a)
        V = V.at[0, 2+p:2+2*p].set(V_gamma_b_eta_a)

        V = V.at[2+2*p:2+3*p, 0].set(V_beta_eta_a)
        V = V.at[0, 2+2*p:2+3*p].set(V_beta_eta_a)

        V = V.at[2+3*p:2+4*p, 0].set(V_sigma_eta_a)
        V = V.at[0, 2+3*p:2+4*p].set(V_sigma_eta_a)

        V = V.at[2:2+p, 1].set(V_gamma_a_eta_b)
        V = V.at[1, 2:2+p].set(V_gamma_a_eta_b)

        V = V.at[2+p:2+2*p, 1].set(V_gamma_b_eta_b)
        V = V.at[1, 2+p:2+2*p].set(V_gamma_b_eta_b)

        V = V.at[2+2*p:2+3*p, 1].set(V_beta_eta_b)
        V = V.at[1, 2+2*p:2+3*p].set(V_beta_eta_b)

        V = V.at[2+3*p:2+4*p, 1].set(V_sigma_eta_b)
        V = V.at[1, 2+3*p:2+4*p].set(V_sigma_eta_b)

        # --- Cross-block diag symmetric filling ---
        i = jnp.arange(p)
        # gamma_b × gamma_a
        V = V.at[2+p + i, 2 + i].set(V_gamma_b_gamma_a)
        V = V.at[2 + i, 2+p + i].set(V_gamma_b_gamma_a)

        # beta × gamma_a
        V = V.at[2+2*p + i, 2 + i].set(V_beta_gamma_a)
        V = V.at[2 + i, 2+2*p + i].set(V_beta_gamma_a)

        # sigma × gamma_a
        V = V.at[2+3*p + i, 2 + i].set(V_sigma_gamma_a)
        V = V.at[2 + i, 2+3*p + i].set(V_sigma_gamma_a)

        # beta × gamma_b
        V = V.at[2+2*p + i, 2+p + i].set(V_beta_gamma_b)
        V = V.at[2+p + i, 2+2*p + i].set(V_beta_gamma_b)

        # sigma × gamma_b
        V = V.at[2+3*p + i, 2+p + i].set(V_sigma_gamma_b)
        V = V.at[2+p + i, 2+3*p + i].set(V_sigma_gamma_b)

        # sigma × beta
        V = V.at[2+3*p + i, 2+2*p + i].set(V_sigma_beta)
        V = V.at[2+2*p + i, 2+3*p + i].set(V_sigma_beta)


    else:
        V = {}
        V["eta_a_eta_a"] = V_eta_a_eta_a
        V["eta_b_eta_b"] = V_eta_b_eta_b
        V["eta_b_eta_a"] = V_eta_b_eta_a

        # scalar–vector
        V["gamma_a_eta_a"] = V_gamma_a_eta_a      # (p,)
        V["gamma_b_eta_a"] = V_gamma_b_eta_a
        V["beta_eta_a"]    = V_beta_eta_a
        V["sigma_eta_a"]   = V_sigma_eta_a

        V["gamma_a_eta_b"] = V_gamma_a_eta_b
        V["gamma_b_eta_b"] = V_gamma_b_eta_b
        V["beta_eta_b"]    = V_beta_eta_b
        V["sigma_eta_b"]   = V_sigma_eta_b

        # vector–vector (diagonal only!)
        V["gamma_a_gamma_a"] = V_gamma_a_gamma_a  # (p,)
        V["gamma_b_gamma_a"] = V_gamma_b_gamma_a
        V["beta_gamma_a"]    = V_beta_gamma_a
        V["sigma_gamma_a"]   = V_sigma_gamma_a

        V["gamma_b_gamma_b"] = V_gamma_b_gamma_b  # matrix p^2
        V["beta_gamma_b"]    = V_beta_gamma_b
        V["sigma_gamma_b"]   = V_sigma_gamma_b

        V["beta_beta"]       = V_beta_beta
        V["sigma_beta"]      = V_sigma_beta
        V["sigma_sigma"]     = V_sigma_sigma

    return V


# ===============================================================
# === Assemble analytical H =====================================
# ===============================================================
def assemble_H(Wa, Wb, Wa_dot, Wb_dot,
               gamma_a, gamma_b,
               G, Gamma0,
               sigma2, T, kappa4=3.0, full=True):

    V = assemble_V(Wa, Wb, Wa_dot, Wb_dot,
                   gamma_a, gamma_b,
                   G, Gamma0,
                   sigma2, T, kappa4, full)

    S = jnp.eye(Wb.shape[0]) - ldiag_mul(gamma_b, Wb)
    WbS = rsolve_S(S, Wb)
    WbS_d = rsolve_S(S, Wb_dot)

    gamma_b_x_Wb_dot = ldiag_mul(gamma_b, Wb_dot)

    sigma2_inv = 1 / sigma2
    sigma_inv4 = 1.0 / (sigma2 ** 2)

    X = rsolve_S(S, gamma_b_x_Wb_dot)
    Delta_eta_b_eta_b = (kappa4 - 3.0) * jnp.sum(jnp.square(jnp.diagonal(X)))

    diag_WbS = jnp.diagonal(WbS)
    Delta_gamma_b_gamma_b = (kappa4 - 3.0) * (diag_WbS ** 2)  # vector p

    Delta_sigma_sigma = (kappa4 - 3) / 4.0 * sigma_inv4  # vector p

    Delta_gamma_b_eta_b = (kappa4 - 3) * (
        gamma_b * jnp.diagonal(WbS) * jnp.diagonal(WbS_d)
    )

    Delta_sigma_eta_b = (kappa4 - 3) / 2.0 * (
        gamma_b * sigma2_inv * jnp.diagonal(WbS_d)
    )

    Delta_sigma_gamma_b = (
        (kappa4 - 3.0) / 2.0
        * sigma2_inv
        * jnp.diagonal(WbS)
    )

    H = V.copy()
    if full:
        i = jnp.arange(p)

        # scalar–scalar
        H = H.at[1, 1].add(-Delta_eta_b_eta_b)

        # gamma_b – gamma_b  (FULL MATRIX)
        H = H.at[2+p:2+2*p, 2+p:2+2*p].add(-Delta_gamma_b_gamma_b)

        # sigma – sigma  (DIAGONAL VECTOR)
        H = H.at[2+3*p + i, 2+3*p + i].add(-Delta_sigma_sigma)

        # gamma_b – eta_b  (vector–scalar)
        H = H.at[2+p:2+2*p, 1].add(-Delta_gamma_b_eta_b)
        H = H.at[1, 2+p:2+2*p].add(-Delta_gamma_b_eta_b)

        # sigma – eta_b  (vector–scalar)
        H = H.at[2+3*p:2+4*p, 1].add(-Delta_sigma_eta_b)
        H = H.at[1, 2+3*p:2+4*p].add(-Delta_sigma_eta_b)

        # sigma – gamma_b  (vector–vector, DIAGONAL ONLY)
        H = H.at[2+3*p + i, 2+p + i].add(-Delta_sigma_gamma_b)
        H = H.at[2+p + i, 2+3*p + i].add(-Delta_sigma_gamma_b)
    else:
        H["eta_b_eta_b"] = H["eta_b_eta_b"] - Delta_eta_b_eta_b
        H["gamma_b_gamma_b"] = H["gamma_b_gamma_b"] - Delta_gamma_b_gamma_b
        H["sigma_sigma"] = H["sigma_sigma"] - Delta_sigma_sigma
        H["gamma_b_eta_b"] = H["gamma_b_eta_b"] - Delta_gamma_b_eta_b
        H["sigma_eta_b"] = H["sigma_eta_b"] - Delta_sigma_eta_b
        H["sigma_gamma_b"] = H["sigma_gamma_b"] - Delta_sigma_gamma_b

    return H



# ===============================================================
# === Sandwich variance =========================================
# ===============================================================
def apply_block_operator(x, M, p):
    """
    M: dict like V or H (same structure), storing scalars and diag vectors.
    x: (d,) where d=2+4p
    returns: (d,) = (full_matrix(M) @ x)  but never materializes the matrix
    """
    xa, xb = x[0], x[1]
    x_ga = x[2:2+p]
    x_gb = x[2+p:2+2*p]
    x_b  = x[2+2*p:2+3*p]
    x_s  = x[2+3*p:2+4*p]

    out = jnp.zeros_like(x)

    # --- eta_a row ---
    out0 = M["eta_a_eta_a"] * xa + M["eta_b_eta_a"] * xb
    out0 = out0 + jnp.dot(M["gamma_a_eta_a"], x_ga)
    out0 = out0 + jnp.dot(M["gamma_b_eta_a"], x_gb)
    out0 = out0 + jnp.dot(M["beta_eta_a"],    x_b)
    out0 = out0 + jnp.dot(M["sigma_eta_a"],   x_s)
    out = out.at[0].set(out0)

    # --- eta_b row ---
    out1 = M["eta_b_eta_b"] * xb + M["eta_b_eta_a"] * xa
    out1 = out1 + jnp.dot(M["gamma_a_eta_b"], x_ga)
    out1 = out1 + jnp.dot(M["gamma_b_eta_b"], x_gb)
    out1 = out1 + jnp.dot(M["beta_eta_b"],    x_b)
    out1 = out1 + jnp.dot(M["sigma_eta_b"],   x_s)
    out = out.at[1].set(out1)

    # --- gamma_a block (diag-only vector–vector) ---
    y_ga = M["gamma_a_eta_a"] * xa + M["gamma_a_eta_b"] * xb
    y_ga = y_ga + M["gamma_a_gamma_a"] * x_ga
    y_ga = y_ga + M["gamma_b_gamma_a"] * x_gb
    y_ga = y_ga + M["beta_gamma_a"]    * x_b
    y_ga = y_ga + M["sigma_gamma_a"]   * x_s
    out = out.at[2:2+p].set(y_ga)

    # --- gamma_b block ---
    y_gb = M["gamma_b_eta_a"] * xa + M["gamma_b_eta_b"] * xb
    y_gb = y_gb + M["gamma_b_gamma_a"] * x_ga
    y_gb = y_gb + M["gamma_b_gamma_b"] @ x_gb      # gamma_b_gamma_b is a matrix, not a scalar
    y_gb = y_gb + M["beta_gamma_b"]    * x_b
    y_gb = y_gb + M["sigma_gamma_b"]   * x_s
    out = out.at[2+p:2+2*p].set(y_gb)

    # --- beta block ---
    y_b = M["beta_eta_a"] * xa + M["beta_eta_b"] * xb
    y_b = y_b + M["beta_gamma_a"] * x_ga
    y_b = y_b + M["beta_gamma_b"] * x_gb
    y_b = y_b + M["beta_beta"]    * x_b
    y_b = y_b + M["sigma_beta"]   * x_s
    out = out.at[2+2*p:2+3*p].set(y_b)

    # --- sigma block ---
    y_s = M["sigma_eta_a"] * xa + M["sigma_eta_b"] * xb
    y_s = y_s + M["sigma_gamma_a"] * x_ga
    y_s = y_s + M["sigma_gamma_b"] * x_gb
    y_s = y_s + M["sigma_beta"]    * x_b
    y_s = y_s + M["sigma_sigma"]   * x_s
    out = out.at[2+3*p:2+4*p].set(y_s)

    return out

def estimate_diag_sandwich_rademacher(
    H, V,
    apply_H, apply_V,
    p,
    num_mc=50,
    cg_tol=1e-6,
    cg_maxiter=200,
    key=jax.random.PRNGKey(0),
):
    """
    Estimate diag(H^{-1} V H^{-1}) using
    Rademacher (Hutchinson) + CG + block-Jacobi preconditioner.

    Parameters
    ----------
    H, V : dict
        Block-structured operators (your current design).
    apply_H, apply_V : callable
        Matrix-free operators: x -> Hx, x -> Vx.
    p : int
        Dimension of each vector block.
    """

    d = 2 + 4 * p

    # ============================================================
    # 1. Block-Jacobi preconditioner Minv
    # ============================================================

    # scalar blocks
    inv_eta = jnp.array([
        1.0 / H["eta_a_eta_a"],
        1.0 / H["eta_b_eta_b"],
    ])

    # diagonal vector blocks
    inv_gamma_a = 1.0 / H["gamma_a_gamma_a"]
    inv_beta    = 1.0 / H["beta_beta"]
    inv_sigma   = 1.0 / H["sigma_sigma"]

    # full gamma_b block
    Lb = jnp.linalg.cholesky(H["gamma_b_gamma_b"])

    def solve_gamma_b(v):
        # solves H_gb x = v
        return cho_solve((Lb, True), v)

    def Minv(x):
        xa, xb = x[0], x[1]
        x_ga = x[2:2+p]
        x_gb = x[2+p:2+2*p]
        x_b  = x[2+2*p:2+3*p]
        x_s  = x[2+3*p:2+4*p]

        ya = inv_eta[0] * xa
        yb = inv_eta[1] * xb
        y_ga = inv_gamma_a * x_ga
        y_gb = solve_gamma_b(x_gb)
        y_b  = inv_beta * x_b
        y_s  = inv_sigma * x_s

        return jnp.concatenate([
            jnp.array([ya, yb]),
            y_ga,
            y_gb,
            y_b,
            y_s
        ])

    # ============================================================
    # 2. Linear solve helper: H x = b
    # ============================================================

    def solve_H(b):
        x, _ = cg(
            apply_H,
            b,
            M=Minv,
            tol=cg_tol,
            maxiter=cg_maxiter,
        )
        return x

    # ============================================================
    # 3. Rademacher (Hutchinson) estimator
    # ============================================================

    diag_est = jnp.zeros((d,))

    for _ in range(num_mc):
        key, sub = jax.random.split(key)
        z = jax.random.rademacher(sub, (d,), dtype=jnp.float32)

        u = solve_H(z)        # H^{-1} z
        w = apply_V(u)        # V H^{-1} z
        v = solve_H(w)        # H^{-1} V H^{-1} z

        diag_est = diag_est + z * v

    return diag_est / num_mc

def estimate_Avar_explicit(
    H, V,
    apply_H, apply_V,
    p,
):
    """
    Explicitly compute A = H^{-1} V H^{-1}.
    WARNING: O(d^3), only feasible for small p.
    """
    d = 2 + 4 * p
    A = jnp.zeros((d, d))

    # Solve H X = I column by column
    I = jnp.eye(d)

    def solve_H(b):
        x, _ = jax.scipy.sparse.linalg.cg(apply_H, b)
        return x

    # First compute H^{-1}
    Hinv = jnp.column_stack([solve_H(I[:, j]) for j in range(d)])

    # Then A = H^{-1} V H^{-1}
    A = Hinv @ jnp.column_stack([apply_V(Hinv[:, j]) for j in range(d)])

    return A

def compute_Avar(H, V):

    HinvV = jnp.linalg.solve(H, V)
    Avar = jnp.linalg.solve(H, HinvV.T).T

    return Avar

def compute_sandwich_variance(Y, Z, theta_hat, params_scale, kappa4=3.0, alpha=0.05, verbose=False):

    T, p = Y.shape

    eta_a, eta_b = theta_hat[0], theta_hat[1]
    gamma_a = theta_hat[2:2+p]
    gamma_b = theta_hat[2+p:2+2*p]
    beta = theta_hat[2+2*p:2+3*p]
    sigma2 = theta_hat[2+3*p:2+4*p]
    # Sigma_e = jnp.diag(sigma2)

    # use consistent attention derivatives
    Wa, Wa_dot, _ = compute_Wa_Wb_dots(Z, eta_a, params_scale["eta_a"])
    Wb, Wb_dot, _ = compute_Wa_Wb_dots(Z, eta_b, params_scale["eta_b"])

    S = jnp.eye(p) - ldiag_mul(gamma_b, Wb)
    G = jnp.diag(beta) + ldiag_mul(gamma_a, Wa)

    # Gamma0 = Gamma0_stationary(S, G, sigma2) # explicit form, slow
    Gamma0 = Gamma0_stationary_with_lyapunov_doubling(S, G, sigma2)

    # full = False
    # V = assemble_V(Wa, Wb, Wa_dot, Wb_dot,
    #                     gamma_a, gamma_b,
    #                     G, Gamma0,
    #                     sigma2, T, kappa4, full)

    # H = assemble_H(Wa, Wb, Wa_dot, Wb_dot,
    #                     gamma_a, gamma_b,
    #                     G, Gamma0,
    #                     sigma2, T, kappa4, full)

    # diag_Avar = estimate_diag_sandwich_rademacher(
    #     H, V,
    #     apply_H=lambda x: apply_block_operator(x, H, p),
    #     apply_V=lambda x: apply_block_operator(x, V, p),
    #     p=p,
    #     num_mc=50)

    # Avar1 = estimate_Avar_explicit(
    #     H, V,
    #     apply_H=lambda x: apply_block_operator(x, H, p),
    #     apply_V=lambda x: apply_block_operator(x, V, p),
    #     p=p
    # )

    full = True
    V = assemble_V(Wa, Wb, Wa_dot, Wb_dot,
                        gamma_a, gamma_b,
                        G, Gamma0,
                        sigma2, T, kappa4, full)

    H = assemble_H(Wa, Wb, Wa_dot, Wb_dot,
                        gamma_a, gamma_b,
                        G, Gamma0,
                        sigma2, T, kappa4, full)
    Avar = compute_Avar(H, V)

    se = jnp.sqrt(jnp.maximum(jnp.diag(Avar), 0.0))
    z_crit = norm.ppf(1 - alpha / 2.0)
    lower = theta_hat - z_crit * se
    upper = theta_hat + z_crit * se

    return dict(theta=theta_hat, se=se, lower=lower, upper=upper)

# ------------------------------------------------------------
# Example
# ------------------------------------------------------------


if __name__ == "__main__":
    import random
    from dgp import simulate_VAR_attention
    from dgp import flatten_params
    from yule_walker_ad import yule_walker

    # jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(threshold=10_000, linewidth=200, suppress=True, precision=6)

    # --- Simulation settings ---
    key = jax.random.PRNGKey(random.randint(0, 10000))
    T, p, q = 100, 10, 5

    # --- True parameters (in natural scale) ---
    params_true = dict(
        eta_a=0.2,
        eta_b=0.2,
        gamma_a=jnp.linspace(-0.15, 0.15, p),
        gamma_b=jnp.linspace(-0.20, 0.20, p),
        beta=jnp.linspace(-0.20, 0.20, p),
        sigma2=jnp.exp(jax.random.normal(key, (p,)))  # positive σ²
    )

    params_scale = dict(
        eta_a=1,
        eta_b=1)

    algorithm = "auto_gradhess"
    # algorithm = "analytical_gradhess"
    # update_idx = [1]
    update_idx = range(0, 3*p+2)
    # update_idx = slice(None)
    # --- Simulate data ---
    Y, Z, W_a, W_b = simulate_VAR_attention(key, params_true, T=T, p=p, q=q, dist="normal")
    print("Simulated Y shape:", Y.shape)

    # --- Convert σ² → σ̃² (softplus inverse) ---
    sigma2_tilde_true = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = params_true.copy()
    params_reparam["sigma2"] = sigma2_tilde_true

    # --- Flatten parameters into reparameterized vector θ̃ ---
    order = ["eta_a", "eta_b", "gamma_a", "gamma_b", "beta", "sigma2"]
    theta_true_reparam, _ = flatten_params(params_reparam, order)
    print("θ length:", theta_true_reparam.size)

    # Only initialize update_idx related initial values
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_true_reparam.size)

    update_idx = jnp.array(update_idx)
    theta_init = theta_true_reparam
    theta_init_update = theta_true_reparam[update_idx]
    theta_init = theta_init.at[update_idx].set(theta_init_update * (1 + 1 * jax.random.normal(key, theta_init_update.shape)))

    # Estimation
    # breakpoint()
    theta_hat_reparam_short, grad_ll, hess_ll, convergence = yule_walker(Y, Z, theta_init, params_scale, update_idx=update_idx)

    theta_hat_reparam = theta_true_reparam
    theta_hat_reparam = theta_hat_reparam.at[update_idx].set(theta_hat_reparam_short)

    # Convert back to natural scale (with softplus)
    sigma2_tilde_hat = theta_hat_reparam[2+3*p:2+4*p]
    sigma2_hat = jax.nn.softplus(sigma2_tilde_hat)

    theta_hat = jnp.concatenate([theta_hat_reparam[:2+3*p], sigma2_hat])

    # Test sandwich variance
    var_theta = compute_sandwich_variance(Y, Z, theta_hat, params_scale)

    print(var_theta["se"])
