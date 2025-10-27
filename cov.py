import jax.numpy as jnp
from jax.numpy.linalg import inv
from jax.scipy.stats import norm

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
def Gamma0_stationary(S, G, Sigma_e):
    """
    Solve for the stationary covariance matrix Γ₀ of Yₜ in:
        Yₜ = S⁻¹ G Yₜ₋₁ + S⁻¹ εₜ ,   εₜ ∼ (0, Σₑ).
    """
    p = S.shape[0]
    Sinv = inv(S)
    A = Sinv @ G
    kron_term = jnp.kron(A, A)
    I = jnp.eye(p * p)
    rhs = vec(Sinv @ Sigma_e @ Sinv.T)
    vecGamma = jnp.linalg.solve(I - kron_term, rhs)
    return unvec(vecGamma, (p, p))


# ===============================================================
# === Assemble analytical V =====================================
# ===============================================================
def assemble_V_full(Wa, Wb, Wa_dot, Wb_dot,
                    gamma_a, gamma_b,
                    Sigma_e, G, Gamma0,
                    sigma2, T, kappa4=3.0):
    """
    Assemble analytical covariance matrix V(θ) = E[s_t s_tᵀ].
    """
    p = Wa.shape[0]
    d = 4 * p + 2
    V = jnp.zeros((d, d))

    S = jnp.eye(Wb.shape[0]) - jnp.diag(gamma_b) @ Wb
    Sinv = jnp.linalg.inv(S)

    Sigma_inv = jnp.diag(1.0 / sigma2)  # ✅ explicit diag for clarity

    # --- Diagonal blocks ---
    V_eta_a_eta_a = T * jnp.trace(
        (jnp.diag(gamma_a) @ Wa_dot).T @ Sigma_inv @
        (jnp.diag(gamma_a) @ Wa_dot) @ Gamma0
    )

    V_eta_b_eta_b = T * jnp.trace((jnp.diag(gamma_b) @ Wb_dot).T @ Sigma_inv @ (jnp.diag(gamma_b) @ Wb_dot) @ Gamma0 +
                                  (jnp.diag(gamma_b) @ Wb_dot @ Sinv) @ (jnp.diag(gamma_b) @ Wb_dot @ Sinv) +
                                  (kappa4 - 3.0) * (jnp.diag(gamma_b) @ Wb_dot @ Sinv) * (jnp.diag(gamma_b) @ Wb_dot @ Sinv))


    V_gamma_a_gamma_a = T * (Sigma_inv * (Wa @ Gamma0 @ Wa.T))

    V_gamma_b_gamma_b = T * (
        Sigma_inv * (Wb @ Gamma0 @ Wb.T) +
        (Wb @ Sinv) * (Wb @ Sinv).T +
        (kappa4 - 3.0) * jnp.eye(p) * (Wb @ Sinv) * (Wb @ Sinv))

    V_beta_beta = T * (Sigma_inv * Gamma0)
    V_sigma_sigma = T * (kappa4 - 1.0) / 4.0 * jnp.diag(1.0 / sigma2**2)

    # --- Cross-covariance examples ---
    V_eta_b_eta_a = T * jnp.trace(
        Wa_dot.T @ jnp.diag(gamma_a) @ Sigma_inv @
        jnp.diag(gamma_b) @ Wb_dot @ Sinv @ G @ Gamma0
    )

    V_gamma_a_eta_a = T * (gamma_a * (1.0 / sigma2) * jnp.diag(Wa @ Gamma0 @ Wa_dot.T))
    V_gamma_b_eta_a = T * (gamma_a * (1.0 / sigma2) *
                           jnp.diag((Wb @ Sinv @ G) @ Gamma0 @ Wa_dot.T))
    V_beta_eta_a = T * (gamma_a * (1.0 / sigma2) * jnp.diag(Gamma0 @ Wa_dot.T))

    V_sigma_eta_a = jnp.zeros((p, ))

    V_gamma_a_eta_b = T * (gamma_b * (1.0 / sigma2) *
                           jnp.diag(Wa_dot @ Gamma0 @ (Wb_dot @ Sinv @ G).T))

    V_gamma_b_eta_b = T * (
        gamma_b * (1.0 / sigma2) * jnp.diag(Wb @ Gamma0 @ Wb_dot) +
        jnp.diag(Wb @ Sinv @ jnp.diag(gamma_b) @ Wb_dot @ Sinv) +
        (kappa4 - 3.0) * gamma_b * jnp.diag(Wb @ Sinv) * jnp.diag(Wb_dot @ Sinv))


    V_beta_eta_b = T * (gamma_b * (1.0 / sigma2) *
                        jnp.diag(Gamma0 @ (Wb_dot @ Sinv @ G).T))

    V_sigma_eta_b = T * (kappa4 - 1.0) / 2.0 * (
        (1.0 / sigma2) * jnp.diag(Wb_dot @ Sinv))

    V_gamma_b_gamma_a = T * (Sigma_inv * ((Wb @ Sinv @ G) @ Gamma0 @ Wa.T))
    V_beta_gamma_a = T * (Sigma_inv * (Gamma0 @ Wa.T))

    V_sigma_gamma_a = jnp.zeros((p, p))
    V_beta_gamma_b = T * (Sigma_inv * (Gamma0 @ (Wb @ Sinv @ G).T))
    V_sigma_gamma_b = T * (kappa4 - 1.0) / 2.0 * (Sigma_inv * (Wb @ Sinv))

    V_sigma_beta    = jnp.zeros((p, p))

    # --- Assemble selected blocks (unchanged) ---
    V = V.at[0, 0].set(V_eta_a_eta_a)
    V = V.at[1, 1].set(V_eta_b_eta_b)
    V = V.at[2:2+p, 2:2+p].set(V_gamma_a_gamma_a)
    V = V.at[2+p:2+2*p, 2+p:2+2*p].set(V_gamma_b_gamma_b)
    V = V.at[2+2*p:2+3*p, 2+2*p:2+3*p].set(V_beta_beta)
    V = V.at[2+3*p:2+4*p, 2+3*p:2+4*p].set(V_sigma_sigma)

    # --- Cross-block symmetric filling (unchanged) ---
    V = V.at[1, 0].set(V_eta_b_eta_a)
    V = V.at[0, 1].set(V_eta_b_eta_a)

    V = V.at[2:2+p, 0].set(V_gamma_a_eta_a)
    V = V.at[0, 2:2+p].set(V_gamma_a_eta_a.T)

    V = V.at[2+p:2+2*p, 0].set(V_gamma_b_eta_a)
    V = V.at[0, 2+p:2+2*p].set(V_gamma_b_eta_a.T)

    V = V.at[2+2*p:2+3*p, 0].set(V_beta_eta_a)
    V = V.at[0, 2+2*p:2+3*p].set(V_beta_eta_a.T)

    V = V.at[2+3*p:2+4*p, 0].set(V_sigma_eta_a)
    V = V.at[0, 2+3*p:2+4*p].set(V_sigma_eta_a.T)

    V = V.at[2:2+p, 1].set(V_gamma_a_eta_b)
    V = V.at[1, 2:2+p].set(V_gamma_a_eta_b.T)

    V = V.at[2+p:2+2*p, 1].set(V_gamma_b_eta_b)
    V = V.at[1, 2+p:2+2*p].set(V_gamma_b_eta_b.T)

    V = V.at[2+2*p:2+3*p, 1].set(V_beta_eta_b)
    V = V.at[1, 2+2*p:2+3*p].set(V_beta_eta_b.T)

    V = V.at[2+3*p:2+4*p, 1].set(V_sigma_eta_b)
    V = V.at[1, 2+3*p:2+4*p].set(V_sigma_eta_b.T)

    V = V.at[2+p:2+2*p, 2:2+p].set(V_gamma_b_gamma_a)
    V = V.at[2:2+p, 2+p:2+2*p].set(V_gamma_b_gamma_a.T)

    V = V.at[2+2*p:2+3*p, 2:2+p].set(V_beta_gamma_a)
    V = V.at[2:2+p, 2+2*p:2+3*p].set(V_beta_gamma_a.T)

    V = V.at[2+3*p:2+4*p, 2:2+p].set(V_sigma_gamma_a)
    V = V.at[2:2+p, 2+3*p:2+4*p].set(V_sigma_gamma_a.T)

    V = V.at[2+2*p:2+3*p, 2+p:2+2*p].set(V_beta_gamma_b)
    V = V.at[2+p:2+2*p, 2+2*p:2+3*p].set(V_beta_gamma_b.T)

    V = V.at[2+3*p:2+4*p, 2+p:2+2*p].set(V_sigma_gamma_b)
    V = V.at[2+p:2+2*p, 2+3*p:2+4*p].set(V_sigma_gamma_b.T)

    V = V.at[2+3*p:2+4*p, 2+2*p:2+3*p].set(V_sigma_beta)
    V = V.at[2+2*p:2+3*p, 2+3*p:2+4*p].set(V_sigma_beta.T)

    # return 0.5 * (V + V.T)
    return V


# ===============================================================
# === Assemble analytical H =====================================
# ===============================================================
def assemble_H_full(Wa, Wb, Wa_dot, Wb_dot,
                    gamma_a, gamma_b,
                    Sigma_e, G, Gamma0,
                    sigma2, T, kappa4=3.0):
    V = assemble_V_full(Wa, Wb, Wa_dot, Wb_dot,
                        gamma_a, gamma_b,
                        Sigma_e, G, Gamma0,
                        sigma2, T, kappa4)
    p = Wa.shape[0]

    S = jnp.eye(Wb.shape[0]) - jnp.diag(gamma_b) @ Wb
    Sinv = jnp.linalg.inv(S)

    Sigma_inv = jnp.diag(1.0 / sigma2)  # ✅ explicit diag
    sigma_e_inv2 = 1.0 / sigma2
    sigma_e_inv4 = 1.0 / (sigma2 ** 2)

    Delta_eta_b_eta_b = (kappa4 - 3) * jnp.trace(
        (jnp.diag(gamma_b) @ Wb_dot @ Sinv) *
        (jnp.diag(gamma_b) @ Wb_dot @ Sinv)
    )

    Delta_gamma_b_gamma_b = (kappa4 - 3) * jnp.eye(p) * (Wb @ Sinv) * (Wb @ Sinv)

    Delta_sigma_sigma = (kappa4 - 3) / 4.0 * jnp.diag(sigma_e_inv4)

    Delta_gamma_b_eta_b = (kappa4 - 3) * (
        gamma_b * jnp.diag(Wb @ Sinv) * jnp.diag(Wb_dot @ Sinv)
    )

    Delta_sigma_eta_b = (kappa4 - 3) / 2.0 * (
        gamma_b * sigma_e_inv2 * jnp.diag(Wb_dot @ Sinv)
    )

    Delta_sigma_gamma_b = (kappa4 - 3) / 2.0 * (Sigma_inv * (Wb @ Sinv))

    H = V.copy()
    H = H.at[1, 1].add(-Delta_eta_b_eta_b)
    H = H.at[2+p:2+2*p, 2+p:2+2*p].add(-Delta_gamma_b_gamma_b)
    H = H.at[2+3*p:2+4*p, 2+3*p:2+4*p].add(-Delta_sigma_sigma)

    for i in range(p):
        H = H.at[2+p+i, 1].add(-Delta_gamma_b_eta_b[i])
        H = H.at[1, 2+p+i].add(-Delta_gamma_b_eta_b[i])
        H = H.at[2+3*p+i, 1].add(-Delta_sigma_eta_b[i])
        H = H.at[1, 2+3*p+i].add(-Delta_sigma_eta_b[i])

    H = H.at[2+3*p:2+4*p, 2+p:2+2*p].add(-Delta_sigma_gamma_b)
    H = H.at[2+p:2+2*p, 2+3*p:2+4*p].add(-Delta_sigma_gamma_b.T)

    # return 0.5 * (H + H.T)
    return H


# ===============================================================
# === Compute Avar ==============================================
# ===============================================================
def compute_Avar(H, V, eps=1e-8):
    H = (H + H.T) / 2
    V = (V + V.T) / 2

    eigvals_H = jnp.linalg.eigvalsh(H)
    min_eig_H = float(eigvals_H.min())

    if min_eig_H <= 0:
        H_reg = H + eps * jnp.eye(H.shape[0])
    else:
        H_reg = H

    H_inv = jnp.linalg.inv(H_reg)
    Avar = H_inv @ V @ H_inv
    Avar = (Avar + Avar.T) / 2

    # eigvals_A = jnp.linalg.eigvalsh(Avar)
    # if ensure_pd and float(eigvals_A.min()) <= 0:
    #     eigvals_clipped = jnp.clip(eigvals_A, eps, None)
    #     eigvecs = jnp.linalg.eigh(Avar)[1]
    #     Avar = eigvecs @ jnp.diag(eigvals_clipped) @ eigvecs.T

    return Avar


# ===============================================================
# === Sandwich variance =========================================
# ===============================================================
def compute_sandwich_variance(Y, Z, theta_hat, params_scale, kappa4=3.0, alpha=0.05, verbose=False):

    T, p = Y.shape

    eta_a, eta_b = theta_hat[0], theta_hat[1]
    gamma_a = theta_hat[2:2+p]
    gamma_b = theta_hat[2+p:2+2*p]
    beta = theta_hat[2+2*p:2+3*p]
    sigma2 = theta_hat[2+3*p:2+4*p]
    Sigma_e = jnp.diag(sigma2)

    # ✅ use consistent attention derivatives
    Wa, Wa_dot, _ = compute_Wa_Wb_dots(Z, eta_a, params_scale["eta_a"])
    Wb, Wb_dot, _ = compute_Wa_Wb_dots(Z, eta_b, params_scale["eta_b"])

    S = jnp.eye(p) - jnp.diag(gamma_b) @ Wb
    G = jnp.diag(beta) + jnp.diag(gamma_a) @ Wa


    Gamma0 = Gamma0_stationary(S, G, Sigma_e)
    V = assemble_V_full(Wa, Wb, Wa_dot, Wb_dot,
                        gamma_a, gamma_b,
                        Sigma_e, G, Gamma0,
                        sigma2, T, kappa4)

    H = assemble_H_full(Wa, Wb, Wa_dot, Wb_dot,
                        gamma_a, gamma_b,
                        Sigma_e, G, Gamma0,
                        sigma2, T, kappa4)


    # breakpoint()

    Avar_theta = compute_Avar(H, V)
    se = jnp.sqrt(jnp.diag(Avar_theta))   # The first value not used, only T-1 left ?
    z_crit = norm.ppf(1 - alpha / 2.0)
    lower = theta_hat - z_crit * se
    upper = theta_hat + z_crit * se

    print(params_scale)

    return dict(theta=theta_hat, Avar=Avar_theta, se=se, lower=lower, upper=upper)
