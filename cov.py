import jax.numpy as jnp
from jax.numpy.linalg import inv
from jax.scipy.stats import norm

from loglik_attn import build_attention


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

    Analytical solution:
        vec(Γ₀) = [I - (S⁻¹G ⊗ S⁻¹G)]⁻¹ vec(S⁻¹ Σₑ (S⁻¹)ᵀ)

    Returns
    -------
    Gamma0 : (p, p) array
        Stationary covariance of Yₜ.
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
                    Sigma_e, S, G, Gamma0,
                    sigma2, T, kappa4=3.0):
    """
    Assemble analytical covariance matrix V(θ) = E[s_t s_tᵀ]
    for the VAR–Attention model.

    Parameter ordering:
        θ = (η_a, η_b, γ_a(p), γ_b(p), β(p), σ²(p))

    Notes
    -----
    • Symmetry enforced (V = ½(V + Vᵀ)).
    • All blocks scaled by T; normalization can be applied later.
    • Assumes S is nonsingular and Γ₀ ≻ 0.
    """
    p = Wa.shape[0]
    d = 4 * p + 2
    V = jnp.zeros((d, d))

    Sinv = inv(S)
    Sigma_inv = inv(Sigma_e)

    # --- Diagonal blocks ---
    V_eta_a_eta_a = T * jnp.trace(
        (jnp.diag(gamma_a) @ Wa_dot).T @ Sigma_inv @
        (jnp.diag(gamma_a) @ Wa_dot) @ Gamma0
    )
    V_eta_b_eta_b = T * jnp.trace(
        (jnp.diag(gamma_b) @ Wb_dot).T @ Sigma_inv @
        (jnp.diag(gamma_b) @ Wb_dot) @ Gamma0
    )

    V_gamma_a_gamma_a = T * (Sigma_inv * (Wa @ Gamma0 @ Wa.T))
    V_gamma_b_gamma_b = T * (Sigma_inv * ((Wb @ Sinv @ G) @ Gamma0 @ (Wb @ Sinv @ G).T))
    V_beta_beta = T * (Sigma_inv * Gamma0)
    V_sigma_sigma = T * (kappa4 - 1.0) / 4.0 * jnp.diag(1.0 / sigma2**2)

    # --- Cross-covariance examples (subset, symmetric fill later) ---
    V_eta_b_eta_a = T * jnp.trace(
        Wa_dot.T @ jnp.diag(gamma_a) @ Sigma_inv @
        jnp.diag(gamma_b) @ Wb_dot @ Sinv @ G @ Gamma0
    )
    V_gamma_a_eta_a = T * (gamma_a * (1.0 / sigma2) * jnp.diag(Wa @ Gamma0 @ Wa_dot.T))
    V_gamma_b_eta_a = T * (gamma_a * (1.0 / sigma2) *
                           jnp.diag((Wb @ Sinv @ G) @ Gamma0 @ Wa_dot.T))
    V_beta_eta_a = T * (gamma_a * (1.0 / sigma2) * jnp.diag(Gamma0 @ Wa_dot.T))
    V_sigma_eta_a = jnp.zeros((p,))

    V_gamma_a_eta_b = T * (gamma_b * (1.0 / sigma2) *
                           jnp.diag(Wa_dot @ Gamma0 @ (Wb_dot @ Sinv @ G).T))
    diag1 = jnp.diag(Wb @ Gamma0 @ Wb_dot)
    diag2 = jnp.diag(Wb @ Sinv @ jnp.diag(gamma_b) @ Wb_dot @ Sinv)
    diag3 = jnp.diag(Wb @ Sinv) * jnp.diag(Wb_dot @ Sinv)
    V_gamma_b_eta_b = T * (gamma_b * (1.0 / sigma2) * diag1 + diag2 +
                           (kappa4 - 3.0) * gamma_b * diag3)
    V_beta_eta_b = T * (gamma_b * (1.0 / sigma2) *
                        jnp.diag(Gamma0 @ (Wb_dot @ Sinv @ G).T))
    V_sigma_eta_b = T * (kappa4 - 1.0) / 2.0 * ((1.0 / sigma2) * jnp.diag(Wb @ Sinv))

    V_gamma_b_gamma_a = T * (Sigma_inv * ((Wb @ Sinv @ G) @ Gamma0 @ Wa.T))
    V_beta_gamma_a = T * (Sigma_inv * (Gamma0 @ Wa.T))
    V_sigma_gamma_a = jnp.zeros((p,))
    V_beta_gamma_b = T * (Sigma_inv * (Gamma0 @ (Wb @ Sinv @ G).T))
    V_sigma_gamma_b = T * (kappa4 - 1.0) / 2.0 * (Sigma_inv * (Wb @ Sinv))
    V_sigma_beta = jnp.zeros((p,))

    # --- Assemble selected blocks ---

    V = V.at[0, 0].set(V_eta_a_eta_a)
    V = V.at[1, 1].set(V_eta_b_eta_b)
    V = V.at[2:2+p, 2:2+p].set(V_gamma_a_gamma_a)
    V = V.at[2+p:2+2*p, 2+p:2+2*p].set(V_gamma_b_gamma_b)
    V = V.at[2+2*p:2+3*p, 2+2*p:2+3*p].set(V_beta_beta)
    V = V.at[2+3*p:2+4*p, 2+3*p:2+4*p].set(V_sigma_sigma)

    # Symmetrically fill all cross-blocks
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

    V = V.at[2+p:2+2*p, 2:2+p].set(V_gamma_b_gamma_a)
    V = V.at[2:2+p, 2+p:2+2*p].set(V_gamma_b_gamma_a.T)

    V = V.at[2+2*p:2+3*p, 2:2+p].set(V_beta_gamma_a)
    V = V.at[2:2+p, 2+2*p:2+3*p].set(V_beta_gamma_a.T)

    V = V.at[2+3*p:2+4*p, 2:2+p].set(V_sigma_gamma_a)
    V = V.at[2:2+p, 2+3*p:2+4*p].set(V_sigma_gamma_a)

    V = V.at[2+2*p:2+3*p, 2+p:2+2*p].set(V_beta_gamma_b)
    V = V.at[2+p:2+2*p, 2+2*p:2+3*p].set(V_beta_gamma_b.T)

    V = V.at[2+3*p:2+4*p, 2+p:2+2*p].set(V_sigma_gamma_b)
    V = V.at[2+p:2+2*p, 2+3*p:2+4*p].set(V_sigma_gamma_b.T)

    V = V.at[2+3*p:2+4*p, 2+2*p:2+3*p].set(V_sigma_beta)
    V = V.at[2+2*p:2+3*p, 2+3*p:2+4*p].set(V_sigma_beta)


    V = 0.5 * (V + V.T)
    return V


# ===============================================================
# === Assemble analytical H =====================================
# ===============================================================
def assemble_H_full(Wa, Wb, Wa_dot, Wb_dot,
                    gamma_a, gamma_b,
                    Sigma_e, S, G, Gamma0,
                    sigma2, T, kappa4=3.0):
    """
    Assemble expected Hessian H(θ) = V(θ) − Δ(θ).

    Δ corrections adjust curvature for non-Gaussian fourth moments (κ₄ ≠ 3)
    to maintain information matrix consistency.

    Returns
    -------
    H : (4p+2, 4p+2) array
        Symmetric expected Hessian.
    """
    V = assemble_V_full(Wa, Wb, Wa_dot, Wb_dot,
                        gamma_a, gamma_b,
                        Sigma_e, S, G, Gamma0,
                        sigma2, T, kappa4)

    p = Wa.shape[0]
    Sinv = inv(S)
    Sigma_inv = inv(Sigma_e)
    sigma_e_inv2 = 1.0 / sigma2
    sigma_e_inv4 = 1.0 / (sigma2 ** 2)

    Delta_eta_b_eta_b = (kappa4 - 3) * jnp.trace(
        (jnp.diag(gamma_b) @ Wb_dot @ Sinv) *
        (jnp.diag(gamma_b) @ Wb_dot @ Sinv)
    )
    Delta_gamma_b_gamma_b = (kappa4 - 3) * jnp.diag(jnp.diag(Wb @ Sinv))
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

    return 0.5 * (H + H.T)


# ===============================================================
# === Compute Avar ==============================================
# ===============================================================
def compute_Avar(H, V, eps=1e-8, ensure_pd=True, verbose=False):
    """
    Compute asymptotic covariance matrix Avar = H⁻¹ V H⁻¹
    with conditional numerical stabilization and PD enforcement.

    Behavior
    --------
    • If H is positive definite, invert it directly.
    • If H has non-positive eigenvalues or near-singularity, regularize as H + eps·I.
    • Optionally project Avar to the nearest positive definite matrix if ensure_pd=True.

    Parameters
    ----------
    H : array_like
        Analytical expected Hessian.
    V : array_like
        Outer-product-of-gradients covariance matrix.
    eps : float, default=1e-8
        Regularization constant (only used if H is not PD).
    ensure_pd : bool, default=True
        If True, enforces PD-ness of Avar by eigenvalue clipping.
    verbose : bool, default=True
        If True, prints diagnostic information.

    Returns
    -------
    Avar : jnp.ndarray
        Symmetric, approximately positive definite asymptotic covariance matrix.
    """
    # Step 1: Symmetrize H and V
    H = (H + H.T) / 2
    V = (V + V.T) / 2

    # Step 2: Check if H is positive definite
    eigvals_H = jnp.linalg.eigvalsh(H)
    min_eig_H = float(eigvals_H.min())
    if verbose:
        print(f"H eigenvalue range: [{eigvals_H.min():.3e}, {eigvals_H.max():.3e}]")

    if min_eig_H <= 0:
        if verbose:
            print(f"H not PD (min eig={min_eig_H:.2e}); regularizing with eps={eps:.1e}.")
        H_reg = H + eps * jnp.eye(H.shape[0])
    else:
        H_reg = H

    # Step 3: Compute inverse and sandwich form
    H_inv = jnp.linalg.inv(H_reg)
    Avar = H_inv @ V @ H_inv
    Avar = (Avar + Avar.T) / 2

    # Step 4: Check Avar PD status
    eigvals_A = jnp.linalg.eigvalsh(Avar)
    min_eig_A = float(eigvals_A.min())
    if verbose:
        print(f"Avar eigenvalue range: [{eigvals_A.min():.3e}, {eigvals_A.max():.3e}]")

    # Step 5: Optionally enforce PD
    if ensure_pd and min_eig_A <= 0:
        if verbose:
            print(f"Avar not PD (min eig={min_eig_A:.2e}); projecting to nearest PD.")
        eigvals_clipped = jnp.clip(eigvals_A, eps, None)
        eigvecs = jnp.linalg.eigh(Avar)[1]
        Avar = eigvecs @ jnp.diag(eigvals_clipped) @ eigvecs.T

    return (Avar + Avar.T) / 2



# ===============================================================
# === Sandwich variance with diagnostics ========================
# ===============================================================
def compute_sandwich_variance(Y, Z, theta_hat, kappa4=3.0, alpha=0.05, verbose=False):
    """
    Compute the robust (sandwich) covariance estimator:
        Var(θ̂) = H⁻¹ V H⁻¹

    Includes diagnostics for stationarity and numerical stability.

    Diagnostics printed:
        - Condition number of S
        - Spectral radius of A = S⁻¹G
        - Eigenvalue ranges of H, V, Avar
    """
    T, p = Y.shape

    eta_a = theta_hat[0]
    eta_b = theta_hat[1]
    gamma_a = theta_hat[2:2+p]
    gamma_b = theta_hat[2+p:2+2*p]
    beta = theta_hat[2+2*p:2+3*p]
    sigma2 = theta_hat[2+3*p:2+4*p]

    Sigma_e = jnp.diag(sigma2)

    # Build attention matrices and derivatives
    Wa = build_attention(Z, eta=eta_a)
    Wb = build_attention(Z, eta=eta_b)
    Wa_dot = Wa * (Z @ Z.T / jnp.sqrt(Z.shape[1]))
    Wb_dot = Wb * (Z @ Z.T / jnp.sqrt(Z.shape[1]))  # corrected derivative

    S = jnp.eye(p) - jnp.diag(gamma_b) @ Wb
    G = jnp.diag(beta) + jnp.diag(gamma_a) @ Wa

    # Diagnostics: stationarity & conditioning
    cond_S = jnp.linalg.cond(S)
    A = inv(S) @ G
    rho = jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    print(f"Condition number of S: {cond_S:.3e}, spectral radius of A: {rho:.3f}")

    Gamma0 = Gamma0_stationary(S, G, Sigma_e)

    V = assemble_V_full(Wa, Wb, Wa_dot, Wb_dot,
                        gamma_a, gamma_b,
                        Sigma_e, S, G, Gamma0,
                        sigma2, T, kappa4)
    H = assemble_H_full(Wa, Wb, Wa_dot, Wb_dot,
                        gamma_a, gamma_b,
                        Sigma_e, S, G, Gamma0,
                        sigma2, T, kappa4)

    # Eigen diagnostics for H and V
    for name, M in [("H", H), ("V", V)]:
        eigvals = jnp.linalg.eigvalsh((M + M.T)/2)
        print(f"{name}: min eig={eigvals.min():.3e}, max eig={eigvals.max():.3e}, cond≈{eigvals.max()/jnp.clip(eigvals.min(),1e-12,None):.2e}")

    Avar_theta = compute_Avar(H, V, ensure_pd=True, verbose=verbose)
    se = jnp.sqrt(jnp.diag(Avar_theta))

    z_crit = norm.ppf(1 - alpha / 2.0)
    lower = theta_hat - z_crit * se
    upper = theta_hat + z_crit * se

    return dict(theta=theta_hat, Avar=Avar_theta, se=se, lower=lower, upper=upper)
