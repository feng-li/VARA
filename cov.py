import jax.numpy as jnp
from jax.numpy.linalg import inv
from jax.scipy.stats import norm


def vec(X):
    """Vectorize matrix X (column-major flatten)."""
    return X.reshape(-1, 1)


def unvec(v, shape):
    """Inverse of vec() — reshape vector to given shape."""
    return v.reshape(shape)


def Gamma0_stationary(S, G, Sigma_e):
    """
    Solve for Γ₀ = cov(Yₜ) given stationary VAR form:
        Γ₀ = (S⁻¹G) Γ₀ (S⁻¹G)ᵀ + S⁻¹ Σₑ (S⁻¹)ᵀ
    Analytical solution:
        vec(Γ₀) = [I - (S⁻¹G ⊗ S⁻¹G)]⁻¹ vec(S⁻¹ Σₑ (S⁻¹)ᵀ)
    """
    p = S.shape[0]
    Sinv = inv(S)
    A = Sinv @ G
    kron_term = jnp.kron(A, A)
    I = jnp.eye(p * p)
    rhs = vec(Sinv @ Sigma_e @ Sinv.T)
    vecGamma = jnp.linalg.solve(I - kron_term, rhs)
    Gamma0 = unvec(vecGamma, (p, p))
    return Gamma0


def assemble_V_full(Wa, Wb, Wa_dot, Wb_dot,
                         gamma_a, gamma_b,
                         Sigma_e, S, G, Gamma0,
                         sigma_e2, T, kappa4=3.0):
    """
    Assemble the complete analytical V(θ) ∈ ℝ^{(4p+2)×(4p+2)}
    for the VAR–Attention model, including all 15 nonzero cross-blocks.

    Parameter ordering:
        θ = (η_a, η_b, γ_a(p), γ_b(p), β(p), σ²_ε(p))

    Notes:
        - Uses all 15 analytical cross-covariance expressions.
        - Symmetry enforced: V = ½(V + Vᵀ)
        - Compatible with JAX autodiff/JIT.
    """
    p = Wa.shape[0]
    d = 4 * p + 2
    V = jnp.zeros((d, d))

    # ================================================================
    # === Diagonal (variance) blocks ================================
    # ================================================================
    # η_aη_a, η_bη_b are scalars
    V_eta_a_eta_a = T * jnp.trace(
        (jnp.diag(gamma_a) @ Wa_dot).T @ inv(Sigma_e) @
        (jnp.diag(gamma_a) @ Wa_dot) @ Gamma0
    )
    V_eta_b_eta_b = T * jnp.trace(
        (jnp.diag(gamma_b) @ Wb_dot).T @ inv(Sigma_e) @
        (jnp.diag(gamma_b) @ Wb_dot) @ Gamma0
    )

    # γ_aγ_a, γ_bγ_b, ββ, σ²σ² are p×p matrices
    V_gamma_a_gamma_a = T * (inv(Sigma_e) * (Wa @ Gamma0 @ Wa.T))
    V_gamma_b_gamma_b = T * (
        inv(Sigma_e) * ((Wb @ inv(S) @ G) @ Gamma0 @ (Wb @ inv(S) @ G).T)
    )
    V_beta_beta = T * (inv(Sigma_e) * Gamma0)
    V_sigma_sigma = T * (kappa4 - 1.0) / 4.0 * jnp.diag(1.0 / sigma_e2**2)

    # ================================================================
    # === Cross-covariance blocks (15 in total) ======================
    # ================================================================

    # (1) cov(η_b, η_a)
    V_eta_b_eta_a = T * jnp.trace(
        Wa_dot.T @ jnp.diag(gamma_a) @ inv(Sigma_e) @
        jnp.diag(gamma_b) @ Wb_dot @ inv(S) @ G @ Gamma0
    )

    # (2) cov(γ_a, η_a)
    V_gamma_a_eta_a = T * (
        gamma_a * (1.0 / sigma_e2) * jnp.diag(Wa @ Gamma0 @ Wa_dot.T)
    )

    # (3) cov(γ_b, η_a)
    V_gamma_b_eta_a = T * (
        gamma_a * (1.0 / sigma_e2) *
        jnp.diag((Wb @ inv(S) @ G) @ Gamma0 @ Wa_dot.T)
    )

    # (4) cov(β, η_a)
    V_beta_eta_a = T * (
        gamma_a * (1.0 / sigma_e2) * jnp.diag(Gamma0 @ Wa_dot.T)
    )

    # (5) cov(σ², η_a) = 0
    V_sigma_eta_a = jnp.zeros((p,))

    # (6) cov(γ_a, η_b)
    V_gamma_a_eta_b = T * (
        gamma_b * (1.0 / sigma_e2) *
        jnp.diag(Wa_dot @ Gamma0 @ (Wb_dot @ inv(S) @ G).T)
    )

    # (7) cov(γ_b, η_b)
    Sinv = inv(S)
    diag1 = jnp.diag(Wb @ Gamma0 @ Wb_dot)
    diag2 = jnp.diag(Wb @ Sinv @ jnp.diag(gamma_b) @ Wb_dot @ Sinv)
    diag3 = jnp.diag(Wb @ Sinv) * jnp.diag(Wb_dot @ Sinv)
    V_gamma_b_eta_b = T * (
        gamma_b * (1.0 / sigma_e2) * diag1 + diag2 +
        (kappa4 - 3.0) * gamma_b * diag3
    )

    # (8) cov(β, η_b)
    V_beta_eta_b = T * (
        gamma_b * (1.0 / sigma_e2) *
        jnp.diag(Gamma0 @ (Wb_dot @ inv(S) @ G).T)
    )

    # (9) cov(σ², η_b)
    V_sigma_eta_b = T * (kappa4 - 1.0) / 2.0 * (
        (1.0 / sigma_e2) * jnp.diag(Wb @ inv(S))
    )

    # (10) cov(γ_b, γ_a)
    V_gamma_b_gamma_a = T * (
        inv(Sigma_e) *
        ((Wb @ inv(S) @ G) @ Gamma0 @ Wa.T)
    )

    # (11) cov(β, γ_a)
    V_beta_gamma_a = T * (inv(Sigma_e) * (Gamma0 @ Wa.T))

    # (12) cov(σ², γ_a) = 0
    V_sigma_gamma_a = jnp.zeros((p,))

    # (13) cov(β, γ_b)
    V_beta_gamma_b = T * (
        inv(Sigma_e) * (Gamma0 @ (Wb @ inv(S) @ G).T)
    )

    # (14) cov(σ², γ_b)
    V_sigma_gamma_b = T * (kappa4 - 1.0) / 2.0 * (
        inv(Sigma_e) * (Wb @ inv(S))
    )

    # (15) cov(σ², β) = 0
    V_sigma_beta = jnp.zeros((p,))

    # ================================================================
    # === Block assembly =============================================
    # ================================================================
    # Block index mapping
    #   0     : η_a
    #   1     : η_b
    #   2:2+p : γ_a
    #   2+p:2+2p : γ_b
    #   2+2p:2+3p : β
    #   2+3p:2+4p : σ²
    #
    # Fill diagonals first
    V = V.at[0, 0].set(V_eta_a_eta_a)
    V = V.at[1, 1].set(V_eta_b_eta_b)
    V = V.at[2:2+p, 2:2+p].set(V_gamma_a_gamma_a)
    V = V.at[2+p:2+2*p, 2+p:2+2*p].set(V_gamma_b_gamma_b)
    V = V.at[2+2*p:2+3*p, 2+2*p:2+3*p].set(V_beta_beta)
    V = V.at[2+3*p:2+4*p, 2+3*p:2+4*p].set(V_sigma_sigma)

    # Now fill cross-blocks (and transposes for symmetry)
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

    # Symmetrize numerically
    V = 0.5 * (V + V.T)
    return V


def assemble_H_full(Wa, Wb, Wa_dot, Wb_dot,
                    gamma_a, gamma_b,
                    Sigma_e, S, G, Gamma0,
                    sigma_e2, T, kappa4=3.0):
    """
    Assemble the full analytical expected Hessian matrix H(θ) = V − Δ
    for the VAR-Attention model.

    Parameters
    ----------
    Wa, Wb : (p×p) attention matrices
    Wa_dot, Wb_dot : (p×p) attention derivatives wrt η_a, η_b
    gamma_a, gamma_b : (p,) vectors
    Sigma_e : (p×p) noise covariance (diagonal)
    S, G : (p×p) system matrices
    Gamma0 : (p×p) stationary covariance of Y_t
    sigma_e2 : (p,) variance vector
    T : int
        Sample size
    kappa4 : float
        Fourth moment coefficient (e.g., 3 for Gaussian)

    Returns
    -------
    H : (4p+2, 4p+2) array
        Expected information matrix, consistent with analytical block structure.
    """

    # ------------------------------------------------------------------
    # Step 1: Start from V (already assembled analytically)
    # ------------------------------------------------------------------
    V = assemble_V_full(Wa, Wb, Wa_dot, Wb_dot,
                             gamma_a, gamma_b,
                             Sigma_e, S, G, Gamma0,
                             sigma_e2, T, kappa4)

    p = Wa.shape[0]
    d = 4 * p + 2

    # ------------------------------------------------------------------
    # Step 2: Compute Δ-blocks (nonzero only for η_b, γ_b, σ²)
    # ------------------------------------------------------------------
    Sinv = inv(S)
    Sigma_inv = inv(Sigma_e)
    sigma_e_inv2 = 1.0 / sigma_e2
    sigma_e_inv4 = 1.0 / (sigma_e2 ** 2)

    # Δ_{ηb,ηb} scalar
    Delta_eta_b_eta_b = (kappa4 - 3) * jnp.trace(
        (jnp.diag(gamma_b) @ Wb_dot @ Sinv)
        * (jnp.diag(gamma_b) @ Wb_dot @ Sinv)
    )

    # Δ_{γb,γb} (p×p)
    Delta_gamma_b_gamma_b = (kappa4 - 3) * (jnp.eye(p) * (Wb @ Sinv))

    # Δ_{σ²,σ²} (p×p)
    Delta_sigma_sigma = (kappa4 - 3) / 4.0 * jnp.diag(sigma_e_inv4)

    # Δ_{γb,ηb} (p,)
    Delta_gamma_b_eta_b = (kappa4 - 3) * (
        gamma_b
        * jnp.diag(Wb @ Sinv)
        * jnp.diag(Wb_dot @ Sinv)
    )

    # Δ_{σ²,ηb} (p,)
    Delta_sigma_eta_b = (kappa4 - 3) / 2.0 * (
        gamma_b
        * sigma_e_inv2
        * jnp.diag(Wb_dot @ Sinv)
    )

    # Δ_{σ²,γb} (p×p)
    Delta_sigma_gamma_b = (kappa4 - 3) / 2.0 * (
        Sigma_inv * (Wb @ Sinv)
    )

    # ------------------------------------------------------------------
    # Step 3: Subtract Δ-blocks from V to get H
    # ------------------------------------------------------------------
    H = V.copy()

    # Scalar η_bη_b block
    H = H.at[1, 1].add(-Delta_eta_b_eta_b)

    # γ_bγ_b block
    H = H.at[2+p:2+2*p, 2+p:2+2*p].add(-Delta_gamma_b_gamma_b)

    # σ²σ² block
    H = H.at[2+3*p:2+4*p, 2+3*p:2+4*p].add(-Delta_sigma_sigma)

    # γ_bη_b and η_bγ_b (vector–scalar)
    for i in range(p):
        H = H.at[2+p+i, 1].add(-Delta_gamma_b_eta_b[i])
        H = H.at[1, 2+p+i].add(-Delta_gamma_b_eta_b[i])

    # σ²η_b and η_bσ²
    for i in range(p):
        H = H.at[2+3*p+i, 1].add(-Delta_sigma_eta_b[i])
        H = H.at[1, 2+3*p+i].add(-Delta_sigma_eta_b[i])

    # σ²γ_b and γ_bσ² (matrix)
    H = H.at[2+3*p:2+4*p, 2+p:2+2*p].add(-Delta_sigma_gamma_b)
    H = H.at[2+p:2+2*p, 2+3*p:2+4*p].add(-Delta_sigma_gamma_b.T)

    # Symmetrize
    H = 0.5 * (H + H.T)

    return H


def compute_sandwich_variance(Wa, Wb, Wa_dot, Wb_dot,
                              gamma_a, gamma_b,
                              Sigma_e, S, G, Gamma0,
                              sigma_e2, T, kappa4=3.0):
    """
    Compute the robust (sandwich) covariance estimator for QMLE parameters.

    Var(θ̂) = H⁻¹ V H⁻¹

    Parameters
    ----------
    Wa, Wb, Wa_dot, Wb_dot : array
        Weight matrices and their derivatives.
    gamma_a, gamma_b : array
        Parameter vectors.
    Sigma_e : array
        Covariance of innovations (p×p, usually diagonal).
    S, G : array
        System matrices.
    Gamma0 : array
        Stationary covariance of Y_t.
    sigma_e2 : array
        Variances of ε components.
    T : int
        Sample size.
    kappa4 : float
        Fourth moment (3 for Gaussian).

    Returns
    -------
    var_theta : array
        Robust covariance matrix of parameter estimates.
    """

    # Step 1: Compute analytical V and H matrices
    V = assemble_V_full(Wa, Wb, Wa_dot, Wb_dot,
                             gamma_a, gamma_b,
                             Sigma_e, S, G, Gamma0,
                             sigma_e2, T, kappa4)

    H = assemble_H_full(Wa, Wb, Wa_dot, Wb_dot,
                             gamma_a, gamma_b,
                             Sigma_e, S, G, Gamma0,
                             sigma_e2, T, kappa4)

    # Step 2: Compute inverse Hessian
    H_inv = inv(H)

    # Step 3: Sandwich variance estimator
    var_theta = H_inv @ V @ H_inv

    # Step 4: Symmetrize for numerical stability
    var_theta = 0.5 * (var_theta + var_theta.T)

    return var_theta


def compute_confidence_intervals(theta_hat, var_theta, alpha=0.05):
    """
    Compute robust (sandwich-based) confidence intervals for QMLE estimates.

    Parameters
    ----------
    theta_hat : (d,) array
        Vector of estimated parameters.
    var_theta : (d,d) array
        Robust covariance matrix (from H⁻¹ V H⁻¹).
    alpha : float, optional
        Significance level, default 0.05 (for 95% CI).

    Returns
    -------
    results : dict
        Contains 'se', 'lower', 'upper', 'z_value', 'alpha'.
    """

    # Step 1: robust standard errors
    se = jnp.sqrt(jnp.diag(var_theta))

    # Step 2: z critical value
    z_crit = norm.ppf(1 - alpha / 2.0)

    # Step 3: confidence bounds
    lower = theta_hat - z_crit * se
    upper = theta_hat + z_crit * se

    # Step 4: z-values (useful for Wald tests)
    z_values = theta_hat / se

    return {
        "alpha": alpha,
        "se": se,
        "z_value": z_values,
        "lower": lower,
        "upper": upper,
    }


def test_full_V_blocks():
    p, T = 3, 100
    Wa = jnp.eye(p)
    Wb = jnp.eye(p) * 0.9
    Wa_dot = jnp.eye(p) * 0.1
    Wb_dot = jnp.eye(p) * 0.2
    gamma_a = jnp.array([0.5, 0.7, 1.0])
    gamma_b = jnp.array([0.3, 0.8, 1.2])
    Sigma_e = jnp.diag(jnp.linspace(0.5, 1.0, p))
    S = jnp.eye(p)
    G = jnp.eye(p) * 0.8
    Gamma0 = Gamma0_stationary(S, G, Sigma_e)
    sigma_e2 = jnp.array([0.5, 0.8, 1.2])

    V_full = assemble_V_full(Wa, Wb, Wa_dot, Wb_dot,
                                  gamma_a, gamma_b,
                                  Sigma_e, S, G, Gamma0,
                                  sigma_e2, T)

    print("Full V shape:", V_full.shape)
    print("Symmetry check:", jnp.allclose(V_full, V_full.T, atol=1e-10))
    print("Mean(abs(V)):", jnp.mean(jnp.abs(V_full)))
    print("Top-left 6×6 submatrix:\n", V_full[:6, :6])


def test_H_full():
    p, T = 3, 100
    Wa = jnp.eye(p)
    Wb = jnp.eye(p) * 0.8
    Wa_dot = jnp.eye(p) * 0.05
    Wb_dot = jnp.eye(p) * 0.1
    gamma_a = jnp.array([0.4, 0.6, 0.8])
    gamma_b = jnp.array([0.5, 0.7, 0.9])
    Sigma_e = jnp.diag(jnp.linspace(0.5, 1.0, p))
    S = jnp.eye(p)
    G = jnp.eye(p) * 0.7
    Gamma0 = Gamma0_stationary(S, G, Sigma_e)
    sigma_e2 = jnp.array([0.5, 0.8, 1.0])

    H = assemble_H_full(Wa, Wb, Wa_dot, Wb_dot,
                             gamma_a, gamma_b,
                             Sigma_e, S, G, Gamma0,
                             sigma_e2, T)

    print("H shape:", H.shape)
    print("Symmetry check:", jnp.allclose(H, H.T, atol=1e-10))
    print("Mean(abs(H)):", jnp.mean(jnp.abs(H)))
    print("Top-left 6×6 submatrix:\n", H[:6, :6])


def test_sandwich_variance():
    p, T = 3, 200
    Wa = jnp.eye(p)
    Wb = jnp.eye(p) * 0.8
    Wa_dot = jnp.eye(p) * 0.05
    Wb_dot = jnp.eye(p) * 0.1
    gamma_a = jnp.array([0.4, 0.6, 0.8])
    gamma_b = jnp.array([0.5, 0.7, 0.9])
    Sigma_e = jnp.diag(jnp.linspace(0.5, 1.0, p))
    S = jnp.eye(p)
    G = jnp.eye(p) * 0.7
    Gamma0 = Gamma0_stationary(S, G, Sigma_e)
    sigma_e2 = jnp.array([0.5, 0.8, 1.0])

    var_theta = compute_sandwich_variance(Wa, Wb, Wa_dot, Wb_dot,
                                          gamma_a, gamma_b,
                                          Sigma_e, S, G, Gamma0,
                                          sigma_e2, T)

    print("Sandwich variance shape:", var_theta.shape)
    print("Symmetry check:", jnp.allclose(var_theta, var_theta.T, atol=1e-10))
    print("Trace(Var):", jnp.trace(var_theta))
    print("Leading 5×5 submatrix:\n", var_theta[:5, :5])


if __name__ == "__main__":
    test_full_V_blocks()
    test_H_full()
    test_sandwich_variance()
