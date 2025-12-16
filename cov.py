import jax
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

    Note that this function is very memory consuming.
    """
    p = S.shape[0]
    Sinv = inv(S)
    A = Sinv @ G
    kron_term = jnp.kron(A, A)
    I = jnp.eye(p * p)
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
    Sigma_e,
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

    # Inverse of S
    Sinv = jnp.linalg.inv(S)

    # Transition matrix
    A = Sinv @ G

    # Innovation covariance for Lyapunov equation
    Q = Sinv @ Sigma_e @ Sinv.T

    # Solve Γ₀ = A Γ₀ Aᵀ + Q
    Gamma0 = lyapunov_doubling(A, Q, max_iter=max_iter, tol=tol)

    return Gamma0


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

    Sigma_inv = jnp.diag(1.0 / sigma2)  # explicit diag for clarity

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

    # --- Assemble selected blocks ---
    V = V.at[0, 0].set(V_eta_a_eta_a)
    V = V.at[1, 1].set(V_eta_b_eta_b)
    V = V.at[2:2+p, 2:2+p].set(V_gamma_a_gamma_a)
    V = V.at[2+p:2+2*p, 2+p:2+2*p].set(V_gamma_b_gamma_b)
    V = V.at[2+2*p:2+3*p, 2+2*p:2+3*p].set(V_beta_beta)
    V = V.at[2+3*p:2+4*p, 2+3*p:2+4*p].set(V_sigma_sigma)

    # --- Cross-block symmetric filling ---
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

    Sigma_inv = jnp.diag(1.0 / sigma2)  # explicit diag
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

    # use consistent attention derivatives
    Wa, Wa_dot, _ = compute_Wa_Wb_dots(Z, eta_a, params_scale["eta_a"])
    Wb, Wb_dot, _ = compute_Wa_Wb_dots(Z, eta_b, params_scale["eta_b"])

    S = jnp.eye(p) - jnp.diag(gamma_b) @ Wb
    G = jnp.diag(beta) + jnp.diag(gamma_a) @ Wa

    # Gamma0 = Gamma0_stationary(S, G, Sigma_e) # explicit form, slow
    Gamma0 = Gamma0_stationary_with_lyapunov_doubling(S, G, Sigma_e)
    breakpoint()

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

    # print(se)

    return dict(theta=theta_hat, Avar=Avar_theta, se=se, lower=lower, upper=upper)

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
    T, p, q = 1000, 3, 3

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
    print("θ̃ length:", theta_true_reparam.size)

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
