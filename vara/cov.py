import jax
import jax.numpy as jnp
from jax.scipy.stats import norm
from jax.scipy.sparse.linalg import cg

from .hess import compute_Wa_Wb_dots
from .cov_auto import compute_V_auto_natural, compute_H_auto_natural

# ===============================================================
# === Utility functions =========================================
# ===============================================================
def vec(X):
    """
    Vectorize a matrix in column-major order.

    Parameters
    ----------
    X : array_like
        Matrix to reshape.

    Returns
    -------
    jax.Array
        Column-stacked vector with shape ``(m * n, 1)``.
    """
    return X.T.reshape(-1, 1)


def unvec(v, shape):
    """
    Invert :func:`vec` for a target matrix shape.

    Parameters
    ----------
    v : array_like
        Vectorized matrix produced by :func:`vec`.
    shape : tuple[int, int]
        Desired matrix shape ``(rows, cols)``.

    Returns
    -------
    jax.Array
        Matrix with the requested shape.
    """
    return v.reshape((shape[1], shape[0])).T

# ===============================================================
# === Stationary covariance =====================================
# ===============================================================

def Gamma0_stationary(G, sigma2):
    """
    Solve for the stationary covariance matrix Γ₀ of Yₜ in:
        Yₜ = G Yₜ₋₁ + εₜ ,   εₜ ∼ (0, Σₑ).

    Note that this function is very memory consuming.
    """
    Sigma_e = jnp.diag(sigma2)
    vecGamma = jnp.linalg.solve(
        jnp.eye(G.shape[0] ** 2) - jnp.kron(G, G),
        vec(Sigma_e)
    )
    return unvec(vecGamma, G.shape)



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


def Gamma0_stationary_with_lyapunov_doubling(G, sigma2, max_iter=30, tol=1e-6,):
    """
    Compute stationary covariance Γ₀ for
        Y_t = G Y_{t-1} + ε_t,
        ε_t ~ (0, diag(sigma2))

    using Lyapunov doubling.
    """
    Q = jnp.diag(sigma2)
    Gamma0 = lyapunov_doubling(G, Q, max_iter=max_iter, tol=tol)
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



# ===============================================================
# === Assemble analytical V =====================================
# ===============================================================
def assemble_V(Wa, Wa_dot, gamma_a, G, Gamma0, sigma2, T, kappa4=3.0, full=True):
    """Assemble analytical covariance matrix V(θ) = E[s_t s_tᵀ]."""

    p = Wa.shape[0]
    d = 1+3*p

    gamma_a_x_Wa_dot = ldiag_mul(gamma_a, Wa_dot)

    # Sigma_inv = jnp.diag(sigma2_inv)  # explicit diag for clarity
    sigma2_inv = 1.0 / sigma2

    # --- Diagonal blocks ---

    SA = ldiag_mul(sigma2_inv, gamma_a_x_Wa_dot)     # Σ⁻¹ A
    AG = gamma_a_x_Wa_dot @ Gamma0                      # A Γ₀
    V_eta_a_eta_a = T * jnp.sum(SA * AG)  # vector 1

    # V_gamma_a_gamma_a = T * (jnp.diag(sigma2_inv) * (Wa @ Gamma0 @ Wa.T))
    diag_M = jnp.sum(Wa * (Wa @ Gamma0), axis=1)
    V_gamma_a_gamma_a = T * sigma2_inv * diag_M  # vector p

    # V_beta_beta = T * (jnp.diag(sigma2_inv) * Gamma0)
    V_beta_beta = T * sigma2_inv * jnp.diag(Gamma0)  # vector p

    V_sigma_sigma = T * (kappa4 - 1.0) / 4.0 * sigma2_inv**2  # vector p

    # --- Cross-covariance examples ---
    diag_term = jnp.sum(Wa * (Gamma0 @ Wa_dot.T).T, axis=1)
    V_gamma_a_eta_a = T * (gamma_a * sigma2_inv * diag_term)  # vector p


    # V_beta_eta_a = T * (gamma_a * (sigma2_inv) * jnp.diag(Gamma0 @ Wa_dot.T))
    diag_term = jnp.sum(Gamma0 * Wa_dot, axis=1)
    V_beta_eta_a = T * gamma_a * sigma2_inv * diag_term  # vector p


    # V_sigma_eta_a = jnp.zeros((p, ))
    V_sigma_eta_a = jnp.zeros(p)

    diag_term = jnp.sum(Gamma0 * Wa, axis=1)
    V_beta_gamma_a = T * (sigma2_inv * diag_term)  # vector p

    V_sigma_gamma_a = jnp.zeros(p)

    V_sigma_beta = jnp.zeros(p)

    # --- Assemble selected blocks ---
    if full:
        # --- Diag-block filling ---
        V = jnp.zeros((d, d))
        V = V.at[0, 0].set(V_eta_a_eta_a)

        V = V.at[jnp.arange(1, 1 + p), jnp.arange(1, 1 + p)].set(V_gamma_a_gamma_a)
        V = V.at[jnp.arange(1+p, 1+2*p), jnp.arange(1+p, 1+2*p)].set(V_beta_beta)
        V = V.at[jnp.arange(1+2*p, 1+3*p), jnp.arange(1+2*p, 1+3*p)].set(V_sigma_sigma)

        # --- Cross-block for vector symmetric filling ---
        V = V.at[1:1+p, 0].set(V_gamma_a_eta_a)
        V = V.at[0, 1:1+p].set(V_gamma_a_eta_a)

        V = V.at[1+p:1+2*p, 0].set(V_beta_eta_a)
        V = V.at[0, 1+p:1+2*p].set(V_beta_eta_a)

        V = V.at[1+2*p:1+3*p, 0].set(V_sigma_eta_a)
        V = V.at[0, 1+2*p:1+3*p].set(V_sigma_eta_a)

        # --- Cross-block diag symmetric filling ---
        i = jnp.arange(p)

        # beta × gamma_a
        V = V.at[1+p + i, 1+i].set(V_beta_gamma_a)
        V = V.at[1+i, 1+p+i].set(V_beta_gamma_a)

        # sigma × gamma_a
        V = V.at[1+2*p+i, 1+i].set(V_sigma_gamma_a)
        V = V.at[1+i, 1+2*p+i].set(V_sigma_gamma_a)

        # sigma × beta
        V = V.at[1+2*p + i, 1+p+i].set(V_sigma_beta)
        V = V.at[1+p + i, 1+2*p + i].set(V_sigma_beta)


    else:
        V = {}
        V["eta_a_eta_a"] = V_eta_a_eta_a

        # scalar–vector
        V["gamma_a_eta_a"] = V_gamma_a_eta_a      # (p,)
        V["beta_eta_a"] = V_beta_eta_a
        V["sigma_eta_a"] = V_sigma_eta_a

        # vector–vector (diagonal only!)
        V["gamma_a_gamma_a"] = V_gamma_a_gamma_a  # (p,)
        V["beta_gamma_a"] = V_beta_gamma_a
        V["sigma_gamma_a"] = V_sigma_gamma_a

        V["beta_beta"] = V_beta_beta
        V["sigma_beta"] = V_sigma_beta
        V["sigma_sigma"] = V_sigma_sigma

    return V


# ===============================================================
# === Assemble analytical H =====================================
# ===============================================================
def assemble_H(Wa, Wa_dot, gamma_a, G, Gamma0, sigma2, T, kappa4=3.0, full=True):
    """
    Assemble the analytical sensitivity matrix ``H``.

    This routine starts from :func:`assemble_V` and then applies the
    fourth-moment correction to the variance block so that the resulting
    matrix matches the analytical Hessian-based sensitivity used in the
    sandwich variance formulas.

    Parameters
    ----------
    Wa, Wa_dot : jax.Array
        Attention matrix and its derivative with respect to ``eta_a``.
    gamma_a : jax.Array
        Lagged attention coefficients of shape ``(p,)``.
    G : jax.Array
        Transition matrix in the VAR recursion.
    Gamma0 : jax.Array
        Stationary covariance matrix of the process.
    sigma2 : jax.Array
        Innovation variances of shape ``(p,)``.
    T : int
        Sample size used to scale the analytical expressions.
    kappa4 : float, default=3.0
        Fourth moment of the innovation distribution.
    full : bool, default=True
        When ``True``, return a dense matrix. Otherwise return the block
        dictionary representation used by the matrix-free helpers.

    Returns
    -------
    jax.Array | dict[str, jax.Array]
        Sensitivity matrix in the requested representation.
    """
    p = sigma2.shape[0]
    V = assemble_V(Wa, Wa_dot, gamma_a, G, Gamma0, sigma2, T, kappa4, full)
    sigma_inv4 = 1.0 / (sigma2 ** 2)

    Delta_sigma_sigma = (kappa4 - 3) / 4.0 * sigma_inv4  # vector p

    H = V.copy()
    if full:
        i = jnp.arange(p)
        # sigma – sigma  (DIAGONAL VECTOR)
        H = H.at[1+2*p + i, 1+2*p + i].add(-Delta_sigma_sigma)

    else:
        H["sigma_sigma"] = H["sigma_sigma"] - Delta_sigma_sigma

    return H



def build_Kp(p, dtype=jnp.float32):
    """
    K_p = diag(p^{-1/2}, 1, ..., 1) in R^{(3p+1)x(3p+1)}
    """
    d = 1 + 3 * p
    diag_entries = jnp.ones((d,), dtype=dtype)
    diag_entries = diag_entries.at[0].set(1.0 / jnp.sqrt(p))
    return jnp.diag(diag_entries)


def build_Kp_adaptive_from_diag(V, H, eps=1e-8, mode="max"):
    """
    Build a simple diagonal scaling matrix Kp from the diagonal magnitudes of V and H.

    Parameters
    ----------
    V, H : (d, d) arrays
        Full covariance/sensitivity matrices.
    eps : float
        Numerical floor to avoid division by zero.
    mode : str
        "max"  -> use max(|diag(V)|, |diag(H)|)
        "avg"  -> use 0.5 * (|diag(V)| + |diag(H)|)

    Returns
    -------
    Kp : (d, d) array
        Diagonal scaling matrix.
    scale_vec : (d,) array
        Diagonal entries of Kp.
    """
    diag_V = jnp.abs(jnp.diag(V))
    diag_H = jnp.abs(jnp.diag(H))

    if mode == "max":
        d = jnp.maximum(diag_V, diag_H)
    elif mode == "avg":
        d = 0.5 * (diag_V + diag_H)
    else:
        raise ValueError("mode must be 'max' or 'avg'")

    d = jnp.maximum(d, eps)
    scale_vec = 1.0 / jnp.sqrt(d)
    Kp = jnp.diag(scale_vec)
    return Kp, scale_vec


def scale_V_H(V, H, p=None, method="theory", eps=1e-8):
    """
    Compute scaled matrices V_tilde and H_tilde.

    Parameters
    ----------
    V, H : full matrices
    p : int or None
        Needed only if method == "theory"
    method : str
        "theory"   -> Kp = diag(p^{-1/2}, 1, ..., 1)
        "adaptive" -> diagonal scaling from diag(V), diag(H)
    """
    if method == "theory":
        if p is None:
            raise ValueError("p must be provided when method='theory'")
        Kp = build_Kp(p, dtype=V.dtype)
        scale_vec = jnp.diag(Kp)
    elif method == "adaptive":
        Kp, scale_vec = build_Kp_adaptive_from_diag(V, H, eps=eps, mode="max")
    else:
        raise ValueError("method must be 'theory' or 'adaptive'")

    V_tilde = Kp @ V @ Kp
    H_tilde = Kp @ H @ Kp
    return V_tilde, H_tilde, Kp, scale_vec


def check_positive_definite_scaled(V_tilde, H_tilde, tol=1e-8):
    """
    Check whether V_tilde and H_tilde are numerically positive definite.

    Returns
    -------
    info : dict
        eigenvalue summaries and boolean flags
    """
    evals_V = jnp.linalg.eigvalsh(V_tilde)
    evals_H = jnp.linalg.eigvalsh(H_tilde)

    info = {
        "evals_V": evals_V,
        "evals_H": evals_H,
        "V_min_eig": jnp.min(evals_V),
        "V_max_eig": jnp.max(evals_V),
        "H_min_eig": jnp.min(evals_H),
        "H_max_eig": jnp.max(evals_H),
        "V_is_pd": jnp.min(evals_V) > tol,
        "H_is_pd": jnp.min(evals_H) > tol,
    }
    return info

# ===============================================================
# === Sandwich variance =========================================
# ===============================================================
def apply_block_operator(x, M, p):
    """
    M: dict like V or H (same structure), storing scalars and diag vectors.
    x: (d,) where d = 1 + 3p

    returns: (d,) = full_matrix(M) @ x, without materializing the matrix
    """
    xa = x[0]
    x_ga = x[1:1+p]
    x_b = x[1+p:1+2*p]
    x_s = x[1+2*p:1+3*p]

    out = jnp.zeros_like(x)

    # --- eta_a row ---
    out0 = M["eta_a_eta_a"] * xa
    out0 = out0 + jnp.dot(M["gamma_a_eta_a"], x_ga)
    out0 = out0 + jnp.dot(M["beta_eta_a"], x_b)
    out0 = out0 + jnp.dot(M["sigma_eta_a"], x_s)
    out = out.at[0].set(out0)

    # --- gamma_a block ---
    y_ga = M["gamma_a_eta_a"] * xa
    y_ga = y_ga + M["gamma_a_gamma_a"] * x_ga
    y_ga = y_ga + M["beta_gamma_a"] * x_b
    y_ga = y_ga + M["sigma_gamma_a"] * x_s
    out = out.at[1:1+p].set(y_ga)

    # --- beta block ---
    y_b = M["beta_eta_a"] * xa
    y_b = y_b + M["beta_gamma_a"] * x_ga
    y_b = y_b + M["beta_beta"] * x_b
    y_b = y_b + M["sigma_beta"] * x_s
    out = out.at[1+p:1+2*p].set(y_b)

    # --- sigma block ---
    y_s = M["sigma_eta_a"] * xa
    y_s = y_s + M["sigma_gamma_a"] * x_ga
    y_s = y_s + M["sigma_beta"] * x_b
    y_s = y_s + M["sigma_sigma"] * x_s
    out = out.at[1+2*p:1+3*p].set(y_s)

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
    Estimate diag(H^{-1} V H^{-1}) using Rademacher (Hutchinson) + CG + block-Jacobi preconditioner.

    Parameters
    ----------
    H, V : dict
        Block-structured operators (your current design).
    apply_H, apply_V : callable
        Matrix-free operators: x -> Hx, x -> Vx.
    p : int
        Dimension of each vector block.
    """

    d = 1 + 3 * p

    inv_eta = 1.0 / H["eta_a_eta_a"]
    inv_gamma_a = 1.0 / H["gamma_a_gamma_a"]
    inv_beta = 1.0 / H["beta_beta"]
    inv_sigma = 1.0 / H["sigma_sigma"]

    def Minv(x):
        xa = x[0]
        x_ga = x[1:1+p]
        x_b  = x[1+p:1+2*p]
        x_s  = x[1+2*p:1+3*p]

        ya = inv_eta * xa
        y_ga = inv_gamma_a * x_ga
        y_b  = inv_beta * x_b
        y_s  = inv_sigma * x_s

        return jnp.concatenate([
            jnp.array([ya]),
            y_ga,
            y_b,
            y_s
        ])

    def solve_H(b):
        x, _ = cg(
            apply_H,
            b,
            M=Minv,
            tol=cg_tol,
            maxiter=cg_maxiter,
        )
        return x

    diag_est = jnp.zeros((d,))

    for _ in range(num_mc):
        key, sub = jax.random.split(key)
        z = jax.random.rademacher(sub, (d,), dtype=jnp.float32)

        u = solve_H(z)
        w = apply_V(u)
        v = solve_H(w)

        diag_est = diag_est + z * v

    return diag_est / num_mc


def estimate_Avar_explicit(H, V, apply_H, apply_V, p,):
    """Explicitly compute A = H^{-1} V H^{-1}.

    WARNING: O(d^3), only feasible for small p.
    """
    d = 1 + 3 * p
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
    """
    Compute the sandwich covariance ``H^{-1} V H^{-1}``.

    Parameters
    ----------
    H, V : jax.Array
        Dense sensitivity and score covariance matrices.

    Returns
    -------
    jax.Array
        Asymptotic covariance matrix on the same parameter scale as ``H``.
    """

    HinvV = jnp.linalg.solve(H, V)
    Avar = jnp.linalg.solve(H, HinvV.T).T

    return Avar


def compute_sandwich_variance(Y, Z, theta_hat, params_scale, kappa4=3.0, alpha=0.05, verbose=False):
    """
    Compute Wald-style uncertainty estimates for a fitted parameter vector.

    Parameters
    ----------
    Y : jax.Array
        Observed data matrix with shape ``(T, p)``.
    Z : jax.Array
        Feature matrix used to build the attention weights.
    theta_hat : jax.Array
        Estimated parameter vector on the natural scale
        ``(eta_a, gamma_a, beta, sigma2)``.
    params_scale : dict
        Scaling constants used by the attention map.
    kappa4 : float, default=3.0
        Assumed fourth moment of the innovation distribution.
    alpha : float, default=0.05
        Two-sided confidence level parameter.
    verbose : bool, default=False
        If ``True``, print eigenvalue diagnostics for the scaled ``V`` and ``H``.

    Returns
    -------
    dict[str, jax.Array]
        Dictionary with the point estimate, standard errors, and lower/upper
        Wald confidence limits.
    """

    T, p = Y.shape

    eta_a = theta_hat[0]

    gamma_a = theta_hat[1:1+p]
    beta = theta_hat[1+p:1+2*p]
    sigma2 = theta_hat[1+2*p:1+3*p]

    # use consistent attention derivatives
    Wa, Wa_dot, _ = compute_Wa_Wb_dots(Z, eta_a, params_scale["eta_a"])

    G = jnp.diag(beta) + ldiag_mul(gamma_a, Wa)

    Gamma0 = Gamma0_stationary_with_lyapunov_doubling(G, sigma2)
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
    V = assemble_V(Wa, Wa_dot, gamma_a, G, Gamma0, sigma2, T, kappa4, full)
    H = assemble_H(Wa, Wa_dot, gamma_a, G, Gamma0, sigma2, T, kappa4, full)

    V_tilde, H_tilde, Kp, scale_vec = scale_V_H(V, H, p)
    pd_info = check_positive_definite_scaled(V_tilde, H_tilde)

    if verbose:
        print("Scaled V/H eigenvalue check:")
        print(f"V_tilde min eig = {pd_info['V_min_eig']:.6e}, max eig = {pd_info['V_max_eig']:.6e}")
        print(f"H_tilde min eig = {pd_info['H_min_eig']:.6e}, max eig = {pd_info['H_max_eig']:.6e}")
        print(f"V_tilde PD? {pd_info['V_is_pd']}")
        print(f"H_tilde PD? {pd_info['H_is_pd']}")

    Avar = compute_Avar(H, V)

    se = jnp.sqrt(jnp.maximum(jnp.diag(Avar), 0.0))
    z_crit = norm.ppf(1 - alpha / 2.0)
    lower = theta_hat - z_crit * se
    upper = theta_hat + z_crit * se

    return dict(theta=theta_hat, se=se, lower=lower, upper=upper)


def make_A(p, kind, j=None, indices=None, j1=None, j2=None):
    """
    Build projection matrix A for the old parameter map.

        theta = (eta_a, gamma_a[1:p], beta[1:p], sigma2[1:p])
              in R^{3p+1}

    Parameter order in Python indexing:
        eta_a      -> 0
        gamma_a[j] -> 1 + (j-1),          j = 1,...,p
        beta[j]    -> 1 + p + (j-1),      j = 1,...,p
        sigma2[j]  -> 1 + 2*p + (j-1),    j = 1,...,p

    Parameters
    ----------
    p : int
        Dimension of each vector block.
    kind : str
        Type of projection. Supported values:
            "eta_a"
            "gamma_a"
            "beta"
            "sigma2"
            "pair"
            "mean_beta"
            "mean_gamma_a"
            "diff_beta"
            "sum_gamma_beta"
    j : int or list[int], optional
        1-based component index, used for:
            "gamma_a", "beta", "sigma2", "sum_gamma_beta"
        For "pair", use `indices` instead.
    indices : list[int], optional
        List of raw 0-based parameter positions in theta, used for "pair".
    j1, j2 : int, optional
        1-based component indices used for "diff_beta".

    Returns
    -------
    A : jnp.ndarray
        Projection matrix of shape (L, 1 + 3*p).
    """
    d = 1 + 3 * p

    def idx_gamma_a(j):
        if not (1 <= j <= p):
            raise ValueError(f"j must be in 1,...,{p}")
        return 1 + (j - 1)

    def idx_beta(j):
        if not (1 <= j <= p):
            raise ValueError(f"j must be in 1,...,{p}")
        return 1 + p + (j - 1)

    def idx_sigma2(j):
        if not (1 <= j <= p):
            raise ValueError(f"j must be in 1,...,{p}")
        return 1 + 2 * p + (j - 1)

    if kind == "eta_a":
        A = jnp.zeros((1, d))
        A = A.at[0, 0].set(1.0)
        return A

    elif kind == "gamma_a":
        if j is None:
            raise ValueError("j must be provided for kind='gamma_a'")
        A = jnp.zeros((1, d))
        A = A.at[0, idx_gamma_a(j)].set(1.0)
        return A

    elif kind == "beta":
        if j is None:
            raise ValueError("j must be provided for kind='beta'")
        A = jnp.zeros((1, d))
        A = A.at[0, idx_beta(j)].set(1.0)
        return A

    elif kind == "sigma2":
        if j is None:
            raise ValueError("j must be provided for kind='sigma2'")
        A = jnp.zeros((1, d))
        A = A.at[0, idx_sigma2(j)].set(1.0)
        return A

    elif kind == "pair":
        if indices is None or len(indices) == 0:
            raise ValueError("indices must be a non-empty list for kind='pair'")
        A = jnp.zeros((len(indices), d))
        for r, idx in enumerate(indices):
            if not (0 <= idx < d):
                raise ValueError(f"index {idx} out of range for theta dimension {d}")
            A = A.at[r, idx].set(1.0)
        return A

    elif kind == "mean_beta":
        A = jnp.zeros((1, d))
        beta_idx = jnp.arange(1 + p, 1 + 2 * p)
        A = A.at[0, beta_idx].set(1.0 / p)
        return A

    elif kind == "mean_gamma_a":
        A = jnp.zeros((1, d))
        gamma_idx = jnp.arange(1, 1 + p)
        A = A.at[0, gamma_idx].set(1.0 / p)
        return A

    elif kind == "diff_beta":
        if j1 is None or j2 is None:
            raise ValueError("j1 and j2 must be provided for kind='diff_beta'")
        A = jnp.zeros((1, d))
        A = A.at[0, idx_beta(j1)].set(1.0)
        A = A.at[0, idx_beta(j2)].set(-1.0)
        return A

    elif kind == "sum_gamma_beta":
        if j is None:
            raise ValueError("j must be provided for kind='sum_gamma_beta'")
        A = jnp.zeros((1, d))
        A = A.at[0, idx_gamma_a(j)].set(1.0)
        A = A.at[0, idx_beta(j)].add(1.0)
        return A

    else:
        raise ValueError(
            "Unsupported kind. Choose from "
            "{'eta_a', 'gamma_a', 'beta', 'sigma2', 'pair', "
            "'mean_beta', 'mean_gamma_a', 'diff_beta', 'sum_gamma_beta'}"
        )


def compute_projected_asymptotic_covariance(A, V, H, p, scaling="theory", eps=1e-8):
    """
    Compute the projected asymptotic covariance.

        J_T = A H_tilde^{-1} V_tilde H_tilde^{-1} A^T

    Parameters
    ----------
    A : (L, d) array
        Projection matrix, where d = 1 + 3p in the current code.
        For a single parameter, A can be a row vector reshaped to (1, d).
    V, H : (d, d) arrays
        Full covariance/sensitivity matrices.
    p : int
        Dimension of the time series.
    scaling : str
        "theory"   -> use K_p = diag(p^{-1/2}, 1, ..., 1)
        "adaptive" -> use your adaptive scaling from diag(V), diag(H)
    eps : float
        Numerical floor used only if scaling="adaptive".

    Returns
    -------
    J_T : (L, L) array
        Projected asymptotic covariance.
    extras : dict
        Useful intermediate objects for debugging.
    """
    A = jnp.atleast_2d(A)
    d = 1 + 3 * p

    if A.shape[1] != d:
        raise ValueError(f"A must have shape (L, {d}), got {A.shape}")

    if scaling == "theory":
        V_tilde, H_tilde, Kp, scale_vec = scale_V_H(V, H, p=p)
    elif scaling == "adaptive":
        Kp, scale_vec = build_Kp_adaptive_from_diag(V, H, eps=eps, mode="max")
        V_tilde = Kp @ V @ Kp
        H_tilde = Kp @ H @ Kp
    else:
        raise ValueError("scaling must be either 'theory' or 'adaptive'")

    # H_tilde_inv = jnp.linalg.inv(H_tilde)
    # J_T = A @ H_tilde_inv @ V_tilde @ H_tilde_inv @ A.T
    X = jnp.linalg.solve(H_tilde, A.T)
    J_T = A @ jnp.linalg.solve(H_tilde, V_tilde @ X)
    J_T = 0.5 * (J_T + J_T.T)

    return J_T, {
        "Kp": Kp,
        "scale_vec": scale_vec,
        "V_tilde": V_tilde,
        "H_tilde": H_tilde,
    }


def build_VH_from_theta_hat(Y, Z, theta_hat, params_scale, kappa4=3.0):
    """
    Build dense analytical ``V`` and ``H`` matrices from a natural-scale estimate.

    Parameters
    ----------
    Y, Z : jax.Array
        Observed data and feature matrix.
    theta_hat : jax.Array
        Parameter estimate on the natural scale.
    params_scale : dict
        Scaling constants for the attention map.
    kappa4 : float, default=3.0
        Assumed innovation fourth moment.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        Dense ``(V, H)`` matrices evaluated at ``theta_hat``.
    """
    T, p = Y.shape

    eta_a = theta_hat[0]
    # eta_max = 2
    # eta_a = jnp.clip(theta_hat[0], -eta_max, eta_max)

    gamma_a = theta_hat[1:1+p]
    beta = theta_hat[1+p:1+2*p]
    sigma2 = theta_hat[1+2*p:1+3*p]

    Wa, Wa_dot, _ = compute_Wa_Wb_dots(Z, eta_a, params_scale["eta_a"])
    G = jnp.diag(beta) + ldiag_mul(gamma_a, Wa)
    Gamma0 = Gamma0_stationary_with_lyapunov_doubling(G, sigma2)

    V = assemble_V(Wa, Wa_dot, gamma_a, G, Gamma0, sigma2, T, kappa4=kappa4, full=True)
    H = assemble_H(Wa, Wa_dot, gamma_a, G, Gamma0, sigma2, T, kappa4=kappa4, full=True)
    return V, H


def compute_projected_asymptotic_covariance_from_theta_hat(Y, Z, theta_hat, A, params_scale,
                                                           kappa4=3.0, scaling="theory", eps=1e-8):
    """
    Project the asymptotic covariance after rebuilding ``V`` and ``H`` from data.

    Parameters
    ----------
    Y, Z : jax.Array
        Observed data and feature matrix.
    theta_hat : jax.Array
        Natural-scale parameter estimate.
    A : jax.Array
        Projection matrix selecting the linear functional of interest.
    params_scale : dict
        Scaling constants for the attention map.
    kappa4 : float, default=3.0
        Assumed innovation fourth moment.
    scaling : {"theory", "adaptive"}, default="theory"
        Scaling rule passed to :func:`compute_projected_asymptotic_covariance`.
    eps : float, default=1e-8
        Numerical floor used by adaptive scaling.

    Returns
    -------
    tuple[jax.Array, dict]
        Projected covariance matrix and a diagnostics dictionary containing the
        reconstructed ``V``/``H`` plus the intermediate scaling objects.
    """
    T, p = Y.shape
    V, H = build_VH_from_theta_hat(Y, Z, theta_hat, params_scale, kappa4=kappa4)
    J_T, extras = compute_projected_asymptotic_covariance(
        A=A, V=V, H=H, p=p, scaling=scaling, eps=eps
    )
    return J_T, {"V": V, "H": H, **extras}


def compute_z_stat_from_theta_hat(
    Y,
    Z,
    theta_hat,
    theta_true,
    A,
    params_scale,
    theta_for_J=None,
    kappa4=3.0,
    eps=1e-12,
):
    """
    Theorem 1 statistic for a 1-row projection A:
        sqrt(T_eff) * A K_p^{-1} (theta_hat - theta_true) / sqrt(J)
    where
        J = A H_tilde^{-1} V_tilde H_tilde^{-1} A^T,
        H_tilde = K_p H K_p,
        V_tilde = K_p V K_p.

    Here H and V are the theorem-normalized matrices:
        H = -(1/T_eff) E[ Hessian(total loglik) ]
        V =  (1/T_eff) Cov[ Score(total loglik) ].
    """
    T_eff = Y.shape[0] - 1
    p = Y.shape[1]

    if theta_for_J is None:
        theta_for_J = theta_true   # cleaner in simulation

    # raw matrices from current code carry a factor of T_eff
    V_raw, H_raw = build_VH_from_theta_hat(
        Y, Z, theta_for_J, params_scale, kappa4=kappa4
    )

    # theorem normalization
    V = V_raw / T_eff
    H = H_raw / T_eff

    J, extras = compute_projected_asymptotic_covariance(
        A=A, V=V, H=H, p=p, scaling="theory"
    )

    # K_p^{-1} = diag(sqrt(p), 1, ..., 1)
    Kp = build_Kp(p, dtype=theta_hat.dtype)
    Kp_inv_diag = 1.0 / jnp.diag(Kp)

    diff_tilde = A @ (Kp_inv_diag * (theta_hat - theta_true))
    z_val = jnp.sqrt(T_eff) * diff_tilde / jnp.sqrt(jnp.maximum(jnp.diag(J), eps))

    return z_val, {
        "J": J,
        "V_theorem": V,
        "H_theorem": H,
        "T_eff": T_eff,
        **extras,
    }


def compare_VH_analytic_vs_auto(theta_hat, Y, Z, params_scale, kappa4=3.0, verbose=True):
    """
    Compare analytical V,H from build_VH_from_theta_hat(...) with
    autodiff-based V_auto,H_auto on the natural scale.
    """
    V_analytic, H_analytic = build_VH_from_theta_hat(
        Y, Z, theta_hat, params_scale, kappa4=kappa4
    )

    V_auto = compute_V_auto_natural(theta_hat, Y, Z, params_scale)
    H_auto = compute_H_auto_natural(theta_hat, Y, Z, params_scale)

    out = {
        "V_analytic": V_analytic,
        "H_analytic": H_analytic,
        "V_auto": V_auto,
        "H_auto": H_auto,
        "V_max_abs_diff": jnp.max(jnp.abs(V_analytic - V_auto)),
        "H_max_abs_diff": jnp.max(jnp.abs(H_analytic - H_auto)),
    }

    if verbose:
        print("Autodiff vs analytical comparison")
        print(f"max abs diff V = {out['V_max_abs_diff']:.6e}")
        print(f"max abs diff H = {out['H_max_abs_diff']:.6e}")

    return out

# ------------------------------------------------------------
# Example
# ------------------------------------------------------------

if __name__ == "__main__":
    import random
    from .dgp import simulate_VAR_attention
    from .dgp import flatten_params
    from .mle_optimize import mle_optimize
    from .params import make_block_order

    # jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(threshold=10_000, linewidth=200, suppress=True, precision=6)

    # --- Simulation settings ---
    key = jax.random.PRNGKey(random.randint(0, 10000))
    keys = jax.random.split(key,10)
    T, p, q = 1000, 10, 5

    # --- True parameters (in natural scale) ---
    params_true = dict(
        eta_a=0.2,
        gamma_a=jnp.linspace(-0.15, 0.15, p),
        beta=jnp.linspace(-0.20, 0.20, p),
        sigma2=jnp.exp(jnp.log(0.5) + 0.2 * jax.random.normal(keys[0], (p,)))
    )

    params_scale = dict(eta_a=1)

    algorithm = "auto_gradhess"
    # algorithm = "analytical_gradhess"
    # update_idx = [1]
    update_idx = range(0, 1+2*p)
    # update_idx = slice(None)
    # --- Simulate data ---
    Y, Z, W_a = simulate_VAR_attention(key, params_true, T=T, p=p, q=q, dist="normal")
    print("Simulated Y shape:", Y.shape)

    # --- Convert σ² → σ̃² (softplus inverse) ---
    sigma2_tilde_true = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = params_true.copy()
    params_reparam["sigma2"] = sigma2_tilde_true

    # --- Flatten parameters into reparameterized vector θ̃ ---
    block_order = make_block_order(p)
    theta_true_reparam, _ = flatten_params(params_reparam, block_order)
    print("θ length:", theta_true_reparam.size)

    # Only initialize update_idx related initial values
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_true_reparam.size)

    update_idx = jnp.array(update_idx)
    theta_init = theta_true_reparam
    theta_init_update = theta_true_reparam[update_idx]
    theta_init = theta_init.at[update_idx].add(0.5 * jax.random.normal(key, theta_init_update.shape))

    # Estimation
    # breakpoint()
    theta_hat_reparam_short, grad_ll, hess_ll, convergence = mle_optimize(Y, Z, theta_init, params_scale, algorithm="lbfgs", update_idx=update_idx)

    theta_hat_reparam = theta_true_reparam
    theta_hat_reparam = theta_hat_reparam.at[update_idx].set(theta_hat_reparam_short)

    # Convert back to natural scale (with softplus)
    sigma2_tilde_hat = theta_hat_reparam[1+2*p:1+3*p]
    sigma2_hat = jax.nn.softplus(sigma2_tilde_hat)

    theta_hat = jnp.concatenate([theta_hat_reparam[:1+2*p], sigma2_hat])

    # Test sandwich variance
    var_theta = compute_sandwich_variance(Y, Z, theta_hat, params_scale)

    # ------------------------------------------------------------
    # Test sandwich variance
    # ------------------------------------------------------------
    var_theta = compute_sandwich_variance(Y, Z, theta_hat, params_scale)
    print("Sandwich SE:")
    print(var_theta["se"])

    # ------------------------------------------------------------
    # Compare analytical V/H against autodiff V/H on natural scale
    # ------------------------------------------------------------
    cmp = compare_VH_analytic_vs_auto(
        theta_hat=theta_hat,    # or theta_hat
        Y=Y,
        Z=Z,
        params_scale=params_scale,
        kappa4=3.0,
        verbose=True,
    )

    print("\nMax absolute differences:")
    print("V_max_abs_diff =", cmp["V_max_abs_diff"])
    print("H_max_abs_diff =", cmp["H_max_abs_diff"])

    T_eff = Y.shape[0] - 1
    print("H normalized diff =", jnp.max(jnp.abs(cmp["H_analytic"] / T_eff - cmp["H_auto"])))
    print("V normalized diff =", jnp.max(jnp.abs(cmp["V_analytic"] / T_eff - cmp["V_auto"])))
    # ------------------------------------------------------------
    # Compare projected asymptotic covariance J_T
    # Example: eta_a, gamma_a[1], beta[1], sigma2[1]
    # ------------------------------------------------------------
    A_eta = make_A(p, "eta_a")
    A_gamma1 = make_A(p, "gamma_a", j=1)
    A_beta1 = make_A(p, "beta", j=1)
    A_sigma1 = make_A(p, "sigma2", j=1)

    J_eta_analytic, _ = compute_projected_asymptotic_covariance(
        A_eta, cmp["V_analytic"], cmp["H_analytic"], p, scaling="theory"
    )
    J_eta_auto, _ = compute_projected_asymptotic_covariance(
        A_eta, cmp["V_auto"], cmp["H_auto"], p, scaling="theory"
    )

    J_gamma1_analytic, _ = compute_projected_asymptotic_covariance(
        A_gamma1, cmp["V_analytic"], cmp["H_analytic"], p, scaling="theory"
    )
    J_gamma1_auto, _ = compute_projected_asymptotic_covariance(
        A_gamma1, cmp["V_auto"], cmp["H_auto"], p, scaling="theory"
    )

    J_beta1_analytic, _ = compute_projected_asymptotic_covariance(
        A_beta1, cmp["V_analytic"], cmp["H_analytic"], p, scaling="theory"
    )
    J_beta1_auto, _ = compute_projected_asymptotic_covariance(
        A_beta1, cmp["V_auto"], cmp["H_auto"], p, scaling="theory"
    )

    J_sigma1_analytic, _ = compute_projected_asymptotic_covariance(
        A_sigma1, cmp["V_analytic"], cmp["H_analytic"], p, scaling="theory"
    )
    J_sigma1_auto, _ = compute_projected_asymptotic_covariance(
        A_sigma1, cmp["V_auto"], cmp["H_auto"], p, scaling="theory"
    )

    print("\nProjected covariance comparison:")
    print("eta_a   analytic =", J_eta_analytic[0, 0],   " auto =", J_eta_auto[0, 0])
    print("gamma_1 analytic =", J_gamma1_analytic[0, 0], " auto =", J_gamma1_auto[0, 0])
    print("beta_1  analytic =", J_beta1_analytic[0, 0],  " auto =", J_beta1_auto[0, 0])
    print("sigma_1 analytic =", J_sigma1_analytic[0, 0], " auto =", J_sigma1_auto[0, 0])


    D_H = cmp["H_analytic"] / T - cmp["H_auto"]
    D_V = cmp["V_analytic"] / T - (T - 1) * cmp["V_auto"]
    print("scaled H max abs diff =", jnp.max(jnp.abs(D_H)))
    print("scaled V max abs diff =", jnp.max(jnp.abs(D_V)))
