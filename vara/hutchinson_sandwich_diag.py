import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from .mle_ad_hvp import make_hvp

def make_solve_H(hvp, *, damping=0.0, tol=1e-6, maxiter=200):
    """
    Return a solver solve(b) that approximately computes x ≈ (H + damping I)^{-1} b
    using CG with matrix-free matvec via hvp.
    """
    def matvec(v):
        return hvp(v) + damping * v

    def solve(b):
        x, info = cg(matvec, b, tol=tol, maxiter=maxiter)
        return x, info

    return solve

def sandwich_diag_hutchinson(
    key,
    solve_H,      # solve_H(b) -> (x, info) where x ≈ H^{-1} b
    V_mv,         # V_mv(u) -> V @ u  (matrix-free or explicit)
    dim,
    *,
    n_probe=128,
):
    """
    Estimate diag(H^{-1} V H^{-1}) using Hutchinson probing:
      diag(A) ≈ (1/m) Σ z ⊙ (A z),  z ~ Rademacher
    with A = H^{-1} V H^{-1}.

    Cost per probe: 2 CG solves + 1 V matvec.
    """
    def one_probe(carry, _):
        key = carry
        key, sub = jax.random.split(key)

        # Rademacher ±1 (paired variance reduction)
        z = jax.random.rademacher(sub, (dim,), dtype=jnp.int8).astype(jnp.float32)

        # --- +z ---
        u1, info1a = solve_H(z)
        w1 = V_mv(u1)
        x1, info1b = solve_H(w1)

        # --- -z ---
        zm = -z
        u2, info2a = solve_H(zm)
        w2 = V_mv(u2)
        x2, info2b = solve_H(w2)

        # paired Hutchinson contribution (unbiased, lower variance)
        contrib = 0.5 * (z * x1 + zm * x2)

        return key, (contrib, info1a, info1b)

    key, (contribs, info1s, info2s) = jax.lax.scan(one_probe, key, xs=None, length=n_probe)
    diag_est = jnp.mean(contribs, axis=0)

    # optionally: report CG statuses
    info = {
        "cg_info_stage1": info1s,
        "cg_info_stage2": info2s,
    }
    return diag_est, info


if __name__ == "__main__":

    import random

    from .loglik_attn import loglik_reparam
    from .dgp import simulate_VAR_attention
    from .dgp import flatten_params

    # jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(threshold=10_000, linewidth=200, suppress=True, precision=6)

    # --- Simulation settings ---
    key = jax.random.PRNGKey(random.randint(0, 10000))
    T = 1000
    p = 100
    q = 50

    # --- True parameters (in natural scale) ---
    params_true = dict(
        eta_a=0.5,
        eta_b=0.1,
        gamma_a=jnp.linspace(-0.15, 0.15, p),
        gamma_b=jnp.linspace(-0.10, 0.10, p),
        beta=jnp.linspace(-0.20, 0.20, p),
        sigma2=jnp.exp(jax.random.normal(key, (p,)))  # positive σ²
    )

    params_scale = dict(
        eta_a=1,
        eta_b=1)

    # update_idx = [0, 1]
    update_idx = slice(None)

    # --- Simulate data ---
    Y, Z, W_a, W_b = simulate_VAR_attention(key, params_true, T=T, p=p, q=q, dist="normal")
    print("Simulated Y shape:", Y.shape)

    # --- Convert σ² → σ̃² (softplus inverse) ---
    sigma2_tilde_true = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = params_true.copy()
    params_reparam["sigma2"] = sigma2_tilde_true

    # --- Flatten parameters into reparameterized vector θ̃ ---
    block_order = {
        "eta_a": [0],
        "gamma_a": list(range(1, 1 + p)),
        "beta": list(range(1 + p, 1 + 2 * p)),
        "sigma2": list(range(1 + 2 * p, 1 + 3 * p)),
    }
    theta_true_reparam, _ = flatten_params(params_reparam, block_order)
    print("θ̃ length:", theta_true_reparam.size)

    # -----------------------
    # Gradient function
    # -----------------------
    loglik_grad = jax.grad(loglik_reparam)

    theta0 = theta_true_reparam
    g_full = loglik_grad(theta0, Y, Z, params_scale)
    g = g_full[update_idx]

    print(f"Parameter dimension: {g.size}")

    # -----------------------
    # HVP construction
    # -----------------------
    hvp = make_hvp(
        theta_full=theta0,
        update_idx=update_idx,
        grad_fn=loglik_grad,
        Y=Y,
        Z=Z,
        params_scale=params_scale,
    )

    # -----------------------
    # Observed V = g gᵀ  (outer product form)
    # -----------------------
    def V_mv(u):
        return g * (g @ u)

    # -----------------------
    # Solver for H^{-1}
    # -----------------------
    solve_H = make_solve_H(
        hvp,
        damping=1e-3,      # important for stability
        tol=1e-6,
        maxiter=300,
    )

    # -----------------------
    # Hutchinson sandwich diagonal
    # -----------------------
    key, subkey = jax.random.split(key)
    diag_sandwich, info = sandwich_diag_hutchinson(
        subkey,
        solve_H=solve_H,
        V_mv=V_mv,
        dim=g.size,
        n_probe=128,
    )

    # -----------------------
    # Output
    # -----------------------
    frac_neg = jnp.mean(diag_sandwich < 0.0)
    print(f"Fraction of negative diagonal entries: {frac_neg:.2%}")

    print("Sandwich variance (diagonal, reduced space):")
    print(diag_sandwich)

    print("\nSummary:")
    print(f"Min variance: {diag_sandwich.min():.4e}")
    print(f"Max variance: {diag_sandwich.max():.4e}")
    print(f"Mean variance: {diag_sandwich.mean():.4e}")
