import jax
from jax import jit
import jax.numpy as jnp
import optax
from loglik_attn import build_attention     # moved import here


# ------------------------------------------------------------
# Compute sample auto-covariances Γ_lag
# ------------------------------------------------------------
def compute_autocovariances(Y, lag=0):
    T, p = Y.shape
    if lag < 0 or lag >= T:
        raise ValueError("lag must be in [0, T-1]")

    if lag == 0:
        return (Y.T @ Y) / (T - 1)

    Y_t = Y[lag:]
    Y_tlag = Y[:-lag]
    return (Y_t.T @ Y_tlag) / (T - lag)



# ------------------------------------------------------------
# Yule-Walker Loss
# ------------------------------------------------------------
def yule_walker_loss(theta, Z, Gamma0_hat, Gamma1_hat, params_scale,
                     loss_mode="full", key=None):

    p = Gamma0_hat.shape[0]

    # unpack theta vector
    eta_a = theta[0]
    gamma_a = theta[1:1+p]
    beta = theta[1+p:1+2*p]
    # sigma2 skipped for YW (not needed)

    # build W_a, W_b
    W_a = build_attention(Z, eta_a, params_scale['eta_a'])

    # S, G matrices
    # S = jnp.eye(p) - jnp.diag(gamma_b) @ W_b
    # G = jnp.diag(beta) + jnp.diag(gamma_a) @ W_a
    G = (beta[:, None] * jnp.eye(p)) + (gamma_a[:, None] * W_a)

    residual = Gamma1_hat - G @ Gamma0_hat

    # loss_mode = "full"
    # loss_mode = "row1"
    # loss_mode = "randomK"
    # loss_mode = "hutchinson"
    if loss_mode == "row1":
        row1 = residual[0:5, :]
        return 0.5 * p * jnp.sum(row1 ** 2)
    elif loss_mode == "randomK":
        key = jax.random.PRNGKey(0) if key is None else key
        K = 10
        p = residual.shape[0]
        idx = jax.random.choice(key, p, shape=(K,), replace=False)
        # print(f"random: {idx}")
        rows = residual[idx, :]
        return 0.5 * p / K * jnp.sum(rows ** 2)

    elif loss_mode == "hutchinson":
        key = jax.random.PRNGKey(0) if key is None else key
        # Random ±1 vector
        r = jax.random.rademacher(key, (p,))

        # Row energy scores: A r
        scores = residual @ r

        # Most informative row
        k = jnp.argmax(jnp.abs(scores))

        # Use only row k but scale to approximate Frobenius
        row = residual[k, :]
        return 0.5 * p * jnp.sum(row ** 2)
    else:
        # Full Frobenius norm (default)
        return 0.5 * jnp.sum(residual ** 2)

# ------------------------------------------------------------
# Fit Yule-Walker
# ------------------------------------------------------------
def yule_walker(Y, Z, theta_init, params_scale, update_idx,
        lr=1e-2,
        steps=200):

    Gamma0_hat = compute_autocovariances(Y, 0)
    Gamma1_hat = compute_autocovariances(Y, 1)

    # p = Y.shape[1]

    # start from theta_init
    theta_shortvec = theta_init[update_idx]

    # loss function
    def loss_fn(vec):
        theta = theta_init
        theta = theta.at[update_idx].set(vec)
        return yule_walker_loss(
            theta,
            Z,
            Gamma0_hat,
            Gamma1_hat,
            params_scale
        )

    grad_fn = jax.grad(loss_fn)
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(theta_shortvec)

    # training step
    @jit
    def train_step(vec, opt_state):
        grads = grad_fn(vec)

        # only update entries in update_idx
        # grads = grads * mask

        updates, opt_state = optimizer.update(grads, opt_state, vec)
        vec = optax.apply_updates(vec, updates)

        return vec, opt_state

    # loop
    convergence = dict(status=1, trace=[])
    for i in range(steps):
        theta_shortvec, opt_state = train_step(theta_shortvec, opt_state)

        # --- SIMPLE CONVERGENCE CHECK ---
        if i > 20 and i % 20 == 0:  # check infrequently to avoid slowing JIT
            # grad_norm = jnp.linalg.norm(grad_fn(param_vec))
            # mu = first moment (EMA of gradients), nu = second moment (EMA of gradient squares).
            # If both are near zero, gradients must be small → algorithm has converged.
            # mu_norm = jnp.max(jnp.abs(opt_state[0].mu))
            mu_norm = float(jnp.max(jnp.abs(opt_state[0].mu)))
            if mu_norm < 1e-2:
                print(f"Adam optimizer converged at step {i}, mu_norm={mu_norm}")
                convergence = dict(status=0, trace=opt_state)
                break

    # grad_ll = grad_fn(param_vec)
    grad_ll = []
    hess_ll = []
    # breakpoint()

    # theta_shortvec = param_vec[update_idx]
    # if convergence["status"] == 1:
    #     breakpoint()

    return theta_shortvec, grad_ll, hess_ll, convergence

# ------------------------------------------------------------
# Example
# ------------------------------------------------------------
if __name__ == "__main__":

    import os
    import math
    import random
    import matplotlib.pyplot as plt

    from dgp import simulate_VAR_attention
    from dgp import flatten_params

    # jax.config.update("jax_enable_x64", True)
    jnp.set_printoptions(threshold=10_000, linewidth=200, suppress=True, precision=6)

    # --- Simulation settings ---
    key = jax.random.PRNGKey(random.randint(0, 10000))
    keys = jax.random.split(key,10)

    T, p, q = 1000, 3, 3

    # --- True parameters (in natural scale) ---
    params_true = dict(
        eta_a=0.2,
        gamma_a=jnp.linspace(-0.15, 0.15, p),
        beta=jnp.linspace(-0.20, 0.20, p),
        sigma2=jnp.exp(jnp.log(0.5) + 0.2 * jax.random.normal(keys[0], (p,)))  # positive σ²
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
    block_order = {
        "eta_a": [0],
        "gamma_a": list(range(1, 1 + p)),
        "beta": list(range(1 + p, 1 + 2 * p)),
        "sigma2": list(range(1 + 2 * p, 1 + 3 * p)),
    }
    theta_true_reparam, _ = flatten_params(params_reparam, block_order)
    print("θ̃ length:", theta_true_reparam.size)

    # Only initialize update_idx related initial values
    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_true_reparam.size)

    update_idx = jnp.array(update_idx)
    theta_init = theta_true_reparam
    theta_init_update = theta_true_reparam[update_idx]
    theta_init = theta_init.at[update_idx].add(0.5 * jax.random.normal(key, theta_init_update.shape))

    # Estimation
    # breakpoint()
    theta_hat_reparam_short, grad_ll, hess_ll, convergence = yule_walker(Y, Z, theta_init, params_scale, update_idx=update_idx)

    theta_hat_reparam = theta_true_reparam
    theta_hat_reparam = theta_hat_reparam.at[update_idx].set(theta_hat_reparam_short)

    # Convert back to natural scale (with softplus)
    sigma2_tilde_hat = theta_hat_reparam[1+2*p:1+3*p]
    sigma2_hat = jax.nn.softplus(sigma2_tilde_hat)

    theta_hat = jnp.concatenate([theta_hat_reparam[:1+2*p], sigma2_hat])


    # --- Plot log-likelihood profile for update_idx parameters ---
    plot_width = 2
    n_points = 100

    n_params = len(update_idx)
    n_cols = min(4, n_params)
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    # --- FIX: make axes iterable in all cases ---
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    Gamma0_hat = compute_autocovariances(Y, 0)
    Gamma1_hat = compute_autocovariances(Y, 1)
    for ax, idx in zip(axes, update_idx):
        theta_grid = jnp.linspace(theta_true_reparam[idx] - plot_width, theta_true_reparam[idx] + plot_width, n_points)
        ll_values = []
        for val in theta_grid:
            theta_temp = theta_true_reparam.at[idx].set(val)
            ll = yule_walker_loss(theta_temp, Z, Gamma0_hat, Gamma1_hat, params_scale)
            ll_values.append(float(ll))

        ax.plot(theta_grid, ll_values, linestyle="-", linewidth=2, label="Loss function")
        ax.axvline(theta_init[idx], color="green", linestyle=":", linewidth=2, label="Initial value")
        ax.axvline(theta_true_reparam[idx], color="red", linestyle="--", linewidth=2, label="True value")
        ax.axvline(theta_hat_reparam[idx], color="blue", linestyle="-.", linewidth=2, label="Estimated value")
        ax.legend()
        ax.set_xlabel(f"θ[{idx}] value")
        ax.set_ylabel("Loss Function")
        ax.set_title(f"Loss Profile for θ[{idx}]")
        ax.grid(True)

    # hide unused subplots
    for ax in axes[n_params:]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.expanduser("Yule_Walker_profiles.pdf"), bbox_inches="tight")
    print("Saved all Yule Walker profiles to Yule_Walker_profiles.pdf")

    print("True theta:", theta_true_reparam)
    print("Estimated theta:", theta_hat_reparam)

    # ------------------------------------------------------------
    # EXTRA PLOTS: Profile W_a(eta_a) and W_b(eta_a)
    # ------------------------------------------------------------
    print("\n=== Plotting W_a and W_b Profiles with respect to eta ===")

    from matplotlib.backends.backend_pdf import PdfPages

    eta_min, eta_max = -2.0, 2.0
    n_eta = 100
    eta_a_grid = jnp.linspace(eta_min, eta_max, n_eta)

    Wa_vals = []

    for eta_a in eta_a_grid:
        Wa = build_attention(Z, eta_a, params_scale["eta_a"])
        Wa_vals.append(float(jnp.linalg.norm(Wa)))

    Wa_vals = jnp.array(Wa_vals)

    # --- Save two separate plots into ONE pdf ---
    with PdfPages("Yule_Walker_Wa_profiles.pdf") as pdf:

        # -------- Plot 1: W_a(eta_a) --------
        plt.figure(figsize=(7,5))
        plt.plot(eta_a_grid, Wa_vals, linewidth=2, label="‖W_a(eta_a)‖_F")
        plt.axvline(params_true["eta_a"], color="red", linestyle="--", label="True eta_a")
        plt.xlabel("eta_a")
        plt.ylabel("‖W_a(eta_a)‖_F")
        plt.title("Profile of W_a with respect to eta_a")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    print("Saved two attention profiles in one file: Yule_Walker_Wa_profiles.pdf")

    # breakpoint()
    # diff = jnp.stack([theta_true_reparam, theta_hat_reparam], axis=0)
    # print("Diff:", diff)
