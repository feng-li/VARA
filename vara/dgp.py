try:
    from memory_profiler import profile
except ImportError:
    def profile(func):
        return func

import jax
import jax.numpy as jnp
import jax.lax as lax
from jax.random import multivariate_normal, normal
import numpy as np
import matplotlib.pyplot as plt
import psutil
import os

from .loglik_attn import build_attention
from .loglik_attn import loglik_reparam
from .hess import compute_Wa_Wb_dots
from .cov import Gamma0_stationary_with_lyapunov_doubling

def mem_gb():
    """Return the current process resident memory usage in gigabytes."""
    return psutil.Process(os.getpid()).memory_info().rss / 1024**3


def create_structured_features(key, p, q):
    """
    Create features with clear block structure for better identifiability.
    """
    Z = jax.random.normal(key, (p, q)) * 0.3

    # Create 2-3 clear groups
    n_groups = min(3, p // 2)
    group_size = p // n_groups

    for g in range(n_groups):
        start_idx = g * group_size
        end_idx = min((g + 1) * group_size, p)
        feature_start = (g * q) // n_groups
        feature_end = ((g + 1) * q) // n_groups

        # Each group has high values in different feature dimensions
        Z = Z.at[start_idx:end_idx, feature_start:feature_end].add(2.5)

    # Normalize to unit norm
    Z = Z / jnp.linalg.norm(Z, axis=1, keepdims=True)
    return Z


def ensure_stationarity(beta, gamma_a, margin=0.98, eps=1e-6):
    """
    Ensure the old-model stationarity condition:
        ||beta||_inf + ||gamma_a||_inf < 1

    If violated, jointly rescale beta and gamma_a so that the sum is below `margin`.

    Args:
        beta: jnp.ndarray, shape (p,)
        gamma_a: jnp.ndarray, shape (p,)
        margin: float, target upper bound after rescaling, should be < 1
        eps: float, small number to avoid division by zero

    Returns:
        beta_new: rescaled beta
        gamma_a_new: rescaled gamma_a
        info: dict with diagnostics
    """
    beta_max = jnp.max(jnp.abs(beta))
    gamma_max = jnp.max(jnp.abs(gamma_a))
    lhs = beta_max + gamma_max

    needs_rescale = lhs >= margin
    scale = jnp.where(needs_rescale, margin / (lhs + eps), 1.0)

    beta_new = beta * scale
    gamma_a_new = gamma_a * scale

    info = {
        "beta_max": beta_max,
        "gamma_max": gamma_max,
        "lhs_before": lhs,
        "scale": scale,
        "lhs_after": jnp.max(jnp.abs(beta_new)) + jnp.max(jnp.abs(gamma_a_new)),
        "was_stationary": lhs < 1.0,
        "rescaled": needs_rescale,
    }

    return beta_new, gamma_a_new, info


def build_param_name_map(block_order):
    """
    Build a dictionary mapping each global parameter index
    to a readable parameter name based on block_order.

    Scalar blocks keep their original name, while vector blocks
    are labeled as name_1, name_2, ...

    Example
    -------
    block_order = {
        "eta_a": [0],
        "gamma_a": [1, 2],
        "beta": [3, 4],
        "sigma2": [5, 6],
    }

    returns
    -------
    {
        0: "eta_a",
        1: "gamma_a_1",
        2: "gamma_a_2",
        3: "beta_1",
        4: "beta_2",
        5: "sigma2_1",
        6: "sigma2_2",
    }
    """
    name_map = {}
    for param_name, block in block_order.items():
        block = list(block)
        if len(block) == 1:
            name_map[int(block[0])] = param_name
        else:
            for local_pos, global_idx in enumerate(block, start=1):
                name_map[int(global_idx)] = f"{param_name}_{local_pos}"
    return name_map


def draw_Z(key, p, q):
    """
    Draw Z according to the paper design:
        Z_k ~ N(0, Sigma_z),  Sigma_z[j1,j2] = 0.5^{|j1-j2|}
    """
    idx = jnp.arange(q)
    Sigma_z = 0.5 ** jnp.abs(idx[:, None] - idx[None, :])
    Z = multivariate_normal(key, jnp.zeros(q), Sigma_z, (p,))
    return Z


def draw_good_Z_for_eta(
    key,
    p,
    q,
    params_true,
    params_scale,
    max_tries=200,
    min_offdiag_std=0.15,
    max_abs_score=3.0,
    min_eta_signal=0.1,
    verbose=True,
):
    """
    Draw a fixed-design Z matrix that is informative enough for eta_a.

    Why this helps
    --------------
    eta_a is identified only through the attention map W_a(eta_a), and in practice
    its information is driven by gamma_a * W_a_dot. Some randomly drawn Z designs
    make this direction nearly flat, which hurts Theorem 1 verification for eta_a.

    Selection rule
    ---------------
    Evaluate all ``max_tries`` candidates and select the candidate with the
    largest eta information proxy among designs that pass:
      1) off-diagonal similarity scores have enough spread,
      2) no score is too extreme,
      3) the eta-direction signal ||diag(gamma_a) W_dot|| is not too small.

    If no candidate passes all screens, fall back to the candidate with the
    largest eta information proxy among all draws.

    Returns
    -------
    Z : jax.Array, shape (p, q)
        Accepted feature matrix.
    info : dict
        Diagnostics for the accepted design.
    """
    best_Z = None
    best_info = None
    best_good_Z = None
    best_good_info = None
    best_good_score = float("-inf")
    best_any_score = float("-inf")

    gamma_a_raw = params_true["gamma_a"]
    beta_raw = params_true.get("beta", jnp.zeros_like(gamma_a_raw))
    sigma2 = params_true.get("sigma2", jnp.ones_like(gamma_a_raw))
    beta, gamma_a, _ = ensure_stationarity(beta_raw, gamma_a_raw, margin=0.98)
    eta_a = params_true["eta_a"]

    for attempt in range(max_tries):
        subkey = jax.random.fold_in(key, attempt)
        Z = draw_Z(subkey, p, q)

        S = (Z @ Z.T) / jnp.sqrt(q)
        off_mask = ~jnp.eye(p, dtype=bool)
        off = S[off_mask]

        Wa, Wdot, _ = compute_Wa_Wb_dots(
            Z, eta_a, params_scale["eta_a"]
        )

        eta_deriv = gamma_a[:, None] * Wdot
        G = (beta[:, None] * jnp.eye(p)) + (gamma_a[:, None] * Wa)
        Gamma0 = Gamma0_stationary_with_lyapunov_doubling(G, sigma2)
        sigma2_inv = 1.0 / sigma2

        eta_signal = jnp.linalg.norm(eta_deriv)
        eta_info = jnp.sum((sigma2_inv[:, None] * eta_deriv) * (eta_deriv @ Gamma0))
        off_std = jnp.std(off)
        max_score = jnp.max(jnp.abs(off))

        good_design = bool(
            (off_std > min_offdiag_std) and
            (max_score < max_abs_score) and
            (eta_signal > min_eta_signal)
        )

        info = {
            "attempt": attempt + 1,
            "offdiag_std": float(off_std),
            "max_abs_score": float(max_score),
            "eta_signal": float(eta_signal),
            "eta_info": float(eta_info),
            "accepted": good_design,
        }

        score_value = float(eta_info)
        if score_value > best_any_score:
            best_any_score = score_value
            best_Z = Z
            best_info = info

        if good_design and score_value > best_good_score:
            best_good_score = score_value
            best_good_Z = Z
            best_good_info = info

    if verbose:
        selected_info = best_good_info if best_good_info is not None else best_info
        message = "[Z design accepted: best eta information]" if best_good_info is not None else "[Z design fallback: no candidate passed screening]"
        print(
            f"{message} "
            f"attempt={selected_info['attempt']}, "
            f"offdiag_std={selected_info['offdiag_std']:.4f}, "
            f"max_abs_score={selected_info['max_abs_score']:.4f}, "
            f"eta_signal={selected_info['eta_signal']:.4f}, "
            f"eta_info={selected_info['eta_info']:.4f}"
        )

    if best_good_Z is not None:
        return best_good_Z, best_good_info
    return best_Z, best_info


# @profile
def simulate_VAR_attention(key, params, T=200, p=10, q=10, dist="normal",
                           params_scale=dict(eta_a=1), save=False, Z=None):
    """
    Simulate data from the VAR attention model.

    save:
        False -> do not save
        True  -> save to default path
        str   -> save to that file path

    Returns: (Y, Z, W_a)
    """
    mem_start = mem_gb()

    keys = jax.random.split(key, 7)
    # --------------------------------------------------
    # Feature matrix Z
    # --------------------------------------------------
    if Z is None:
        # default: paper-style DGP
        Z = draw_Z(keys[0], p, q)
    else:
        # sanity check
        assert Z.shape == (p, q), f"Z must have shape ({p},{q})"

    eta_a = params["eta_a"]
    gamma_a = params["gamma_a"]
    beta = params["beta"]
    sigma2 = params["sigma2"]

    beta, gamma_a, stat_info = ensure_stationarity(beta, gamma_a, margin=0.98)
    print("Stationarity check:", stat_info)

    W_a = build_attention(Z, eta_a, params_scale['eta_a'])
    G = (beta[:, None] * jnp.eye(p)) + (gamma_a[:, None] * W_a)

    if dist == "normal":
        eps_raw = normal(keys[5], (T, p))
    elif dist == "t5":
        k1, k2 = jax.random.split(keys[5])
        z = normal(k1, (T, p))
        u = jax.random.chisquare(k2, df=5, shape=(T, p))
        eps_raw = z / jnp.sqrt(u / 5.0)
        eps_raw = eps_raw * jnp.sqrt(3.0 / 5.0)
    else:
        raise ValueError("dist must be 'normal' or 't5'.")

    eps = eps_raw * jnp.sqrt(sigma2)

    def step(y_prev, eps_t):
        y_t = G @ y_prev + eps_t
        return y_t, y_t

    y0 = jnp.zeros((p,))
    _, Ys = lax.scan(step, y0, eps[1:])
    Y = jnp.vstack([y0[None, :], Ys])

    params = dict(eta_a=eta_a, gamma_a=gamma_a, beta=beta, sigma2=sigma2)

    mem_used = mem_gb() - mem_start
    print(f"DGP RAM used: {mem_used:.3f} GB")

    if save is not False:
        if save is True:
            save_path = f"T={T}_p={p}_q={q}.npz"
        elif isinstance(save, str):
            save_path = save
        else:
            raise ValueError("save must be False, True, or a file path string.")

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        save_obj = dict(
            Y=Y,
            Z=Z,
            W_a=W_a,
            params=params,
            T=T,
            p=p,
            q=q,
            dist=dist,
            key=key
        )

        np.savez_compressed(save_path, payload=np.array(save_obj, dtype=object))
        print(f"Saved simulated object to: {save_path}")

    return Y, Z, W_a


def flatten_params(params: dict, block_order: dict):
    """
    Flatten a parameter dictionary into a single 1D JAX array using
    the key order of `block_order`, and return parameter slices.

    Also checks that parameter lengths match the block sizes implied by block_order.
    """
    flat_list = []
    param_slices = {}

    start = 0
    for k, blk in block_order.items():
        if k not in params:
            continue

        v = params[k]
        v_arr = jnp.atleast_1d(v).ravel()

        expected_size = len(jnp.atleast_1d(blk))
        if v_arr.size != expected_size:
            raise ValueError(
                f"Parameter '{k}' has size {v_arr.size}, "
                f"but block_order implies size {expected_size}."
            )

        flat_list.append(v_arr)

        end = start + v_arr.size
        param_slices[k] = slice(start, end)
        start = end

    if len(flat_list) == 0:
        theta_flat = jnp.array([])
    else:
        theta_flat = jnp.concatenate(flat_list)

    return theta_flat, param_slices


def plot_likelihood_profiles(
    Y,
    Z,
    params_true,
    params_scale,
    loglik_fn,
    update_idx=slice(None),
    theta_init=None,
    theta_hat_dict=None,
    plot_natural_scale=True,
    plot_width=2.0,
    n_points=100,
    save_path=None,
):
    """
    Plot 1D likelihood profiles around true parameters.

    Parameters
    ----------
    Y, Z : arrays
        Data used in the likelihood.
    params_true : dict
        True parameters on natural scale.
    params_scale : dict
        Parameter scaling dict.
    loglik_fn : callable
        Typically loglik_reparam(theta, Y, Z, params_scale, update_idx).
    update_idx : slice or array-like
        Indices to plot.
    theta_init : 1D array, optional
        Initial parameter vector on reparameterized scale.
    theta_hat_dict : dict[str, 1D array], optional
        Estimated parameter vectors on reparameterized scale.
        Example:
            {
                "Full Newton": theta_hat_reparam_full,
                "4-block Newton": theta_hat_reparam_gibbs,
            }
    plot_natural_scale : bool
        If True, sigma2 is shown on natural scale.
    plot_width : float
        Half-width of grid around true value.
    n_points : int
        Number of grid points.
    save_path : str, optional
        If provided, save figure to this path.

    Returns
    -------
    fig, axes
    """
    import math
    import matplotlib.pyplot as plt

    p = Y.shape[1]

    block_order = {
        "eta_a": [0],
        "gamma_a": list(range(1, 1 + p)),
        "beta": list(range(1 + p, 1 + 2 * p)),
        "sigma2": list(range(1 + 2 * p, 1 + 3 * p)),
    }

    param_name_map = build_param_name_map(block_order)

    sigma2_slice = slice(1 + 2 * p, 1 + 3 * p)

    sigma2_tilde_true = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = params_true.copy()
    params_reparam["sigma2"] = sigma2_tilde_true

    theta_true_reparam, _ = flatten_params(params_reparam, block_order)

    if isinstance(update_idx, slice):
        update_idx = jnp.arange(theta_true_reparam.size)
    update_idx = jnp.array(update_idx)

    if plot_natural_scale:
        theta_true_plot, _ = flatten_params(params_true, block_order)
    else:
        theta_true_plot = theta_true_reparam

    if theta_init is not None:
        if plot_natural_scale:
            theta_init_plot = theta_init.at[sigma2_slice].set(
                jax.nn.softplus(theta_init[sigma2_slice])
            )
        else:
            theta_init_plot = theta_init
    else:
        theta_init_plot = None

    theta_hat_plot_dict = {}
    if theta_hat_dict is not None:
        for label, theta_hat in theta_hat_dict.items():
            if plot_natural_scale:
                theta_hat_plot = theta_hat.at[sigma2_slice].set(
                    jax.nn.softplus(theta_hat[sigma2_slice])
                )
            else:
                theta_hat_plot = theta_hat
            theta_hat_plot_dict[label] = theta_hat_plot

    n_params = len(update_idx)
    n_cols = min(4, n_params)
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten().tolist()

    styles = [
        ("blue", "-."),
        ("purple", "--"),
        ("orange", ":"),
        ("brown", "-"),
        ("black", "--"),
    ]

    for ax, idx in zip(axes, update_idx):
        idx = int(idx)

        ll_values = []

        if plot_natural_scale and (1 + 2 * p <= idx < 1 + 3 * p):
            theta_grid_reparam = jnp.linspace(
                theta_true_reparam[idx] - plot_width,
                theta_true_reparam[idx] + plot_width,
                n_points
            )
            theta_grid_plot = jax.nn.softplus(theta_grid_reparam)

            for val_reparam in theta_grid_reparam:
                theta_temp = theta_true_reparam.at[idx].set(val_reparam)
                ll = loglik_fn(theta_temp, Y, Z, params_scale, update_idx)
                ll_values.append(float(ll))

            ax.plot(theta_grid_plot, ll_values, linewidth=2, label="Likelihood")

        else:
            theta_grid = jnp.linspace(
                theta_true_plot[idx] - plot_width,
                theta_true_plot[idx] + plot_width,
                n_points
            )

            for val in theta_grid:
                theta_temp = theta_true_reparam.at[idx].set(val)
                ll = loglik_fn(theta_temp, Y, Z, params_scale, update_idx)
                ll_values.append(float(ll))

            ax.plot(theta_grid, ll_values, linewidth=2, label="Likelihood")

        if theta_init_plot is not None:
            ax.axvline(
                theta_init_plot[idx],
                color="green",
                linestyle=":",
                linewidth=2,
                label="Initial value"
            )

        ax.axvline(
            theta_true_plot[idx],
            color="red",
            linestyle="--",
            linewidth=2,
            label="True value"
        )

        for k, (label, theta_hat_plot) in enumerate(theta_hat_plot_dict.items()):
            color, linestyle = styles[k % len(styles)]
            ax.axvline(
                theta_hat_plot[idx],
                color=color,
                linestyle=linestyle,
                linewidth=2,
                label=label
            )

        param_name = param_name_map[idx]
        suffix = " (natural scale)" if (plot_natural_scale and "sigma2" in param_name) else ""
        ax.set_xlabel(f"{param_name} value{suffix}")
        ax.set_ylabel("Log-likelihood")
        ax.set_title(f"Likelihood Profile for {param_name}")
        ax.grid(True)
        ax.legend()

    for ax in axes[n_params:]:
        ax.axis("off")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(os.path.expanduser(save_path), bbox_inches="tight")
        print(f"Saved likelihood profiles to: {save_path}")

    return fig, axes


if __name__ == "__main__":

    import random

    # Set random number seed
    seed = random.randrange(10**9)
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 10)

    # Example parameter dictionary (in natural scale)
    T, p, q = 500, 10, 50

    # --- True parameters (in natural scale) ---
    params_true = dict(
        eta_a=0.5,
        gamma_a=jnp.linspace(-0.15, 0.15, p),
        beta=jnp.linspace(-0.20, 0.20, p),
        sigma2=jnp.exp(jnp.log(0.5) + 0.2 * jax.random.normal(keys[0], (p,)))  # positive σ²
    )

    # Ensure stationarity
    beta, gamma_a, stat_info = ensure_stationarity(params_true["beta"], params_true["gamma_a"], margin=0.98)
    params_true["beta"] = beta
    params_true["gamma_a"] = gamma_a
    print("Stationarity check:", stat_info)

    params_scale = dict(eta_a=1)

    # Simulate dataset
    Y, Z, W_a = simulate_VAR_attention(
        key, params=params_true, T=T, p=p, q=q, dist="normal",
        params_scale=params_scale, save=True
    )

    # data_path = f"T={T}_p={p}_q={q}.npz"
    # print(f"Using {data_path}")
    # with np.load(data_path, allow_pickle=True) as obj:
    #     data = obj["payload"].item()

    # Y = data["Y"]
    # Z = data["Z"]
    # W_a = data["W_a"]
    # params_true = data["params"]

    print("Simulated Y shape:", Y.shape)

    # Convert σ² → σ̃² for reparameterized likelihood
    sigma2_tilde_true = jnp.log(jnp.expm1(params_true["sigma2"]))
    params_reparam = params_true.copy()
    params_reparam["sigma2"] = sigma2_tilde_true

    # Flatten parameter vector θ̃
    block_order = {
        "eta_a": [0],
        "gamma_a": list(range(1, 1 + p)),
        "beta": list(range(1 + p, 1 + 2 * p)),
        "sigma2": list(range(1 + 2 * p, 1 + 3 * p)),
    }

    theta_tilde, param_slices = flatten_params(params_reparam, block_order)

    print("Parameter vector θ (reparameterized) length:", theta_tilde.shape[0])
    print("First few sigma2_tilde:", sigma2_tilde_true[:5])

    # ---------- Time series visualization ----------
    fig, axes = plt.subplots(10, 10, figsize=(20, 20), sharex=True)
    axes = axes.flatten()

    print("Plotting first 100 series")
    for i in range(100):
        axes[i].plot(Y[:, i], lw=1)
        axes[i].set_title(f"Series {i+1}", fontsize=8)
        axes[i].tick_params(axis="both", which="major", labelsize=6)

    plt.tight_layout()
    plt.show()

    # ---------- Likelihood profile plot ----------
    plot_likelihood_profiles(
        Y=Y,
        Z=Z,
        params_true=params_true,
        params_scale=params_scale,
        loglik_fn=loglik_reparam,
        update_idx=slice(None),
        theta_init=None,
        theta_hat_dict=None,
        plot_natural_scale=True,
        plot_width=2.0,
        n_points=100,
        save_path="likelihood_profiles_true.pdf",
    )

    plt.show()
