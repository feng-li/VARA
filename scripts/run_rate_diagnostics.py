import os
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["RAY_memory_usage_threshold"] = "0.9"

import gc
import math
import time
from datetime import datetime

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd
import ray
from tqdm import trange

try:
    from ._bootstrap import RESULTS_DIR
except ImportError:
    from _bootstrap import RESULTS_DIR

from vara.dgp import (
    simulate_VAR_attention,
    draw_good_Z_for_eta,
    ensure_stationarity,
)
from vara.simulation import (
    estimate_eta_concentrated,
    estimate_eta_curvature as estimate_omega_gamma,
)


# =====================================================================
# 1. Rate utilities for convergence diagnostics
# =====================================================================

def convergence_rate_bound(T_eff, p):
    """
    Convergence rate for sqrt(T) * (eta_hat_c - eta):
        1 / sqrt(p) + sqrt(log(p) log(Tp) / T).
    """
    T_eff = float(T_eff)
    p = float(p)
    return 1.0 / math.sqrt(p) + math.sqrt(
        math.log(p) * math.log(T_eff * p) / T_eff
    )


def convergence_condition_numbers(T_eff, p):
    """
    Return the two rate-condition quantities for convergence diagnostics.

    Statement (i):  log(p) log(Tp) / T -> 0.
    Statement (ii): p log(p) log(Tp) / T -> 0.
    """
    T_eff = float(T_eff)
    p = float(p)
    c1 = math.log(p) * math.log(T_eff * p) / T_eff
    c2 = p * c1
    return c1, c2


# =====================================================================
# 2. Single replication
# =====================================================================

def convergence_single_replication(
    key,
    params_true,
    Z,
    T,
    p,
    q,
    params_scale,
    dist="normal",
    eta_bounds=(-2.0, 2.0),
    grid_size=81,
    curvature_step=0.02,
    ridge=1e-10,
):
    """
    One Monte Carlo replication for convergence diagnostics.

    These diagnostics study the concentrated estimator of the shared attention
    parameter. In the current code notation, paper gamma is eta_a.

    Returns a single-row dict with:
      - eta_hat_c and eta error,
      - convergence-rate diagnostics,
      - standardized statistics using omega at the truth
        and omega at the estimator.
    """
    params_true = dict(params_true)
    beta, gamma_a, stat_info = ensure_stationarity(
        params_true["beta"], params_true["gamma_a"], margin=0.98
    )
    params_true["beta"] = beta
    params_true["gamma_a"] = gamma_a

    Y, Z_used, _ = simulate_VAR_attention(
        key,
        params=params_true,
        T=T,
        p=p,
        q=q,
        dist=dist,
        params_scale=params_scale,
        Z=Z,
    )

    eta_true = float(params_true["eta_a"])
    T_eff = T - 1

    eta_hat, info = estimate_eta_concentrated(
        Y=Y,
        Z=Z_used,
        params_scale=params_scale,
        eta_bounds=eta_bounds,
        grid_size=grid_size,
        ridge=ridge,
    )
    eta_hat = float(eta_hat)
    eta_error = eta_hat - eta_true

    omega_true, l2_true = estimate_omega_gamma(
        eta_center=eta_true,
        Y=Y,
        Z=Z_used,
        p=p,
        params_scale=params_scale,
        h=curvature_step,
        eta_bounds=eta_bounds,
        ridge=ridge,
    )
    omega_hat, l2_hat = estimate_omega_gamma(
        eta_center=eta_hat,
        Y=Y,
        Z=Z_used,
        p=p,
        params_scale=params_scale,
        h=curvature_step,
        eta_bounds=eta_bounds,
        ridge=ridge,
    )

    omega_true_f = float(omega_true) if jnp.isfinite(omega_true) else float("nan")
    omega_hat_f = float(omega_hat) if jnp.isfinite(omega_hat) else float("nan")

    # Rate convergence: sqrt(T)(eta_hat - eta) has the displayed O_p rate.
    rate_i = convergence_rate_bound(T_eff, p)
    sqrtT_error = math.sqrt(T_eff) * eta_error
    rate_ratio = abs(sqrtT_error) / (rate_i + 1e-12)

    # Normality diagnostic: sqrt(Tp)(eta_hat - eta) -> N(0, omega^{-1}).
    sqrtTp_error = math.sqrt(T_eff * p) * eta_error

    z_trueomega = float("nan")
    if math.isfinite(omega_true_f) and omega_true_f > 0:
        z_trueomega = sqrtTp_error * math.sqrt(omega_true_f)

    z_hatomega = float("nan")
    if math.isfinite(omega_hat_f) and omega_hat_f > 0:
        z_hatomega = sqrtTp_error * math.sqrt(omega_hat_f)

    cond_i, cond_ii = convergence_condition_numbers(T_eff, p)

    return {
        "eta_true": eta_true,
        "eta_hat": eta_hat,
        "eta_error": eta_error,
        "abs_eta_error": abs(eta_error),
        "sqrtT_error": sqrtT_error,
        "sqrtTp_error": sqrtTp_error,
        "rate_i": rate_i,
        "rate_ratio": rate_ratio,
        "omega_true": omega_true_f,
        "omega_hat": omega_hat_f,
        "lc_second_true": float(l2_true) if jnp.isfinite(l2_true) else float("nan"),
        "lc_second_hat": float(l2_hat) if jnp.isfinite(l2_hat) else float("nan"),
        "z_trueomega": z_trueomega,
        "z_hatomega": z_hatomega,
        "grid_best_eta": float(info.get("grid_best_eta", float("nan"))),
        "objective": float(info.get("objective", float("nan"))),
        "on_boundary": bool(info.get("on_boundary", False)),
        "cond_i_loglog_over_T": cond_i,
        "cond_ii_p_loglog_over_T": cond_ii,
        "stationary_lhs_after": float(stat_info["lhs_after"]),
        "status": 0,
    }


def failed_convergence_row(T, p, q, dist):
    """Return one failed placeholder row."""
    T_eff = T - 1
    cond_i, cond_ii = convergence_condition_numbers(T_eff, p)
    return {
        "eta_true": float("nan"),
        "eta_hat": float("nan"),
        "eta_error": float("nan"),
        "abs_eta_error": float("nan"),
        "sqrtT_error": float("nan"),
        "sqrtTp_error": float("nan"),
        "rate_i": convergence_rate_bound(T_eff, p),
        "rate_ratio": float("nan"),
        "omega_true": float("nan"),
        "omega_hat": float("nan"),
        "lc_second_true": float("nan"),
        "lc_second_hat": float("nan"),
        "z_trueomega": float("nan"),
        "z_hatomega": float("nan"),
        "grid_best_eta": float("nan"),
        "objective": float("nan"),
        "on_boundary": True,
        "cond_i_loglog_over_T": cond_i,
        "cond_ii_p_loglog_over_T": cond_ii,
        "stationary_lhs_after": float("nan"),
        "status": -1,
    }


# =====================================================================
# 3. Ray actor
# =====================================================================

@ray.remote
class ConvergenceReplicationActor:
    def __init__(self):
        self._cache = None

    def run(self, *args, **kwargs):
        try:
            result = convergence_single_replication(*args, **kwargs)
            self._cache = None
            gc.collect()
            return result
        except Exception as e:
            print(f"[Ray actor error] {e}")
            T = kwargs.get("T", args[3] if len(args) > 3 else 0)
            p = kwargs.get("p", args[4] if len(args) > 4 else 0)
            q = kwargs.get("q", args[5] if len(args) > 5 else 0)
            dist = kwargs.get("dist", "unknown")
            return failed_convergence_row(T, p, q, dist)


# =====================================================================
# 4. Monte Carlo driver for one (T, p, q) setting
# =====================================================================

def run_convergence_verification(
    seed=2026,
    R=500,
    T=1000,
    p=10,
    q=50,
    dist="normal",
    eta_bounds=(-2.0, 2.0),
    grid_size=81,
    curvature_step=0.02,
    ray_num_cpus=52,
    max_cpus_per_task=1,
):
    """
    Run Monte Carlo convergence diagnostics for one (T, p, q) setting.

    Main outputs:
      - rate diagnostic: distribution of |sqrt(T)(eta_hat_c-eta)| / rate_i.
      - normality diagnostic: z = sqrt(Tp)(eta_hat_c-eta) sqrt(omega_hat),
        which should have mean about 0, variance about 1, and rejection
        probability about 0.05.
    """
    if ray_num_cpus < 1:
        raise ValueError("ray_num_cpus must be >= 1.")
    use_ray = ray_num_cpus > 1
    if use_ray and max_cpus_per_task > ray_num_cpus:
        raise ValueError("max_cpus_per_task cannot exceed ray_num_cpus in Ray mode.")

    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, 10)
    rep_keys = jax.random.split(keys[4], R)

    params_scale = {"eta_a": 1.0}

    params_true = {
        "eta_a": 0.5,
        # Keep the attention-loading coefficients away from zero to identify eta_a.
        "gamma_a": jax.random.uniform(keys[1], (p,), minval=0.15, maxval=0.30),
        "beta": jax.random.uniform(keys[2], (p,), minval=-0.20, maxval=0.20),
        "sigma2": jax.random.uniform(keys[3], (p,), minval=0.50, maxval=1.00),
    }
    beta, gamma_a, stat_info = ensure_stationarity(
        params_true["beta"], params_true["gamma_a"], margin=0.98
    )
    params_true["beta"] = beta
    params_true["gamma_a"] = gamma_a

    Z, Z_info = draw_good_Z_for_eta(
        key=keys[0],
        p=p,
        q=q,
        params_true=params_true,
        params_scale=params_scale,
        max_tries=200,
        min_offdiag_std=0.15,
        max_abs_score=3.0,
        min_eta_signal=0.05,
        verbose=True,
    )

    T_eff = T - 1
    cond_i, cond_ii = convergence_condition_numbers(T_eff, p)

    print("\n" + "=" * 70)
    print("Convergence verification setup")
    print("=" * 70)
    print(f"seed       = {seed}")
    print(f"R          = {R}")
    print(f"T_eff      = {T_eff}")
    print(f"p, q       = {p}, {q}")
    print(f"dist       = {dist}")
    print(f"eta_true   = {params_true['eta_a']}")
    print(f"eta_bounds = {eta_bounds}")
    print(f"grid_size  = {grid_size}")
    print(f"curv step  = {curvature_step}")
    print(f"condition (i)  log(p)log(Tp)/T      = {cond_i:.6f}")
    print(f"condition (ii) p log(p)log(Tp)/T    = {cond_ii:.6f}")
    print(f"stationarity lhs after              = {float(stat_info['lhs_after']):.6f}")
    print(f"Z_info     = {Z_info}")
    print("=" * 70 + "\n")

    if use_ray:
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=ray_num_cpus,
            _node_ip_address="127.0.0.1",
        )
        available = ray.available_resources()
        if float(available.get("CPU", 0.0)) < float(max_cpus_per_task):
            raise RuntimeError(
                f"Ray has no available CPUs after initialization: {available}. "
                "Stop stale Ray workers with `ray stop --force` or reduce other Ray jobs."
            )
        n_actors = max(1, min(R, int(ray_num_cpus / max_cpus_per_task)))
        print(f"Running with Ray using {n_actors} actors")
        actors = []
        progress = None
        rows = []
        try:
            actors = [
                ConvergenceReplicationActor.options(num_cpus=max_cpus_per_task).remote()
                for _ in range(n_actors)
            ]

            def submit_replication(rep):
                return actors[rep % n_actors].run.remote(
                    rep_keys[rep],
                    params_true,
                    Z,
                    T,
                    p,
                    q,
                    params_scale,
                    dist,
                    eta_bounds,
                    grid_size,
                    curvature_step,
                )

            max_pending = min(R, max(1, 2 * n_actors))
            print(f"Submitting Ray tasks with max_pending={max_pending}")
            pending = []
            submitted = 0
            completed = 0
            submit_times = {}
            progress = trange(R, desc="Convergence replications", ncols=100)
            progress_update = getattr(progress, "update", None)

            while submitted < R and len(pending) < max_pending:
                ref = submit_replication(submitted)
                pending.append(ref)
                submit_times[ref] = time.time()
                submitted += 1

            start_time = time.time()
            while pending:
                ready, pending = ray.wait(pending, num_returns=1, timeout=60.0)
                if not ready:
                    slowest_wait = time.time() - min(submit_times.values()) if submit_times else 0.0
                    print(
                        "Ray replication wait: "
                        f"done={completed}/{R}, submitted={submitted}/{R}, "
                        f"pending={len(pending)}, elapsed_sec={time.time() - start_time:.1f}, "
                        f"slowest_pending_sec={slowest_wait:.1f}, "
                        f"available={ray.available_resources()}",
                        flush=True,
                    )
                    continue

                for ref in ready:
                    rows.append(ray.get(ref))
                    submit_times.pop(ref, None)
                    completed += 1
                    if progress_update is not None:
                        progress_update(1)
                    while submitted < R and len(pending) < max_pending:
                        next_ref = submit_replication(submitted)
                        pending.append(next_ref)
                        submit_times[next_ref] = time.time()
                        submitted += 1
        finally:
            if progress is not None and hasattr(progress, "close"):
                progress.close()
            for actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception:
                    pass
            ray.shutdown()
    else:
        print("Running sequentially")
        rows = []
        for r in trange(R, desc="Convergence replications", ncols=100):
            rows.append(
                convergence_single_replication(
                    rep_keys[r],
                    params_true,
                    Z,
                    T,
                    p,
                    q,
                    params_scale,
                    dist=dist,
                    eta_bounds=eta_bounds,
                    grid_size=grid_size,
                    curvature_step=curvature_step,
                )
            )

    df = pd.DataFrame(rows)
    df["rep"] = range(R)
    df["seed"] = seed
    df["T"] = T
    df["T_eff"] = T_eff
    df["p"] = p
    df["q"] = q
    df["dist"] = dist
    df["Z_offdiag_std"] = Z_info.get("offdiag_std", float("nan"))
    df["Z_max_abs_score"] = Z_info.get("max_abs_score", float("nan"))
    df["Z_eta_signal"] = Z_info.get("eta_signal", float("nan"))

    df_valid = df[df["status"] == 0].copy()

    print("\n" + "=" * 70)
    print("Convergence rate diagnostics")
    print("Target: rate_ratio = O_p(1), not exploding across larger settings")
    print("=" * 70)
    rate_summary = {
        "valid_rate": len(df_valid) / max(len(df), 1),
        "eta_bias": df_valid["eta_error"].mean(),
        "eta_rmse": math.sqrt((df_valid["eta_error"] ** 2).mean()),
        "sqrtT_rmse": math.sqrt((df_valid["sqrtT_error"] ** 2).mean()),
        "sqrtTp_rmse": math.sqrt((df_valid["sqrtTp_error"] ** 2).mean()),
        "rate_ratio_mean": df_valid["rate_ratio"].mean(),
        "rate_ratio_median": df_valid["rate_ratio"].median(),
        "rate_ratio_q90": df_valid["rate_ratio"].quantile(0.90),
        "rate_ratio_q95": df_valid["rate_ratio"].quantile(0.95),
        "boundary_share": df_valid["on_boundary"].mean(),
    }
    print(pd.Series(rate_summary).round(6).to_string())

    print("\n" + "=" * 70)
    print("Convergence normality diagnostics")
    print("Target: mean ~ 0, variance ~ 1, P(|Z|>1.96) ~ 0.05")
    print("=" * 70)
    z_rows = []
    for stat_name in ["z_trueomega", "z_hatomega"]:
        z = df_valid[stat_name].dropna()
        z_rows.append({
            "stat": stat_name,
            "n": len(z),
            "mean": z.mean(),
            "var": z.var(ddof=0),
            "P_abs_gt_1.96": (z.abs() > 1.959963984540054).mean(),
        })
    z_summary = pd.DataFrame(z_rows)
    print(z_summary.round(6).to_string(index=False))

    print("\n" + "=" * 70)
    print("Omega diagnostics")
    print("=" * 70)
    print(
        df_valid[["omega_true", "omega_hat", "lc_second_true", "lc_second_hat"]]
        .agg(["mean", "std", "min", "max"])
        .round(6)
        .to_string()
    )

    run_tag = datetime.now().strftime("%Y%m%d_%H.%M.%S")
    out_path = RESULTS_DIR / (
        f"results_convergence_R={R}_p={p}_q={q}_T={T}_seed={seed}_{run_tag}.csv"
    )
    df.to_csv(out_path, index=False)
    print(f"\nSaved detailed results to: {out_path}")

    return df, pd.Series(rate_summary), z_summary


# =====================================================================
# 5. CLI
# =====================================================================

def _parse_int_list(value):
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _format_config_values(values):
    values = list(dict.fromkeys(values))
    if not values:
        return "none"
    return "-".join(str(value) for value in values)


def _p_from_rule(T, rule):
    if rule == "sqrt":
        return int(math.floor(math.sqrt(T)))
    if rule == "logT":
        return max(2, int(math.floor(math.log(T))))
    raise ValueError("Unsupported p_rule. Use 'sqrt', 'logT', or pass --p-list.")


def summarize_convergence_grid(df):
    """
    Summarize convergence diagnostics across a grid of (T, p) settings.

    The table is designed to check the two main convergence diagnostics:

      1. Rate statement: eta_rmse should decrease and the rate_ratio
         distribution should remain bounded as T and p increase.
      2. Normality statement: z_hatomega and z_trueomega should have mean
         close to 0, variance close to 1, and rejection probability close to 0.05.

    Parameters
    ----------
    df : pandas.DataFrame
        Combined row-level simulation output returned by run_convergence_verification.

    Returns
    -------
    summary : pandas.DataFrame
        One row per (T, p, q, dist) setting.
    trend : pandas.DataFrame
        First/last/max diagnostics for selected metrics, useful for quickly
        checking whether quantities decrease or remain bounded.
    """
    if df.empty:
        print("No rows available for convergence grid summary.")
        return pd.DataFrame(), pd.DataFrame()

    group_cols = ["T", "T_eff", "p", "q", "dist"]
    rows = []

    for keys, sub in df.groupby(group_cols, sort=True):
        T, T_eff, p, q, dist = keys
        valid = sub[sub["status"] == 0].copy()

        row = {
            "T": T,
            "T_eff": T_eff,
            "p": p,
            "q": q,
            "dist": dist,
            "valid_rate": len(valid) / max(len(sub), 1),
            "cond_i_loglog_over_T": sub["cond_i_loglog_over_T"].iloc[0],
            "cond_ii_p_loglog_over_T": sub["cond_ii_p_loglog_over_T"].iloc[0],
        }

        if len(valid) == 0:
            rows.append(row)
            continue

        z_hat = valid["z_hatomega"].dropna()
        z_true = valid["z_trueomega"].dropna()

        row.update({
            # Rate / consistency diagnostics.
            "eta_bias": valid["eta_error"].mean(),
            "eta_rmse": math.sqrt((valid["eta_error"] ** 2).mean()),
            "sqrtT_rmse": math.sqrt((valid["sqrtT_error"] ** 2).mean()),
            "sqrtTp_rmse": math.sqrt((valid["sqrtTp_error"] ** 2).mean()),
            "rate_ratio_mean": valid["rate_ratio"].mean(),
            "rate_ratio_median": valid["rate_ratio"].median(),
            "rate_ratio_q90": valid["rate_ratio"].quantile(0.90),
            "rate_ratio_q95": valid["rate_ratio"].quantile(0.95),
            "boundary_share": valid["on_boundary"].mean(),

            # Normality diagnostics using estimated omega.
            "z_hatomega_n": len(z_hat),
            "z_hatomega_mean": z_hat.mean(),
            "z_hatomega_var": z_hat.var(ddof=0),
            "z_hatomega_reject_5": (z_hat.abs() > 1.959963984540054).mean(),

            # Same statistic using curvature at the true eta, mostly diagnostic
            "z_trueomega_n": len(z_true),
            "z_trueomega_mean": z_true.mean(),
            "z_trueomega_var": z_true.var(ddof=0),
            "z_trueomega_reject_5": (z_true.abs() > 1.959963984540054).mean(),

            # Curvature diagnostics
            "omega_hat_mean": valid["omega_hat"].mean(),
            "omega_hat_min": valid["omega_hat"].min(),
            "omega_true_mean": valid["omega_true"].mean(),
            "omega_true_min": valid["omega_true"].min(),
            "omega_hat_nonpos_share": (valid["omega_hat"] <= 0).mean(),
            "omega_true_nonpos_share": (valid["omega_true"] <= 0).mean(),

            # Fixed-design diagnostics
            "Z_offdiag_std": valid["Z_offdiag_std"].iloc[0],
            "Z_max_abs_score": valid["Z_max_abs_score"].iloc[0],
            "Z_eta_signal": valid["Z_eta_signal"].iloc[0],
        })
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["T", "p"]).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("Convergence grid summary across increasing T and p")
    print("Check: eta_rmse should decrease; rate ratios should stay bounded; z stats target N(0,1).")
    print("=" * 90)

    display_cols = [
        "T", "p", "q",
        "cond_i_loglog_over_T", "cond_ii_p_loglog_over_T",
        "valid_rate", "eta_bias", "eta_rmse",
        "sqrtT_rmse", "sqrtTp_rmse",
        "rate_ratio_mean", "rate_ratio_q90", "rate_ratio_q95",
        "boundary_share",
        "z_hatomega_mean", "z_hatomega_var", "z_hatomega_reject_5",
        "z_trueomega_mean", "z_trueomega_var", "z_trueomega_reject_5",
        "omega_hat_mean", "omega_hat_min", "omega_true_mean", "omega_true_min",
    ]
    existing_display_cols = [c for c in display_cols if c in summary.columns]
    print(summary[existing_display_cols].round(6).to_string(index=False))

    # Compact trend table: first, last, max, and last/first ratios.
    trend_metrics = [
        "cond_i_loglog_over_T",
        "cond_ii_p_loglog_over_T",
        "eta_rmse",
        "sqrtT_rmse",
        "sqrtTp_rmse",
        "rate_ratio_mean",
        "rate_ratio_q95",
        "z_hatomega_var",
        "z_hatomega_reject_5",
        "boundary_share",
    ]
    trend_rows = []
    for metric in trend_metrics:
        if metric not in summary.columns or summary[metric].dropna().empty:
            continue
        vals = summary[metric].astype(float)
        first = vals.iloc[0]
        last = vals.iloc[-1]
        trend_rows.append({
            "metric": metric,
            "first": first,
            "last": last,
            "max": vals.max(),
            "last_over_first": last / first if first not in (0, 0.0) else float("nan"),
            "nonincreasing": bool(vals.is_monotonic_decreasing),
        })

    trend = pd.DataFrame(trend_rows)
    print("\n" + "=" * 90)
    print("Convergence trend check")
    print("nonincreasing=True is ideal for RMSE / condition terms; rate ratios mainly need to stay bounded.")
    print("=" * 90)
    print(trend.round(6).to_string(index=False))

    return summary, trend


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run convergence diagnostics.")
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=52,
        help="Total Ray CPU budget. Use 1 for sequential execution.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--R", type=int, default=500)
    parser.add_argument("--T-list", type=str, default="500,1000,2000,4000")
    parser.add_argument("--p-list", type=str, default="")
    parser.add_argument("--p-rule", type=str, default="sqrt", choices=["sqrt", "logT"])
    parser.add_argument("--q", type=int, default=50)
    parser.add_argument("--dist", type=str, default="normal", choices=["normal", "laplace", "t5"])
    parser.add_argument("--grid-size", type=int, default=81)
    parser.add_argument("--curvature-step", type=float, default=0.02)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    args = parser.parse_args()

    T_values = _parse_int_list(args.T_list)
    if args.p_list:
        p_values = _parse_int_list(args.p_list)
        if len(p_values) != len(T_values):
            raise ValueError("--p-list must have the same length as --T-list.")
    else:
        p_values = [_p_from_rule(T, args.p_rule) for T in T_values]

    all_dfs = []
    for offset, (T, p) in enumerate(zip(T_values, p_values)):
        df, _, _ = run_convergence_verification(
            seed=args.seed + offset,
            R=args.R,
            T=T,
            p=p,
            q=args.q,
            dist=args.dist,
            grid_size=args.grid_size,
            curvature_step=args.curvature_step,
            ray_num_cpus=args.ray_num_cpus,
            max_cpus_per_task=args.max_cpus_per_task,
        )
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)

    summary, trend = summarize_convergence_grid(combined)

    run_tag = datetime.now().strftime("%Y%m%d_%H.%M.%S")
    config_tag = (
        f"R={args.R}_"
        f"p={_format_config_values(p_values)}_"
        f"q={args.q}_"
        f"T={_format_config_values(T_values)}_"
        f"seed={args.seed}_"
        f"{run_tag}"
    )
    combined_path = RESULTS_DIR / f"results_convergence_combined_{config_tag}.csv"
    summary_path = RESULTS_DIR / f"summary_convergence_grid_{config_tag}.csv"
    trend_path = RESULTS_DIR / f"summary_convergence_trend_{config_tag}.csv"

    combined.to_csv(combined_path, index=False)
    summary.to_csv(summary_path, index=False)
    trend.to_csv(trend_path, index=False)

    print(f"Saved combined results to: {combined_path}")
    print(f"Saved grid summary to: {summary_path}")
    print(f"Saved trend summary to: {trend_path}")
