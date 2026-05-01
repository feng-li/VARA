"""
Full VARA simulation driver for the paper experiments.

Code notation reminder:
    paper gamma  = code eta_a
    paper alpha  = code beta
    paper beta   = code gamma_a
    paper sigma2 = code sigma2
"""

from __future__ import annotations

import argparse
import gc
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("RAY_memory_usage_threshold", "0.9")

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import pandas as pd
import ray
from tqdm import trange

try:
    from ._bootstrap import RESULTS_DIR
except Exception:
    try:
        from _bootstrap import RESULTS_DIR
    except Exception:
        RESULTS_DIR = Path("results")

from vara.dgp import draw_Z, draw_good_Z_for_eta, ensure_stationarity
from vara.simulation import (
    PARAMS_SCALE,
    full_single_replication,
    make_paper_table,
    p_for_setting,
    pack_result_row,
    summarize_setting,
)


SIMULATION_ETA_A = 0.5
SIMULATION_GAMMA_A_MIN = 0.15
SIMULATION_GAMMA_A_MAX = 0.30
SIMULATION_BETA_MIN = -0.20
SIMULATION_BETA_MAX = 0.20
SIMULATION_SIGMA2_MIN = 0.50
SIMULATION_SIGMA2_MAX = 1.00
SIMULATION_STATIONARITY_MARGIN = 0.98

SIMULATION_Z_MAX_TRIES = 200
SIMULATION_Z_MIN_OFFDIAG_STD = 0.15
SIMULATION_Z_MAX_ABS_SCORE = 3.0
SIMULATION_Z_MIN_ETA_SIGNAL = 0.05


def _parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_str_list(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _unique_in_order(values: Sequence[Any]) -> List[Any]:
    seen = set()
    out = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _format_config_values(values: Sequence[Any]) -> str:
    values = _unique_in_order(values)
    if not values:
        return "none"
    return "-".join(str(value) for value in values)


def _full_simulation_config_tag(
    *,
    R: int,
    T_list: Sequence[int],
    settings: Sequence[str],
    q: int,
    seed: int,
    high_max_p: Optional[int],
    run_tag: str,
) -> str:
    p_values = []
    for setting in settings:
        for T in T_list:
            p = p_for_setting(T, setting)
            if setting == "high" and high_max_p is not None and p > high_max_p:
                continue
            p_values.append(p)

    return (
        f"R={R}_"
        f"p={_format_config_values(p_values)}_"
        f"q={q}_"
        f"T={_format_config_values(T_list)}_"
        f"seed={seed}_"
        f"{run_tag}"
    )


def _full_simulation_output_paths(
    results_dir: Path,
    *,
    R: int,
    T_list: Sequence[int],
    settings: Sequence[str],
    q: int,
    seed: int,
    high_max_p: Optional[int],
    run_tag: str,
) -> Tuple[Path, Path, Path]:
    config_tag = _full_simulation_config_tag(
        R=R,
        T_list=T_list,
        settings=settings,
        q=q,
        seed=seed,
        high_max_p=high_max_p,
        run_tag=run_tag,
    )
    return (
        results_dir / f"results_full_simulation_detail_{config_tag}.csv",
        results_dir / f"summary_full_simulation_{config_tag}.csv",
        results_dir / f"paper_table_full_simulation_{config_tag}.csv",
    )


def _simulation_reporting_output_paths(results_dir: Path, config_tag: str) -> Dict[str, Path]:
    return {
        "setup": results_dir / f"table_simulation_setup_{config_tag}.csv",
        "low_rmse": results_dir / f"table_simulation_low_rmse_{config_tag}.csv",
        "low_ecp": results_dir / f"table_simulation_low_ecp_{config_tag}.csv",
        "high_rmse": results_dir / f"table_simulation_high_rmse_{config_tag}.csv",
        "high_ecp": results_dir / f"table_simulation_high_ecp_{config_tag}.csv",
        "eta_diagnostics": results_dir / f"table_simulation_eta_diagnostics_{config_tag}.csv",
        "publication_wide": results_dir / f"table_simulation_publication_wide_{config_tag}.csv",
    }


def draw_simulation_dgp_params(key: jax.Array, p: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Draw the strengthened DGP used by the full simulation driver."""
    k_gamma, k_beta, k_sig = jax.random.split(key, 3)
    params_true = {
        "eta_a": jnp.asarray(SIMULATION_ETA_A),
        "gamma_a": jax.random.uniform(
            k_gamma, (p,), minval=SIMULATION_GAMMA_A_MIN, maxval=SIMULATION_GAMMA_A_MAX
        ),
        "beta": jax.random.uniform(
            k_beta, (p,), minval=SIMULATION_BETA_MIN, maxval=SIMULATION_BETA_MAX
        ),
        "sigma2": jax.random.uniform(
            k_sig, (p,), minval=SIMULATION_SIGMA2_MIN, maxval=SIMULATION_SIGMA2_MAX
        ),
    }
    beta, gamma_a, stat_info = ensure_stationarity(
        params_true["beta"],
        params_true["gamma_a"],
        margin=SIMULATION_STATIONARITY_MARGIN,
    )
    params_true["beta"] = beta
    params_true["gamma_a"] = gamma_a
    return params_true, stat_info


def make_simulation_setup_table(
    *,
    R: int,
    T_list: Sequence[int],
    settings: Sequence[str],
    dist_list: Sequence[str],
    q: int,
    high_max_p: Optional[int],
    screen_z: bool,
) -> pd.DataFrame:
    rows = []
    z_rule = (
        f"max_tries={SIMULATION_Z_MAX_TRIES}; "
        f"min_offdiag_std={SIMULATION_Z_MIN_OFFDIAG_STD}; "
        f"max_abs_score={SIMULATION_Z_MAX_ABS_SCORE}; "
        f"min_eta_signal={SIMULATION_Z_MIN_ETA_SIGNAL}"
    )
    for setting in settings:
        p_rule = "floor(sqrt(T))" if setting == "low" else "floor(T^1.5)"
        for dist in dist_list:
            for T in T_list:
                p = p_for_setting(T, setting)
                if setting == "high" and high_max_p is not None and p > high_max_p:
                    continue
                rows.append({
                    "setting": setting,
                    "p_rule": p_rule,
                    "dist": dist,
                    "T": int(T),
                    "p": int(p),
                    "q": int(q),
                    "R": int(R),
                    "gamma": SIMULATION_ETA_A,
                    "alpha_range": f"U({SIMULATION_BETA_MIN:.2f},{SIMULATION_BETA_MAX:.2f})",
                    "beta_range": f"U({SIMULATION_GAMMA_A_MIN:.2f},{SIMULATION_GAMMA_A_MAX:.2f})",
                    "sigma2_range": f"U({SIMULATION_SIGMA2_MIN:.2f},{SIMULATION_SIGMA2_MAX:.2f})",
                    "stationarity_margin": SIMULATION_STATIONARITY_MARGIN,
                    "Z_screening": bool(screen_z),
                    "Z_screening_rule": z_rule if screen_z else "none",
                })
    return pd.DataFrame(rows)


def make_simulation_reporting_tables(summary: pd.DataFrame, setup: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Build publication-oriented simulation reporting tables."""
    id_cols = ["setting", "dist", "T", "p", "q", "method"]
    rmse_cols = ["RMSE_gamma", "RMSE_alpha", "RMSE_beta", "RMSE_sigma2"]
    ecp_cols = ["ECP_gamma", "ECP_alpha", "ECP_beta", "ECP_sigma2"]
    high_ecp_cols = ["ECP_alpha", "ECP_beta", "ECP_sigma2"]
    diag_cols = [
        "valid_rate",
        "eta_boundary_rate",
        "Z_screened",
        "Z_accepted",
        "Z_attempt",
        "Z_offdiag_std",
        "Z_max_abs_score",
        "Z_eta_signal",
        "Z_eta_info",
        "stationarity_rescaled",
        "stationary_lhs_after",
    ]

    if summary.empty:
        empty = pd.DataFrame()
        return {
            "setup": setup,
            "low_rmse": empty,
            "low_ecp": empty,
            "high_rmse": empty,
            "high_ecp": empty,
            "eta_diagnostics": empty,
            "publication_wide": empty,
        }

    def select(setting: str, cols: Sequence[str]) -> pd.DataFrame:
        existing = [c for c in [*id_cols, *cols] if c in summary.columns]
        return (
            summary.loc[summary["setting"] == setting, existing]
            .sort_values(["dist", "T", "method"])
            .reset_index(drop=True)
        )

    publication_cols = [c for c in [*id_cols, *rmse_cols, *ecp_cols, *diag_cols] if c in summary.columns]
    diagnostics_cols = [c for c in [*id_cols, *diag_cols] if c in summary.columns]

    return {
        "setup": setup,
        "low_rmse": select("low", rmse_cols),
        "low_ecp": select("low", ecp_cols),
        "high_rmse": select("high", rmse_cols),
        "high_ecp": select("high", high_ecp_cols),
        "eta_diagnostics": (
            summary.loc[:, diagnostics_cols]
            .sort_values(["setting", "dist", "T", "method"])
            .reset_index(drop=True)
        ),
        "publication_wide": (
            summary.loc[:, publication_cols]
            .sort_values(["setting", "dist", "T", "method"])
            .reset_index(drop=True)
        ),
    }


@ray.remote
class FullSimulationActor:
    def run(self, *args, **kwargs):
        try:
            out = full_single_replication(*args, **kwargs)
            gc.collect()
            return out
        except Exception as e:
            rep = kwargs.get("rep", args[0] if len(args) > 0 else -1)
            methods = kwargs.get("methods", ["unknown"])
            setting = kwargs.get("setting", "unknown")
            T = kwargs.get("T", -1)
            p = kwargs.get("p", -1)
            q = kwargs.get("q", -1)
            dist = kwargs.get("dist", "unknown")
            seed = kwargs.get("seed", -1)
            rows = []
            dummy_params = {
                "eta_a": jnp.asarray(float("nan")),
                "gamma_a": jnp.full((max(p, 1),), jnp.nan),
                "beta": jnp.full((max(p, 1),), jnp.nan),
                "sigma2": jnp.full((max(p, 1),), jnp.nan),
            }
            for method in methods:
                rows.append(pack_result_row(
                    setting=setting,
                    method=method,
                    rep=rep,
                    T=T,
                    p=p,
                    q=q,
                    dist=dist,
                    seed=seed,
                    status=-1,
                    elapsed_sec=math.nan,
                    params_true=dummy_params,
                    extra={"error": repr(e)},
                ))
            return rows


def run_full_simulation(
    *,
    seed: int = 2026,
    R: int = 500,
    T_list: Sequence[int] = (100, 200, 500),
    settings: Sequence[str] = ("low", "high"),
    dist_list: Sequence[str] = ("normal", "laplace"),
    q: int = 50,
    alpha: float = 0.05,
    ray_num_cpus: int = 52,
    max_cpus_per_task: int = 1,
    qmle_algorithm: str = "lbfgs",
    qmle_perturb_scale: float = 0.05,
    eta_bounds: Tuple[float, float] = (-2.0, 2.0),
    grid_size: int = 41,
    high_grid_size: Optional[int] = None,
    curvature_step: float = 0.02,
    ridge: float = 1e-10,
    batch_size: int = 512,
    dense_threshold: int = 2500,
    screen_z: bool = True,
    high_max_p: Optional[int] = None,
    results_dir: Path = RESULTS_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full simulation grid and save detailed/summary CSVs."""
    if ray_num_cpus < 1:
        raise ValueError("ray_num_cpus must be >= 1.")
    use_ray = ray_num_cpus > 1
    if use_ray and max_cpus_per_task > ray_num_cpus:
        raise ValueError("max_cpus_per_task cannot exceed ray_num_cpus in Ray mode.")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    key = jax.random.PRNGKey(seed)
    all_detail_rows: List[Dict[str, Any]] = []
    setting_summaries: List[pd.DataFrame] = []
    run_tag = datetime.now().strftime("%Y%m%d_%H.%M.%S")
    config_tag = _full_simulation_config_tag(
        R=R,
        T_list=T_list,
        settings=settings,
        q=q,
        seed=seed,
        high_max_p=high_max_p,
        run_tag=run_tag,
    )
    detail_path, summary_path, paper_path = _full_simulation_output_paths(
        results_dir,
        R=R,
        T_list=T_list,
        settings=settings,
        q=q,
        seed=seed,
        high_max_p=high_max_p,
        run_tag=run_tag,
    )
    reporting_paths = _simulation_reporting_output_paths(results_dir, config_tag)
    setup_table = make_simulation_setup_table(
        R=R,
        T_list=T_list,
        settings=settings,
        dist_list=dist_list,
        q=q,
        high_max_p=high_max_p,
        screen_z=screen_z,
    )

    for setting in settings:
        if setting not in {"low", "high"}:
            raise ValueError("settings must contain only 'low' and/or 'high'")

        for dist in dist_list:
            for T in T_list:
                p = p_for_setting(T, setting)
                if setting == "high" and high_max_p is not None and p > high_max_p:
                    print(f"[skip] high-dimensional setting T={T} gives p={p} > high_max_p={high_max_p}")
                    continue

                methods = ["two_stage"] if setting == "high" else ["qmle", "two_stage"]
                grid_size_use = int(
                    high_grid_size
                    if (setting == "high" and high_grid_size is not None)
                    else grid_size
                )

                key, k_params, k_Z, k_rep = jax.random.split(key, 4)
                params_true, stat_info = draw_simulation_dgp_params(k_params, p)
                if screen_z:
                    Z, Z_info = draw_good_Z_for_eta(
                        key=k_Z,
                        p=p,
                        q=q,
                        params_true=params_true,
                        params_scale=PARAMS_SCALE,
                        max_tries=SIMULATION_Z_MAX_TRIES,
                        min_offdiag_std=SIMULATION_Z_MIN_OFFDIAG_STD,
                        max_abs_score=SIMULATION_Z_MAX_ABS_SCORE,
                        min_eta_signal=SIMULATION_Z_MIN_ETA_SIGNAL,
                        verbose=True,
                    )
                else:
                    Z = draw_Z(k_Z, p, q)
                    Z_info = {"screened": False}

                print("\n" + "=" * 80)
                print("Full VARA simulation setting")
                print("=" * 80)
                print(f"setting      : {setting}")
                print(f"dist         : {dist}")
                print(f"T, p, q      : {T}, {p}, {q}")
                print(f"R            : {R}")
                print(f"methods      : {methods}")
                print(f"seed         : {seed}")
                print(f"grid_size    : {grid_size_use}")
                print(f"ray_num_cpus : {ray_num_cpus} ({'ray' if use_ray else 'sequential'})")
                print(f"stationary   : {not bool(stat_info['rescaled'])}")
                print(f"screen_z     : {screen_z}, Z_info={Z_info}")
                print("=" * 80)

                rep_keys = jax.random.split(k_rep, R)
                setting_start = datetime.now()

                if use_ray:
                    ray.init(
                        ignore_reinit_error=True,
                        include_dashboard=False,
                        num_cpus=ray_num_cpus,
                        _node_ip_address="127.0.0.1",
                    )
                    n_actors = max(1, min(R, int(ray_num_cpus / max_cpus_per_task)))
                    actors = [
                        FullSimulationActor.options(num_cpus=max_cpus_per_task).remote()
                        for _ in range(n_actors)
                    ]
                    futures = [
                        actors[r % n_actors].run.remote(
                            rep=r,
                            key=rep_keys[r],
                            params_true=params_true,
                            Z=Z,
                            setting=setting,
                            T=T,
                            p=p,
                            q=q,
                            dist=dist,
                            seed=seed,
                            methods=methods,
                            alpha=alpha,
                            qmle_algorithm=qmle_algorithm,
                            qmle_perturb_scale=qmle_perturb_scale,
                            eta_bounds=eta_bounds,
                            grid_size=grid_size_use,
                            curvature_step=curvature_step,
                            ridge=ridge,
                            batch_size=batch_size,
                            dense_threshold=dense_threshold,
                        )
                        for r in range(R)
                    ]
                    setting_rows: List[Dict[str, Any]] = []
                    for r in trange(R, desc=f"{setting}/{dist}/T={T}", ncols=100):
                        setting_rows.extend(ray.get(futures[r]))
                    ray.shutdown()
                else:
                    setting_rows = []
                    for r in trange(R, desc=f"{setting}/{dist}/T={T}", ncols=100):
                        setting_rows.extend(full_single_replication(
                            rep=r,
                            key=rep_keys[r],
                            params_true=params_true,
                            Z=Z,
                            setting=setting,
                            T=T,
                            p=p,
                            q=q,
                            dist=dist,
                            seed=seed,
                            methods=methods,
                            alpha=alpha,
                            qmle_algorithm=qmle_algorithm,
                            qmle_perturb_scale=qmle_perturb_scale,
                            eta_bounds=eta_bounds,
                            grid_size=grid_size_use,
                            curvature_step=curvature_step,
                            ridge=ridge,
                            batch_size=batch_size,
                            dense_threshold=dense_threshold,
                        ))

                elapsed_setting = (datetime.now() - setting_start).total_seconds()
                for row in setting_rows:
                    row["setting_elapsed_sec"] = elapsed_setting
                    row["Z_screened"] = bool(screen_z)
                    row["Z_accepted"] = bool(Z_info.get("accepted", False))
                    row["Z_attempt"] = Z_info.get("attempt", math.nan)
                    row["Z_offdiag_std"] = Z_info.get("offdiag_std", math.nan)
                    row["Z_max_abs_score"] = Z_info.get("max_abs_score", math.nan)
                    row["Z_eta_signal"] = Z_info.get("eta_signal", math.nan)
                    row["Z_eta_info"] = Z_info.get("eta_info", math.nan)
                    row["stationarity_rescaled"] = bool(stat_info["rescaled"])
                    row["stationary_lhs_before"] = float(stat_info["lhs_before"])
                    row["stationary_lhs_after"] = float(stat_info["lhs_after"])

                all_detail_rows.extend(setting_rows)
                df_setting = pd.DataFrame(setting_rows)
                summary_current = summarize_setting(df_setting)
                setting_summaries.append(summary_current)

                print("\nSummary for current setting:")
                print(summary_current.round(4).to_string(index=False))

                pd.DataFrame(all_detail_rows).to_csv(detail_path, index=False)
                pd.concat(setting_summaries, ignore_index=True).to_csv(summary_path, index=False)
                print(f"Saved incremental detail to: {detail_path}")
                print(f"Saved incremental summary to: {summary_path}")

                gc.collect()

    df_detail = pd.DataFrame(all_detail_rows)
    df_summary = pd.concat(setting_summaries, ignore_index=True) if setting_summaries else pd.DataFrame()
    df_paper = make_paper_table(df_summary) if len(df_summary) else pd.DataFrame()
    df_reporting = make_simulation_reporting_tables(df_summary, setup_table)

    df_detail.to_csv(detail_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    df_paper.to_csv(paper_path, index=False)
    for name, table in df_reporting.items():
        table.to_csv(reporting_paths[name], index=False)

    print("\n" + "=" * 80)
    print("Full VARA simulation summary")
    print("=" * 80)
    if len(df_summary):
        print(df_summary.round(4).to_string(index=False))
    print("\nPaper-style table:")
    if len(df_paper):
        print(df_paper.round(4).to_string(index=False))
    print("=" * 80)
    print(f"Detailed results : {detail_path}")
    print(f"Summary results  : {summary_path}")
    print(f"Paper table      : {paper_path}")
    print("Simulation reporting tables:")
    for name, path in reporting_paths.items():
        print(f"  {name:<16s}: {path}")

    return df_detail, df_summary, df_paper


def main():
    parser = argparse.ArgumentParser(description="Run full VARA paper simulations.")
    parser.add_argument(
        "--ray-num-cpus",
        type=int,
        default=52,
        help="Total Ray CPU budget. Use 1 to disable Ray and run sequentially.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--R", type=int, default=500)
    parser.add_argument("--T-list", type=str, default="100,200,500")
    parser.add_argument("--settings", type=str, default="low,high", help="Comma list: low,high")
    parser.add_argument("--dist-list", type=str, default="normal,laplace")
    parser.add_argument("--q", type=int, default=50)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)

    parser.add_argument("--qmle-algorithm", type=str, default="lbfgs")
    parser.add_argument("--qmle-perturb-scale", type=float, default=0.05)
    parser.add_argument("--eta-lower", type=float, default=-2.0)
    parser.add_argument("--eta-upper", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=41)
    parser.add_argument("--high-grid-size", type=int, default=None)
    parser.add_argument("--curvature-step", type=float, default=0.02)
    parser.add_argument("--ridge", type=float, default=1e-10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--dense-threshold", type=int, default=2500)
    parser.add_argument(
        "--screen-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Screen Z for eta identification. Use --no-screen-z to draw an unscreened design.",
    )
    parser.add_argument(
        "--high-max-p",
        type=int,
        default=None,
        help="Optional safety cap; skip high-dimensional settings with p above this value.",
    )
    parser.add_argument("--results-dir", type=str, default=str(RESULTS_DIR))
    args = parser.parse_args()

    run_full_simulation(
        seed=args.seed,
        R=args.R,
        T_list=_parse_int_list(args.T_list),
        settings=_parse_str_list(args.settings),
        dist_list=_parse_str_list(args.dist_list),
        q=args.q,
        alpha=args.alpha,
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
        qmle_algorithm=args.qmle_algorithm,
        qmle_perturb_scale=args.qmle_perturb_scale,
        eta_bounds=(args.eta_lower, args.eta_upper),
        grid_size=args.grid_size,
        high_grid_size=args.high_grid_size,
        curvature_step=args.curvature_step,
        ridge=args.ridge,
        batch_size=args.batch_size,
        dense_threshold=args.dense_threshold,
        screen_z=args.screen_z,
        high_max_p=args.high_max_p,
        results_dir=Path(args.results_dir),
    )


if __name__ == "__main__":
    main()
