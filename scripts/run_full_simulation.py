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
    p_values = []
    for setting in settings:
        for T in T_list:
            p = p_for_setting(T, setting)
            if setting == "high" and high_max_p is not None and p > high_max_p:
                continue
            p_values.append(p)

    config_tag = (
        f"R={R}_"
        f"p={_format_config_values(p_values)}_"
        f"q={q}_"
        f"T={_format_config_values(T_list)}_"
        f"seed={seed}_"
        f"{run_tag}"
    )
    return (
        results_dir / f"results_full_simulation_detail_{config_tag}.csv",
        results_dir / f"summary_full_simulation_{config_tag}.csv",
        results_dir / f"paper_table_full_simulation_{config_tag}.csv",
    )


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

                key, k_Z, k_gamma, k_beta, k_sig, k_rep = jax.random.split(key, 6)
                params_true = {
                    "eta_a": jnp.asarray(0.5),
                    "gamma_a": jax.random.uniform(k_gamma, (p,), minval=0.15, maxval=0.30),
                    "beta": jax.random.uniform(k_beta, (p,), minval=-0.20, maxval=0.20),
                    "sigma2": jax.random.uniform(k_sig, (p,), minval=0.50, maxval=1.00),
                }
                beta, gamma_a, stat_info = ensure_stationarity(
                    params_true["beta"], params_true["gamma_a"], margin=0.98
                )
                params_true["beta"] = beta
                params_true["gamma_a"] = gamma_a
                if screen_z:
                    Z, Z_info = draw_good_Z_for_eta(
                        key=k_Z,
                        p=p,
                        q=q,
                        params_true=params_true,
                        params_scale=PARAMS_SCALE,
                        max_tries=200,
                        min_offdiag_std=0.15,
                        max_abs_score=3.0,
                        min_eta_signal=0.05,
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

    df_detail.to_csv(detail_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    df_paper.to_csv(paper_path, index=False)

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
    parser.add_argument("--dist-list", type=str, default="normal,laplace,t5")
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
