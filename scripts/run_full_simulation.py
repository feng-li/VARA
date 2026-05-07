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
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("RAY_memory_usage_threshold", "0.9")

import pandas as pd
from tqdm import trange

try:
    import ray
except Exception as exc:
    ray = None
    RAY_IMPORT_ERROR = exc
else:
    RAY_IMPORT_ERROR = None

try:
    from ._bootstrap import RESULTS_DIR
except Exception:
    try:
        from _bootstrap import RESULTS_DIR
    except Exception:
        RESULTS_DIR = Path("results")

def _ensure_runtime_imports() -> None:
    """Import JAX-dependent modules after optional Ray initialization."""
    if "jax" in globals():
        return

    import jax as _jax
    _jax.config.update("jax_enable_x64", True)
    import jax.numpy as _jnp

    from vara import attention as _attention
    from vara import dgp as _dgp
    from vara import simulation as _simulation

    globals().update({
        "jax": _jax,
        "jnp": _jnp,
        "build_attention": _attention.build_attention,
        "build_topk_attention_cache": _attention.build_topk_attention_cache,
        "evaluate_topk_cache_quality": _attention.evaluate_topk_cache_quality,
        "draw_Z": _dgp.draw_Z,
        "draw_good_Z_for_eta": _dgp.draw_good_Z_for_eta,
        "ensure_stationarity": _dgp.ensure_stationarity,
        "PARAMS_SCALE": _simulation.PARAMS_SCALE,
        "full_single_replication": _simulation.full_single_replication,
        "make_paper_table": _simulation.make_paper_table,
        "p_for_setting": _simulation.p_for_setting,
        "pack_result_row": _simulation.pack_result_row,
        "summarize_setting": _simulation.summarize_setting,
    })


SIMULATION_ETA_A = 1.0
SIMULATION_GAMMA_A_MIN = 0.15
SIMULATION_GAMMA_A_MAX = 0.30
SIMULATION_BETA_MIN = -0.20
SIMULATION_BETA_MAX = 0.20
SIMULATION_SIGMA2_MIN = 0.50
SIMULATION_SIGMA2_MAX = 1.00
SIMULATION_STATIONARITY_MARGIN = 0.98

SIMULATION_Z_MAX_TRIES = 200
SIMULATION_Z_MIN_OFFDIAG_STD = 0.15
SIMULATION_Z_MAX_ABS_SCORE = 5.0
SIMULATION_Z_MIN_ETA_SIGNAL = 0.05
SIMULATION_Z_SCORE_QUANTILE = 0.999
SIMULATION_Z_MIN_ETA_INFO = 0.05
SIMULATION_Z_MIN_ATTENTION_X2_STD = 0.02
SIMULATION_Z_MIN_ATTENTION_X2_STD_ADAPTIVE_FACTOR = 1.1
SIMULATION_Z_MAX_ATTENTION_EFFECTIVE_NEIGHBORS = 1000.0
SIMULATION_Z_MAX_ATTENTION_EFFECTIVE_NEIGHBOR_FRACTION = 0.25
SIMULATION_Z_APPROX_THRESHOLD = 5000
SIMULATION_Z_APPROX_ROWS = 512
SIMULATION_Z_APPROX_PAIRS = 200_000
SIMULATION_ATTENTION_MODE = "dense"
SIMULATION_ATTENTION_TOP_K = None
SIMULATION_ATTENTION_CACHE_MODE = "exact"
SIMULATION_ATTENTION_CACHE_CLUSTERS = 512
SIMULATION_ATTENTION_CACHE_PROBES = 4
SIMULATION_ATTENTION_CACHE_CANDIDATE_SIZE = None
SIMULATION_ATTENTION_CACHE_CANDIDATE_MULTIPLIER = 5.0
SIMULATION_ATTENTION_CACHE_DIAGNOSTIC_ROWS = 64


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


def _print_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


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
    _ensure_runtime_imports()
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


def _full_simulation_config_path(results_dir: Path, config_tag: str) -> Path:
    return results_dir / f"config_full_simulation_{config_tag}.json"


def draw_simulation_dgp_params(
    key: jax.Array,
    p: int,
    eta_a: float = SIMULATION_ETA_A,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Draw the strengthened DGP used by the full simulation driver."""
    _ensure_runtime_imports()
    k_gamma, k_beta, k_sig = jax.random.split(key, 3)
    params_true = {
        "eta_a": jnp.asarray(eta_a),
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
    precompute_dgp_attention: bool,
    require_good_z: bool,
    dgp_eta_a: float,
    z_max_tries: int,
    z_score_quantile: float,
    z_max_abs_score: float,
    z_min_eta_signal: float,
    z_min_eta_info: float,
    z_min_attention_x2_std: float,
    z_min_attention_x2_std_adaptive_factor: float,
    z_max_attention_effective_neighbors: float,
    z_max_attention_effective_neighbor_fraction: float,
    z_approx_threshold: int,
    z_approx_rows: int,
    z_approx_pairs: int,
    attention_mode: str,
    attention_top_k: Optional[int],
    attention_cache_mode: str,
    attention_cache_clusters: int,
    attention_cache_probes: int,
    attention_cache_candidate_size: Optional[int],
    attention_cache_candidate_multiplier: float,
    attention_cache_diagnostic_rows: int,
) -> pd.DataFrame:
    _ensure_runtime_imports()
    rows = []
    z_rule = (
        f"max_tries={z_max_tries}; "
        f"min_offdiag_std={SIMULATION_Z_MIN_OFFDIAG_STD}; "
        f"q{z_score_quantile:g}_abs_score<={z_max_abs_score}; "
        f"min_eta_signal={z_min_eta_signal}; "
        f"min_eta_info={z_min_eta_info}; "
        f"min_attention_x2_std={z_min_attention_x2_std}; "
        f"min_attention_x2_std_adaptive_factor={z_min_attention_x2_std_adaptive_factor}; "
        f"max_effective_neighbors=max({z_max_attention_effective_neighbors},"
        f"{z_max_attention_effective_neighbor_fraction}*p); "
        f"approx_threshold={z_approx_threshold}; "
        f"approx_rows={z_approx_rows}; "
        f"approx_pairs={z_approx_pairs}"
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
                    "gamma": dgp_eta_a,
                    "alpha_range": f"U({SIMULATION_BETA_MIN:.2f},{SIMULATION_BETA_MAX:.2f})",
                    "beta_range": f"U({SIMULATION_GAMMA_A_MIN:.2f},{SIMULATION_GAMMA_A_MAX:.2f})",
                    "sigma2_range": f"U({SIMULATION_SIGMA2_MIN:.2f},{SIMULATION_SIGMA2_MAX:.2f})",
                    "stationarity_margin": SIMULATION_STATIONARITY_MARGIN,
                    "Z_screening": bool(screen_z),
                    "Z_screening_rule": z_rule if screen_z else "none",
                    "Z_require_accepted": bool(require_good_z),
                    "attention_mode": attention_mode,
                    "attention_top_k": attention_top_k if attention_top_k is not None else "",
                    "attention_cache_mode": attention_cache_mode,
                    "attention_cache_clusters": attention_cache_clusters,
                    "attention_cache_probes": attention_cache_probes,
                    "attention_cache_candidate_size": (
                        attention_cache_candidate_size
                        if attention_cache_candidate_size is not None
                        else ""
                    ),
                    "attention_cache_candidate_multiplier": attention_cache_candidate_multiplier,
                    "attention_cache_diagnostic_rows": attention_cache_diagnostic_rows,
                    "DGP_attention_precomputed": bool(precompute_dgp_attention),
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
        "Z_parallel",
        "Z_parallel_workers",
        "Z_parallel_chunks",
        "Z_parallel_accepted_chunks",
        "Z_approximate",
        "Z_approx_rows",
        "Z_approx_pairs",
        "Z_approx_threshold",
        "Z_accepted",
        "Z_attempt",
        "Z_offdiag_std",
        "Z_max_abs_score",
        "Z_score_quantile",
        "Z_abs_score_quantile",
        "Z_eta_signal",
        "Z_eta_info",
        "Z_attention_x2_std_mean",
        "Z_attention_x2_std_min",
        "Z_attention_x2_std_requested_min",
        "Z_attention_x2_std_effective_min",
        "Z_attention_x2_std_adaptive_min",
        "Z_attention_x2_std_adaptive_factor",
        "Z_attention_effective_neighbors_mean",
        "Z_attention_effective_neighbors_p95",
        "Z_attention_effective_neighbors_limit",
        "Z_attention_max_weight_mean",
        "Z_selection_score",
        "DGP_attention_precomputed",
        "TopK_attention_cached",
        "attention_cache_mode",
        "attention_cache_clusters",
        "attention_cache_probes",
        "attention_cache_candidate_size",
        "TopK_cache_quality_rows",
        "TopK_cache_recall_pos_mean",
        "TopK_cache_recall_pos_min",
        "TopK_cache_score_loss_pos_mean",
        "TopK_cache_score_loss_pos_max",
        "TopK_cache_score_loss_pos_abs_mean",
        "TopK_cache_selected_valid_count_pos_mean",
        "TopK_cache_selected_valid_count_pos_min",
        "TopK_cache_selected_duplicate_pair_count_pos_mean",
        "TopK_cache_selected_duplicate_pair_count_pos_max",
        "TopK_cache_recall_neg_mean",
        "TopK_cache_recall_neg_min",
        "TopK_cache_score_loss_neg_mean",
        "TopK_cache_score_loss_neg_max",
        "TopK_cache_score_loss_neg_abs_mean",
        "TopK_cache_selected_valid_count_neg_mean",
        "TopK_cache_selected_valid_count_neg_min",
        "TopK_cache_selected_duplicate_pair_count_neg_mean",
        "TopK_cache_selected_duplicate_pair_count_neg_max",
        "attention_mode",
        "attention_top_k",
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


class FullSimulationActor:
    def run(self, *args, **kwargs):
        _ensure_runtime_imports()
        ray_debug = bool(kwargs.pop("ray_debug", False))
        rep = kwargs.get("rep", args[0] if len(args) > 0 else -1)
        setting = kwargs.get("setting", "unknown")
        T = kwargs.get("T", -1)
        p = kwargs.get("p", -1)
        q = kwargs.get("q", -1)
        methods = kwargs.get("methods", ["unknown"])
        started = time.time()
        if ray_debug:
            print(
                f"[Ray replication start] rep={rep}, setting={setting}, "
                f"T={T}, p={p}, q={q}, methods={methods}",
                flush=True,
            )
        try:
            if isinstance(kwargs.get("dgp_attention"), ray.ObjectRef):
                kwargs["dgp_attention"] = ray.get(kwargs["dgp_attention"])
            if isinstance(kwargs.get("attention_cache"), ray.ObjectRef):
                kwargs["attention_cache"] = ray.get(kwargs["attention_cache"])
            out = full_single_replication(*args, **kwargs)
            gc.collect()
            if ray_debug:
                statuses = [row.get("status") for row in out]
                print(
                    f"[Ray replication done] rep={rep}, statuses={statuses}, "
                    f"elapsed_sec={time.time() - started:.2f}",
                    flush=True,
                )
            return out
        except Exception as e:
            if ray_debug:
                print(
                    f"[Ray replication error] rep={rep}, elapsed_sec={time.time() - started:.2f}, "
                    f"error={repr(e)}",
                    flush=True,
                )
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


def draw_good_z_chunk_remote(
    *,
    key,
    chunk_idx: int,
    p: int,
    q: int,
    params_true,
    params_scale,
    attempt_start: int,
    max_tries: int,
    min_offdiag_std: float,
    max_abs_score: float,
    min_eta_signal: float,
    score_quantile: float,
    min_eta_info: float,
    min_attention_x2_std: float,
    min_attention_x2_std_adaptive_factor: Optional[float],
    max_attention_effective_neighbors: Optional[float],
    max_attention_effective_neighbor_fraction: Optional[float],
    approximate_threshold: Optional[int],
    approx_rows: int,
    approx_pairs: int,
    attention_mode: str,
    attention_top_k: Optional[int],
    ray_debug: bool = False,
):
    """Score a contiguous chunk of candidate Z designs."""
    _ensure_runtime_imports()
    started = time.time()
    if ray_debug:
        print(
            f"[Ray Z chunk start] chunk={chunk_idx}, attempt_start={attempt_start}, "
            f"tries={max_tries}, p={p}, q={q}, approx_rows={approx_rows}, "
            f"approx_pairs={approx_pairs}",
            flush=True,
        )
    Z, info = draw_good_Z_for_eta(
        key=key,
        p=p,
        q=q,
        params_true=params_true,
        params_scale=params_scale,
        max_tries=max_tries,
        min_offdiag_std=min_offdiag_std,
        max_abs_score=max_abs_score,
        min_eta_signal=min_eta_signal,
        score_quantile=score_quantile,
        min_eta_info=min_eta_info,
        min_attention_x2_std=min_attention_x2_std,
        min_attention_x2_std_adaptive_factor=min_attention_x2_std_adaptive_factor,
        max_attention_effective_neighbors=max_attention_effective_neighbors,
        max_attention_effective_neighbor_fraction=max_attention_effective_neighbor_fraction,
        attempt_start=attempt_start,
        approximate_threshold=approximate_threshold,
        approx_rows=approx_rows,
        approx_pairs=approx_pairs,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        require_accepted=False,
        verbose=False,
    )
    info = dict(info)
    info["parallel_chunk"] = chunk_idx
    info["parallel_attempt_start"] = attempt_start
    info["parallel_chunk_tries"] = max_tries
    if ray_debug:
        print(
            f"[Ray Z chunk done] chunk={chunk_idx}, accepted={bool(info.get('accepted', False))}, "
            f"best_attempt={info.get('attempt')}, eta_info={info.get('eta_info', math.nan):.4f}, "
            f"x2_std_mean={info.get('attention_x2_std_mean', math.nan):.4f}, "
            f"eff_neighbors_mean={info.get('attention_effective_neighbors_mean', math.nan):.1f}, "
            f"elapsed_sec={time.time() - started:.2f}",
            flush=True,
        )
    return Z, info


if ray is not None:
    FullSimulationActor = ray.remote(FullSimulationActor)
    draw_good_z_chunk_remote = ray.remote(draw_good_z_chunk_remote)


def _select_best_z_result(
    results: Sequence[Tuple[jnp.ndarray, Dict[str, Any]]],
    *,
    require_good_z: bool,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    accepted = [(Z, info) for Z, info in results if bool(info.get("accepted", False))]
    pool = accepted if accepted else list(results)
    if not pool:
        raise RuntimeError("No candidate Z was evaluated.")

    Z, info = max(pool, key=lambda item: float(item[1].get("selection_score", float("-inf"))))
    info = dict(info)
    info["parallel_chunks"] = len(results)
    info["parallel_accepted_chunks"] = len(accepted)

    if require_good_z and not accepted:
        best_msg = (
            f"Best rejected candidate: attempt={info.get('attempt')}, "
            f"offdiag_std={info.get('offdiag_std', math.nan):.4f}, "
            f"max_abs_score={info.get('max_abs_score', math.nan):.4f}, "
            f"q{info.get('score_quantile', math.nan):.3f}_abs_score={info.get('abs_score_quantile', math.nan):.4f}, "
            f"eta_signal={info.get('eta_signal', math.nan):.4f}, "
            f"eta_info={info.get('eta_info', math.nan):.4f}, "
            f"x2_std_mean={info.get('attention_x2_std_mean', math.nan):.4f}, "
            f"x2_std_min_required={info.get('attention_x2_std_effective_min', math.nan):.4f}, "
            f"eff_neighbors_mean={info.get('attention_effective_neighbors_mean', math.nan):.1f}, "
            f"eff_neighbors_limit={info.get('attention_effective_neighbors_limit', math.nan):.1f}, "
            f"approximate={bool(info.get('approximate', False))}, "
            f"approx_rows={info.get('approx_rows', math.nan)}, "
            f"approx_pairs={info.get('approx_pairs', math.nan)}, "
            f"parallel_chunks={len(results)}, "
            f"parallel_accepted_chunks={len(accepted)}."
        )
        raise RuntimeError(
            "No candidate Z passed eta-identification screening. "
            f"{best_msg} "
            "Try increasing eta_a, using structured/sparser attention, relaxing "
            "the screening thresholds, or increasing max_tries."
        )
    return Z, info


def _print_selected_z_info(info: Dict[str, Any]) -> None:
    message = (
        "[Z design accepted: best eta information]"
        if bool(info.get("accepted", False))
        else "[Z design fallback: no candidate passed screening]"
    )
    suffix = ""
    if bool(info.get("approximate", False)):
        suffix = (
            f", approximate=True, "
            f"approx_rows={info.get('approx_rows', math.nan)}, "
            f"approx_pairs={info.get('approx_pairs', math.nan)}"
        )
    print(
        f"{message} "
        f"attempt={info.get('attempt')}, "
        f"offdiag_std={info.get('offdiag_std', math.nan):.4f}, "
        f"max_abs_score={info.get('max_abs_score', math.nan):.4f}, "
        f"q{info.get('score_quantile', math.nan):.3f}_abs_score={info.get('abs_score_quantile', math.nan):.4f}, "
        f"eta_signal={info.get('eta_signal', math.nan):.4f}, "
        f"eta_info={info.get('eta_info', math.nan):.4f}, "
        f"x2_std_mean={info.get('attention_x2_std_mean', math.nan):.4f}, "
        f"x2_std_min_required={info.get('attention_x2_std_effective_min', math.nan):.4f}, "
        f"eff_neighbors_mean={info.get('attention_effective_neighbors_mean', math.nan):.1f}, "
        f"eff_neighbors_limit={info.get('attention_effective_neighbors_limit', math.nan):.1f}"
        f"{suffix}"
    )


def draw_good_Z_for_eta_with_ray(
    *,
    key,
    p: int,
    q: int,
    params_true,
    params_scale,
    max_tries: int,
    min_offdiag_std: float,
    max_abs_score: float,
    min_eta_signal: float,
    score_quantile: float,
    min_eta_info: float,
    min_attention_x2_std: float,
    min_attention_x2_std_adaptive_factor: Optional[float],
    max_attention_effective_neighbors: Optional[float],
    max_attention_effective_neighbor_fraction: Optional[float],
    approximate_threshold: Optional[int],
    approx_rows: int,
    approx_pairs: int,
    attention_mode: str,
    attention_top_k: Optional[int],
    require_accepted: bool,
    ray_num_cpus: int,
    max_cpus_per_task: int,
    ray_debug: bool = False,
    verbose: bool = True,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """Parallel candidate-Z screening using the active global Ray cluster."""
    if ray is None:
        raise RuntimeError(f"Ray could not be imported: {RAY_IMPORT_ERROR!r}")
    if max_tries < 1:
        raise ValueError("max_tries must be >= 1.")
    n_workers = max(1, min(max_tries, int(ray_num_cpus / max_cpus_per_task)))
    chunk_size = int(math.ceil(max_tries / n_workers))

    futures = []
    for worker_idx in range(n_workers):
        attempt_start = worker_idx * chunk_size
        tries = min(chunk_size, max_tries - attempt_start)
        if tries <= 0:
            continue
        futures.append(draw_good_z_chunk_remote.options(num_cpus=max_cpus_per_task).remote(
            key=key,
            chunk_idx=worker_idx,
            p=p,
            q=q,
            params_true=params_true,
            params_scale=params_scale,
            attempt_start=attempt_start,
            max_tries=tries,
            min_offdiag_std=min_offdiag_std,
            max_abs_score=max_abs_score,
            min_eta_signal=min_eta_signal,
            score_quantile=score_quantile,
            min_eta_info=min_eta_info,
            min_attention_x2_std=min_attention_x2_std,
            min_attention_x2_std_adaptive_factor=min_attention_x2_std_adaptive_factor,
            max_attention_effective_neighbors=max_attention_effective_neighbors,
            max_attention_effective_neighbor_fraction=max_attention_effective_neighbor_fraction,
            approximate_threshold=approximate_threshold,
            approx_rows=approx_rows,
            approx_pairs=approx_pairs,
            attention_mode=attention_mode,
            attention_top_k=attention_top_k,
            ray_debug=ray_debug,
        ))

    if ray_debug:
        pending = list(futures)
        results = []
        accepted_count = 0
        best_info = None
        started = time.time()
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=30.0)
            if not ready:
                _print_progress(
                    f"Z Ray progress: chunks_done={len(results)}/{len(futures)}, "
                    f"pending={len(pending)}, elapsed_sec={time.time() - started:.1f}"
                )
                continue
            for ref in ready:
                result = ray.get(ref)
                results.append(result)
                info = result[1]
                if bool(info.get("accepted", False)):
                    accepted_count += 1
                if (
                    best_info is None or
                    float(info.get("selection_score", float("-inf"))) >
                    float(best_info.get("selection_score", float("-inf")))
                ):
                    best_info = info
                _print_progress(
                    f"Z Ray progress: chunks_done={len(results)}/{len(futures)}, "
                    f"accepted_chunks={accepted_count}, "
                    f"best_eta_info={best_info.get('eta_info', math.nan):.4f}, "
                    f"best_x2_std={best_info.get('attention_x2_std_mean', math.nan):.4f}, "
                    f"best_eff_neighbors={best_info.get('attention_effective_neighbors_mean', math.nan):.1f}, "
                    f"elapsed_sec={time.time() - started:.1f}"
                )
    else:
        results = ray.get(futures)
    Z, info = _select_best_z_result(results, require_good_z=require_accepted)
    info["parallel"] = True
    info["parallel_workers"] = len(futures)
    if verbose:
        _print_selected_z_info(info)
    return Z, info


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
    precompute_dgp_attention: bool = False,
    require_good_z: bool = False,
    dgp_eta_a: float = SIMULATION_ETA_A,
    z_max_tries: int = SIMULATION_Z_MAX_TRIES,
    z_score_quantile: float = SIMULATION_Z_SCORE_QUANTILE,
    z_max_abs_score: float = SIMULATION_Z_MAX_ABS_SCORE,
    z_min_eta_signal: float = SIMULATION_Z_MIN_ETA_SIGNAL,
    z_min_eta_info: float = SIMULATION_Z_MIN_ETA_INFO,
    z_min_attention_x2_std: float = SIMULATION_Z_MIN_ATTENTION_X2_STD,
    z_min_attention_x2_std_adaptive_factor: float = SIMULATION_Z_MIN_ATTENTION_X2_STD_ADAPTIVE_FACTOR,
    z_max_attention_effective_neighbors: float = SIMULATION_Z_MAX_ATTENTION_EFFECTIVE_NEIGHBORS,
    z_max_attention_effective_neighbor_fraction: float = SIMULATION_Z_MAX_ATTENTION_EFFECTIVE_NEIGHBOR_FRACTION,
    z_approx_threshold: int = SIMULATION_Z_APPROX_THRESHOLD,
    z_approx_rows: int = SIMULATION_Z_APPROX_ROWS,
    z_approx_pairs: int = SIMULATION_Z_APPROX_PAIRS,
    attention_mode: str = SIMULATION_ATTENTION_MODE,
    attention_top_k: Optional[int] = SIMULATION_ATTENTION_TOP_K,
    attention_cache_mode: str = SIMULATION_ATTENTION_CACHE_MODE,
    attention_cache_clusters: int = SIMULATION_ATTENTION_CACHE_CLUSTERS,
    attention_cache_probes: int = SIMULATION_ATTENTION_CACHE_PROBES,
    attention_cache_candidate_size: Optional[int] = SIMULATION_ATTENTION_CACHE_CANDIDATE_SIZE,
    attention_cache_candidate_multiplier: float = SIMULATION_ATTENTION_CACHE_CANDIDATE_MULTIPLIER,
    attention_cache_diagnostic_rows: int = SIMULATION_ATTENTION_CACHE_DIAGNOSTIC_ROWS,
    ray_debug: bool = False,
    high_max_p: Optional[int] = None,
    results_dir: Path = RESULTS_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the full simulation grid and save detailed/summary CSVs."""
    if ray_num_cpus < 1:
        raise ValueError("ray_num_cpus must be >= 1.")
    use_ray = ray_num_cpus > 1
    if use_ray and ray is None:
        raise RuntimeError(f"Ray could not be imported: {RAY_IMPORT_ERROR!r}")
    if use_ray and max_cpus_per_task > ray_num_cpus:
        raise ValueError("max_cpus_per_task cannot exceed ray_num_cpus in Ray mode.")
    if attention_mode not in {"dense", "topk"}:
        raise ValueError("attention_mode must be 'dense' or 'topk'.")
    if attention_mode == "topk" and attention_top_k is None:
        raise ValueError("--attention-top-k must be set when --attention-mode=topk.")
    if attention_cache_mode not in {"exact", "cluster"}:
        raise ValueError("attention_cache_mode must be 'exact' or 'cluster'.")

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    ray_started_here = False
    if use_ray and not ray.is_initialized():
        _print_progress(f"Starting Ray before JAX import: ray_num_cpus={ray_num_cpus}, max_cpus_per_task={max_cpus_per_task}")
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=ray_num_cpus,
            _node_ip_address="127.0.0.1",
        )
        ray_started_here = True
        _print_progress("Ray started")
    if use_ray:
        available = ray.available_resources()
        available_cpus = float(available.get("CPU", 0.0))
        if available_cpus < float(max_cpus_per_task):
            raise RuntimeError(
                f"Ray has no available CPUs after initialization: {available}. "
                "Stop stale Ray workers with `ray stop --force` or reduce other Ray jobs."
            )

    _ensure_runtime_imports()

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
    config_path = _full_simulation_config_path(results_dir, config_tag)
    setup_table = make_simulation_setup_table(
        R=R,
        T_list=T_list,
        settings=settings,
        dist_list=dist_list,
        q=q,
        high_max_p=high_max_p,
        screen_z=screen_z,
        precompute_dgp_attention=precompute_dgp_attention and attention_mode != "topk",
        require_good_z=require_good_z,
        dgp_eta_a=dgp_eta_a,
        z_max_tries=z_max_tries,
        z_score_quantile=z_score_quantile,
        z_max_abs_score=z_max_abs_score,
        z_min_eta_signal=z_min_eta_signal,
        z_min_eta_info=z_min_eta_info,
        z_min_attention_x2_std=z_min_attention_x2_std,
        z_min_attention_x2_std_adaptive_factor=z_min_attention_x2_std_adaptive_factor,
        z_max_attention_effective_neighbors=z_max_attention_effective_neighbors,
        z_max_attention_effective_neighbor_fraction=z_max_attention_effective_neighbor_fraction,
        z_approx_threshold=z_approx_threshold,
        z_approx_rows=z_approx_rows,
        z_approx_pairs=z_approx_pairs,
        attention_mode=attention_mode,
        attention_top_k=attention_top_k,
        attention_cache_mode=attention_cache_mode,
        attention_cache_clusters=attention_cache_clusters,
        attention_cache_probes=attention_cache_probes,
        attention_cache_candidate_size=attention_cache_candidate_size,
        attention_cache_candidate_multiplier=attention_cache_candidate_multiplier,
        attention_cache_diagnostic_rows=attention_cache_diagnostic_rows,
    )
    run_config = {
        "command": " ".join(sys.argv),
        "config_tag": config_tag,
        "run_tag": run_tag,
        "seed": seed,
        "R": R,
        "T_list": list(T_list),
        "settings": list(settings),
        "dist_list": list(dist_list),
        "q": q,
        "alpha": alpha,
        "ray_num_cpus": ray_num_cpus,
        "max_cpus_per_task": max_cpus_per_task,
        "qmle_algorithm": qmle_algorithm,
        "qmle_perturb_scale": qmle_perturb_scale,
        "eta_bounds": list(eta_bounds),
        "grid_size": grid_size,
        "high_grid_size": high_grid_size,
        "curvature_step": curvature_step,
        "ridge": ridge,
        "batch_size": batch_size,
        "dense_threshold": dense_threshold,
        "screen_z": screen_z,
        "precompute_dgp_attention": precompute_dgp_attention,
        "require_good_z": require_good_z,
        "dgp_eta_a": dgp_eta_a,
        "z_max_tries": z_max_tries,
        "z_score_quantile": z_score_quantile,
        "z_max_abs_score": z_max_abs_score,
        "z_min_eta_signal": z_min_eta_signal,
        "z_min_eta_info": z_min_eta_info,
        "z_min_attention_x2_std": z_min_attention_x2_std,
        "z_min_attention_x2_std_adaptive_factor": z_min_attention_x2_std_adaptive_factor,
        "z_max_attention_effective_neighbors": z_max_attention_effective_neighbors,
        "z_max_attention_effective_neighbor_fraction": z_max_attention_effective_neighbor_fraction,
        "z_approx_threshold": z_approx_threshold,
        "z_approx_rows": z_approx_rows,
        "z_approx_pairs": z_approx_pairs,
        "attention_mode": attention_mode,
        "attention_top_k": attention_top_k,
        "attention_cache_mode": attention_cache_mode,
        "attention_cache_clusters": attention_cache_clusters,
        "attention_cache_probes": attention_cache_probes,
        "attention_cache_candidate_size": attention_cache_candidate_size,
        "attention_cache_candidate_multiplier": attention_cache_candidate_multiplier,
        "attention_cache_diagnostic_rows": attention_cache_diagnostic_rows,
        "ray_debug": ray_debug,
        "high_max_p": high_max_p,
        "results_dir": str(results_dir),
        "detail_path": str(detail_path),
        "summary_path": str(summary_path),
        "paper_path": str(paper_path),
        "reporting_paths": {name: str(path) for name, path in reporting_paths.items()},
    }
    config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True), encoding="utf-8")
    _print_progress(f"Saved run config to: {config_path}")

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
                p_rule = "floor(T^1.5)" if setting == "high" else "floor(sqrt(T))"
                grid_size_use = int(
                    high_grid_size
                    if (setting == "high" and high_grid_size is not None)
                    else grid_size
                )

                key, k_params, k_Z, k_rep = jax.random.split(key, 4)
                _print_progress(f"Starting setting: setting={setting}, dist={dist}, T={T}, p={p}, q={q}")
                _print_progress("Drawing DGP parameters")
                params_true, stat_info = draw_simulation_dgp_params(k_params, p, eta_a=dgp_eta_a)
                _print_progress("DGP parameters ready")

                print("\n" + "=" * 80)
                print("Full VARA simulation setting")
                print("=" * 80)
                print(f"setting      : {setting}")
                print(f"dist         : {dist}")
                print(f"T, p, q      : {T}, {p}, {q}")
                print(f"p_rule       : {p_rule}")
                print(f"R            : {R}")
                print(f"methods      : {methods}")
                print(f"seed         : {seed}")
                print(f"grid_size    : {grid_size_use}")
                print(f"ray_num_cpus : {ray_num_cpus} ({'ray' if use_ray else 'sequential'})")
                print(f"ray_debug    : {ray_debug}")
                print(f"attention    : {attention_mode}" + (f", top_k={attention_top_k}" if attention_mode == "topk" else ""))
                if attention_mode == "topk":
                    cache_detail = f"mode={attention_cache_mode}"
                    if attention_cache_mode == "cluster":
                        cache_detail += (
                            f", clusters={attention_cache_clusters}, "
                            f"probes={attention_cache_probes}, "
                            f"candidate_size={attention_cache_candidate_size}, "
                            f"candidate_multiplier={attention_cache_candidate_multiplier}"
                        )
                    print(f"attention cache: {cache_detail}")
                print(f"dgp_eta_a    : {float(params_true['eta_a']):.4f}")
                print(f"stationary   : {not bool(stat_info['rescaled'])}")
                print(f"stationary_lhs: {float(stat_info['lhs_after']):.4f}")
                print(f"screen_z     : {screen_z}")
                if screen_z:
                    print(
                        "Z screening  : "
                        f"max_tries={z_max_tries}, "
                        f"q{z_score_quantile:g}_abs_score<={z_max_abs_score}, "
                        f"min_eta_signal={z_min_eta_signal}, "
                        f"min_eta_info={z_min_eta_info}, "
                        f"min_x2_std={z_min_attention_x2_std}, "
                        f"x2_adaptive_factor={z_min_attention_x2_std_adaptive_factor}, "
                        f"max_eff_neighbors=max({z_max_attention_effective_neighbors},"
                        f"{z_max_attention_effective_neighbor_fraction}*p), "
                        f"approx_threshold={z_approx_threshold}, "
                        f"approx_rows={z_approx_rows}, "
                        f"approx_pairs={z_approx_pairs}, "
                        f"require_good_z={require_good_z}, "
                        f"parallel_z={use_ray}"
                    )
                print(f"dgp_attention: {'precompute requested' if precompute_dgp_attention else 'on demand'}")
                print("=" * 80)

                if screen_z:
                    z_mode = "parallel Ray" if use_ray else "sequential"
                    _print_progress(f"Starting Z screening ({z_mode}): max_tries={z_max_tries}")
                    if use_ray:
                        Z, Z_info = draw_good_Z_for_eta_with_ray(
                            key=k_Z,
                            p=p,
                            q=q,
                            params_true=params_true,
                            params_scale=PARAMS_SCALE,
                            max_tries=z_max_tries,
                            min_offdiag_std=SIMULATION_Z_MIN_OFFDIAG_STD,
                            max_abs_score=z_max_abs_score,
                            min_eta_signal=z_min_eta_signal,
                            score_quantile=z_score_quantile,
                            min_eta_info=z_min_eta_info,
                            min_attention_x2_std=z_min_attention_x2_std,
                            min_attention_x2_std_adaptive_factor=z_min_attention_x2_std_adaptive_factor,
                            max_attention_effective_neighbors=z_max_attention_effective_neighbors,
                            max_attention_effective_neighbor_fraction=z_max_attention_effective_neighbor_fraction,
                            approximate_threshold=z_approx_threshold,
                            approx_rows=z_approx_rows,
                            approx_pairs=z_approx_pairs,
                            attention_mode=attention_mode,
                            attention_top_k=attention_top_k,
                            require_accepted=require_good_z,
                            ray_num_cpus=ray_num_cpus,
                            max_cpus_per_task=max_cpus_per_task,
                            ray_debug=ray_debug,
                            verbose=True,
                        )
                    else:
                        Z, Z_info = draw_good_Z_for_eta(
                            key=k_Z,
                            p=p,
                            q=q,
                            params_true=params_true,
                            params_scale=PARAMS_SCALE,
                            max_tries=z_max_tries,
                            min_offdiag_std=SIMULATION_Z_MIN_OFFDIAG_STD,
                            max_abs_score=z_max_abs_score,
                            min_eta_signal=z_min_eta_signal,
                            score_quantile=z_score_quantile,
                            min_eta_info=z_min_eta_info,
                            min_attention_x2_std=z_min_attention_x2_std,
                            min_attention_x2_std_adaptive_factor=z_min_attention_x2_std_adaptive_factor,
                            max_attention_effective_neighbors=z_max_attention_effective_neighbors,
                            max_attention_effective_neighbor_fraction=z_max_attention_effective_neighbor_fraction,
                            approximate_threshold=z_approx_threshold,
                            approx_rows=z_approx_rows,
                            approx_pairs=z_approx_pairs,
                            attention_mode=attention_mode,
                            attention_top_k=attention_top_k,
                            require_accepted=require_good_z,
                            verbose=True,
                        )
                else:
                    _print_progress("Drawing unscreened Z")
                    Z = draw_Z(k_Z, p, q)
                    Z_info = {"screened": False}
                _print_progress(f"Z ready: accepted={bool(Z_info.get('accepted', False))}")

                dgp_attention = None
                attention_cache = None
                attention_cache_quality: Dict[str, Any] = {}
                if precompute_dgp_attention and attention_mode == "topk":
                    _print_progress("Skipping DGP attention precompute for top-k attention")
                if attention_mode == "topk":
                    _print_progress("Starting top-k attention cache build")
                    attention_cache = build_topk_attention_cache(
                        Z,
                        int(attention_top_k),
                        batch_size=batch_size,
                        include_negative=True,
                        cache_mode=attention_cache_mode,
                        cluster_count=attention_cache_clusters,
                        cluster_probe=attention_cache_probes,
                        cluster_candidate_size=attention_cache_candidate_size,
                        cluster_candidate_multiplier=attention_cache_candidate_multiplier,
                    )
                    attention_cache.indices_pos.block_until_ready()
                    attention_cache.scores_pos.block_until_ready()
                    if attention_cache.indices_neg is not None:
                        attention_cache.indices_neg.block_until_ready()
                    if attention_cache.scores_neg is not None:
                        attention_cache.scores_neg.block_until_ready()
                    _print_progress("Top-k attention cache ready")
                    if attention_cache.mode == "cluster" and attention_cache_diagnostic_rows > 0:
                        _print_progress(
                            f"Starting top-k cache quality diagnostics: rows={attention_cache_diagnostic_rows}"
                        )
                        attention_cache_quality = evaluate_topk_cache_quality(
                            Z,
                            attention_cache,
                            sample_rows=attention_cache_diagnostic_rows,
                            batch_size=min(batch_size, max(1, attention_cache_diagnostic_rows)),
                            include_negative=True,
                        )
                        _print_progress(
                            "Top-k cache quality: "
                            f"pos_recall={attention_cache_quality.get('recall_pos_mean', math.nan):.4f}, "
                            f"pos_loss={attention_cache_quality.get('score_loss_pos_mean', math.nan):.4g}, "
                            f"pos_valid_min={attention_cache_quality.get('selected_valid_count_pos_min', math.nan):.0f}, "
                            f"pos_dup_max={attention_cache_quality.get('selected_duplicate_pair_count_pos_max', math.nan):.0f}, "
                            f"neg_recall={attention_cache_quality.get('recall_neg_mean', math.nan):.4f}, "
                            f"neg_loss={attention_cache_quality.get('score_loss_neg_mean', math.nan):.4g}, "
                            f"neg_valid_min={attention_cache_quality.get('selected_valid_count_neg_min', math.nan):.0f}, "
                            f"neg_dup_max={attention_cache_quality.get('selected_duplicate_pair_count_neg_max', math.nan):.0f}"
                        )
                elif precompute_dgp_attention:
                    _print_progress("Starting DGP attention precompute")
                    dgp_attention = build_attention(
                        Z,
                        float(params_true["eta_a"]),
                        PARAMS_SCALE["eta_a"],
                        attention_mode=attention_mode,
                        top_k=attention_top_k,
                    )
                    dgp_attention.block_until_ready()
                    _print_progress("DGP attention precompute finished")

                print(f"Z_info       : {Z_info}")
                print(f"dgp_attention: {'precomputed' if dgp_attention is not None else 'on demand'}")

                rep_keys = jax.random.split(k_rep, R)
                setting_start = datetime.now()

                if use_ray:
                    _print_progress(f"Starting Ray replications: R={R}")
                    n_actors = max(1, min(R, int(ray_num_cpus / max_cpus_per_task)))
                    _print_progress(f"Creating {n_actors} replication actors")
                    actors = []
                    progress = None
                    try:
                        actors = [
                            FullSimulationActor.options(num_cpus=max_cpus_per_task).remote()
                            for _ in range(n_actors)
                        ]
                        dgp_attention_arg = ray.put(dgp_attention) if dgp_attention is not None else None
                        attention_cache_arg = ray.put(attention_cache) if attention_cache is not None else None

                        def submit_replication(rep: int):
                            return actors[rep % n_actors].run.remote(
                                rep=rep,
                                key=rep_keys[rep],
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
                                dgp_attention=dgp_attention_arg,
                                attention_mode=attention_mode,
                                attention_top_k=attention_top_k,
                                attention_cache=attention_cache_arg,
                                ray_debug=ray_debug,
                            )

                        max_pending = min(R, max(1, 2 * n_actors))
                        _print_progress(
                            f"Submitting replication tasks: max_pending={max_pending}, n_actors={n_actors}"
                        )
                        setting_rows = []
                        pending = []
                        submitted = 0
                        completed = 0
                        submit_times = {}
                        progress = trange(R, desc=f"{setting}/{dist}/T={T}", ncols=100)

                        while submitted < R and len(pending) < max_pending:
                            ref = submit_replication(submitted)
                            pending.append(ref)
                            submit_times[ref] = time.time()
                            submitted += 1

                        while pending:
                            ready, pending = ray.wait(pending, num_returns=1, timeout=60.0)
                            if not ready:
                                now = time.time()
                                slowest_wait = now - min(submit_times.values()) if submit_times else 0.0
                                elapsed = (datetime.now() - setting_start).total_seconds()
                                _print_progress(
                                    "Ray replication wait: "
                                    f"done={completed}/{R}, submitted={submitted}/{R}, "
                                    f"pending={len(pending)}, elapsed_sec={elapsed:.1f}, "
                                    f"slowest_pending_sec={slowest_wait:.1f}, "
                                    f"available={ray.available_resources()}"
                                )
                                continue

                            for ref in ready:
                                setting_rows.extend(ray.get(ref))
                                submit_times.pop(ref, None)
                                completed += 1
                                progress.update(1)

                                while submitted < R and len(pending) < max_pending:
                                    next_ref = submit_replication(submitted)
                                    pending.append(next_ref)
                                    submit_times[next_ref] = time.time()
                                    submitted += 1
                    finally:
                        if progress is not None:
                            progress.close()
                        for actor in actors:
                            try:
                                ray.kill(actor, no_restart=True)
                            except Exception:
                                pass
                else:
                    _print_progress(f"Starting sequential replications: R={R}")
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
                            dgp_attention=dgp_attention,
                            attention_mode=attention_mode,
                            attention_top_k=attention_top_k,
                            attention_cache=attention_cache,
                        ))

                elapsed_setting = (datetime.now() - setting_start).total_seconds()
                _print_progress(f"Replications finished in {elapsed_setting:.2f} seconds")
                _print_progress("Annotating replication rows")
                for row in setting_rows:
                    row["setting_elapsed_sec"] = elapsed_setting
                    row["Z_screened"] = bool(screen_z)
                    row["Z_parallel"] = bool(Z_info.get("parallel", False))
                    row["Z_parallel_workers"] = Z_info.get("parallel_workers", math.nan)
                    row["Z_parallel_chunks"] = Z_info.get("parallel_chunks", math.nan)
                    row["Z_parallel_accepted_chunks"] = Z_info.get("parallel_accepted_chunks", math.nan)
                    row["Z_approximate"] = bool(Z_info.get("approximate", False))
                    row["Z_approx_rows"] = Z_info.get("approx_rows", math.nan)
                    row["Z_approx_pairs"] = Z_info.get("approx_pairs", math.nan)
                    row["Z_approx_threshold"] = z_approx_threshold
                    row["Z_accepted"] = bool(Z_info.get("accepted", False))
                    row["Z_attempt"] = Z_info.get("attempt", math.nan)
                    row["Z_offdiag_std"] = Z_info.get("offdiag_std", math.nan)
                    row["Z_max_abs_score"] = Z_info.get("max_abs_score", math.nan)
                    row["Z_score_quantile"] = Z_info.get("score_quantile", math.nan)
                    row["Z_abs_score_quantile"] = Z_info.get("abs_score_quantile", math.nan)
                    row["Z_eta_signal"] = Z_info.get("eta_signal", math.nan)
                    row["Z_eta_info"] = Z_info.get("eta_info", math.nan)
                    row["Z_attention_x2_std_mean"] = Z_info.get("attention_x2_std_mean", math.nan)
                    row["Z_attention_x2_std_min"] = Z_info.get("attention_x2_std_min", math.nan)
                    row["Z_attention_x2_std_requested_min"] = Z_info.get("attention_x2_std_requested_min", math.nan)
                    row["Z_attention_x2_std_effective_min"] = Z_info.get("attention_x2_std_effective_min", math.nan)
                    row["Z_attention_x2_std_adaptive_min"] = Z_info.get("attention_x2_std_adaptive_min", math.nan)
                    row["Z_attention_x2_std_adaptive_factor"] = Z_info.get("attention_x2_std_adaptive_factor", math.nan)
                    row["Z_attention_effective_neighbors_mean"] = Z_info.get("attention_effective_neighbors_mean", math.nan)
                    row["Z_attention_effective_neighbors_p95"] = Z_info.get("attention_effective_neighbors_p95", math.nan)
                    row["Z_attention_effective_neighbors_limit"] = Z_info.get("attention_effective_neighbors_limit", math.nan)
                    row["Z_attention_max_weight_mean"] = Z_info.get("attention_max_weight_mean", math.nan)
                    row["Z_selection_score"] = Z_info.get("selection_score", math.nan)
                    row["DGP_attention_precomputed"] = dgp_attention is not None
                    row["TopK_attention_cached"] = attention_cache is not None
                    row["attention_cache_mode"] = attention_cache.mode if attention_cache is not None else ""
                    row["attention_cache_clusters"] = (
                        attention_cache.cluster_count if attention_cache is not None else math.nan
                    )
                    row["attention_cache_probes"] = (
                        attention_cache.cluster_probe if attention_cache is not None else math.nan
                    )
                    row["attention_cache_candidate_size"] = (
                        attention_cache.cluster_candidate_size if attention_cache is not None else math.nan
                    )
                    row["TopK_cache_quality_rows"] = attention_cache_quality.get("sample_rows", math.nan)
                    row["TopK_cache_recall_pos_mean"] = attention_cache_quality.get("recall_pos_mean", math.nan)
                    row["TopK_cache_recall_pos_min"] = attention_cache_quality.get("recall_pos_min", math.nan)
                    row["TopK_cache_score_loss_pos_mean"] = attention_cache_quality.get("score_loss_pos_mean", math.nan)
                    row["TopK_cache_score_loss_pos_max"] = attention_cache_quality.get("score_loss_pos_max", math.nan)
                    row["TopK_cache_score_loss_pos_abs_mean"] = attention_cache_quality.get("score_loss_pos_abs_mean", math.nan)
                    row["TopK_cache_selected_valid_count_pos_mean"] = attention_cache_quality.get("selected_valid_count_pos_mean", math.nan)
                    row["TopK_cache_selected_valid_count_pos_min"] = attention_cache_quality.get("selected_valid_count_pos_min", math.nan)
                    row["TopK_cache_selected_duplicate_pair_count_pos_mean"] = attention_cache_quality.get("selected_duplicate_pair_count_pos_mean", math.nan)
                    row["TopK_cache_selected_duplicate_pair_count_pos_max"] = attention_cache_quality.get("selected_duplicate_pair_count_pos_max", math.nan)
                    row["TopK_cache_recall_neg_mean"] = attention_cache_quality.get("recall_neg_mean", math.nan)
                    row["TopK_cache_recall_neg_min"] = attention_cache_quality.get("recall_neg_min", math.nan)
                    row["TopK_cache_score_loss_neg_mean"] = attention_cache_quality.get("score_loss_neg_mean", math.nan)
                    row["TopK_cache_score_loss_neg_max"] = attention_cache_quality.get("score_loss_neg_max", math.nan)
                    row["TopK_cache_score_loss_neg_abs_mean"] = attention_cache_quality.get("score_loss_neg_abs_mean", math.nan)
                    row["TopK_cache_selected_valid_count_neg_mean"] = attention_cache_quality.get("selected_valid_count_neg_mean", math.nan)
                    row["TopK_cache_selected_valid_count_neg_min"] = attention_cache_quality.get("selected_valid_count_neg_min", math.nan)
                    row["TopK_cache_selected_duplicate_pair_count_neg_mean"] = attention_cache_quality.get("selected_duplicate_pair_count_neg_mean", math.nan)
                    row["TopK_cache_selected_duplicate_pair_count_neg_max"] = attention_cache_quality.get("selected_duplicate_pair_count_neg_max", math.nan)
                    row["attention_mode"] = attention_mode
                    row["attention_top_k"] = attention_top_k if attention_top_k is not None else math.nan
                    row["stationarity_rescaled"] = bool(stat_info["rescaled"])
                    row["stationary_lhs_before"] = float(stat_info["lhs_before"])
                    row["stationary_lhs_after"] = float(stat_info["lhs_after"])

                all_detail_rows.extend(setting_rows)
                _print_progress("Summarizing current setting")
                df_setting = pd.DataFrame(setting_rows)
                summary_current = summarize_setting(df_setting)
                setting_summaries.append(summary_current)

                print("\nSummary for current setting:")
                print(summary_current.round(4).to_string(index=False))

                _print_progress("Saving incremental results")
                pd.DataFrame(all_detail_rows).to_csv(detail_path, index=False)
                pd.concat(setting_summaries, ignore_index=True).to_csv(summary_path, index=False)
                print(f"Saved incremental detail to: {detail_path}")
                print(f"Saved incremental summary to: {summary_path}")
                _print_progress(f"Finished setting: setting={setting}, dist={dist}, T={T}")

                gc.collect()

    if ray_started_here:
        _print_progress("Shutting down Ray")
        ray.shutdown()
        _print_progress("Ray shut down")

    _print_progress("Building final result tables")
    df_detail = pd.DataFrame(all_detail_rows)
    df_summary = pd.concat(setting_summaries, ignore_index=True) if setting_summaries else pd.DataFrame()
    df_paper = make_paper_table(df_summary) if len(df_summary) else pd.DataFrame()
    df_reporting = make_simulation_reporting_tables(df_summary, setup_table)

    _print_progress("Saving final result tables")
    df_detail.to_csv(detail_path, index=False)
    df_summary.to_csv(summary_path, index=False)
    df_paper.to_csv(paper_path, index=False)
    for name, table in df_reporting.items():
        table.to_csv(reporting_paths[name], index=False)
    _print_progress("Final result tables saved")

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
    print(f"Run config       : {config_path}")
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
    parser.add_argument(
        "--dgp-eta-a",
        type=float,
        default=SIMULATION_ETA_A,
        help="True DGP eta_a. Paper gamma in repository mapping.",
    )

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
        "--attention-mode",
        choices=["dense", "topk"],
        default=SIMULATION_ATTENTION_MODE,
        help="Attention backend for DGP simulation, eta likelihood, and Z screening.",
    )
    parser.add_argument(
        "--attention-top-k",
        type=int,
        default=SIMULATION_ATTENTION_TOP_K,
        help="Number of neighbors for --attention-mode topk.",
    )
    parser.add_argument(
        "--attention-cache-mode",
        choices=["exact", "cluster"],
        default=SIMULATION_ATTENTION_CACHE_MODE,
        help="Top-k cache builder. exact preserves current behavior; cluster uses approximate candidate clusters.",
    )
    parser.add_argument(
        "--attention-cache-clusters",
        type=int,
        default=SIMULATION_ATTENTION_CACHE_CLUSTERS,
        help="Number of deterministic anchor clusters for --attention-cache-mode cluster.",
    )
    parser.add_argument(
        "--attention-cache-probes",
        type=int,
        default=SIMULATION_ATTENTION_CACHE_PROBES,
        help="Number of nearest clusters searched per row for --attention-cache-mode cluster.",
    )
    parser.add_argument(
        "--attention-cache-candidate-size",
        type=int,
        default=SIMULATION_ATTENTION_CACHE_CANDIDATE_SIZE,
        help="Rows retained per cluster for --attention-cache-mode cluster. Default uses multiplier * top_k.",
    )
    parser.add_argument(
        "--attention-cache-candidate-multiplier",
        type=float,
        default=SIMULATION_ATTENTION_CACHE_CANDIDATE_MULTIPLIER,
        help="Default cluster candidate size as multiplier * top_k.",
    )
    parser.add_argument(
        "--attention-cache-diagnostic-rows",
        type=int,
        default=SIMULATION_ATTENTION_CACHE_DIAGNOSTIC_ROWS,
        help="Sampled rows used to compare cluster top-k cache against exact top-k. Use 0 to disable.",
    )
    parser.add_argument(
        "--screen-z",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Screen Z for eta identification. Use --no-screen-z to draw an unscreened design.",
    )
    parser.add_argument(
        "--precompute-dgp-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build W(eta_true) once per setting and reuse it during DGP simulation.",
    )
    parser.add_argument(
        "--require-good-z",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail a setting if screened Z does not pass all eta-identification checks.",
    )
    parser.add_argument("--z-max-tries", type=int, default=SIMULATION_Z_MAX_TRIES)
    parser.add_argument("--z-score-quantile", type=float, default=SIMULATION_Z_SCORE_QUANTILE)
    parser.add_argument("--z-max-abs-score", type=float, default=SIMULATION_Z_MAX_ABS_SCORE)
    parser.add_argument("--z-min-eta-signal", type=float, default=SIMULATION_Z_MIN_ETA_SIGNAL)
    parser.add_argument("--z-min-eta-info", type=float, default=SIMULATION_Z_MIN_ETA_INFO)
    parser.add_argument("--z-min-attention-x2-std", type=float, default=SIMULATION_Z_MIN_ATTENTION_X2_STD)
    parser.add_argument(
        "--z-min-attention-x2-std-adaptive-factor",
        type=float,
        default=SIMULATION_Z_MIN_ATTENTION_X2_STD_ADAPTIVE_FACTOR,
        help=(
            "Cap the effective x2 std threshold at this multiple of "
            "sqrt(mean(Gamma0_diag_proxy) / effective_neighbor_limit). "
            "Use 0 to disable the adaptive cap."
        ),
    )
    parser.add_argument(
        "--z-max-attention-effective-neighbors",
        type=float,
        default=SIMULATION_Z_MAX_ATTENTION_EFFECTIVE_NEIGHBORS,
    )
    parser.add_argument(
        "--z-max-attention-effective-neighbor-fraction",
        type=float,
        default=SIMULATION_Z_MAX_ATTENTION_EFFECTIVE_NEIGHBOR_FRACTION,
        help="Adaptive high-p effective-neighbor cap as a fraction of p.",
    )
    parser.add_argument(
        "--z-approx-threshold",
        type=int,
        default=SIMULATION_Z_APPROX_THRESHOLD,
        help="Use approximate Z screening when p is at least this value. Use 0 to disable.",
    )
    parser.add_argument(
        "--z-approx-rows",
        type=int,
        default=SIMULATION_Z_APPROX_ROWS,
        help="Number of sampled rows for approximate high-p Z screening.",
    )
    parser.add_argument(
        "--z-approx-pairs",
        type=int,
        default=SIMULATION_Z_APPROX_PAIRS,
        help="Number of sampled off-diagonal pairs for approximate high-p Z screening.",
    )
    parser.add_argument(
        "--ray-debug",
        action="store_true",
        help="Print Ray Z-screening chunk progress and replication actor start/end messages.",
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
        precompute_dgp_attention=args.precompute_dgp_attention,
        require_good_z=args.require_good_z,
        dgp_eta_a=args.dgp_eta_a,
        z_max_tries=args.z_max_tries,
        z_score_quantile=args.z_score_quantile,
        z_max_abs_score=args.z_max_abs_score,
        z_min_eta_signal=args.z_min_eta_signal,
        z_min_eta_info=args.z_min_eta_info,
        z_min_attention_x2_std=args.z_min_attention_x2_std,
        z_min_attention_x2_std_adaptive_factor=args.z_min_attention_x2_std_adaptive_factor,
        z_max_attention_effective_neighbors=args.z_max_attention_effective_neighbors,
        z_max_attention_effective_neighbor_fraction=args.z_max_attention_effective_neighbor_fraction,
        z_approx_threshold=args.z_approx_threshold,
        z_approx_rows=args.z_approx_rows,
        z_approx_pairs=args.z_approx_pairs,
        attention_mode=args.attention_mode,
        attention_top_k=args.attention_top_k,
        attention_cache_mode=args.attention_cache_mode,
        attention_cache_clusters=args.attention_cache_clusters,
        attention_cache_probes=args.attention_cache_probes,
        attention_cache_candidate_size=args.attention_cache_candidate_size,
        attention_cache_candidate_multiplier=args.attention_cache_candidate_multiplier,
        attention_cache_diagnostic_rows=args.attention_cache_diagnostic_rows,
        ray_debug=args.ray_debug,
        high_max_p=args.high_max_p,
        results_dir=Path(args.results_dir),
    )


if __name__ == "__main__":
    main()
