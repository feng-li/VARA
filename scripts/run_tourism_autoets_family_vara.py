#!/usr/bin/env python3
"""Run one fixed VARA strategy for multiple saved AutoETS family panels."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

try:
    from _bootstrap import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT  # noqa: F401

try:
    from search_tourism_oos_error_vara import _select_strategies, run_sweep
except ModuleNotFoundError:
    from scripts.search_tourism_oos_error_vara import (
        _select_strategies,
        run_sweep,
    )


def _candidate_ids(value: str) -> list[str]:
    ids = [part.strip() for part in value.split(",") if part.strip()]
    if not ids:
        raise ValueError("At least one candidate id is required.")
    if len(ids) != len(set(ids)):
        raise ValueError("Candidate ids must be unique.")
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-root",
        type=Path,
        default=Path("outputs/tourism_autoets_family_full"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/tourism_vara_autoets_family"),
    )
    parser.add_argument("--candidate-ids", required=True)
    parser.add_argument(
        "--strategy-id",
        default="predictive_avg_topk10_w24_stable05",
    )
    parser.add_argument("--validation-start", default="2013-01-01")
    parser.add_argument("--validation-end", default="2016-12-01")
    parser.add_argument("--selection-end", default="2015-12-01")
    parser.add_argument("--ray-num-cpus", type=int, default=104)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    args = parser.parse_args()

    candidate_ids = _candidate_ids(args.candidate_ids)
    strategies = _select_strategies(args.strategy_id)
    if len(strategies) != 1:
        raise ValueError("Family screening requires exactly one fixed strategy.")

    import ray

    started_here = False
    if not ray.is_initialized():
        ray.init(
            num_cpus=int(args.ray_num_cpus),
            include_dashboard=False,
            ignore_reinit_error=True,
            _node_ip_address="127.0.0.1",
        )
        started_here = True
    try:
        for candidate_id in candidate_ids:
            forecast_path = (
                args.forecast_root
                / candidate_id
                / "rolling_forecasts.csv"
            )
            if not forecast_path.exists():
                raise FileNotFoundError(
                    f"Missing AutoETS panel for {candidate_id}: {forecast_path}"
                )
            output_dir = args.output_root / candidate_id
            output_dir.mkdir(parents=True, exist_ok=True)
            config = {
                "candidate_id": candidate_id,
                "forecast_panel_path": str(forecast_path),
                "output_dir": str(output_dir),
                "validation_start": args.validation_start,
                "validation_end": args.validation_end,
                "selection_end": args.selection_end,
                "ray_num_cpus": int(args.ray_num_cpus),
                "max_cpus_per_task": int(args.max_cpus_per_task),
                "skip_scoring": True,
                "strategies": [
                    asdict(strategy)
                    for strategy in strategies
                ],
                "leakage_control": (
                    "each VARA worker receives rolling AutoETS errors "
                    "through t-1 only"
                ),
            }
            (output_dir / "config.json").write_text(
                json.dumps(config, indent=2) + "\n"
            )
            results, failures = run_sweep(
                purpose_forecasts_path=forecast_path,
                output_dir=output_dir,
                strategies=strategies,
                validation_start=args.validation_start,
                validation_end=args.validation_end,
                ray_num_cpus=args.ray_num_cpus,
                max_cpus_per_task=args.max_cpus_per_task,
            )
            if not failures.empty:
                raise RuntimeError(
                    f"{candidate_id} produced {len(failures)} VARA failures."
                )
            expected_origins = (
                12
                * (
                    int(args.validation_end[:4])
                    - int(args.validation_start[:4])
                    + 1
                )
            )
            observed_origins = int(results["date"].nunique())
            if observed_origins != expected_origins:
                raise RuntimeError(
                    f"{candidate_id} produced {observed_origins} origins; "
                    f"expected {expected_origins}."
                )
            print(
                f"Completed {candidate_id}: {observed_origins} origins.",
                flush=True,
            )
    finally:
        if started_here:
            ray.shutdown()


if __name__ == "__main__":
    main()
