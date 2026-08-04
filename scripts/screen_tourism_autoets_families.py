#!/usr/bin/env python3
"""Screen a small prespecified AutoETS family grid before the 2016 holdout."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from _bootstrap import REPO_ROOT  # noqa: F401
except ModuleNotFoundError:
    from scripts._bootstrap import REPO_ROOT  # noqa: F401

try:
    from evaluate_tourism_76_autoets import run_backtest
except ModuleNotFoundError:
    from scripts.evaluate_tourism_76_autoets import run_backtest


@dataclass(frozen=True)
class AutoETSCandidate:
    candidate_id: str
    model_specification: str
    damped: bool | None
    rationale: str


def default_candidates() -> tuple[AutoETSCandidate, ...]:
    """Return an interpretable grid fixed before examining 2016."""
    return (
        AutoETSCandidate(
            "zzz_standard",
            "ZZZ",
            None,
            "standard unrestricted AutoETS reference",
        ),
        AutoETSCandidate(
            "znz_no_trend_auto",
            "ZNZ",
            False,
            "no trend; error and seasonality selected automatically",
        ),
        AutoETSCandidate(
            "mnz_no_trend_mult_error",
            "MNZ",
            False,
            "multiplicative error, no trend, automatic seasonality",
        ),
        AutoETSCandidate(
            "anz_no_trend_add_error",
            "ANZ",
            False,
            "additive error, no trend, automatic seasonality",
        ),
        AutoETSCandidate(
            "zna_no_trend_add_season",
            "ZNA",
            False,
            "automatic error, no trend, additive seasonality",
        ),
        AutoETSCandidate(
            "znm_no_trend_mult_season",
            "ZNM",
            False,
            "automatic error, no trend, multiplicative seasonality",
        ),
        AutoETSCandidate(
            "mnm_fixed",
            "MNM",
            False,
            "multiplicative error and seasonality with no trend",
        ),
        AutoETSCandidate(
            "mna_fixed",
            "MNA",
            False,
            "multiplicative error, no trend, additive seasonality",
        ),
        AutoETSCandidate(
            "ana_fixed",
            "ANA",
            False,
            "additive error, no trend, additive seasonality",
        ),
        AutoETSCandidate(
            "maa_auto_damping",
            "MAA",
            None,
            "multiplicative error with additive trend and seasonality",
        ),
        AutoETSCandidate(
            "mam_auto_damping",
            "MAM",
            None,
            "multiplicative error with additive trend and multiplicative seasonality",
        ),
        AutoETSCandidate(
            "azz_add_error_auto",
            "AZZ",
            None,
            "additive error with automatic trend and seasonality",
        ),
        AutoETSCandidate(
            "mzz_mult_error_auto",
            "MZZ",
            None,
            "multiplicative error with automatic trend and seasonality",
        ),
        AutoETSCandidate(
            "zaz_add_trend_auto",
            "ZAZ",
            None,
            "additive trend with automatic error and seasonality",
        ),
        AutoETSCandidate(
            "aaa_undamped",
            "AAA",
            False,
            "fixed additive error, trend, and seasonality without damping",
        ),
        AutoETSCandidate(
            "aaa_damped",
            "AAA",
            True,
            "fixed additive error, trend, and seasonality with damping",
        ),
        AutoETSCandidate(
            "aaz_undamped",
            "AAZ",
            False,
            "additive error and trend with automatic seasonality, undamped",
        ),
        AutoETSCandidate(
            "aaz_damped",
            "AAZ",
            True,
            "additive error and trend with automatic seasonality, damped",
        ),
        AutoETSCandidate(
            "aza_auto_damping",
            "AZA",
            None,
            "additive error and seasonality with automatic trend/damping",
        ),
        AutoETSCandidate(
            "zaa_auto_damping",
            "ZAA",
            None,
            "additive trend and seasonality with automatic error/damping",
        ),
        AutoETSCandidate(
            "zza_add_season_auto",
            "ZZA",
            None,
            "additive seasonality with automatic error, trend, and damping",
        ),
        AutoETSCandidate(
            "zzz_undamped",
            "ZZZ",
            False,
            "standard AutoETS candidates with damped trends excluded",
        ),
        AutoETSCandidate(
            "zzz_damped",
            "ZZZ",
            True,
            "standard AutoETS candidates with trends forced to be damped",
        ),
        AutoETSCandidate(
            "azz_undamped",
            "AZZ",
            False,
            "additive-error AutoETS with damped trends excluded",
        ),
        AutoETSCandidate(
            "azz_damped",
            "AZZ",
            True,
            "additive-error AutoETS with trends forced to be damped",
        ),
        AutoETSCandidate(
            "zaz_undamped",
            "ZAZ",
            False,
            "additive-trend AutoETS without damping",
        ),
        AutoETSCandidate(
            "zaz_damped",
            "ZAZ",
            True,
            "additive-trend AutoETS with damping",
        ),
        AutoETSCandidate(
            "aan_undamped",
            "AAN",
            False,
            "additive error and trend without seasonality or damping",
        ),
        AutoETSCandidate(
            "aan_damped",
            "AAN",
            True,
            "additive error and damped trend without seasonality",
        ),
    )


def _select_candidates(value: str | None) -> list[AutoETSCandidate]:
    candidates = default_candidates()
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        raise RuntimeError("AutoETS candidate identifiers must be unique.")
    if value is None or not value.strip():
        return list(candidates)
    requested = {
        part.strip()
        for part in value.split(",")
        if part.strip()
    }
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    missing = sorted(requested.difference(by_id))
    if missing:
        raise ValueError(f"Unknown AutoETS candidate ids: {missing}")
    return [
        candidate
        for candidate in candidates
        if candidate.candidate_id in requested
    ]


def _matches_saved_config(
    forecast_path: Path,
    candidate: AutoETSCandidate,
) -> bool:
    metadata_path = forecast_path.with_suffix(
        forecast_path.suffix + ".metadata.json"
    )
    if not forecast_path.exists() or not metadata_path.exists():
        return False
    metadata = json.loads(metadata_path.read_text())
    config = metadata.get("autoets_config", {})
    return (
        config.get("model_specification") == candidate.model_specification
        and config.get("damped") is candidate.damped
        and config.get("phi") is None
    )


def validation_metrics(forecasts: pd.DataFrame) -> dict[str, float]:
    """Compute pooled original-scale accuracy for the saved validation panel."""
    required = {"actual_original", "direct_auto_ets"}
    missing = sorted(required.difference(forecasts.columns))
    if missing:
        raise ValueError(f"AutoETS validation panel is missing columns: {missing}")
    actual = forecasts["actual_original"].to_numpy(dtype=float)
    prediction = forecasts["direct_auto_ets"].to_numpy(dtype=float)
    error = prediction - actual
    if not np.isfinite(error).all():
        raise ValueError("AutoETS validation errors must all be finite.")
    return {
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/tourism/tourism.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_autoets_family_screen"),
    )
    parser.add_argument("--candidate-ids", default=None)
    parser.add_argument("--first-year", type=int, default=2013)
    parser.add_argument("--final-year", type=int, default=2015)
    parser.add_argument("--series-depth", type=int, default=3)
    parser.add_argument("--expected-n-series", type=int, default=76)
    parser.add_argument("--season-length", type=int, default=12)
    parser.add_argument("--ray-num-cpus", type=int, default=104)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    candidates = _select_candidates(args.candidate_ids)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    missing_candidates = [
        candidate
        for candidate in candidates
        if not (
            args.resume
            and _matches_saved_config(
                args.output_dir
                / candidate.candidate_id
                / "rolling_forecasts.csv",
                candidate,
            )
        )
    ]

    ray_started_here = False
    if missing_candidates and int(args.ray_num_cpus) > 1:
        import ray

        if not ray.is_initialized():
            ray.init(
                num_cpus=int(args.ray_num_cpus),
                include_dashboard=False,
                ignore_reinit_error=True,
                _node_ip_address="127.0.0.1",
            )
            ray_started_here = True
    try:
        rows = []
        for candidate in candidates:
            forecast_path = (
                args.output_dir
                / candidate.candidate_id
                / "rolling_forecasts.csv"
            )
            base_row = {
                "candidate_id": candidate.candidate_id,
                "model_specification": candidate.model_specification,
                "damped": candidate.damped,
                "phi": None,
                "rationale": candidate.rationale,
                "first_year": int(args.first_year),
                "final_year": int(args.final_year),
            }
            try:
                if candidate in missing_candidates:
                    forecasts = run_backtest(
                        data_path=args.data_path,
                        output_path=forecast_path,
                        first_year=args.first_year,
                        final_year=args.final_year,
                        series_depth=args.series_depth,
                        expected_n_series=args.expected_n_series,
                        season_length=args.season_length,
                        ray_num_cpus=args.ray_num_cpus,
                        max_cpus_per_task=args.max_cpus_per_task,
                        clip_nonnegative=True,
                        response_transform="none",
                        model_specification=candidate.model_specification,
                        damped=candidate.damped,
                        phi=None,
                    )
                else:
                    forecasts = pd.read_csv(forecast_path)
                metrics = validation_metrics(forecasts)
                rows.append({
                    **base_row,
                    "status": "ok",
                    "error": None,
                    "n_origins": int(forecasts["date"].nunique()),
                    "n_series": int(forecasts["series_id"].nunique()),
                    **metrics,
                })
            except Exception as exc:
                rows.append({
                    **base_row,
                    "status": "invalid",
                    "error": f"{type(exc).__name__}: {exc}",
                    "n_origins": 0,
                    "n_series": 0,
                    "rmse": np.nan,
                    "mae": np.nan,
                    "bias": np.nan,
                })
                print(
                    f"Skipping invalid candidate {candidate.candidate_id}: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            pd.DataFrame(rows).sort_values(
                ["status", "rmse"],
                na_position="last",
            ).to_csv(
                args.output_dir / "direct_validation_screen.csv",
                index=False,
            )
    finally:
        if ray_started_here:
            import ray

            ray.shutdown()

    ranking = pd.DataFrame(rows)
    successful = ranking[ranking["status"] == "ok"].sort_values(
        ["rmse", "mae", "candidate_id"]
    ).reset_index(drop=True)
    successful.insert(
        0,
        "direct_rank",
        np.arange(1, len(successful) + 1),
    )
    invalid = ranking[ranking["status"] != "ok"].sort_values(
        "candidate_id"
    ).copy()
    invalid.insert(0, "direct_rank", np.nan)
    ranking = pd.concat(
        [successful, invalid],
        ignore_index=True,
        sort=False,
    )
    ranking.to_csv(
        args.output_dir / "direct_validation_screen.csv",
        index=False,
    )
    metadata = {
        "selection_period": (
            f"{args.first_year}-01-01 through {args.final_year}-12-01"
        ),
        "holdout_year_examined": False,
        "response_transform": "none",
        "evaluation_scale": "original visitor nights",
        "candidate_count": len(candidates),
        "candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "model_specification": candidate.model_specification,
                "damped": candidate.damped,
                "phi": None,
                "rationale": candidate.rationale,
            }
            for candidate in candidates
        ],
    }
    (args.output_dir / "screen.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(ranking.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
