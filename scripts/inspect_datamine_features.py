#!/usr/bin/env python3
"""Audit DataMine feature geometry and top-k neighbor quality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

from vara.attention import build_topk_attention_cache, evaluate_topk_cache_quality
from vara.datamine import write_json


def _snapshot_path(output_dir: Path, requested: str | None) -> Path:
    if requested is not None:
        path = output_dir / "features" / requested
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    snapshots = sorted((output_dir / "features").glob("snapshot_*"))
    if not snapshots:
        raise FileNotFoundError(f"No feature snapshots under {output_dir / 'features'}")
    return snapshots[-1]


def audit_features(
    output_dir: Path,
    *,
    snapshot: str | None,
    sample_rows: int,
    attention_top_k: int | None = None,
    attention_cache_mode: str = "exact",
    attention_batch_size: int = 512,
) -> tuple[dict[str, Any], pd.DataFrame]:
    feature_dir = _snapshot_path(output_dir, snapshot)
    matrix = np.load(feature_dir / "Z.npy")
    columns = pd.read_csv(feature_dir / "feature_columns.csv")
    metadata = pd.read_parquet(output_dir / "processed" / "strategy_metadata.parquet")
    provenance = json.loads((feature_dir / "feature_provenance.json").read_text(encoding="utf-8"))
    if matrix.shape != (len(metadata), len(columns)):
        raise ValueError("Feature matrix, metadata, and column artifacts are inconsistent.")

    quantiles = np.quantile(matrix, (0.0, 0.01, 0.25, 0.5, 0.75, 0.99, 1.0), axis=0).T
    dominant = np.argmax(np.square(matrix), axis=1)
    column_audit = pd.DataFrame(
        quantiles,
        columns=("minimum", "q01", "q25", "median", "q75", "q99", "maximum"),
    )
    column_audit.insert(0, "feature_name", columns["feature_name"].astype(str))
    if "feature_block" in columns:
        column_audit.insert(1, "feature_block", columns["feature_block"].astype(str))
    column_audit["mean"] = matrix.mean(axis=0)
    column_audit["standard_deviation"] = matrix.std(axis=0)
    column_audit["dominant_rows"] = np.bincount(
        dominant, minlength=matrix.shape[1]
    )
    column_audit["dominant_fraction"] = column_audit["dominant_rows"] / matrix.shape[0]

    row_norm = np.linalg.norm(matrix, axis=1)
    formula_group_sizes = metadata.groupby(
        ["numerator", "denominator", "formula_type"], observed=True
    ).size()
    rounded = np.round(matrix, 6)
    _, rounded_inverse, rounded_counts = np.unique(
        rounded,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    duplicate_groups: list[dict[str, Any]] = []
    for group_index in np.flatnonzero(rounded_counts > 1):
        member_indices = np.flatnonzero(rounded_inverse == group_index)
        reference = matrix[member_indices[0]]
        duplicate_groups.append({
            "rounded_group_size": int(len(member_indices)),
            "maximum_absolute_feature_difference": float(
                np.max(np.abs(matrix[member_indices] - reference[None, :]))
            ),
            "components": metadata.iloc[member_indices][
                ["component_id", "signalid", "numerator", "denominator", "formula_type"]
            ].to_dict(orient="records"),
        })
    report: dict[str, Any] = {
        "output_dir": output_dir.as_posix(),
        "snapshot": feature_dir.name,
        "shape": list(matrix.shape),
        "all_finite": bool(np.isfinite(matrix).all()),
        "unique_rows_rounded_1e6": int(len(rounded_counts)),
        "duplicate_row_groups_rounded_1e6": duplicate_groups,
        "row_norm": {
            "minimum": float(row_norm.min()),
            "median": float(np.median(row_norm)),
            "maximum": float(row_norm.max()),
        },
        "largest_dominant_feature_fraction": float(column_audit["dominant_fraction"].max()),
        "largest_dominant_feature": str(
            column_audit.loc[column_audit["dominant_fraction"].idxmax(), "feature_name"]
        ),
        "metadata": {
            "components": int(len(metadata)),
            "formula_type_counts": {
                str(key): int(value)
                for key, value in metadata["formula_type"].value_counts().items()
            },
            "numerators": int(metadata["numerator"].nunique()),
            "denominators": int(metadata["denominator"].nunique()),
            "unique_formula_tuples": int(len(formula_group_sizes)),
            "formula_tuple_size_minimum": int(formula_group_sizes.min()),
            "formula_tuple_size_median": float(formula_group_sizes.median()),
            "formula_tuple_size_maximum": int(formula_group_sizes.max()),
        },
        "preprocessing": provenance.get("preprocessing", {}),
        "formula_svd": {
            "dimension": provenance.get("formula", {}).get("svd_dimension"),
            "explained_variance_ratio": provenance.get("formula", {}).get(
                "svd_explained_variance_ratio", []
            ),
            "explained_variance_total": float(
                np.sum(provenance.get("formula", {}).get("svd_explained_variance_ratio", []))
            ),
        },
    }

    locked_path = output_dir / "forecasts" / "locked_test_config.json"
    warnings: list[str] = []
    if report["largest_dominant_feature_fraction"] > 0.25:
        warnings.append("One feature is the largest squared row contribution for over 25% of series.")
    formula_type_rows = column_audit["feature_name"].str.startswith("formula_type=")
    if formula_type_rows.any() and float(
        column_audit.loc[formula_type_rows, ["minimum", "maximum"]].abs().to_numpy().max()
    ) < 0.05:
        warnings.append("Formula-type indicators have negligible magnitude after row normalization.")

    attention_settings: dict[str, Any] | None = None
    if attention_top_k is not None:
        attention_settings = {
            "attention_top_k": int(attention_top_k),
            "attention_batch_size": int(attention_batch_size),
            "attention_cache_mode": str(attention_cache_mode),
        }
    elif locked_path.exists():
        locked = json.loads(locked_path.read_text(encoding="utf-8"))
        if locked.get("resolved_attention_mode") == "topk":
            attention_settings = {
                "attention_top_k": int(locked["attention_top_k"]),
                "attention_batch_size": int(locked["attention_batch_size"]),
                "attention_cache_mode": str(locked["attention_cache_mode"]),
            }

    if attention_settings is not None:
        cache = build_topk_attention_cache(
            jnp.asarray(matrix),
            attention_settings["attention_top_k"],
            batch_size=attention_settings["attention_batch_size"],
            include_negative=True,
            cache_mode=attention_settings["attention_cache_mode"],
        )
        cache_quality = evaluate_topk_cache_quality(
            jnp.asarray(matrix),
            cache,
            sample_rows=int(sample_rows),
            batch_size=min(32, int(sample_rows)),
            include_negative=True,
        )
        indices = np.asarray(cache.indices_pos)
        valid = indices >= 0
        safe = np.where(valid, indices, 0)
        rows = np.arange(len(metadata))[:, None]
        numerator = metadata["numerator"].astype(str).to_numpy()
        denominator = metadata["denominator"].astype(str).to_numpy()
        formula_type = metadata["formula_type"].astype(str).to_numpy()
        valid_count = max(int(valid.sum()), 1)
        cache_quality.update({
            "cache_mode": cache.mode,
            "cluster_count": cache.cluster_count,
            "cluster_probe": cache.cluster_probe,
            "cluster_candidate_size": cache.cluster_candidate_size,
            "same_numerator_fraction": float(
                ((numerator[rows] == numerator[safe]) & valid).sum() / valid_count
            ),
            "same_denominator_fraction": float(
                ((denominator[rows] == denominator[safe]) & valid).sum() / valid_count
            ),
            "same_formula_type_fraction": float(
                ((formula_type[rows] == formula_type[safe]) & valid).sum() / valid_count
            ),
        })
        if cache.indices_neg is not None:
            negative_indices = np.asarray(cache.indices_neg)
            negative_valid = negative_indices >= 0
            negative_safe = np.where(negative_valid, negative_indices, 0)
            negative_valid_count = max(int(negative_valid.sum()), 1)
            cache_quality.update({
                "negative_same_numerator_fraction": float(
                    (
                        (numerator[rows] == numerator[negative_safe])
                        & negative_valid
                    ).sum()
                    / negative_valid_count
                ),
                "negative_same_denominator_fraction": float(
                    (
                        (denominator[rows] == denominator[negative_safe])
                        & negative_valid
                    ).sum()
                    / negative_valid_count
                ),
                "negative_same_formula_type_fraction": float(
                    (
                        (formula_type[rows] == formula_type[negative_safe])
                        & negative_valid
                    ).sum()
                    / negative_valid_count
                ),
            })
        report["topk_cache"] = cache_quality
        if cache_quality["recall_pos_mean"] < 0.9:
            warnings.append("Approximate positive-neighbor recall is below 90%.")
        if cache_quality["selected_valid_count_pos_min"] < attention_settings["attention_top_k"]:
            warnings.append("Some rows have fewer valid positive neighbors than requested.")

    report["warnings"] = warnings
    return report, column_audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--snapshot")
    parser.add_argument("--sample-rows", type=int, default=128)
    parser.add_argument("--report-dir", type=Path)
    parser.add_argument("--attention-top-k", type=int)
    parser.add_argument("--attention-cache-mode", choices=("exact", "cluster"), default="exact")
    parser.add_argument("--attention-batch-size", type=int, default=512)
    args = parser.parse_args()
    report, columns = audit_features(
        args.output_dir,
        snapshot=args.snapshot,
        sample_rows=args.sample_rows,
        attention_top_k=args.attention_top_k,
        attention_cache_mode=args.attention_cache_mode,
        attention_batch_size=args.attention_batch_size,
    )
    report_dir = args.report_dir or args.output_dir / "metrics" / "feature_audit"
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(report_dir / "feature_audit.json", report)
    columns.to_csv(report_dir / "feature_column_audit.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
