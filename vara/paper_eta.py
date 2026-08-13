"""Paper-scale attention parameterization for the DataMine applications.

The reusable VARA core accepts the coefficient that directly multiplies an
attention score in the softmax exponent.  The manuscript instead defines that
effective coefficient as ``eta * log(p)``.  This application-layer adapter
keeps the core untouched while making command-line arguments, diagnostics, and
saved contracts use the manuscript's eta scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


__all__ = [
    "APPLICATION_CONTRACT_SCHEMA",
    "DEFAULT_PAPER_ETA_BOUNDS",
    "DEFAULT_PAPER_ETA_GRID_SIZE",
    "DEFAULT_PAPER_ETA_LOCAL_TOLERANCE",
    "ETA_EFFECTIVE_SLOPE_DEFINITION",
    "ETA_PARAMETERIZATION",
    "ETA_SCORE_SCALE_RULE",
    "PaperEtaSpecification",
    "_boolean_values",
    "attention_support_name",
    "file_fingerprints",
]


ETA_PARAMETERIZATION = "paper_eta_log_component_count_v1"
ETA_SCORE_SCALE_RULE = "log_component_count"
ETA_EFFECTIVE_SLOPE_DEFINITION = "paper_eta * log(component_count)"
APPLICATION_CONTRACT_SCHEMA = "datamine_vara_paper_eta_v2"
DEFAULT_PAPER_ETA_BOUNDS = (-2.0, 2.0)
DEFAULT_PAPER_ETA_GRID_SIZE = 7
DEFAULT_PAPER_ETA_LOCAL_TOLERANCE = 1e-4


def file_fingerprints(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Hash application inputs whose contents determine a saved correction."""
    fingerprints: dict[str, dict[str, Any]] = {}
    for name, path in sorted(paths.items()):
        resolved = Path(path).resolve()
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        fingerprints[str(name)] = {
            "path": resolved.as_posix(),
            "size_bytes": int(resolved.stat().st_size),
            "sha256": digest.hexdigest(),
        }
    return fingerprints


def _boolean_values(series: pd.Series) -> np.ndarray:
    """Parse saved Boolean diagnostics without treating ``'False'`` as true."""
    parsed: list[bool] = []
    for value in series.to_numpy():
        if isinstance(value, (bool, np.bool_)):
            parsed.append(bool(value))
        elif isinstance(value, (int, np.integer, float, np.floating)) and value in (0, 1):
            parsed.append(bool(value))
        elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
            parsed.append(value.strip().lower() == "true")
        else:
            raise ValueError("Saved eta_on_boundary values must be Boolean.")
    return np.asarray(parsed, dtype=bool)


@dataclass(frozen=True)
class PaperEtaSpecification:
    """Map the paper's eta to the effective softmax slope used by the core."""

    component_count: int
    lower: float
    upper: float
    grid_size: int
    local_tolerance: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.component_count, bool)
            or int(self.component_count) != self.component_count
        ):
            raise ValueError("component_count must be an integer.")
        if int(self.component_count) < 2:
            raise ValueError("component_count must be at least two.")
        if not math.isfinite(float(self.lower)) or not math.isfinite(float(self.upper)):
            raise ValueError("Paper eta bounds must be finite.")
        if float(self.lower) >= float(self.upper):
            raise ValueError("Paper eta bounds must satisfy lower < upper.")
        if isinstance(self.grid_size, bool) or int(self.grid_size) != self.grid_size:
            raise ValueError("grid_size must be an integer.")
        if int(self.grid_size) < 2:
            raise ValueError("grid_size must be at least two.")
        if not math.isfinite(float(self.local_tolerance)) or float(self.local_tolerance) <= 0.0:
            raise ValueError("Paper eta local_tolerance must be positive and finite.")

    @property
    def score_scale(self) -> float:
        return math.log(int(self.component_count))

    @property
    def effective_bounds(self) -> tuple[float, float]:
        scale = self.score_scale
        return float(self.lower) * scale, float(self.upper) * scale

    @property
    def effective_local_tolerance(self) -> float:
        return float(self.local_tolerance) * self.score_scale

    def paper_eta(self, effective_slope: float) -> float:
        value = float(effective_slope)
        if not math.isfinite(value):
            raise ValueError("The effective attention slope must be finite.")
        return value / self.score_scale

    def effective_slope(self, paper_eta: float) -> float:
        value = float(paper_eta)
        if not math.isfinite(value):
            raise ValueError("The paper eta must be finite.")
        return value * self.score_scale

    def contract(self, *, attention_support: str) -> dict[str, Any]:
        """Return one canonical contract used for execution and reuse checks."""
        return {
            "method": "ridge_penalized_concentrated_qmle",
            "parameterization": ETA_PARAMETERIZATION,
            "paper_parameter": "eta",
            "bounds": [float(self.lower), float(self.upper)],
            "grid_size": int(self.grid_size),
            "local_tolerance": float(self.local_tolerance),
            "score_scale": {
                "rule": ETA_SCORE_SCALE_RULE,
                "component_count": int(self.component_count),
                "value": self.score_scale,
            },
            "effective_slope": {
                "definition": ETA_EFFECTIVE_SLOPE_DEFINITION,
                "bounds": [float(value) for value in self.effective_bounds],
                "local_tolerance": self.effective_local_tolerance,
            },
            "attention_support": str(attention_support),
        }

    def convert_core_fit_summary(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Convert a fresh core summary from effective-slope to paper-eta units."""
        if any(column.startswith("eta_effective_slope_") for column in frame.columns):
            raise ValueError("The fit summary already contains paper-eta diagnostics.")
        pairs = {
            "eta_hat": "eta_effective_slope_hat",
            "eta_grid_best": "eta_effective_slope_grid_best",
            "eta_bounds_lower": "eta_effective_slope_bounds_lower",
            "eta_bounds_upper": "eta_effective_slope_bounds_upper",
            "eta_local_tolerance": "eta_effective_slope_local_tolerance",
        }
        missing = [column for column in pairs if column not in frame.columns]
        if missing:
            raise ValueError(f"Core fit summary is missing eta columns: {missing}.")

        converted = frame.copy()
        for paper_column, effective_column in pairs.items():
            values = pd.to_numeric(converted[paper_column], errors="coerce").to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"Core fit summary column {paper_column} must be finite.")
            converted[effective_column] = values
            converted[paper_column] = values / self.score_scale

        expected_lower, expected_upper = self.effective_bounds
        if not np.allclose(
            converted["eta_effective_slope_bounds_lower"], expected_lower, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("Core fit summary has unexpected effective eta lower bounds.")
        if not np.allclose(
            converted["eta_effective_slope_bounds_upper"], expected_upper, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("Core fit summary has unexpected effective eta upper bounds.")
        if not np.allclose(
            converted["eta_effective_slope_local_tolerance"],
            self.effective_local_tolerance,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Core fit summary has an unexpected effective eta tolerance.")

        converted["eta_parameterization"] = ETA_PARAMETERIZATION
        converted["eta_score_scale_rule"] = ETA_SCORE_SCALE_RULE
        converted["eta_score_scale_value"] = self.score_scale
        converted["eta_component_count"] = int(self.component_count)
        self.validate_persisted_fit_summary(converted)
        return converted

    def validate_persisted_fit_summary(self, frame: pd.DataFrame) -> None:
        """Validate that a saved fit summary consistently records both scales."""
        pairs = {
            "eta_hat": "eta_effective_slope_hat",
            "eta_grid_best": "eta_effective_slope_grid_best",
            "eta_bounds_lower": "eta_effective_slope_bounds_lower",
            "eta_bounds_upper": "eta_effective_slope_bounds_upper",
            "eta_local_tolerance": "eta_effective_slope_local_tolerance",
        }
        required = [
            "refit_index",
            *pairs,
            *pairs.values(),
            "eta_grid_size",
            "eta_parameterization",
            "eta_score_scale_rule",
            "eta_score_scale_value",
            "eta_component_count",
            "eta_on_boundary",
        ]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Saved fit summary is missing paper-eta columns: {missing}.")
        if frame.empty:
            raise ValueError("Saved fit summary must contain at least one eta refit.")

        if not frame["eta_parameterization"].eq(ETA_PARAMETERIZATION).all():
            raise ValueError("Saved fit summary has the wrong eta parameterization.")
        if not frame["eta_score_scale_rule"].eq(ETA_SCORE_SCALE_RULE).all():
            raise ValueError("Saved fit summary has the wrong eta score-scale rule.")
        if not np.allclose(
            pd.to_numeric(frame["eta_score_scale_value"], errors="coerce"),
            self.score_scale,
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("Saved fit summary has the wrong eta score-scale value.")
        if not np.equal(
            pd.to_numeric(frame["eta_component_count"], errors="coerce").to_numpy(),
            int(self.component_count),
        ).all():
            raise ValueError("Saved fit summary has the wrong eta component count.")

        for paper_column, effective_column in pairs.items():
            paper = pd.to_numeric(frame[paper_column], errors="coerce").to_numpy(dtype=float)
            effective = pd.to_numeric(
                frame[effective_column], errors="coerce"
            ).to_numpy(dtype=float)
            if not np.all(np.isfinite(paper)) or not np.all(np.isfinite(effective)):
                raise ValueError("Saved paper-eta diagnostics must be finite.")
            if not np.allclose(effective, paper * self.score_scale, rtol=1e-9, atol=1e-10):
                raise ValueError(
                    f"Saved columns {paper_column} and {effective_column} use "
                    "inconsistent eta scales."
                )

        expected_paper_values = {
            "eta_bounds_lower": float(self.lower),
            "eta_bounds_upper": float(self.upper),
            "eta_local_tolerance": float(self.local_tolerance),
        }
        for column, expected in expected_paper_values.items():
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            if not np.allclose(values, expected, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Saved fit summary has an unexpected configured {column}.")
        grid_sizes = pd.to_numeric(frame["eta_grid_size"], errors="coerce").to_numpy(dtype=float)
        if not np.equal(grid_sizes, int(self.grid_size)).all():
            raise ValueError("Saved fit summary has the wrong eta grid size.")

        eta_hat = pd.to_numeric(frame["eta_hat"], errors="coerce").to_numpy(dtype=float)
        tolerance = max(float(self.local_tolerance), 1e-10)
        if np.any(eta_hat < float(self.lower) - tolerance) or np.any(
            eta_hat > float(self.upper) + tolerance
        ):
            raise ValueError("Saved paper eta estimates lie outside the configured bounds.")
        boundary = _boolean_values(frame["eta_on_boundary"])
        expected_boundary = (np.abs(eta_hat - float(self.lower)) <= tolerance + 1e-12) | (
            np.abs(eta_hat - float(self.upper)) <= tolerance + 1e-12
        )
        if not np.array_equal(boundary, expected_boundary):
            raise ValueError("Saved eta_on_boundary values are inconsistent with paper eta.")

    def validate_refit_coverage(
        self,
        frame: pd.DataFrame,
        refit_index: np.ndarray,
        evaluation_indices: np.ndarray,
    ) -> None:
        """Cross-check fit rows against refits assigned to forecast origins."""
        self.validate_persisted_fit_summary(frame)
        assignments = np.asarray(refit_index)
        evaluation = np.asarray(evaluation_indices)
        if assignments.ndim != 1:
            raise ValueError("Saved VARA refit indices must be one-dimensional.")
        if evaluation.ndim != 1 or evaluation.size == 0:
            raise ValueError("Evaluation indices must be a nonempty one-dimensional array.")
        if not np.issubdtype(evaluation.dtype, np.integer):
            raise ValueError("Evaluation indices must be integers.")
        if np.any(evaluation < 0) or np.any(evaluation >= len(assignments)):
            raise ValueError("Evaluation indices are outside the saved refit-index array.")
        assigned = assignments[evaluation]
        if not np.issubdtype(assigned.dtype, np.integer) or np.any(assigned < 0):
            raise ValueError("Saved refit indices do not cover every evaluation origin.")

        summary_values = pd.to_numeric(frame["refit_index"], errors="coerce").to_numpy()
        if not np.all(np.isfinite(summary_values)) or not np.equal(
            summary_values, np.floor(summary_values)
        ).all():
            raise ValueError("Fit-summary refit indices must be finite integers.")
        summary_indices = summary_values.astype(np.int64)
        if len(np.unique(summary_indices)) != len(summary_indices):
            raise ValueError("Fit-summary refit indices must be unique.")
        if len(summary_indices) > 1 and np.any(np.diff(summary_indices) <= 0):
            raise ValueError("Fit-summary refit indices must be strictly increasing.")
        if not np.array_equal(summary_indices, np.unique(assigned.astype(np.int64))):
            raise ValueError("Fit-summary rows do not match assigned VARA refits.")

    def refit_summary(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Summarize rolling estimates on paper and effective-slope scales."""
        self.validate_persisted_fit_summary(frame)

        def values(column: str) -> dict[str, float]:
            series = pd.to_numeric(frame[column], errors="raise").astype(float)
            return {
                "minimum": float(series.min()),
                "median": float(series.median()),
                "mean": float(series.mean()),
                "maximum": float(series.max()),
            }

        return {
            "refits": int(len(frame)),
            **values("eta_hat"),
            "boundary_refits": int(_boolean_values(frame["eta_on_boundary"]).sum()),
            "effective_slope": values("eta_effective_slope_hat"),
        }


def attention_support_name(
    cache_mode: str,
    *,
    attention_mode: str = "topk",
) -> str:
    """Return the application-contract name for the selected support.

    ``attention_mode`` defaults to ``"topk"`` so existing application code
    and persisted-contract tests retain their original sparse-support meaning.
    Dense attention always uses every off-diagonal score and therefore has no
    cache-dependent support.
    """
    mode = str(attention_mode)
    if mode == "dense":
        return "full_dense_off_diagonal_scores"
    if mode != "topk":
        raise ValueError("attention_mode must be topk or dense.")
    return (
        "deterministic_fixed_smallest_scores"
        if str(cache_mode) == "exact"
        else "fixed_smallest_scores"
    )
