"""Minimal end-to-end base-plus-VARA forecasting framework."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable

import numpy as np

from .base import BaseForecast, BaseForecaster
from .correction import VARAForecast, VARAResidualCorrector
from .data import PanelData, RollingSplit
from .evaluation import EvaluationResult, ForecastEvaluator, PointForecastEvaluator
from .forecast_contracts import ExecutionConfig, RollingVARAConfig, StaticFeatureSet


def _filename(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return value or "method"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


@dataclass(frozen=True, slots=True)
class ForecastStudyResult:
    """Structured outputs from all base and base-plus-VARA families."""

    data: PanelData
    split: RollingSplit
    bases: Mapping[str, BaseForecast]
    vara: Mapping[str, VARAForecast]
    forecasts: Mapping[str, np.ndarray]
    evaluation: EvaluationResult

    @property
    def metrics(self):
        return self.evaluation.metrics

    def save(self, output_dir: str | Path) -> Path:
        """Persist forecasts, metrics, refit summaries, and a small manifest."""
        output = Path(output_dir)
        forecast_dir = output / "forecasts"
        vara_dir = output / "vara"
        forecast_dir.mkdir(parents=True, exist_ok=True)
        vara_dir.mkdir(parents=True, exist_ok=True)

        names: dict[str, str] = {}
        used: set[str] = set()
        for name, values in self.forecasts.items():
            stem = _filename(name)
            if stem in used:
                raise ValueError(f"forecast names collide after filename normalization: {name!r}.")
            used.add(stem)
            names[name] = f"forecasts/{stem}.npy"
            np.save(output / names[name], np.asarray(values), allow_pickle=False)

        self.evaluation.metrics.to_csv(output / "metrics.csv", index=False)
        self.evaluation.by_origin.to_csv(output / "metrics_by_origin.csv", index=False)
        for base_name, result in self.vara.items():
            stem = _filename(base_name)
            np.save(vara_dir / f"{stem}_correction.npy", result.correction, allow_pickle=False)
            np.save(vara_dir / f"{stem}_lambda.npy", result.lambda_weights, allow_pickle=False)
            np.save(vara_dir / f"{stem}_unit_forecast.npy", result.unit_forecast, allow_pickle=False)
            result.eta_refits.to_csv(vara_dir / f"{stem}_eta_refits.csv", index=False)
            result.lambda_refits.to_csv(vara_dir / f"{stem}_lambda_refits.csv", index=False)

        plan = self.split.plan(self.data)
        manifest = {
            "schema": "vara_forecast_study_v1",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "shape": list(self.data.shape),
            "validation_indices": [int(plan.validation_indices[0]), int(plan.validation_indices[-1])],
            "test_indices": [int(plan.test_indices[0]), int(plan.test_indices[-1])],
            "base_models": list(self.bases),
            "forecast_files": names,
            "evaluation_details": dict(self.evaluation.details),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=_json_default) + "\n",
            encoding="utf-8",
        )
        return output


@dataclass(slots=True)
class ForecastStudy:
    """Run base forecasts, residual VARA corrections, and final evaluation."""

    data: PanelData
    split: RollingSplit
    features: StaticFeatureSet
    vara: RollingVARAConfig = field(default_factory=RollingVARAConfig)
    evaluator: ForecastEvaluator = field(default_factory=PointForecastEvaluator)
    corrector: VARAResidualCorrector | None = None

    def __post_init__(self) -> None:
        if self.features.component_count != self.data.component_count:
            raise ValueError("feature rows must match the panel component count.")
        self.split.plan(self.data)
        if self.corrector is None:
            self.corrector = VARAResidualCorrector(self.vara)

    def run(
        self,
        base_models: Sequence[BaseForecaster],
        *,
        execution: ExecutionConfig = ExecutionConfig(),
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> ForecastStudyResult:
        if not base_models:
            raise ValueError("at least one base model is required.")
        bases: dict[str, BaseForecast] = {}
        vara_results: dict[str, VARAForecast] = {}
        forecasts: dict[str, np.ndarray] = {}
        assert self.corrector is not None

        for model in base_models:
            base = model.rolling_forecast(self.data, self.split, execution)
            if base.name in bases:
                raise ValueError(f"duplicate base-model name: {base.name!r}.")
            # This validates authoritative errors against target - forecast at
            # every evaluation origin before expensive correction begins.
            corrected = self.corrector.forecast(
                self.data,
                self.split,
                self.features,
                base,
                execution,
                progress_callback=progress_callback,
            )
            if corrected.name in forecasts or corrected.name == base.name:
                raise ValueError(f"duplicate corrected-model name: {corrected.name!r}.")
            bases[base.name] = base
            vara_results[base.name] = corrected
            forecasts[base.name] = base.point
            forecasts[corrected.name] = corrected.combined_forecast

        evaluation = self.evaluator.evaluate(self.data, self.split, forecasts)
        return ForecastStudyResult(
            data=self.data,
            split=self.split,
            bases=bases,
            vara=vara_results,
            forecasts=forecasts,
            evaluation=evaluation,
        )
