"""Risk-response construction and risk-specific forecast evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from .base import BaseForecast, PrecomputedForecaster
from .data import PanelData, RollingSplit
from .datamine import evaluate_risk_forecasts
from .evaluation import EvaluationResult
from .forecast_contracts import ExecutionConfig, PaperEtaConfig, RollingVARAConfig


HAR_RIDGE_RISK_MODEL = "har_ridge"
POOLED_LIGHTGBM_RISK_MODEL = "pooled_lightgbm"
RISKMETRICS_EWMA_RISK_MODEL = "riskmetrics_ewma_validation_offset"
GARCH11_RISK_MODEL = "garch11_offset"
STUDENT_T_GARCH11_RISK_MODEL = "garch11_student_t_offset"
RISK_BENCHMARK_MODELS = (
    HAR_RIDGE_RISK_MODEL,
    POOLED_LIGHTGBM_RISK_MODEL,
    RISKMETRICS_EWMA_RISK_MODEL,
    GARCH11_RISK_MODEL,
    STUDENT_T_GARCH11_RISK_MODEL,
)
RISK_VARA_MODELS = MappingProxyType({
    HAR_RIDGE_RISK_MODEL: "har_vara_lambda",
    POOLED_LIGHTGBM_RISK_MODEL: "pooled_lightgbm_vara_lambda",
    RISKMETRICS_EWMA_RISK_MODEL: "riskmetrics_ewma_vara_lambda",
    GARCH11_RISK_MODEL: "garch_vara_lambda",
    STUDENT_T_GARCH11_RISK_MODEL: "garch11_student_t_vara_lambda",
})
RISK_PAPER_TABLE_MODELS = tuple(
    model
    for base_model in RISK_BENCHMARK_MODELS
    for model in (base_model, RISK_VARA_MODELS[base_model])
)

RISK_PAPER_PROFILE = "accounting_signal_risk_primary_v1"
RISK_PAPER_COMPONENT_COUNT = 21_800
RISK_PAPER_VALIDATION_MONTHS = 60
RISK_PAPER_TEST_MONTHS = 120
RISK_PAPER_RAY_NUM_CPUS = 15
RISK_PAPER_MAX_CPUS_PER_TASK = 1


def risk_paper_split() -> RollingSplit:
    """Return the validation/test split used by the primary Risk table."""
    return RollingSplit(
        validation=RISK_PAPER_VALIDATION_MONTHS,
        test=RISK_PAPER_TEST_MONTHS,
    )


def risk_paper_vara_config(
    component_count: int = RISK_PAPER_COMPONENT_COUNT,
) -> RollingVARAConfig:
    """Return the primary full-dense Risk VARA specification.

    ``component_count`` permits a smaller diagnostic panel while preserving
    all other paper settings. Dense attention still uses every off-diagonal
    component in that panel.
    """
    if isinstance(component_count, (bool, np.bool_)) or not isinstance(
        component_count, (int, np.integer)
    ):
        raise TypeError("component_count must be an integer.")
    count = int(component_count)
    if count < 2:
        raise ValueError("component_count must be at least two.")
    return RollingVARAConfig(
        error_window_months=240,
        response_standardization="median_mad",
        response_clip=8.0,
        refit_frequency_months=12,
        ridge=0.001,
        attention_top_k=count - 1,
        attention_cache_mode="exact",
        attention_batch_size=min(512, count),
        attention_mode="dense",
        pilot_size=min(1000, count),
        pilot_seed=20260924,
        lambda_window_months=24,
        lambda_ridge=0.001,
        eta=PaperEtaConfig(
            lower=-2.0,
            upper=2.0,
            grid_size=7,
            local_tolerance=1e-4,
        ),
    )


def risk_paper_execution_config() -> ExecutionConfig:
    """Return the Ray resource allocation used by the primary Risk run."""
    return ExecutionConfig(
        ray_num_cpus=RISK_PAPER_RAY_NUM_CPUS,
        max_cpus_per_task=RISK_PAPER_MAX_CPUS_PER_TASK,
    )


def risk_paper_corrector(component_count: int = RISK_PAPER_COMPONENT_COUNT):
    """Return a VARA corrector with the paper config and model identifiers."""
    from .correction import VARAResidualCorrector

    return VARAResidualCorrector(
        risk_paper_vara_config(component_count),
        model_names=RISK_VARA_MODELS,
    )


def _artifact_location(value: Any, *, path_root: Path) -> Path:
    location = Path(str(value)).expanduser()
    if not location.is_absolute():
        location = path_root / location
    return location


@dataclass(frozen=True, slots=True)
class RiskBenchmarkArtifacts:
    """The five base forecasts used by the Accounting-Signal Risk study.

    The artifact adapter preserves the exact fitted forecasts used to construct
    the paper table. Each loaded array becomes an ordinary ``BaseForecaster``
    and can therefore receive a fresh VARA correction through ``ForecastStudy``.
    """

    files: Mapping[str, str | Path]
    contract_path: str | Path | None = None
    contract: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.files, Mapping):
            raise TypeError("risk benchmark files must be a mapping.")
        files = {str(name): Path(path) for name, path in self.files.items()}
        expected = set(RISK_BENCHMARK_MODELS)
        if set(files) != expected:
            missing = sorted(expected.difference(files))
            extra = sorted(set(files).difference(expected))
            raise ValueError(
                "risk benchmark files must contain exactly the five paper models; "
                f"missing={missing}, extra={extra}."
            )
        if not isinstance(self.contract, Mapping):
            raise TypeError("risk benchmark contract must be a mapping.")
        contract_path = (
            None if self.contract_path is None else Path(self.contract_path)
        )
        object.__setattr__(self, "files", MappingProxyType(files))
        object.__setattr__(self, "contract_path", contract_path)
        object.__setattr__(self, "contract", MappingProxyType(dict(self.contract)))

    @classmethod
    def from_paper_contract(
        cls,
        contract_path: str | Path,
        *,
        path_root: str | Path | None = None,
    ) -> RiskBenchmarkArtifacts:
        """Resolve the five base artifacts from a paper-result contract."""
        source = Path(contract_path)
        contract = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(contract, dict):
            raise TypeError("paper contract must contain a JSON object.")
        required = {
            "source_output",
            "pooled_lightgbm_vara_output",
            "riskmetrics_ewma_vara_output",
            "garch_vara_output",
            "student_t_garch_output",
        }
        missing = sorted(required.difference(contract))
        if missing:
            raise ValueError(
                "paper contract is missing risk benchmark locations: "
                + ", ".join(missing)
            )
        root = Path.cwd() if path_root is None else Path(path_root)
        source_output = _artifact_location(contract["source_output"], path_root=root)
        lightgbm_output = _artifact_location(
            contract["pooled_lightgbm_vara_output"], path_root=root
        )
        ewma_output = _artifact_location(
            contract["riskmetrics_ewma_vara_output"], path_root=root
        )
        garch_output = _artifact_location(
            contract["garch_vara_output"], path_root=root
        )
        student_t_output = _artifact_location(
            contract["student_t_garch_output"], path_root=root
        )
        return cls(
            files={
                HAR_RIDGE_RISK_MODEL: (
                    source_output / "baseline" / "baseline_forecast.npy"
                ),
                POOLED_LIGHTGBM_RISK_MODEL: (
                    lightgbm_output
                    / "forecasts"
                    / "pooled_lightgbm_forecast.npy"
                ),
                RISKMETRICS_EWMA_RISK_MODEL: (
                    ewma_output
                    / "forecasts"
                    / "riskmetrics_ewma_validation_offset_forecast.npy"
                ),
                GARCH11_RISK_MODEL: (
                    garch_output
                    / "forecasts"
                    / "garch_validation_offset_forecast.npy"
                ),
                STUDENT_T_GARCH11_RISK_MODEL: (
                    student_t_output
                    / "forecasts"
                    / "garch11_t_validation_offset_forecast.npy"
                ),
            },
            contract_path=source,
            contract=contract,
        )

    def forecasters(
        self,
        data: PanelData,
        split: RollingSplit,
        *,
        component_indices: np.ndarray | None = None,
        mmap_mode: str | None = "r",
    ) -> tuple[PrecomputedForecaster, ...]:
        """Load the five forecasts in stable paper-table order.

        ``component_indices`` may select a deterministic subset from the full
        artifact panel. The caller remains responsible for constructing
        ``data`` in exactly that selected source-component order.
        """
        if mmap_mode not in (None, "r"):
            raise ValueError("mmap_mode must be None or 'r'.")
        component_count = self.contract.get("component_count")
        indices = None
        if component_indices is not None:
            raw_indices = np.asarray(component_indices)
            if (
                raw_indices.ndim != 1
                or np.issubdtype(raw_indices.dtype, np.bool_)
                or not np.issubdtype(raw_indices.dtype, np.integer)
            ):
                raise TypeError("component_indices must be a one-dimensional integer array.")
            indices = np.asarray(raw_indices, dtype=np.int64)
            if len(indices) != data.component_count:
                raise ValueError("component_indices must contain one index per data series.")
            if len(np.unique(indices)) != len(indices) or np.any(indices < 0):
                raise ValueError("component_indices must be nonnegative and unique.")
        if (
            indices is None
            and component_count is not None
            and int(component_count) != data.component_count
        ):
            raise ValueError(
                "paper contract component_count does not match the panel."
            )
        plan = split.plan(data)
        contract_split = self.contract.get("split")
        if isinstance(contract_split, Mapping):
            expected_dates = {
                "validation_start": data.dates[int(plan.validation_indices[0])],
                "validation_end": data.dates[int(plan.validation_indices[-1])],
                "test_start": data.dates[int(plan.test_indices[0])],
                "test_end": data.dates[int(plan.test_indices[-1])],
            }
            for field_name, expected_date in expected_dates.items():
                observed = contract_split.get(field_name)
                if observed is not None and np.datetime64(observed, "ns") != expected_date:
                    raise ValueError(
                        f"paper contract {field_name} does not match the panel split."
                    )

        evaluation = plan.evaluation_indices
        models: list[PrecomputedForecaster] = []
        for name in RISK_BENCHMARK_MODELS:
            path = self.files[name]
            if not path.is_file():
                raise FileNotFoundError(path)
            source_point = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
            expected_source_components = (
                int(component_count)
                if component_count is not None
                else source_point.shape[1]
            )
            if source_point.shape != (data.time_count, expected_source_components):
                raise ValueError(
                    f"risk forecast {name!r} has shape {source_point.shape}; "
                    "it does not match the source panel contract."
                )
            if indices is not None:
                if len(indices) and int(indices.max()) >= source_point.shape[1]:
                    raise ValueError("component_indices are outside the artifact panel.")
                point = np.asarray(source_point[:, indices])
            else:
                point = source_point
            if point.shape != data.shape:
                raise ValueError(
                    f"selected risk forecast {name!r} has shape {point.shape}; "
                    f"expected {data.shape}."
                )
            error = np.asarray(data.target) - np.asarray(point)
            observed = np.isfinite(error[evaluation])
            if np.any(observed.sum(axis=1) == 0):
                raise ValueError(
                    f"risk forecast {name!r} has no observed error at an "
                    "evaluation origin."
                )
            models.append(PrecomputedForecaster(BaseForecast(
                name=name,
                point=point,
                error=error,
                diagnostics={
                    "source": "risk_paper_artifact",
                    "artifact": path.as_posix(),
                    "paper_contract": (
                        None
                        if self.contract_path is None
                        else self.contract_path.as_posix()
                    ),
                },
            )))
        return tuple(models)

    def validate_primary_profile(self) -> None:
        """Require the artifact contract used by the primary paper table."""
        contract = self.contract
        expected_fields = {
            "component_count": RISK_PAPER_COMPONENT_COUNT,
            "response_standardization": "median_mad",
            "lightgbm_response_normalization": "none",
            "table_methods": list(RISK_PAPER_TABLE_MODELS),
        }
        for name, expected in expected_fields.items():
            if contract.get(name) != expected:
                raise ValueError(
                    f"paper contract has {name}={contract.get(name)!r}; "
                    f"expected {expected!r}."
                )
        eta = contract.get("eta_estimation")
        expected_eta = risk_paper_vara_config().eta
        if not isinstance(eta, Mapping):
            raise ValueError("paper contract has no eta_estimation mapping.")
        eta_fields = {
            "attention_support": "full_dense_off_diagonal_scores",
            "bounds": [expected_eta.lower, expected_eta.upper],
            "grid_size": expected_eta.grid_size,
            "local_tolerance": expected_eta.local_tolerance,
            "parameterization": "paper_eta_log_component_count_v1",
        }
        for name, expected in eta_fields.items():
            if eta.get(name) != expected:
                raise ValueError(
                    f"paper contract eta_estimation has {name}={eta.get(name)!r}; "
                    f"expected {expected!r}."
                )
        table = contract.get("standalone_table")
        if (
            not isinstance(table, Mapping)
            or table.get("attention_support") != "full_dense_off_diagonal_scores"
            or table.get("full_dense_attention_comparison") is not True
        ):
            raise ValueError("paper contract does not identify a full-dense table.")


def log_risk_target(returns: np.ndarray, *, epsilon: float = 1e-6) -> np.ndarray:
    """Return ``log(epsilon + return**2)`` while preserving missing cells."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("returns must have shape (T, p).")
    if not np.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("epsilon must be finite and positive.")
    return np.log(float(epsilon) + np.square(values))


@dataclass(frozen=True, slots=True)
class RiskForecastEvaluator:
    """Evaluate log-risk forecasts and validation-calibrated return variance."""

    benchmark_name: str | None = None
    returns_field: str = "returns"
    epsilon: float = 1e-6
    variance_floor: float = 1e-8

    def evaluate(
        self,
        data: PanelData,
        split: RollingSplit,
        forecasts: Mapping[str, np.ndarray],
    ) -> EvaluationResult:
        if self.returns_field not in data.fields:
            raise ValueError(
                f"risk evaluation requires panel field {self.returns_field!r}."
            )
        if len(forecasts) < 2:
            raise ValueError("risk evaluation requires at least two forecasts.")
        benchmark = self.benchmark_name or next(iter(forecasts))
        if benchmark not in forecasts:
            raise ValueError(f"risk benchmark {benchmark!r} is not a forecast method.")
        for name, forecast in forecasts.items():
            if np.asarray(forecast).shape != data.shape:
                raise ValueError(
                    f"forecast {name!r} has shape {np.asarray(forecast).shape}; "
                    f"expected {data.shape}."
                )
        plan = split.plan(data)
        metrics, monthly, calibration = evaluate_risk_forecasts(
            target=data.target,
            returns=data.fields[self.returns_field],
            dates=data.dates,
            forecasts=forecasts,
            validation_indices=plan.validation_indices,
            test_indices=plan.test_indices,
            benchmark_name=benchmark,
            epsilon=float(self.epsilon),
            variance_floor=float(self.variance_floor),
        )
        return EvaluationResult(
            metrics=metrics,
            by_origin=monthly,
            details={
                "benchmark": benchmark,
                "variance_calibration": calibration,
                "returns_field": self.returns_field,
                "target": "log(epsilon + squared return)",
                "evaluation_scale": "original_log_risk",
                "squared_error_metric": "log_risk_mspe",
            },
        )
