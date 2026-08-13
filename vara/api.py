"""Stable public imports for the panel forecasting framework."""

from .base import (
    BaseForecast,
    BaseForecaster,
    CachedForecaster,
    HARRidgeForecaster,
    PrecomputedForecaster,
)
from .benchmark_models import AutoARIMAForecaster, AutoETSForecaster, NBEATSForecaster
from .correction import VARAForecast, VARAResidualCorrector
from .data import PanelData, RollingSplit
from .evaluation import EvaluationResult, ForecastEvaluator, PointForecastEvaluator
from .forecast_contracts import ExecutionConfig, PaperEtaConfig, RollingVARAConfig, StaticFeatureSet
from .forecast_baselines import NeuralModelConfig
from .pipeline import ForecastStudy, ForecastStudyResult
from .risk import (
    GARCH11_RISK_MODEL,
    HAR_RIDGE_RISK_MODEL,
    POOLED_LIGHTGBM_RISK_MODEL,
    RISKMETRICS_EWMA_RISK_MODEL,
    RISK_BENCHMARK_MODELS,
    STUDENT_T_GARCH11_RISK_MODEL,
    RiskBenchmarkArtifacts,
    RiskForecastEvaluator,
    log_risk_target,
)

VARAConfig = RollingVARAConfig

__all__ = [
    "BaseForecast",
    "BaseForecaster",
    "AutoARIMAForecaster",
    "AutoETSForecaster",
    "CachedForecaster",
    "EvaluationResult",
    "ExecutionConfig",
    "ForecastEvaluator",
    "ForecastStudy",
    "ForecastStudyResult",
    "GARCH11_RISK_MODEL",
    "HAR_RIDGE_RISK_MODEL",
    "HARRidgeForecaster",
    "NBEATSForecaster",
    "NeuralModelConfig",
    "PanelData",
    "PaperEtaConfig",
    "PointForecastEvaluator",
    "POOLED_LIGHTGBM_RISK_MODEL",
    "PrecomputedForecaster",
    "RISKMETRICS_EWMA_RISK_MODEL",
    "RISK_BENCHMARK_MODELS",
    "RiskBenchmarkArtifacts",
    "RiskForecastEvaluator",
    "RollingSplit",
    "StaticFeatureSet",
    "STUDENT_T_GARCH11_RISK_MODEL",
    "VARAConfig",
    "VARAForecast",
    "VARAResidualCorrector",
    "log_risk_target",
]
