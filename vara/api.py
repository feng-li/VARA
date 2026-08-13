"""Stable public imports for the panel forecasting framework."""

from .base import BaseForecast, BaseForecaster, HARRidgeForecaster, PrecomputedForecaster
from .correction import VARAForecast, VARAResidualCorrector
from .data import PanelData, RollingSplit
from .evaluation import EvaluationResult, ForecastEvaluator, PointForecastEvaluator
from .forecast_contracts import ExecutionConfig, PaperEtaConfig, RollingVARAConfig, StaticFeatureSet
from .pipeline import ForecastStudy, ForecastStudyResult
from .risk import RiskForecastEvaluator, log_risk_target

VARAConfig = RollingVARAConfig

__all__ = [
    "BaseForecast",
    "BaseForecaster",
    "EvaluationResult",
    "ExecutionConfig",
    "ForecastEvaluator",
    "ForecastStudy",
    "ForecastStudyResult",
    "HARRidgeForecaster",
    "PanelData",
    "PaperEtaConfig",
    "PointForecastEvaluator",
    "PrecomputedForecaster",
    "RiskForecastEvaluator",
    "RollingSplit",
    "StaticFeatureSet",
    "VARAConfig",
    "VARAForecast",
    "VARAResidualCorrector",
    "log_risk_target",
]
