# Vector Autoregression with Attentions (VARA)

VARA is a research codebase for a network-augmented vector autoregression that uses structured attention weights to model cross-series dependence.

## Repository Layout

```text
.
├── vara/          # core model, likelihood, covariance, and estimation code
├── scripts/       # experiment runners, plotting helpers, and setup utilities
├── data/          # small checked-in input data files
├── patches/       # reference patch files kept with the repo
├── outputs/       # generated results, figures, and simulation artifacts
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Quick Start

1. Create an environment and install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run an experiment:

   ```bash
   python scripts/run_ecp_ray.py
   ```

3. Summarize a result file:

   ```bash
   python scripts/summarize_ecp.py outputs/results/results_coverage_@YYYY-MM-DD_HH.MM.SS.csv
   ```

4. Plot the checked-in varying-`p` example:

   ```bash
   python scripts/plot_varying_p.py
   ```

Generated files are written under `outputs/` instead of cluttering the repository root.

## Model

The model is

$$Y_t = \text{diag}(\beta)Y_{t-1} + \text{diag}(\gamma_a)W_aY_{t-1} + \varepsilon_t.$$

where:

- $Y_t \in \mathbb{R}^p$ is the vector of endogenous series at time $t$.
- $\beta$ controls own-lag effects.
- $W_a$ captures lagged cross-series dependence.
- $\gamma_a$ scales the lagged attention term.
- $\varepsilon_t$ is the innovation vector.

## Notes

- The reusable code lives in the `vara` package and can be imported directly from notebooks or other scripts.
- The scripts directory is intentionally lightweight; it contains runnable entrypoints rather than a large CLI framework.
- `data/varying_p.csv` is the small checked-in plotting example that used to live at the repository root.

## Forecasting API

The stable array-first interface is available from `vara.api`. A study receives
one panel, one trailing validation/test split, static series features, and one or
more base forecasters. The existing risk runners can enter through
`PrecomputedForecaster`; `HARRidgeForecaster` is the first native adapter.

```python
from vara.api import (
    ExecutionConfig,
    ForecastStudy,
    HARRidgeForecaster,
    PanelData,
    RiskForecastEvaluator,
    RollingSplit,
    StaticFeatureSet,
    VARAConfig,
    log_risk_target,
)

panel = PanelData(
    target=log_risk_target(returns),
    dates=dates,
    series_ids=series_ids,
    metadata=metadata,
    fields={"returns": returns},
)
study = ForecastStudy(
    data=panel,
    split=RollingSplit(validation=60, test=120),
    features=StaticFeatureSet(feature_matrix, feature_names, feature_provenance),
    vara=VARAConfig(attention_mode="dense"),
    evaluator=RiskForecastEvaluator(),
)
result = study.run(
    [HARRidgeForecaster(ridge_candidates=(0.01, 0.1, 1.0))],
    execution=ExecutionConfig(ray_num_cpus=16),
)
result.save("outputs/risk_study")
```
