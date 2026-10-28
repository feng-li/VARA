# Vector Autoregression with Attentions (VARA)

VARA is a network-augmented vector autoregression that uses structured attention weights to model cross-series dependence.

## Repository Layout

```text
.
├── vara/          # core model, likelihood, covariance, and estimation code
├── scripts/       # experiment runners, plotting helpers, and setup utilities
├── data/          # small checked-in input data files
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

Scripts using the shared bootstrap write generated files under
`~/running/VARA/outputs/`. Set `VARA_OUTPUTS_DIR` to override this location.

## Manuscript simulations

`scripts/run_full_simulation.py` implements the settings documented in
`docs/vara_simu/` using the existing JAX estimators and Ray workers. The default
table run uses 500 replications, 20 workers, seed `20260919`, and normal,
uniform, and Laplace innovations. Features and parameters are redrawn for each
replication and shared across innovation cases. Each panel contains the initial
innovation plus `T` transitions, with no burn-in. High-dimensional runs compare
full estimation with `ceil(log2(p)^2)` sampled response equations; both retain
all series as attention sources and fit every component in the second stage.

```bash
python scripts/run_full_simulation.py
```

Run the six-size pilot comparison separately:

```bash
python scripts/run_full_simulation.py --settings high --T-list 800 \
  --dist-list normal --R 100 --pilot-sizes 50,100,200,400,800,1600
python scripts/plot_simulation_pilot_sensitivity.py /path/to/summary_full_simulation.csv
```

Pilot subsets are nested and use the same data for each replication. Ray workers
are recreated between sizes. Summary CSVs include pooled component RMSE
(excluding gamma), its Monte Carlo standard error, valid-fit and standard-error
counts, and CPU timings for both estimation stages. CPU timing excludes data
generation but includes standard-error calculation. JAX and NumPy use different
random-number generators, so these runs follow the documented design without
reproducing the reference scripts' draws or hardware timings exactly.

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
    StaticFeatureSet,
    log_risk_target,
    risk_paper_split,
    risk_paper_vara_config,
    risk_paper_execution_config,
    risk_paper_corrector,
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
    split=risk_paper_split(),
    features=StaticFeatureSet(feature_matrix, feature_names, feature_provenance),
    vara=risk_paper_vara_config(panel.component_count),
    evaluator=RiskForecastEvaluator(),
    corrector=risk_paper_corrector(panel.component_count),
)
result = study.run(
    [HARRidgeForecaster(ridge_candidates=(0.01, 0.1, 1.0))],
    execution=risk_paper_execution_config(),
)
result.save("outputs/risk_study")
```

To reuse the exact five fitted risk benchmarks from a completed paper run:

```python
from vara.api import RiskBenchmarkArtifacts

benchmarks = RiskBenchmarkArtifacts.from_paper_contract(
    "outputs/datamine_vara/<paper-run>/paper/paper_contract.json"
)
result = study.run(benchmarks.forecasters(panel, study.split))
```

This loads HAR ridge, pooled LightGBM, RiskMetrics EWMA, Gaussian GARCH(1,1),
and Student-t GARCH(1,1) in the fixed paper-table order. The framework then
applies the configured VARA correction to each model's genuine forecast errors.

Each target is forecast from observations ending at the preceding origin. The
adapters also generate the pre-validation error history required by VARA;
`error_history_months` should match `VARAConfig.error_window_months`.

The risk application has one thin entry point. Its defaults use
the complete 21,800-series panel, the five fitted paper
benchmarks, a 60-month validation and 120-month test split, full-dense attention,
a 240-month median--MAD standardized residual window, annual refits, and a
24-month rolling LSE combination coefficient. The 15 annual refit tasks use
Ray with one CPU per task. The revised first stage estimates the attention
parameter from 1,000 equations sampled without replacement, using the same
seed and component subset across models and refits. Dense attention still uses
all other series; the second stage fits every component. Set `--pilot-size`
and `--pilot-seed` to override these defaults; a pilot size at least as large
as the panel gives the full criterion. `--attention-batch-size` controls only
memory use (default 512). Refit CSVs record pilot indices (zero-based in the
selected panel), seed, objective, and boundary status. Existing regressors,
normalization, and lambda calibration are unchanged. Run into a new directory:

```bash
python scripts/run_risk_study.py --output-dir outputs/risk_subsampled_pilot
```

The saved `run_config.json` records the paper profile and reports whether any
command-line override departed from it. For a quick diagnostic, add (for
example) `--max-series 200 --models paper_har_ridge`; use a separate
`--output-dir` for every materially different model or VARA configuration.

## Simulations

The existing simulation driver uses low-dimensional `(T,p,G)=(100,8,2),(200,12,3),(500,16,4)` and high-dimensional
`(200,200,10),(400,500,20),(800,1600,40)`. Defaults include normal, uniform, and Laplace innovations, 500 replications,
dense attention scaled by `log(p)`, and no feature screening. Each replication redraws features and parameters; the
initial observation is its first innovation and there are `T` transitions.

```bash
python scripts/run_full_simulation.py --results-dir outputs/simulation_revised
```

Low-dimensional runs use joint log-variance QMLE with three data-based starts.
High-dimensional runs compare full and subsampled concentrated likelihood on
the same panel. The subsampled pilot selects `ceil(log2(p)^2)` component rows
once; every selected row still attends to all other series. Both methods then
fit all component equations. Standard errors use plug-in moments rather than
the known innovation fourth moment. CPU timings include JAX compilation, so
they are not directly comparable with the NumPy reference implementation.

For the pilot-size sensitivity experiment (CSV output includes `RMSE_pool`):

```bash
python scripts/run_full_simulation.py --settings high --T-list 800 \
  --dist-list normal --R 100 --pilot-sizes 50,100,200,400,800,1600 \
  --results-dir outputs/simulation_pilot_sensitivity
```

Use `--ray-num-cpus 1` for sequential execution; Ray and sequential runs both
report completed replications with progress bars. Explicit `--T-list` overrides
the default time grid for every requested setting. Saved JSON records effective
designs and pilot settings. Legacy screened-feature runs remain available but
are not the manuscript default. These commands do not modify the manuscript
or the saved results under `docs/vara_simu/`.
