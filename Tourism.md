# Tourism Forecasting Notes

## Current Setup

- Data source: `data/tourism/tourism.csv`.
- Forecast target: detailed coded `All` tourism series, `series_depth=3`.
- Number of forecasted series: 76.
- Evaluation window: last 12 monthly observations, 2016-01-01 to 2016-12-01.
- Training window: 1998-01-01 to 2015-12-01.
- Forecast mode: rolling-origin one-step refitting across the 12 test months.
- Main VARA transformation: `log1p-seasonal-diff`, then inverse transformed back to the original scale for metrics.
- Reported metrics are computed on the held-out test window.

## Implemented Forecasting Components

- `scripts/run_tourism_forecast.py` runs VARA forecasts and writes:
  - `tourism_forecast.csv`
  - `tourism_test.csv`
  - `tourism_errors.csv`
  - `tourism_attention_features.csv`
  - `metrics_summary.csv`
  - `metrics_by_series.csv`
  - `metrics_by_origin.csv`
  - `tourism_metrics.csv`
- `scripts/run_benchmarks.py` runs baseline models with the same summary-table structure.
- Benchmark models currently include:
  - `naive`
  - `seasonal_naive`
  - `drift`
  - `auto_ets`
  - `auto_arima`
  - `ar1`
  - `var1`
- VARA attention feature sources currently include:
  - `tsfeatures`
  - `tourism`
  - `hybrid`
  - `seasonal-corr`

## Current Result Snapshot

The main saved comparison uses `series_depth=3`, `horizon=12`, and rolling-origin evaluation.

| Model / VARA source | Output folder | RMSE | MAE | MASE | RMSSE | Bias |
|---|---:|---:|---:|---:|---:|---:|
| naive | `outputs/tourism_benchmark_statsforecast` | 270.56 | 131.13 | 1.445 | 1.376 | 0.41 |
| seasonal_naive | `outputs/tourism_benchmark_statsforecast` | 153.92 | 92.64 | 1.109 | 0.997 | -12.74 |
| drift | `outputs/tourism_benchmark_statsforecast` | 271.22 | 131.45 | 1.449 | 1.379 | -0.65 |
| auto_ets | `outputs/tourism_benchmark_statsforecast` | 121.12 | 72.32 | 0.842 | 0.785 | -18.43 |
| auto_arima | `outputs/tourism_benchmark_statsforecast` | 124.33 | 75.54 | 0.888 | 0.801 | -26.31 |
| ar1 | `outputs/tourism_benchmark_statsforecast` | 209.05 | 101.19 | 1.098 | 1.080 | -44.20 |
| var1 | `outputs/tourism_benchmark_statsforecast` | 252.94 | 123.27 | 1.316 | 1.287 | -59.28 |
| VARA, tourism features | `outputs/tourism_forecast_tourism_features` | 155.60 | 92.82 | 1.121 | 1.016 | -7.78 |
| VARA, hybrid features | `outputs/tourism_forecast_hybrid_features` | 155.03 | 92.91 | 1.125 | 1.016 | -7.94 |
| VARA, seasonal-corr features | `outputs/tourism_forecast_seasonal_corr` | 155.83 | 93.79 | 1.125 | 1.008 | -8.42 |

## Main Findings

1. Seasonal structure dominates the tourism forecasting task.
   - `seasonal_naive` is a strong baseline and is hard for the current VARA specifications to beat.
   - Plain `naive`, `drift`, `ar1`, and unrestricted `var1` are much worse.

2. Classical univariate seasonal models are currently best.
   - `auto_ets` is the strongest saved benchmark, with RMSE 121.12.
   - `auto_arima` is close behind, with RMSE 124.33.
   - These models substantially outperform the current VARA variants.

3. Current VARA improves over weak dynamic baselines but not over seasonal naive.
   - VARA RMSE is around 155 across the tested attention feature sources.
   - This is far better than `var1` and `naive`, but slightly worse than `seasonal_naive`.

4. Feature engineering changed VARA only marginally.
   - `hybrid` improved RMSE slightly relative to tourism-only features.
   - `seasonal-corr` improved RMSSE slightly but worsened RMSE and MAE.
   - This suggests the feature matrix is not yet capturing forecast-relevant cross-series predictive structure.

5. VARA bias is moderate and less negative than several benchmarks.
   - Current VARA bias is around -8.
   - `auto_arima`, `ar1`, and `var1` show larger negative bias.
   - However, lower bias alone is not enough because RMSE and MAE remain worse than seasonal naive and auto-ETS.

## Main Issues

1. Attention is mostly descriptive rather than predictive.
   - Current `Z` features describe hierarchy, seasonality, tsfeatures, and residual similarity.
   - They do not directly estimate whether one series helps predict another series' next seasonal residual.

2. The VARA model is competing against a very strong seasonal baseline.
   - After log seasonal differencing, the remaining signal may be small.
   - VARA must add useful cross-series correction, not only reproduce seasonal persistence.

3. Symmetric similarity may be the wrong attention object.
   - Similar series are not necessarily useful predictors.
   - Tourism relationships may be directed: one group can lead another even when their levels or seasonal profiles differ.

4. Dense VAR-style cross-series modeling is fragile for this data.
   - `var1` performs poorly, which indicates unrestricted cross-series dependence is noisy.
   - VARA must be sparse, predictive, and strongly regularized to help.

5. Current evaluation is honest but limited.
   - The test period is only 12 months.
   - Model choices should be selected on a validation window before evaluating 2016.

