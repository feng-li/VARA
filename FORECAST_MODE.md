# Forecast Modes

This note describes the `--forecast-mode` option used by
`scripts/run_benchmarks.py` and `scripts/run_vara_forecast.py`.

## Summary

| Mode | Training scheme | Forecast per fit | Test observations used during evaluation? | Current use | Main caveat |
| --- | --- | --- | --- | --- | --- |
| `recursive` | Fit once on the training sample before the test window. | Full horizon, for example 12 or 28 steps. | No. The model forecasts the entire test window without seeing earlier test observations. | Measures true multi-step forecasting behavior from one origin. | Later horizons compound model error, especially for AR(1), VAR(1), and VARA(1). |
| `rolling-origin` | Refit or rerun at each test origin using an expanding window. | One step ahead at each origin. | Yes. After each test point is observed, it is included in the next training window. | Measures sequential one-step-ahead forecasting with data updates. | Usually easier than recursive multi-step forecasting, so it should not be compared as the same experiment. |

## Example

Suppose the final 12 observations are the test set.

With `--forecast-mode recursive`:

| Fit | Training sample | Forecast target |
| --- | --- | --- |
| 1 | All observations before the test window | All 12 test observations |

With `--forecast-mode rolling-origin`:

| Origin | Training sample | Forecast target |
| --- | --- | --- |
| 1 | All observations before the test window | Test observation 1 |
| 2 | Training sample plus actual test observation 1 | Test observation 2 |
| 3 | Training sample plus actual test observations 1-2 | Test observation 3 |
| ... | ... | ... |
| 12 | Training sample plus actual test observations 1-11 | Test observation 12 |

## Implementation Notes

In `scripts/run_benchmarks.py`, `recursive` calls `_forecast_recursive`.
That function uses `wide.iloc[: len(wide) - horizon]` as the training set and
calls `_forecast_once(..., h=horizon)` once.

`rolling-origin` calls `_forecast_rolling_origin`. That function loops over
`step in range(horizon)`, sets
`origin_end = len(wide) - horizon + step`, fits on `wide.iloc[:origin_end]`,
and calls `_forecast_once(..., h=1)` for the next test date.

In `scripts/run_vara_forecast.py`, the same distinction is used. Recursive mode
fits VARA once and calls `predict(h=horizon)`. Rolling-origin mode loops over
the test window, refits VARA at each origin, and calls `predict(h=1)`.

## Ray Parallelism In Benchmarks

Ray parallelism is separate from the forecast mode.

| Model type | Ray task unit | Rolling-origin behavior inside the task |
| --- | --- | --- |
| Univariate models such as `naive`, `seasonal_naive`, `drift`, `auto_ets`, `auto_arima`, `ar1`, `ar1_snaive` | One task per `(model, series_id)` | The task runs all origins for that model and series. |
| Panel models such as `var1`, `var1_snaive`, `deepar`, `nbeats` | One task per model | The task runs all origins for the full panel. |

So the benchmark script is currently parallelized by model and series for
univariate models, not by forecast origin.

## Related Option: Forecast Scenario

`--forecast-scenario` in `scripts/run_vara_forecast.py` is not the same as
`--forecast-mode`.

| Option | Meaning |
| --- | --- |
| `--forecast-mode` | Defines the evaluation protocol: one recursive multi-step forecast or repeated rolling one-step forecasts. |
| `--forecast-scenario direct` | VARA models the target panel directly. |
| `--forecast-scenario seasonal-naive-residual` | A seasonal-naive baseline is computed first; VARA models the residual panel, and the final forecast is baseline plus residual forecast. |

