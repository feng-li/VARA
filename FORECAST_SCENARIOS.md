# Forecast Scenarios

This note describes how `scripts/run_vara_forecast.py` uses `--forecast-scenario`
to define the target that VARA models.

## Summary

| Scenario | VARA target | Final forecast |
|:--|:--|:--|
| `direct` | The selected response panel after `--y-transform` | VARA forecast directly |
| `seasonal-naive-residual` | Residuals after a seasonal-naive baseline | Seasonal-naive forecast plus VARA residual forecast |
| `auto_ets` | Residuals after an AutoETS baseline | AutoETS forecast plus VARA residual forecast |
| `statsmodels_ets` | Residuals after the default `statsmodels` `ETSModel` baseline | statsmodels ETS forecast plus VARA residual forecast |

The residual scenarios decompose the forecast into a univariate baseline and a
cross-series residual correction. The baseline captures each series' own level,
trend, and seasonality. VARA then models dependence left in the residual panel.

## Direct VARA

In the direct scenario, VARA is fitted to the transformed training panel:

```text
Y_model = transform(Y_train).
```

The final forecast is just the inverse-transformed VARA forecast:

```text
y_hat[t+h, i] = inverse_transform(VARA_forecast[t+h, i]).
```

## Seasonal-Naive Residual VARA

In the seasonal-naive residual scenario, the baseline forecast for series `i` is
the value from one seasonal lag earlier:

```text
y_hat_snaive[t+h, i] = y[t+h-seasonal_lag, i].
```

The residual panel is

```text
r[t, i] = y[t, i] - y[t-seasonal_lag, i].
```

VARA is fitted to `r`, and the final forecast is

```text
y_hat[t+h, i] = y_hat_snaive[t+h, i] + r_hat_vara[t+h, i].
```

## AutoETS Residual VARA

In the AutoETS residual scenario, AutoETS is first fitted separately to each
series using only the current training window.

For each series `i`, AutoETS produces in-sample fitted values:

```text
y_hat_auto_ets[t, i].
```

The residual panel is

```text
r[t, i] = y[t, i] - y_hat_auto_ets[t, i].
```

VARA is then fitted to the residual panel `r`, not to the original series `y`.
VARA forecasts future residuals:

```text
r_hat_vara[t+1, i], ..., r_hat_vara[t+H, i].
```

AutoETS also produces baseline future forecasts:

```text
y_hat_auto_ets[t+1, i], ..., y_hat_auto_ets[t+H, i].
```

The final forecast adds the two components:

```text
y_hat[t+h, i] = y_hat_auto_ets[t+h, i] + r_hat_vara[t+h, i].
```

Conceptually, AutoETS handles each series' own univariate structure. VARA is
responsible only for forecasting what AutoETS missed, using the cross-series
attention-weighted lag structure.

## statsmodels ETS Residual VARA

The `statsmodels_ets` scenario follows the same residual-correction structure as
`auto_ets`, but the univariate baseline is `statsmodels.tsa.exponential_smoothing.ets.ETSModel`
with its default settings. In the installed API this means additive errors, no
trend, no seasonal component, and estimated initialization. This gives a weaker,
fixed ETS filter that can be useful when the goal is to test whether VARA adds
cross-series information beyond a simple univariate smoother.

## Feature Use and Leakage

The forecast scenario changes the response panel that VARA fits. It does not
automatically change the attention feature matrix `Z`.

For example, with

```text
--forecast-scenario auto_ets
--feature-path data/m5/m5_attention_features_all_stores.parquet
```

VARA fits AutoETS residuals, but attention is still built from the supplied M5
feature file. This is leakage-safe only if the feature file was computed from
the training window. For M5 with a 28-day test set, the prepared feature file
should be generated with `holdout=28`.

For a fully residual-specific attention design, the no-leakage workflow should
be:

1. Fit the baseline model using only the current training window.
2. Compute fitted residuals inside that training window.
3. Extract residual features from those fitted residuals.
4. Combine residual features with any allowed static/domain features.
5. Build attention from the combined feature matrix.
6. Forecast future residuals with VARA.
7. Add the VARA residual forecast back to the baseline forecast.

For rolling-origin evaluation, steps 1-5 should be repeated at each origin using
only data observed by that origin.
