# Tourism Data

This directory contains the cleaner wide monthly tourism source:

- `tourism.csv`: one row per month and one column per tourism series.
- `metadata.json`: basic shape and date-range metadata for `tourism.csv`.

The forecasting target used by `scripts/run_tourism_forecast_smoke.py` is the
76 detailed base series whose names end in `All` and whose hierarchy code has
depth 3, for example `AAAAll`, `AABAll`, and `GBDAll`. The source also contains
the full 111 `All`-suffix hierarchy nodes, including `TotalAll` and intermediate
aggregates, but those aggregate nodes are not forecast by default. Purpose-specific
columns ending in `Hol`, `Vis`, `Bus`, and `Oth` are kept in the source file but
are not forecast by default.

The smoke script builds a small deterministic feature matrix from the hierarchy
encoded in the base-series names and passes it with the response matrix to
`vara.forecast.VARAForecaster`.
