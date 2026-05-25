# Forecasting Data Requirements

The forecasting scripts use one general panel time-series schema. Inputs can be wide or long, CSV or parquet. Internally, the scripts pivot long input to a wide panel with one `date` column and one numeric column per series.

## Required long-panel schema

A long input table must contain at least these columns:

| Column | Type | Requirement |
| --- | --- | --- |
| `date` | date-like | Parseable by pandas, one timestamp per observation. |
| `series_id` | string | Stable identifier for one time series. |
| `value` | numeric | Observed value at `date` for `series_id`. |

Requirements:

- `(date, series_id)` must be unique.
- The panel must be balanced for the currently supported VARA and benchmark scripts: each selected `series_id` must have one value at every selected `date`.
- `value` must be numeric and finite after loading.
- Dates should be sorted or sortable; the loader sorts by date before modeling.
- Extra columns are allowed. The current forecasting scripts ignore them unless a model-specific feature builder reads them separately.

Use long input with:

```bash
python scripts/run_vara_forecast.py \
  --data-path path/to/panel_long.parquet \
  --data-format long \
  --date-col date \
  --series-col series_id \
  --value-col value
```

The benchmark script accepts the same data flags:

```bash
python scripts/run_benchmarks.py \
  --data-path path/to/panel_long.parquet \
  --data-format long \
  --date-col date \
  --series-col series_id \
  --value-col value
```

## Required wide-panel schema

A wide input table must contain:

- one date column, default `date`;
- one numeric column per time series.

Use wide input with `--data-format wide`, or leave `--data-format auto` when the file has a `date` column but no `series_id,value` pair.

## M5 raw-data requirements

The M5 converter expects the standard competition parquet files under `data/m5`:

| File | Purpose |
| --- | --- |
| `calendar.parquet` | Maps `d_1`, `d_2`, ... day ids to real calendar dates. Must contain `d` and `date`. |
| `sales_train_evaluation.parquet` | Evaluation sales history with metadata columns and `d_*` sales columns. |
| `sales_train_validation.parquet` | Validation sales history with metadata columns and `d_*` sales columns. |

Each sales file must contain:

- `id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id`;
- one or more `d_*` columns, for example `d_1`, `d_2`, ...;
- numeric sales counts in the `d_*` columns.

Run the converter with:

```bash
python scripts/prepare_m5_long.py \
  --raw-dir data/m5 \
  --output-dir data/m5/long \
  --splits evaluation,validation
```

The converter writes parquet files with this schema:

| Column | Meaning |
| --- | --- |
| `date` | Real calendar date from `calendar.parquet`. |
| `series_id` | M5 item-store series id, copied from `id`. |
| `value` | Unit sales count for the date and series. |
| `d` | Original M5 day id. |
| `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` | M5 hierarchy metadata. |

Full M5 outputs are large because they contain about 30,490 series by 1,913 or 1,941 days. For smoke tests, use `--max-series N`; for full forecasting, start with filtered subsets before running large panel models.

## Prepared attention feature tables

VARA can use a precomputed series-level feature table for attention construction:

```bash
python scripts/run_vara_forecast.py \
  --feature-source file \
  --feature-path path/to/features.parquet
```

The feature table must contain one row per selected `series_id`. The loader aligns rows to the selected forecast series, one-hot encodes non-metadata categorical feature columns, cleans non-finite numeric values, and standardizes numeric features before constructing the attention matrix.

For M5, create this table with:

```bash
python scripts/prepare_m5_attention_features.py \
  --input-dir data/m5 \
  --out data/m5/m5_attention_features_all_stores.parquet \
  --holdout 28
```

M5 attention features include hierarchy dummies, demand summaries, day-of-week response, SNAP/event/weekend response, and price summaries computed before the holdout window.
