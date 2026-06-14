# M5 Data Files for the VARA Application

This folder contains the M5 data used by the high-dimensional forecasting application in `docs/VAR_Atten.tex`.

## Essential files for the full M5 application

Keep the following files if you want to reproduce the full M5 application from scratch.

| File | Purpose |
| --- | --- |
| `sales_train_evaluation.parquet` | Raw M5 evaluation sales history. This is the main wide-format source file with one row per item-store series and daily columns `d_1`, `d_2`, ... . It is needed to regenerate the long forecasting panel and to compute M5 attention features. |
| `calendar.parquet` | M5 calendar table. It maps `d_*` day identifiers to calendar dates and provides day-of-week, event, and state-level SNAP indicators. It is required for long-format conversion and calendar/event/SNAP attention features. |
| `sell_prices.parquet` | Weekly selling-price table by store and item. It is required for M5 domain attention features involving price level, price variation, release timing, and recent price behavior. |
| `long/sales_train_evaluation_long.parquet` | Forecasting-ready long panel used directly by `scripts/run_forecast_workflow.py`, `scripts/run_vara_forecast.py`, and `scripts/run_benchmarks.py`. It contains `date`, `series_id`, `value`, `d`, and hierarchy metadata columns. |

The first three files are the raw inputs. The long parquet file is generated from `sales_train_evaluation.parquet` and `calendar.parquet`, but it is kept because it is the direct input used by the forecasting scripts.

## Fetching the raw CSV files

The original M5 CSV files can be downloaded from the Kaggle M5 Forecasting Accuracy competition data page:

```text
https://www.kaggle.com/c/m5-forecasting-accuracy/data
```

The essential CSV files for the full M5 application are:

```text
data/m5/sales_train_evaluation.csv
data/m5/calendar.csv
data/m5/sell_prices.csv
```

Convert them to the raw parquet files expected by the repository with:

```bash
python scripts/prepare_m5_raw_parquet.py \
  --csv-dir data/m5/ \
  --output-dir data/m5 \
  --files sales_train_evaluation,calendar,sell_prices \
  --overwrite
```

## Scripts that generate M5 parquet files

The raw M5 parquet files under `data/m5` are generated from the extracted M5 CSV files:

```text
data/m5/sales_train_evaluation.parquet
data/m5/calendar.parquet
data/m5/sell_prices.parquet
```

The scripts that generate raw and derived parquet files are:

| Script | Inputs | Outputs | Purpose |
| --- | --- | --- | --- |
| `scripts/prepare_m5_raw_parquet.py` | Extracted M5 CSV files from Kaggle, especially `sales_train_evaluation.csv`, `calendar.csv`, and `sell_prices.csv` | `sales_train_evaluation.parquet`, `calendar.parquet`, `sell_prices.parquet`, and `m5_raw_parquet_metadata.json` | Converts the original M5 CSV files into the raw parquet inputs used by the forecasting and feature-preparation scripts. |
| `scripts/prepare_m5_long.py` | `sales_train_evaluation.parquet`, `sales_train_validation.parquet`, and `calendar.parquet` | `long/sales_train_evaluation_long.parquet`, `long/sales_train_validation_long.parquet`, and matching `*_series_metadata.parquet` files | Converts the wide M5 sales matrices with `d_*` columns into forecasting-ready long panels with `date`, `series_id`, `value`, `d`, and hierarchy columns. |
| `scripts/prepare_m5_attention_features.py` | `sales_train_evaluation.parquet`, `calendar.parquet`, and `sell_prices.parquet` | M5 attention feature parquet files, plus `.metadata.json` companions | Computes domain-feature rows for VARA attention, including demand summaries, calendar/event/SNAP response, price summaries, release timing, and hierarchy indicators. Supports `direct`, `seasonal_naive`, and `auto_ets` filters. |

For the current paper-style M5 application, `scripts/prepare_m5_long.py` is used to create the long forecasting panel, and `scripts/prepare_m5_attention_features.py` is used to create the AutoETS-filtered feature table used by VARA.

## Regenerating the long panel

If `long/sales_train_evaluation_long.parquet` is missing, first make sure the raw parquet files have been created from the CSV files, then regenerate the long panel with:

```bash
python scripts/prepare_m5_long.py \
  --raw-dir data/m5 \
  --output-dir data/m5/long \
  --splits evaluation
```

## Attention feature files

VARA can use precomputed M5 domain-feature tables for the attention matrix. These feature files are cached artifacts and should be kept with their metadata JSON companion when they are used in a run.

Examples:

| File | Status |
| --- | --- |
| `m5_attention_features_all_stores.parquet` and `.metadata.json` | Cached all-store feature table, usually for direct/raw-demand features. Useful, but not necessarily the final AutoETS-residual feature table used by the paper workflow. |
| `m5_attention_features_auto_ets_first500.parquet` and `.metadata.json` | First-500-series pilot feature table. Not the full application. |
| `m5_attention_features_CA_1.parquet` and `.metadata.json` | Single-store pilot feature table. Not the full application. |

For the paper M5 setup with `--forecast-scenario auto_ets`, the feature file should be generated with `--filter-method auto_ets`. The `.metadata.json` file is important because `run_vara_forecast.py` uses it to verify that the feature filter and seasonal lag match the forecasting scenario.

A full AutoETS-filtered feature table can be created with:

```bash
python scripts/prepare_m5_attention_features.py \
  --input-dir data/m5 \
  --out outputs/m5_vara_auto_ets_dense_full/features/m5_attention_features_auto_ets.parquet \
  --holdout 28 \
  --filter-method auto_ets \
  --seasonal-lag 7 \
  --max-series 0 \
  --ray-num-cpus 104 \
  --max-cpus-per-task 1 \
  --standardize-output
```

## Minimal keep list

For the full paper-style M5 application, the minimal data inputs are:

```text
data/m5/sales_train_evaluation.parquet
data/m5/calendar.parquet
data/m5/sell_prices.parquet
data/m5/long/sales_train_evaluation_long.parquet
```

If using precomputed VARA attention features, also keep the exact feature parquet used by the run and its `.metadata.json` companion.
