#!/usr/bin/env python3
"""Search leakage-safe VARA--AutoETS combinations on 76 Tourism series.

The candidate models use only the direct rolling AutoETS forecasts, their
previously observed errors, and the 76-series OOS-error VARA correction.  The
purpose-component forecasts are deliberately excluded.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


MODEL = "state_calibrated_vara_auto_ets_oos"
BLENDS = (0.25, 0.5, 0.75, 1.0)
WINDOWS = (12, 18, 24, 36, 48)
RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0)


@dataclass(frozen=True)
class StateBlendConfig:
    window_months: int
    feature_set: str
    ridge_alpha: float


def _shifted_rolling_mean(
    values: pd.core.groupby.SeriesGroupBy,
    window: int,
) -> pd.Series:
    return values.transform(
        lambda series: series.shift(1).rolling(
            int(window),
            min_periods=int(window),
        ).mean()
    )


def build_state_panel(
    autoets_forecasts_path: Path,
    vara_results_path: Path,
    strategy_id: str,
    *,
    expected_n_series: int = 76,
    drop_incomplete_state_origins: bool = False,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    autoets = pd.read_csv(autoets_forecasts_path)
    vara = pd.read_csv(vara_results_path)
    required_autoets = {
        "date",
        "series_id",
        "group",
        "actual",
        "direct_auto_ets",
        "autoets_history_end",
    }
    required_vara = {
        "date",
        "series_id",
        "strategy_id",
        "vara_correction",
        "history_end",
    }
    missing_autoets = sorted(required_autoets.difference(autoets.columns))
    missing_vara = sorted(required_vara.difference(vara.columns))
    if missing_autoets:
        raise ValueError(f"AutoETS forecasts are missing columns: {missing_autoets}")
    if missing_vara:
        raise ValueError(f"VARA results are missing columns: {missing_vara}")

    autoets["date"] = pd.to_datetime(autoets["date"])
    autoets["autoets_history_end"] = pd.to_datetime(
        autoets["autoets_history_end"]
    )
    autoets["series_id"] = autoets["series_id"].astype(str)
    if autoets.duplicated(["date", "series_id"]).any():
        raise ValueError("AutoETS forecasts contain duplicate date/series rows.")
    if autoets["series_id"].nunique() != int(expected_n_series):
        raise ValueError(
            f"Expected {expected_n_series} AutoETS series, found "
            f"{autoets['series_id'].nunique()}."
        )
    if not autoets["series_id"].str.endswith("All").all():
        raise ValueError("The 76-series search accepts all-purpose series only.")
    if not (autoets["autoets_history_end"] < autoets["date"]).all():
        raise ValueError("AutoETS histories must end before their target dates.")

    vara = vara[vara["strategy_id"].astype(str) == str(strategy_id)].copy()
    if vara.empty:
        raise ValueError(f"No VARA results use strategy {strategy_id!r}.")
    vara["date"] = pd.to_datetime(vara["date"])
    vara["history_end"] = pd.to_datetime(vara["history_end"])
    vara["series_id"] = vara["series_id"].astype(str)
    if vara.duplicated(["date", "series_id"]).any():
        raise ValueError("VARA results contain duplicate date/series rows.")
    if vara["series_id"].nunique() != int(expected_n_series):
        raise ValueError(
            f"Expected {expected_n_series} VARA series, found "
            f"{vara['series_id'].nunique()}."
        )
    if not (vara["history_end"] < vara["date"]).all():
        raise ValueError("VARA histories must end before their target dates.")

    history = autoets.loc[:, [
        "date",
        "series_id",
        "group",
        "actual",
        "direct_auto_ets",
        "autoets_history_end",
    ]].sort_values(["series_id", "date"]).copy()
    history["error"] = history["actual"] - history["direct_auto_ets"]
    by_series = history.groupby("series_id", sort=False)["error"]
    state_columns: dict[str, pd.Series] = {
        "lag1": by_series.shift(1),
        "lag2": by_series.shift(2),
        "lag12": by_series.shift(12),
    }
    for window in (3, 6, 12, 24, 60, 120):
        state_columns[f"mean{window}"] = _shifted_rolling_mean(
            by_series,
            window,
        )
    by_series_month = history.assign(
        _month=history["date"].dt.month
    ).groupby(["series_id", "_month"], sort=False)["error"]
    for years in (3, 5, 10):
        state_columns[f"seasonal_mean{years}"] = _shifted_rolling_mean(
            by_series_month,
            years,
        )
    history = pd.concat(
        [history, pd.DataFrame(state_columns, index=history.index)],
        axis=1,
    )

    group_error = (
        history.groupby(["date", "group"], as_index=False)["error"]
        .mean()
        .sort_values(["group", "date"])
    )
    by_group = group_error.groupby("group", sort=False)["error"]
    group_state = pd.DataFrame({
        "group_lag1": by_group.shift(1),
        "group_mean3": _shifted_rolling_mean(by_group, 3),
        "group_mean12": _shifted_rolling_mean(by_group, 12),
    }, index=group_error.index)
    group_error = pd.concat([group_error, group_state], axis=1)
    history = history.merge(
        group_error.drop(columns="error"),
        on=["date", "group"],
        how="left",
        validate="many_to_one",
    )

    global_error = (
        history.groupby("date", as_index=False)["error"].mean().sort_values("date")
    )
    global_error["global_lag1"] = global_error["error"].shift(1)
    for window in (3, 12):
        global_error[f"global_mean{window}"] = (
            global_error["error"].shift(1).rolling(
                window,
                min_periods=window,
            ).mean()
        )
    history = history.merge(
        global_error.drop(columns="error"),
        on="date",
        how="left",
        validate="many_to_one",
    )

    panel = vara.loc[:, [
        "date",
        "series_id",
        "vara_correction",
        "history_end",
    ]].merge(
        history,
        on=["date", "series_id"],
        how="inner",
        validate="one_to_one",
    )
    if len(panel) != len(vara):
        raise ValueError("Some VARA corrections do not match AutoETS forecasts.")
    panel["error_target"] = panel["actual"] - panel["direct_auto_ets"]
    panel["month"] = panel["date"].dt.month.astype(int)
    panel = panel.sort_values(["date", "series_id"]).reset_index(drop=True)

    state = [
        "vara_correction",
        "lag1",
        "lag12",
        "mean3",
        "mean6",
        "mean12",
        "group_lag1",
        "group_mean3",
        "group_mean12",
        "global_lag1",
        "global_mean12",
    ]
    long_state = [
        "vara_correction",
        "mean12",
        "mean24",
        "mean60",
        "mean120",
        "seasonal_mean3",
        "seasonal_mean5",
        "seasonal_mean10",
        "lag12",
    ]
    state_check_columns = list(dict.fromkeys([*state, *long_state]))
    complete_state = np.isfinite(
        panel[state_check_columns].to_numpy(dtype=float)
    ).all(axis=1)
    if not complete_state.all():
        if not drop_incomplete_state_origins:
            raise ValueError(
                "State features are incomplete; at least 120 prior AutoETS errors "
                "are required for every VARA origin."
            )
        panel = panel.loc[complete_state].copy()
        if panel.empty:
            raise ValueError("No complete state-feature origins remain.")
        rows_per_date = panel.groupby("date", observed=True).size()
        if not (rows_per_date == int(expected_n_series)).all():
            raise ValueError(
                "Dropping incomplete state histories left a partial forecast origin."
            )
        panel = panel.sort_values(["date", "series_id"]).reset_index(drop=True)

    group_dummies = pd.get_dummies(
        panel["group"],
        prefix="group",
        dtype=float,
    ).reset_index(drop=True)
    series_dummies = pd.get_dummies(
        panel["series_id"],
        prefix="series",
        dtype=float,
    ).reset_index(drop=True)
    month_dummies = pd.get_dummies(
        panel["month"],
        prefix="month",
        dtype=float,
    ).reset_index(drop=True)
    group_month_dummies = pd.get_dummies(
        panel["group"].astype(str) + "_m" + panel["month"].astype(str),
        prefix="group_month",
        dtype=float,
    ).reset_index(drop=True)
    interactions = group_dummies.mul(panel["vara_correction"], axis=0)
    interactions.columns = [f"vara_x_{column}" for column in group_dummies]
    series_interactions = series_dummies.mul(
        panel["vara_correction"],
        axis=0,
    )
    series_interactions.columns = [
        f"vara_x_{column}" for column in series_dummies
    ]
    group_month_interactions = group_month_dummies.mul(
        panel["vara_correction"],
        axis=0,
    )
    group_month_interactions.columns = [
        f"vara_x_{column}" for column in group_month_dummies
    ]
    seasonal_group_month_interactions = group_month_dummies.mul(
        panel["seasonal_mean10"],
        axis=0,
    )
    seasonal_group_month_interactions.columns = [
        f"seasonal10_x_{column}" for column in group_month_dummies
    ]
    panel = pd.concat(
        [
            panel,
            group_dummies,
            series_dummies,
            month_dummies,
            group_month_dummies,
            interactions,
            series_interactions,
            group_month_interactions,
            seasonal_group_month_interactions,
        ],
        axis=1,
    ).copy()

    group_columns = list(group_dummies.columns)
    series_columns = list(series_dummies.columns)
    month_columns = list(month_dummies.columns)
    group_month_columns = list(group_month_dummies.columns)
    interaction_columns = list(interactions.columns)
    series_interaction_columns = list(series_interactions.columns)
    group_month_interaction_columns = list(group_month_interactions.columns)
    seasonal_group_month_interaction_columns = list(
        seasonal_group_month_interactions.columns
    )
    feature_sets = {
        "vara": ["vara_correction"],
        "vara_lag1": ["vara_correction", "lag1"],
        "vara_lag12": ["vara_correction", "lag12"],
        "vara_mean3": ["vara_correction", "mean3"],
        "vara_mean6": ["vara_correction", "mean6"],
        "vara_mean12": ["vara_correction", "mean12"],
        "vara_mean24": ["vara_correction", "mean24"],
        "vara_mean60": ["vara_correction", "mean60"],
        "vara_mean120": ["vara_correction", "mean120"],
        "vara_seasonal3": ["vara_correction", "seasonal_mean3"],
        "vara_seasonal5": ["vara_correction", "seasonal_mean5"],
        "vara_seasonal10": ["vara_correction", "seasonal_mean10"],
        "vara_long_bias": long_state,
        "vara_long_bias_series": [*long_state, *series_columns],
        "vara_long_bias_series_month": [
            *long_state,
            *series_columns,
            *month_columns,
        ],
        "vara_long_bias_group_month": [
            *long_state,
            *group_month_columns,
        ],
        "vara_long_bias_group_month_slope": [
            *long_state,
            *group_month_columns,
            *group_month_interaction_columns,
        ],
        "vara_long_bias_group_month_two_slopes": [
            *long_state,
            *group_month_columns,
            *group_month_interaction_columns,
            *seasonal_group_month_interaction_columns,
        ],
        "vara_lag12_mean12": ["vara_correction", "lag12", "mean12"],
        "vara_recent_seasonal": [
            "vara_correction",
            "lag1",
            "lag12",
            "mean3",
            "mean12",
        ],
        "vara_seasonal10_group_month": [
            "vara_correction",
            "seasonal_mean10",
            *group_month_columns,
        ],
        "vara_seasonal10_group_month_slope": [
            "vara_correction",
            "seasonal_mean10",
            *group_month_columns,
            *group_month_interaction_columns,
            *seasonal_group_month_interaction_columns,
        ],
        "vara_group_state": [
            "vara_correction",
            "group_lag1",
            "group_mean3",
            "group_mean12",
        ],
        "vara_global_state": [
            "vara_correction",
            "global_lag1",
            "global_mean12",
        ],
        "vara_series_intercept": ["vara_correction", *series_columns],
        "vara_series_intercept_slope": [
            "vara_correction",
            *series_columns,
            *series_interaction_columns,
        ],
        "vara_series_intercept_slope_lag12": [
            "vara_correction",
            "lag12",
            *series_columns,
            *series_interaction_columns,
        ],
        "state": state,
        "state_group": [*state, *group_columns],
        "state_series": [*state, *series_columns],
        "state_series_month": [*state, *series_columns, *month_columns],
        "state_series_group_slope": [
            *state,
            *series_columns,
            *interaction_columns,
        ],
        "state_series_month_group_slope": [
            *state,
            *series_columns,
            *month_columns,
            *interaction_columns,
        ],
    }
    return panel, feature_sets


def forecast_config(
    panel: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    config: StateBlendConfig,
    target_dates: list[pd.Timestamp],
    *,
    blends: tuple[float, ...] = BLENDS,
) -> pd.DataFrame:
    columns = feature_sets[config.feature_set]
    pieces = []
    for target_date in sorted(pd.Timestamp(value) for value in target_dates):
        previous_dates = sorted(panel.loc[panel["date"] < target_date, "date"].unique())
        previous_dates = previous_dates[-int(config.window_months):]
        if len(previous_dates) < 12:
            raise ValueError(
                f"Only {len(previous_dates)} training months precede {target_date.date()}."
            )
        train = panel[panel["date"].isin(previous_dates)]
        test = panel[panel["date"] == target_date].sort_values("series_id").copy()
        if test.empty:
            raise ValueError(f"No forecasts are available for {target_date.date()}.")

        scaler = StandardScaler()
        design = scaler.fit_transform(train[columns].to_numpy(dtype=float))
        test_design = scaler.transform(test[columns].to_numpy(dtype=float))
        model = Ridge(alpha=float(config.ridge_alpha), fit_intercept=True)
        model.fit(design, train["error_target"].to_numpy(dtype=float))
        state_correction = model.predict(test_design)
        for blend in blends:
            prediction = test.loc[:, [
                "date",
                "series_id",
                "group",
                "actual",
                "direct_auto_ets",
                "autoets_history_end",
                "history_end",
            ]].copy()
            prediction["blend"] = float(blend)
            prediction["state_correction"] = state_correction
            prediction[MODEL] = np.maximum(
                prediction["direct_auto_ets"].to_numpy(dtype=float)
                + float(blend) * state_correction,
                0.0,
            )
            prediction["training_start"] = pd.Timestamp(previous_dates[0])
            prediction["training_end"] = pd.Timestamp(previous_dates[-1])
            prediction["n_training_months"] = int(len(previous_dates))
            pieces.append(prediction)
    return pd.concat(pieces, ignore_index=True)


def summarize_predictions(
    predictions: pd.DataFrame,
    config: StateBlendConfig,
) -> list[dict]:
    rows = []
    for blend, frame in predictions.groupby("blend", sort=True):
        error = (
            frame[MODEL].to_numpy(dtype=float)
            - frame["actual"].to_numpy(dtype=float)
        )
        annual = frame.assign(_squared_error=error**2).groupby(
            frame["date"].dt.year,
            observed=True,
        )["_squared_error"].mean().pow(0.5)
        rows.append({
            **asdict(config),
            "blend": float(blend),
            "rmse": float(np.sqrt(np.mean(error**2))),
            "mae": float(np.mean(np.abs(error))),
            "bias": float(np.mean(error)),
            "worst_year_rmse": float(annual.max()),
            "year_rmse_sd": float(annual.std(ddof=0)),
            **{
                f"rmse_{int(year)}": float(value)
                for year, value in annual.items()
            },
        })
    return rows


def _evaluate_config(
    config_dict: dict,
    panel: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    target_dates: list[pd.Timestamp],
) -> list[dict]:
    config = StateBlendConfig(**config_dict)
    return summarize_predictions(
        forecast_config(panel, feature_sets, config, target_dates),
        config,
    )


def run_search(
    *,
    autoets_forecasts_path: Path,
    vara_results_path: Path,
    strategy_id: str,
    output_dir: Path,
    validation_start: str,
    validation_end: str,
    expected_n_series: int,
    ray_num_cpus: int,
    max_cpus_per_task: int,
    drop_incomplete_state_origins: bool,
) -> pd.DataFrame:
    panel, feature_sets = build_state_panel(
        autoets_forecasts_path,
        vara_results_path,
        strategy_id,
        expected_n_series=expected_n_series,
        drop_incomplete_state_origins=drop_incomplete_state_origins,
    )
    target_dates = sorted(
        panel.loc[
            panel["date"].between(
                pd.Timestamp(validation_start),
                pd.Timestamp(validation_end),
            ),
            "date",
        ].unique()
    )
    if not target_dates:
        raise ValueError("No validation origins are available.")
    configs = [
        StateBlendConfig(window, feature_set, alpha)
        for window, feature_set, alpha in product(
            WINDOWS,
            feature_sets,
            RIDGE_ALPHAS,
        )
    ]

    start = time.time()
    rows: list[dict] = []
    if int(ray_num_cpus) > 1:
        import ray

        started_here = False
        if not ray.is_initialized():
            ray.init(
                num_cpus=int(ray_num_cpus),
                include_dashboard=False,
                ignore_reinit_error=True,
                _node_ip_address="127.0.0.1",
            )
            started_here = True
        try:
            remote = ray.remote(num_cpus=int(max_cpus_per_task))(_evaluate_config)
            panel_ref = ray.put(panel)
            feature_sets_ref = ray.put(feature_sets)
            dates_ref = ray.put(target_dates)
            futures = [
                remote.remote(
                    asdict(config),
                    panel_ref,
                    feature_sets_ref,
                    dates_ref,
                )
                for config in configs
            ]
            pending = list(futures)
            while pending:
                ready, pending = ray.wait(
                    pending,
                    num_returns=min(25, len(pending)),
                    timeout=30.0,
                )
                for result in ray.get(ready):
                    rows.extend(result)
                print(
                    f"State-combination search: {len(futures) - len(pending)}/"
                    f"{len(futures)} configs complete.",
                    flush=True,
                )
        finally:
            if started_here:
                ray.shutdown()
    else:
        for index, config in enumerate(configs, start=1):
            rows.extend(
                _evaluate_config(
                    asdict(config),
                    panel,
                    feature_sets,
                    target_dates,
                )
            )
            if index % 25 == 0 or index == len(configs):
                print(
                    f"State-combination search: {index}/{len(configs)} configs complete.",
                    flush=True,
                )

    results = pd.DataFrame(rows)
    results["rank_rmse"] = results["rmse"].rank(method="min")
    results["rank_mae"] = results["mae"].rank(method="min")
    results["rank_worst_year_rmse"] = results["worst_year_rmse"].rank(method="min")
    results["rank_year_rmse_sd"] = results["year_rmse_sd"].rank(method="min")
    results["robust_rank"] = results[[
        "rank_rmse",
        "rank_mae",
        "rank_worst_year_rmse",
        "rank_year_rmse_sd",
    ]].mean(axis=1)
    results = results.sort_values(
        ["rmse", "mae", "robust_rank"],
    ).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_dir / "validation_results.csv", index=False)
    selection = {
        "status": "selected without final-year outcomes",
        "selection_rule": "minimum pooled validation RMSE",
        "selected": json.loads(results.iloc[0].to_json()),
        "autoets_forecasts_path": str(autoets_forecasts_path),
        "vara_results_path": str(vara_results_path),
        "strategy_id": str(strategy_id),
        "validation_start": str(pd.Timestamp(validation_start).date()),
        "validation_end": str(pd.Timestamp(validation_end).date()),
        "n_validation_origins": len(target_dates),
        "n_series": int(expected_n_series),
        "purpose_component_forecasts_used": False,
        "feature_timing": "every error-state feature is shifted by at least one month",
        "drop_incomplete_state_origins": bool(drop_incomplete_state_origins),
        "elapsed_seconds": float(time.time() - start),
    }
    (output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2) + "\n"
    )
    print(results.head(30).to_string(index=False), flush=True)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--autoets-forecasts-path",
        type=Path,
        default=Path(
            "outputs/tourism_76_autoets_long/"
            "rolling_forecasts.csv"
        ),
    )
    parser.add_argument(
        "--vara-results-path",
        type=Path,
        default=Path(
            "outputs/tourism_oos_error_vara_leakage_safe_search/"
            "origin_results.csv"
        ),
    )
    parser.add_argument(
        "--strategy-id",
        default="predictive_avg_topk10_w24_stable05",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_76_state_blend_search"),
    )
    parser.add_argument("--validation-start", default="2013-01-01")
    parser.add_argument("--validation-end", default="2015-12-01")
    parser.add_argument("--expected-n-series", type=int, default=76)
    parser.add_argument(
        "--drop-incomplete-state-origins",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Drop complete early origins that lack 120 prior AutoETS errors; "
            "strict validation remains the default."
        ),
    )
    parser.add_argument("--ray-num-cpus", type=int, default=104)
    parser.add_argument("--max-cpus-per-task", type=int, default=1)
    args = parser.parse_args()
    run_search(
        autoets_forecasts_path=args.autoets_forecasts_path,
        vara_results_path=args.vara_results_path,
        strategy_id=args.strategy_id,
        output_dir=args.output_dir,
        validation_start=args.validation_start,
        validation_end=args.validation_end,
        expected_n_series=args.expected_n_series,
        ray_num_cpus=args.ray_num_cpus,
        max_cpus_per_task=args.max_cpus_per_task,
        drop_incomplete_state_origins=args.drop_incomplete_state_origins,
    )


if __name__ == "__main__":
    main()
