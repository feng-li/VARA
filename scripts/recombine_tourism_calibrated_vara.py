#!/usr/bin/env python3
"""Build a leakage-safe calibrated Tourism AutoETS/VARA forecast stack.

Candidate stacks are selected using expanding-window historical origins only.
The final year is scored after selection and is never used to estimate stack
coefficients or choose the winning specification.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

from vara.forecast_metrics import forecast_metric_tables


PURPOSES = ("Hol", "Vis", "Bus", "Oth")
INTERCEPT_COLUMN = "intercept"


@dataclass(frozen=True)
class StackSpecification:
    name: str
    correction_columns: tuple[str, ...]
    group_columns: tuple[str, ...] = ()


BASE_SPECIFICATIONS = (
    StackSpecification("auto_ets", ()),
    StackSpecification("calibrated_auto_ets", (INTERCEPT_COLUMN,)),
    StackSpecification("purpose_blend", ("purpose_correction",)),
    StackSpecification(
        "calibrated_purpose",
        (INTERCEPT_COLUMN, "purpose_correction"),
    ),
    StackSpecification(
        "calibrated_purpose_vara",
        (INTERCEPT_COLUMN, "purpose_correction", "all_vara_correction"),
    ),
)


def _melt_wide(path: Path, value_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "date" not in frame.columns:
        raise ValueError(f"Expected a date column in {path}.")
    out = frame.melt(id_vars="date", var_name="series_id", value_name=value_name)
    out["date"] = pd.to_datetime(out["date"])
    return out


def _forecast_dir_for_year(
    root: Path,
    year: int,
    *,
    first_year: int,
    final_year: int,
) -> Path:
    if int(year) == int(final_year):
        return Path(root) / "vara"
    origin_index = int(year) - int(first_year) + 1
    return Path(root) / "validation" / "origins" / f"origin_{origin_index:03d}" / "vara"


def _load_vara_correction_panel(
    root: Path,
    *,
    first_year: int,
    final_year: int,
    value_name: str,
) -> pd.DataFrame:
    frames = []
    for year in range(int(first_year), int(final_year) + 1):
        forecast_dir = _forecast_dir_for_year(
            root,
            year,
            first_year=first_year,
            final_year=final_year,
        )
        path = forecast_dir / "residual_forecast_unshrunk.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing unshrunk VARA residual forecast: {path}")
        frame = _melt_wide(path, value_name)
        frame["year"] = frame["date"].dt.year.astype(int)
        if set(frame["year"].unique()) != {int(year)}:
            raise ValueError(f"Unexpected forecast year in {path}.")
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _load_purpose_vara_corrections(
    root: Path,
    *,
    first_year: int,
    final_year: int,
) -> pd.DataFrame:
    rows = []
    for year in range(int(first_year), int(final_year) + 1):
        forecast_dir = _forecast_dir_for_year(
            root,
            year,
            first_year=first_year,
            final_year=final_year,
        )
        path = forecast_dir / "residual_forecast_unshrunk.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing purpose-aware VARA residual forecast: {path}")
        residuals = pd.read_csv(path)
        residuals["date"] = pd.to_datetime(residuals["date"])
        all_ids = [column for column in residuals.columns if column.endswith("All")]
        for _, row in residuals.iterrows():
            for series_id in all_ids:
                code = series_id[:-3]
                rows.append({
                    "date": row["date"],
                    "series_id": series_id,
                    "purpose_panel_all_vara_correction": float(row[series_id]),
                    "component_vara_correction": float(
                        sum(row[f"{code}{purpose}"] for purpose in PURPOSES)
                    ),
                })
    return pd.DataFrame(rows)


def build_strategy_panel(
    purpose_forecasts_path: Path,
    all_vara_root: Path,
    *,
    first_year: int,
    final_year: int,
    purpose_vara_root: Path | None = None,
) -> pd.DataFrame:
    panel = pd.read_csv(purpose_forecasts_path)
    required = {
        "date",
        "year",
        "month",
        "series_id",
        "group",
        "actual",
        "direct_auto_ets",
        "purpose_sum_auto_ets",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Purpose forecast panel is missing columns: {missing}")
    panel["date"] = pd.to_datetime(panel["date"])
    panel["year"] = panel["year"].astype(int)
    panel["month"] = panel["month"].astype(int)
    panel["series_id"] = panel["series_id"].astype(str)
    panel[INTERCEPT_COLUMN] = 1.0
    panel["purpose_correction"] = (
        panel["purpose_sum_auto_ets"] - panel["direct_auto_ets"]
    )

    all_vara = _load_vara_correction_panel(
        all_vara_root,
        first_year=first_year,
        final_year=final_year,
        value_name="all_vara_correction",
    )
    panel = panel.merge(
        all_vara.loc[:, ["date", "series_id", "all_vara_correction"]],
        on=["date", "series_id"],
        how="inner",
        validate="one_to_one",
    )

    if purpose_vara_root is not None:
        purpose_vara = _load_purpose_vara_corrections(
            purpose_vara_root,
            first_year=first_year,
            final_year=final_year,
        )
        panel = panel.merge(
            purpose_vara,
            on=["date", "series_id"],
            how="inner",
            validate="one_to_one",
        )

    expected_years = set(range(int(first_year), int(final_year) + 1))
    if set(panel["year"].unique()) != expected_years:
        raise ValueError(
            "Combined strategy panel does not contain every requested year: "
            f"expected {sorted(expected_years)}, got {sorted(panel['year'].unique())}."
        )
    return panel.sort_values(["date", "series_id"]).reset_index(drop=True)


def available_specifications(panel: pd.DataFrame) -> tuple[StackSpecification, ...]:
    specifications = list(BASE_SPECIFICATIONS)
    if "purpose_panel_all_vara_correction" in panel.columns:
        specifications.extend([
            StackSpecification(
                "calibrated_purpose_cross_vara",
                (INTERCEPT_COLUMN, "purpose_correction", "purpose_panel_all_vara_correction"),
            ),
            StackSpecification(
                "calibrated_purpose_component_vara",
                (INTERCEPT_COLUMN, "purpose_correction", "component_vara_correction"),
            ),
            StackSpecification(
                "calibrated_purpose_full_vara_stack",
                (
                    INTERCEPT_COLUMN,
                    "purpose_correction",
                    "all_vara_correction",
                    "purpose_panel_all_vara_correction",
                    "component_vara_correction",
                ),
            ),
        ])
    return tuple(specifications)


def _fit_coefficients(frame: pd.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    if not columns:
        return np.empty(0, dtype=float)
    design = frame.loc[:, list(columns)].to_numpy(dtype=float)
    target = (
        frame["actual"].to_numpy(dtype=float)
        - frame["direct_auto_ets"].to_numpy(dtype=float)
    )
    lower = np.asarray([-np.inf if column == INTERCEPT_COLUMN else 0.0 for column in columns])
    upper = np.asarray([np.inf if column == INTERCEPT_COLUMN else 1.0 for column in columns])
    return np.asarray(lsq_linear(design, target, bounds=(lower, upper)).x, dtype=float)


def fit_stack_coefficients(
    frame: pd.DataFrame,
    specification: StackSpecification,
) -> pd.DataFrame:
    columns = specification.correction_columns
    if specification.group_columns:
        rows = []
        grouper = (
            specification.group_columns[0]
            if len(specification.group_columns) == 1
            else list(specification.group_columns)
        )
        for key, group in frame.groupby(grouper, sort=True, observed=True):
            keys = (key,) if len(specification.group_columns) == 1 else tuple(key)
            row = {
                column: value
                for column, value in zip(specification.group_columns, keys, strict=True)
            }
            row.update({
                f"coef_{column}": value
                for column, value in zip(columns, _fit_coefficients(group, columns), strict=True)
            })
            row["n_training_observations"] = int(len(group))
            rows.append(row)
        return pd.DataFrame(rows)

    row = {
        f"coef_{column}": value
        for column, value in zip(columns, _fit_coefficients(frame, columns), strict=True)
    }
    row["n_training_observations"] = int(len(frame))
    return pd.DataFrame([row])


def apply_stack(
    frame: pd.DataFrame,
    specification: StackSpecification,
    coefficients: pd.DataFrame,
) -> pd.DataFrame:
    if specification.group_columns:
        out = frame.merge(
            coefficients,
            on=list(specification.group_columns),
            how="left",
            validate="many_to_one",
        )
    else:
        out = frame.copy()
        for column in specification.correction_columns:
            out[f"coef_{column}"] = float(coefficients[f"coef_{column}"].iloc[0])

    prediction = out["direct_auto_ets"].to_numpy(dtype=float).copy()
    for column in specification.correction_columns:
        coefficient_column = f"coef_{column}"
        if coefficient_column not in out or out[coefficient_column].isna().any():
            raise ValueError(f"Missing coefficient {coefficient_column} for {specification.name}.")
        prediction += (
            out[column].to_numpy(dtype=float)
            * out[coefficient_column].to_numpy(dtype=float)
        )
    out["forecast"] = np.maximum(prediction, 0.0)
    out["model"] = specification.name
    return out


def _point_metric_row(frame: pd.DataFrame, model: str, period: str) -> dict:
    error = frame["forecast"].to_numpy(dtype=float) - frame["actual"].to_numpy(dtype=float)
    return {
        "period": period,
        "model": model,
        "n_observations": int(len(frame)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def run_nested_backtest(
    panel: pd.DataFrame,
    specifications: tuple[StackSpecification, ...],
    *,
    first_year: int,
    final_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    predictions = []
    coefficient_rows = []
    fold_metric_rows = []
    for test_year in range(int(first_year) + 1, int(final_year)):
        train = panel[panel["year"] < test_year]
        test = panel[panel["year"] == test_year]
        for specification in specifications:
            coefficients = fit_stack_coefficients(train, specification)
            scored = apply_stack(test, specification, coefficients)
            scored["test_year"] = int(test_year)
            predictions.append(scored)
            fold_metric_rows.append(
                _point_metric_row(scored, specification.name, str(test_year))
            )
            coefficient_copy = coefficients.copy()
            coefficient_copy.insert(0, "model", specification.name)
            coefficient_copy.insert(1, "test_year", int(test_year))
            coefficient_rows.append(coefficient_copy)

    nested_predictions = pd.concat(predictions, ignore_index=True)
    nested_coefficients = pd.concat(coefficient_rows, ignore_index=True, sort=False)
    fold_metrics = pd.DataFrame(fold_metric_rows)

    pooled_rows = []
    for specification in specifications:
        model_predictions = nested_predictions[
            nested_predictions["model"] == specification.name
        ]
        pooled_rows.append(
            _point_metric_row(model_predictions, specification.name, "pooled_nested")
        )
    nested_summary = pd.DataFrame(pooled_rows).sort_values("rmse").reset_index(drop=True)
    baseline_rmse = float(
        nested_summary.loc[nested_summary["model"] == "auto_ets", "rmse"].iloc[0]
    )
    nested_summary["rmse_improvement_vs_auto_ets_pct"] = (
        100.0 * (baseline_rmse - nested_summary["rmse"]) / baseline_rmse
    )
    nested_summary["rank_rmse"] = np.arange(1, len(nested_summary) + 1)
    return nested_predictions, nested_coefficients, pd.concat(
        [fold_metrics, nested_summary], ignore_index=True, sort=False
    )


def _wide_values(frame: pd.DataFrame, value_column: str, series_ids: list[str]) -> np.ndarray:
    return (
        frame.pivot(index="date", columns="series_id", values=value_column)
        .sort_index()
        .loc[:, series_ids]
        .to_numpy(dtype=float)
    )


def run_recombination(
    *,
    data_path: Path,
    purpose_forecasts_path: Path,
    all_vara_root: Path,
    output_dir: Path,
    first_year: int,
    final_year: int,
    purpose_vara_root: Path | None,
) -> dict:
    panel = build_strategy_panel(
        purpose_forecasts_path,
        all_vara_root,
        first_year=first_year,
        final_year=final_year,
        purpose_vara_root=purpose_vara_root,
    )
    specifications = available_specifications(panel)
    nested_predictions, nested_coefficients, nested_metrics = run_nested_backtest(
        panel,
        specifications,
        first_year=first_year,
        final_year=final_year,
    )
    pooled = nested_metrics[nested_metrics["period"] == "pooled_nested"].copy()
    selected_model = str(pooled.sort_values("rmse").iloc[0]["model"])

    historical = panel[panel["year"] < int(final_year)]
    final = panel[panel["year"] == int(final_year)]
    final_predictions = []
    final_coefficient_rows = []
    for specification in specifications:
        coefficients = fit_stack_coefficients(historical, specification)
        scored = apply_stack(final, specification, coefficients)
        final_predictions.append(scored)
        coefficient_copy = coefficients.copy()
        coefficient_copy.insert(0, "model", specification.name)
        final_coefficient_rows.append(coefficient_copy)
    final_predictions_long = pd.concat(final_predictions, ignore_index=True)
    final_coefficients = pd.concat(final_coefficient_rows, ignore_index=True, sort=False)

    series_ids = sorted(final["series_id"].unique())
    dates = sorted(final["date"].unique())
    forecasts = {
        specification.name: _wide_values(
            final_predictions_long[final_predictions_long["model"] == specification.name],
            "forecast",
            series_ids,
        )
        for specification in specifications
    }
    test = _wide_values(final, "actual", series_ids)
    raw = pd.read_csv(data_path)
    raw["date"] = pd.to_datetime(raw["date"])
    train = (
        raw[raw["date"].dt.year < int(final_year)]
        .sort_values("date")
        .loc[:, series_ids]
        .to_numpy(dtype=float)
    )
    by_series, by_origin, final_summary = forecast_metric_tables(
        forecasts,
        test,
        train,
        dates,
        series_ids,
        season_length=12,
        forecast_mode="rolling-origin",
        horizon=12,
        summary_context={"data": str(data_path), "test_year": int(final_year)},
    )
    baseline_rmse = float(
        final_summary.loc[final_summary["model"] == "auto_ets", "rmse"].iloc[0]
    )
    final_summary["rmse_improvement_vs_auto_ets_pct"] = (
        100.0 * (baseline_rmse - final_summary["rmse"]) / baseline_rmse
    )
    final_summary = final_summary.sort_values("rmse").reset_index(drop=True)
    final_summary["rank_rmse"] = np.arange(1, len(final_summary) + 1)
    final_summary["selected_on_nested_validation"] = final_summary["model"].eq(selected_model)

    output_dir.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_dir / "strategy_panel.csv", index=False)
    nested_predictions.to_csv(output_dir / "nested_predictions.csv", index=False)
    nested_coefficients.to_csv(output_dir / "nested_coefficients.csv", index=False)
    nested_metrics.to_csv(output_dir / "nested_metrics.csv", index=False)
    final_predictions_long.to_csv(output_dir / "final_predictions_long.csv", index=False)
    final_coefficients.to_csv(output_dir / "final_coefficients.csv", index=False)
    final_summary.to_csv(output_dir / "metrics_summary.csv", index=False)
    by_series.to_csv(output_dir / "metrics_by_series.csv", index=False)
    by_origin.to_csv(output_dir / "metrics_by_origin.csv", index=False)

    selected_forecast = pd.DataFrame(forecasts[selected_model], columns=series_ids)
    selected_forecast.insert(0, "date", dates)
    selected_forecast.to_csv(output_dir / "forecast.csv", index=False)
    selection = {
        "selected_model": selected_model,
        "selection_metric": "pooled expanding-window nested RMSE",
        "selection_years": list(range(int(first_year) + 1, int(final_year))),
        "coefficient_training_years": list(range(int(first_year), int(final_year))),
        "final_test_year": int(final_year),
        "purpose_forecasts_path": str(purpose_forecasts_path),
        "all_vara_root": str(all_vara_root),
        "purpose_vara_root": None if purpose_vara_root is None else str(purpose_vara_root),
    }
    (output_dir / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("Nested strategy ranking:", flush=True)
    print(pooled.sort_values("rmse").to_string(index=False), flush=True)
    print("\nFinal test ranking:", flush=True)
    print(final_summary.to_string(index=False), flush=True)
    return {
        "panel": panel,
        "nested_metrics": nested_metrics,
        "final_summary": final_summary,
        "selection": selection,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path("data/tourism/tourism.csv"))
    parser.add_argument(
        "--purpose-forecasts-path",
        type=Path,
        default=Path("outputs/tourism_purpose_autoets_rolling_2011_2016/rolling_forecasts.csv"),
    )
    parser.add_argument(
        "--all-vara-root",
        type=Path,
        default=Path("outputs/tourism_vara_group_standardized_nested_5origins"),
    )
    parser.add_argument("--purpose-vara-root", type=Path, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tourism_calibrated_purpose_vara"),
    )
    parser.add_argument("--first-year", type=int, default=2011)
    parser.add_argument("--final-year", type=int, default=2016)
    args = parser.parse_args()
    run_recombination(
        data_path=args.data_path,
        purpose_forecasts_path=args.purpose_forecasts_path,
        all_vara_root=args.all_vara_root,
        output_dir=args.output_dir,
        first_year=args.first_year,
        final_year=args.final_year,
        purpose_vara_root=args.purpose_vara_root,
    )


if __name__ == "__main__":
    main()
