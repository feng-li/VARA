import numpy as np
import pandas as pd

from scripts.recombine_tourism_calibrated_vara import (
    INTERCEPT_COLUMN,
    StackSpecification,
    apply_stack,
    fit_stack_coefficients,
    run_nested_backtest,
)


def _synthetic_panel() -> pd.DataFrame:
    rows = []
    for year in range(2011, 2016):
        for month in (1, 2):
            for series_id, group, scale in (("AAAAll", "A", 1.0), ("BAAAll", "B", 2.0)):
                direct = 100.0 * scale
                purpose_correction = float(month * scale)
                vara_correction = float((month - 1) * scale)
                actual = direct + 3.0 + 0.5 * purpose_correction + 0.25 * vara_correction
                rows.append({
                    "date": pd.Timestamp(year=year, month=month, day=1),
                    "year": year,
                    "month": month,
                    "series_id": series_id,
                    "group": group,
                    "actual": actual,
                    "direct_auto_ets": direct,
                    INTERCEPT_COLUMN: 1.0,
                    "purpose_correction": purpose_correction,
                    "all_vara_correction": vara_correction,
                })
    return pd.DataFrame(rows)


def test_calibrated_vara_stack_recovers_bounded_coefficients():
    panel = _synthetic_panel()
    specification = StackSpecification(
        "calibrated_purpose_vara",
        (INTERCEPT_COLUMN, "purpose_correction", "all_vara_correction"),
    )

    coefficients = fit_stack_coefficients(panel, specification)
    scored = apply_stack(panel, specification, coefficients)

    assert np.isclose(coefficients["coef_intercept"].iloc[0], 3.0)
    assert np.isclose(coefficients["coef_purpose_correction"].iloc[0], 0.5)
    assert np.isclose(coefficients["coef_all_vara_correction"].iloc[0], 0.25)
    assert np.allclose(scored["forecast"], scored["actual"])


def test_non_intercept_stack_coefficients_are_clipped_to_unit_interval():
    panel = _synthetic_panel()
    panel["actual"] = panel["direct_auto_ets"] + 4.0 * panel["purpose_correction"]
    specification = StackSpecification("purpose", ("purpose_correction",))

    coefficients = fit_stack_coefficients(panel, specification)

    assert np.isclose(coefficients["coef_purpose_correction"].iloc[0], 1.0)


def test_nested_backtest_excludes_final_year_from_selection_metrics():
    panel = _synthetic_panel()
    specifications = (
        StackSpecification("auto_ets", ()),
        StackSpecification(
            "calibrated_purpose_vara",
            (INTERCEPT_COLUMN, "purpose_correction", "all_vara_correction"),
        ),
    )
    _, _, metrics_before = run_nested_backtest(
        panel,
        specifications,
        first_year=2011,
        final_year=2015,
    )

    changed = panel.copy()
    changed.loc[changed["year"] == 2015, "actual"] += 1_000_000.0
    _, _, metrics_after = run_nested_backtest(
        changed,
        specifications,
        first_year=2011,
        final_year=2015,
    )

    pooled_before = metrics_before[metrics_before["period"] == "pooled_nested"].reset_index(drop=True)
    pooled_after = metrics_after[metrics_after["period"] == "pooled_nested"].reset_index(drop=True)
    pd.testing.assert_frame_equal(pooled_before, pooled_after)
