import numpy as np
import pandas as pd

from scripts.build_tourism_table7_pairwise_dm import (
    FINAL_DATES,
    METHOD_ORDER,
    calculate_pairwise_dm_tests,
    render_latex,
    render_markdown,
)


def test_pairwise_dm_builds_complete_lower_triangle() -> None:
    time = np.arange(len(FINAL_DATES), dtype=float)
    losses = pd.DataFrame(
        {
            model: (
                100.0
                + 5.0 * model_index
                + (1.0 + model_index / 10.0) * np.sin(time)
            )
            for model_index, model in enumerate(METHOD_ORDER)
        },
        index=FINAL_DATES,
    )

    results = calculate_pairwise_dm_tests(losses)

    assert len(results) == 55
    assert (results["row_number"] > results["column_number"]).all()
    assert (results["dm_statistic_raw"] > 0.0).all()
    assert results["p_value_raw_two_sided"].between(0.0, 1.0).all()
    assert (
        results["p_value_raw_two_sided_holm_55"]
        >= results["p_value_raw_two_sided"]
    ).all()


def test_pairwise_dm_renderers_explain_direction_and_multiplicity() -> None:
    time = np.arange(len(FINAL_DATES), dtype=float)
    losses = pd.DataFrame(
        {
            model: (
                100.0
                + 4.0 * model_index
                + (1.0 + model_index / 8.0) * np.cos(time)
            )
            for model_index, model in enumerate(METHOD_ORDER)
        },
        index=FINAL_DATES,
    )
    results = calculate_pairwise_dm_tests(losses)

    latex = render_latex(results)
    markdown = render_markdown(results)

    assert r"\begin{tabular}{@{}l*{10}{c}@{}}" in latex
    assert "all 55 tests" in latex
    assert "row method minus column method" in latex
    assert "Holm correction" in latex
    assert latex.count(r"\\") >= 11
    assert markdown.splitlines()[0].count("|") == 12
