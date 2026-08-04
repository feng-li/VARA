import pandas as pd

from scripts.select_tourism_autoets_family_vara import select_candidate


def test_validation_selection_prefers_least_constrained_eligible_family() -> None:
    summary = pd.DataFrame({
        "candidate_id": [
            "zzz_standard",
            "fixed_better",
            "automatic_better",
            "automatic_not_stable",
        ],
        "automatic_family_components": [3, 0, 2, 3],
        "baseline_rmse_w36": [114.0, 112.0, 113.0, 113.0],
        "unit_rmse_w24": [115.0, 110.0, 111.0, 109.0],
        "rmse_gain_w24": [-1.0, 1.0, 1.0, 1.0],
        "rmse_gain_w12": [-1.0, 1.0, 1.0, -0.1],
    })

    selected, annotated = select_candidate(summary)

    assert selected == "automatic_better"
    selected_row = annotated[annotated["selected"]].iloc[0]
    assert selected_row["eligible"]
    assert selected_row["automatic_family_components"] == 2
