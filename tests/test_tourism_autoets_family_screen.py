import numpy as np
import pandas as pd

from scripts.screen_tourism_autoets_families import (
    _select_candidates,
    default_candidates,
    validation_metrics,
)


def test_autoets_family_grid_is_unique_and_contains_dominant_no_trend_families() -> None:
    candidates = default_candidates()
    ids = [candidate.candidate_id for candidate in candidates]
    specifications = {candidate.model_specification for candidate in candidates}

    assert len(ids) == len(set(ids))
    assert {"ZZZ", "ZNZ", "MNM", "MNA", "ANA"}.issubset(specifications)
    assert [
        candidate.candidate_id
        for candidate in _select_candidates("mnm_fixed,ana_fixed")
    ] == ["mnm_fixed", "ana_fixed"]


def test_validation_metrics_use_pooled_original_scale_errors() -> None:
    forecasts = pd.DataFrame({
        "actual_original": [10.0, 20.0],
        "direct_auto_ets": [8.0, 24.0],
    })

    metrics = validation_metrics(forecasts)

    assert metrics["rmse"] == np.sqrt(10.0)
    assert metrics["mae"] == 3.0
    assert metrics["bias"] == 1.0
