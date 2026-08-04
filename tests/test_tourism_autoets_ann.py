import pandas as pd

from scripts.compare_tourism_autoets_ann import build_ann_comparison


def test_ann_comparison_recomputes_rank_score_with_distinct_methods() -> None:
    base = pd.DataFrame({
        "model": ["auto_ets", "other"],
        "rmse": [2.0, 4.0],
        "mae": [2.0, 4.0],
        "mase": [2.0, 4.0],
        "rmsse": [2.0, 4.0],
        "rank_rmse": [1.0, 2.0],
        "rank_mae": [1.0, 2.0],
        "rank_mase": [1.0, 2.0],
        "rank_rmsse": [1.0, 2.0],
        "mean_accuracy_rank": [1.0, 2.0],
        "rank_score": [100.0, 0.0],
        "overall_rank": [1, 2],
    })
    ann = pd.DataFrame({
        "model": list((
            "reliability_gated_vara_auto_ets_oos",
            "raw_oos_vara",
            "direct_auto_ets",
        )),
        "rmse": [1.0, 3.0, 5.0],
        "mae": [1.0, 3.0, 5.0],
        "mase": [1.0, 3.0, 5.0],
        "rmsse": [1.0, 3.0, 5.0],
    })

    comparison = build_ann_comparison(base, ann)

    assert comparison["model"].tolist() == [
        "vara_auto_ets_ann_lambda",
        "auto_ets",
        "vara_auto_ets_ann_o",
        "other",
        "auto_ets_ann",
    ]
    assert comparison["rank_score"].tolist() == [100.0, 75.0, 50.0, 25.0, 0.0]
    ann_lambda = comparison.set_index("model").loc[
        "vara_auto_ets_ann_lambda"
    ]
    assert ann_lambda["rmse_improvement_vs_auto_ets_pct"] == 50.0
    assert ann_lambda["rmse_improvement_vs_auto_ets_ann_pct"] == 80.0
