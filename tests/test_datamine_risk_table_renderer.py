from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_datamine_risk_paper_results import (
    FULL_DENSE_ATTENTION_SUPPORT,
    TABLE_METHODS,
    render_risk_table,
    render_mspe_comparison_table,
    render_training_zscore_table,
    write_risk_table,
    write_mspe_comparison_table,
)
from scripts.plot_datamine_table5_mse import load_table5_metrics


def _table_metrics() -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for pair_index in range(len(TABLE_METHODS) // 2):
        base_model = TABLE_METHODS[2 * pair_index]
        vara_model = TABLE_METHODS[2 * pair_index + 1]
        rows.extend([
            {
                "model": base_model,
                "log_risk_mspe": 4.0 + pair_index,
                "log_risk_mae": 1.0 + pair_index,
                "gaussian_return_crps": 0.01 + pair_index / 100.0,
                "log_risk_bias": -0.1,
                "absolute_r2": 0.10,
                "rank_score": 60.0,
            },
            {
                "model": vara_model,
                "log_risk_mspe": 2.25 + pair_index,
                "log_risk_mae": 1.2 + pair_index,
                "gaussian_return_crps": 0.01 + pair_index / 100.0,
                "log_risk_bias": 0.2,
                "absolute_r2": 0.20,
                "rank_score": 50.0,
            },
        ])
    # Rendering must use TABLE_METHODS rather than incoming row order.
    return pd.DataFrame(rows).iloc[::-1].reset_index(drop=True)


def test_dense_risk_table_uses_paper_order_caption_label_and_pairwise_bolding(
    tmp_path,
) -> None:
    markdown_path, latex_path = write_risk_table(
        _table_metrics(),
        output_dir=tmp_path,
        attention_support=FULL_DENSE_ATTENTION_SUPPORT,
    )

    assert markdown_path == tmp_path / "paper_table.md"
    assert latex_path == tmp_path / "paper_table.tex"
    markdown = markdown_path.read_text(encoding="utf-8")
    latex = latex_path.read_text(encoding="utf-8")

    assert "Parallel full-dense-attention Table 5 comparison" in markdown
    assert "uses all off-diagonal component scores" in markdown
    assert (
        r"\label{tab:datamine_forecasting_results_full_dense_attention}"
        in latex
    )
    assert "Parallel full-dense-attention comparison" in latex
    assert latex.count(r"\addlinespace") == 4

    har_base = next(
        line for line in markdown.splitlines() if line.startswith("| HAR ridge |")
    )
    har_vara = next(
        line
        for line in markdown.splitlines()
        if line.startswith("| HAR ridge + lambda*VARA |")
    )
    # Lower MSPE and larger R2 are on VARA; larger RankScore is on base.
    assert har_base.endswith("| 4.0000 | 10.0000 | **60.00** |")
    assert har_vara.endswith("| **2.2500** | **20.0000** | 50.00 |")
    assert "Log-risk MSPE" in markdown
    assert "Log-risk MSPE" in latex
    assert "Log-risk RMSE" not in markdown
    assert "Log-risk RMSE" not in latex
    for removed_column in ("Log-risk MAE", "CRPS", "Bias"):
        assert removed_column not in markdown
        assert removed_column not in latex

    positions = [markdown.index(f"| {model}") for model in (
        "HAR ridge",
        "Pooled LightGBM",
        "RiskMetrics-style EWMA",
        "GARCH(1,1)",
        "GARCH(1,1)-t",
    )]
    assert positions == sorted(positions)


def test_neutral_risk_table_has_standalone_table5_identity() -> None:
    markdown, latex = render_risk_table(
        _table_metrics(),
        attention_support="deterministic_fixed_smallest_scores",
    )
    assert "Standalone Table 5 comparison" in markdown
    assert "full-dense-attention" not in markdown
    assert (
        r"\label{tab:datamine_forecasting_results_table10_comparison}" in latex
    )


def test_plot_loader_accepts_legacy_rmse_artifact(tmp_path) -> None:
    metrics = _table_metrics().drop(columns="log_risk_mspe")
    metrics["log_risk_rmse"] = 2.0
    path = tmp_path / "metrics.csv"
    metrics.to_csv(path, index=False)

    _, values = load_table5_metrics(path)

    np.testing.assert_allclose(values["mse"][0], 4.0)
    np.testing.assert_allclose(values["mse"][1], 4.0)


def test_active_mspe_table_has_five_paired_rows(tmp_path) -> None:
    paired = pd.DataFrame([
        {
            "base_model": base_model,
            "vara_model": vara_model,
            "base_log_risk_mspe": 4.0 + pair_index,
            "vara_log_risk_mspe": 3.5 + pair_index,
            "dm_statistic": -2.0 - pair_index,
            "dm_p_value_two_sided": 0.01,
            "paired_r2_t_statistic": 2.5 + pair_index,
            "paired_r2_p_value_two_sided": 0.02,
        }
        for pair_index, (base_model, vara_model) in enumerate(
            zip(TABLE_METHODS[::2], TABLE_METHODS[1::2], strict=True)
        )
    ]).iloc[::-1]

    markdown, latex = render_mspe_comparison_table(paired)
    markdown_path, latex_path = write_mspe_comparison_table(
        paired,
        output_dir=tmp_path,
    )

    assert markdown.count("\n|") == 7
    assert "| HAR ridge | 4.0000 | 3.5000 | -2.000 | 0.0100 |" in markdown
    assert r"\label{tab:mspe-comparison}" in latex
    assert latex.count("(0.0100)") == 5
    assert latex.index("HAR ridge") < latex.index("Pooled LightGBM")
    assert markdown_path == tmp_path / "paper_table.md"
    assert latex_path == tmp_path / "paper_table.tex"


def test_training_zscore_table_omits_mae_crps_and_bias() -> None:
    metrics = _table_metrics().assign(
        z_log_risk_mse=0.9,
        z_log_risk_rmse=0.95,
        z_log_risk_mae=0.7,
        z_log_risk_bias=-0.1,
        z_absolute_r2=0.15,
    )

    markdown, latex = render_training_zscore_table(metrics)

    assert "| Method | z-RMSE | z-absolute R2 (%) | RankScore (%) |" in markdown
    assert r"\begin{tabular}{lrrr}" in latex
    for removed_column in ("z-MSE", "z-MAE", "z-Bias", "CRPS"):
        assert removed_column not in markdown
        assert removed_column not in latex


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda frame: frame.drop(columns="rank_score"), "missing columns"),
        (lambda frame: frame.iloc[:-1], "wrong method set"),
        (
            lambda frame: pd.concat([frame, frame.iloc[[0]]], ignore_index=True),
            "one row per model",
        ),
        (
            lambda frame: frame.assign(
                log_risk_mspe=lambda current: current["log_risk_mspe"].mask(
                    current.index == 0
                )
            ),
            "finite metric values",
        ),
    ],
)
def test_risk_table_renderer_rejects_invalid_metrics(mutate, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        render_risk_table(
            mutate(_table_metrics()),
            attention_support=FULL_DENSE_ATTENTION_SUPPORT,
        )
