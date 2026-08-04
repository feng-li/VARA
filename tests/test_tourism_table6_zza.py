import numpy as np
import pandas as pd

from scripts.build_tourism_table6_zza import (
    INSAMPLE_VARA_MODEL,
    NON_AUTOETS_MODELS,
    ZZA_BENCHMARK_MODELS,
    ZZA_VARA_MODEL_NAMES,
    build_table6_zza,
    calculate_origin_absolute_error_rank_scores,
    render_latex,
    render_markdown,
)


def _summary(models: list[str], start: float) -> pd.DataFrame:
    values = start + np.arange(len(models), dtype=float)
    return pd.DataFrame({
        "model": models,
        "rmse": values,
        "mae": values,
        "mase": values,
        "rmsse": values,
        "oos_r2": -values,
        "crps": values,
        "bias": values,
    })


def _origins(models: list[str], start: float) -> pd.DataFrame:
    rows = []
    for date in pd.date_range("2016-01-01", periods=12, freq="MS"):
        for offset, model in enumerate(models):
            rows.append({
                "model": model,
                "date": date,
                "rmse": start + float(offset),
                "mae": start + float(offset),
            })
    return pd.DataFrame(rows)


def test_table6_zza_uses_exactly_eleven_methods_and_recomputes_rankscore() -> None:
    standard = _summary(list(NON_AUTOETS_MODELS), 3.0)
    zza_benchmark = _summary(list(ZZA_BENCHMARK_MODELS), 9.0)
    zza_vara = _summary(list(ZZA_VARA_MODEL_NAMES), 1.0)
    insample_vara = _summary([INSAMPLE_VARA_MODEL], 2.0)
    standard_origins = _origins(list(NON_AUTOETS_MODELS), 3.0)
    zza_benchmark_origins = _origins(list(ZZA_BENCHMARK_MODELS), 9.0)
    zza_vara_origins = _origins(list(ZZA_VARA_MODEL_NAMES), 1.0)
    insample_vara_origins = _origins([INSAMPLE_VARA_MODEL], 2.0)

    table = build_table6_zza(
        standard,
        zza_benchmark,
        zza_vara,
        insample_vara,
        standard_origins,
        zza_benchmark_origins,
        zza_vara_origins,
        insample_vara_origins,
    )

    assert len(table) == 11
    assert table["model"].nunique() == 11
    assert table["rmse"].is_monotonic_increasing
    assert table.iloc[0]["model"] == "vara_auto_ets_lambda"
    assert table.iloc[0]["rank_score"] == 100.0
    assert table.iloc[-1]["rank_score"] == 0.0
    assert table.iloc[0]["mean_monthly_absolute_error_wins"] == 10.0
    zza_methods = table[table["autoets_family"] == "ZZA"]
    assert set(zza_methods["model"]) == {
        *ZZA_BENCHMARK_MODELS,
        *ZZA_VARA_MODEL_NAMES.values(),
        INSAMPLE_VARA_MODEL,
    }
    assert zza_methods["response_normalization"].eq("training_zscore").all()
    assert table.loc[
        table["autoets_family"] == "not_applicable",
        "response_normalization",
    ].eq("not_applicable").all()
    latex = render_latex(table)
    markdown = render_markdown(table)
    assert "Method & RMSE & RankScore" in latex
    assert "Rank & Method" not in latex
    assert markdown.splitlines()[0] == "| Method | RMSE | RankScore |"
    assert r"\label{tab:tourism_forecasting_results}" in latex
    assert "original visitor-night scale" in latex
    assert "ZZA" not in latex
    assert "in-sample errors" not in latex
    assert "MAE" not in latex
    assert "RMSSE" not in markdown


def test_origin_absolute_error_rankscore_uses_strict_pairwise_wins() -> None:
    standard = _origins(list(NON_AUTOETS_MODELS), 3.0)
    zza_benchmark = _origins(list(ZZA_BENCHMARK_MODELS), 9.0)
    zza_vara = _origins(list(ZZA_VARA_MODEL_NAMES), 1.0)
    insample_vara = _origins([INSAMPLE_VARA_MODEL], 1.0)

    scores, audit = calculate_origin_absolute_error_rank_scores(
        standard,
        zza_benchmark,
        zza_vara,
        insample_vara,
    )

    observed = scores.set_index("model")["rank_score"]
    assert observed["vara_auto_ets_lambda"] == 90.0
    assert observed[INSAMPLE_VARA_MODEL] == 90.0
    tied = audit[audit["model"].isin({
        "vara_auto_ets_lambda",
        INSAMPLE_VARA_MODEL,
    })]
    assert tied["absolute_error_wins"].eq(9).all()
