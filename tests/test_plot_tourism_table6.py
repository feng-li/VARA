from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_tourism_table6_zza import (
    DISPLAY_NAMES,
    INSAMPLE_VARA_MODEL,
    NON_AUTOETS_MODELS,
    ZZA_BENCHMARK_MODELS,
    ZZA_VARA_MODEL_NAMES,
)
from scripts.plot_tourism_table6 import (
    AUTOETS_RELATED_MODELS,
    load_plot_data,
    plot_table6_comparison,
)


def _origin_metrics(models: tuple[str, ...], offset: float) -> pd.DataFrame:
    rows = []
    for model_index, model in enumerate(models):
        for date_index, date in enumerate(
            pd.date_range("2016-01-01", periods=12, freq="MS")
        ):
            rows.append({
                "model": model,
                "date": date,
                "rmse": offset + model_index + date_index / 10.0,
            })
    return pd.DataFrame(rows)


def test_plot_table6_loads_exact_inputs_and_writes_figure(tmp_path: Path) -> None:
    standard_path = tmp_path / "standard.csv"
    zza_path = tmp_path / "zza.csv"
    vara_path = tmp_path / "vara.csv"
    insample_path = tmp_path / "insample.csv"
    table_path = tmp_path / "table.csv"
    _origin_metrics(NON_AUTOETS_MODELS, 80.0).to_csv(
        standard_path,
        index=False,
    )
    _origin_metrics(ZZA_BENCHMARK_MODELS, 90.0).to_csv(
        zza_path,
        index=False,
    )
    _origin_metrics(tuple(ZZA_VARA_MODEL_NAMES), 70.0).to_csv(
        vara_path,
        index=False,
    )
    _origin_metrics((INSAMPLE_VARA_MODEL,), 75.0).to_csv(
        insample_path,
        index=False,
    )

    models = list(DISPLAY_NAMES)
    table = pd.DataFrame({
        "model": models,
        "method": [DISPLAY_NAMES[model] for model in models],
        "rmse": np.linspace(100.0, 200.0, len(models)),
        "response_normalization": [
            "training_zscore"
            if model in AUTOETS_RELATED_MODELS
            else "not_applicable"
            for model in models
        ],
    })
    table.to_csv(table_path, index=False)

    origins, loaded_table = load_plot_data(
        standard_path,
        zza_path,
        vara_path,
        insample_path,
        table_path,
    )

    assert len(origins) == 11 * 12
    assert origins["model"].nunique() == 11
    assert origins["date"].nunique() == 12
    assert loaded_table["model"].nunique() == 11
    assert loaded_table["mse"].is_monotonic_increasing
    assert not loaded_table["plot_label"].str.contains("ZZA", regex=False).any()
    assert loaded_table.loc[
        loaded_table["model"].eq(INSAMPLE_VARA_MODEL),
        "plot_label",
    ].item() == "VARA–AutoETS"
    assert set(
        loaded_table.loc[
            loaded_table["response_normalization"].eq("training_zscore"),
            "model",
        ]
    ) == AUTOETS_RELATED_MODELS

    output_pdf = tmp_path / "figure.pdf"
    output_png = tmp_path / "figure.png"
    plot_data_csv = tmp_path / "plot_data.csv"
    plot_table6_comparison(
        origins,
        loaded_table,
        output_pdf=output_pdf,
        output_png=output_png,
        plot_data_csv=plot_data_csv,
    )

    assert output_pdf.stat().st_size > 0
    assert output_png.stat().st_size > 0
    plotted = pd.read_csv(plot_data_csv)
    assert len(plotted) == 11 * 12
    assert "mse" in plotted
    assert "rmse" not in plotted
    assert "rank_score" not in plotted
    expected_mse = np.square(
        _origin_metrics(NON_AUTOETS_MODELS, 80.0)["rmse"].to_numpy()
    )
    observed_mse = plotted.loc[
        plotted["source"].eq("unchanged_non_autoets_benchmark"),
        "mse",
    ].sort_values().to_numpy()
    np.testing.assert_allclose(observed_mse, np.sort(expected_mse))
