from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.build_datamine_risk_paper_results import (
    DM_COMPARISONS,
    LEGACY_LIGHTGBM_RESPONSE_NORMALIZATION,
    LEGACY_RESPONSE_STANDARDIZATION,
    POOLED_LIGHTGBM_METHOD,
    POOLED_LIGHTGBM_VARA_METHOD,
    RISKMETRICS_EWMA_METHOD,
    RISKMETRICS_EWMA_VARA_METHOD,
    STUDENT_T_GARCH_VARA_METHOD,
    TABLE_METHODS,
    TRAINING_ZSCORE_SCALE_FLOOR,
    build_argument_parser,
    build_paper_dm_results,
    evaluate_training_zscore_risk_table,
    validate_paper_input_contracts,
    validate_pooled_lightgbm_vara_contract,
    validate_riskmetrics_ewma_vara_contract,
    validate_student_t_vara_contract,
)
from scripts.datamine_paper_eta import (
    APPLICATION_CONTRACT_SCHEMA,
    DEFAULT_PAPER_ETA_BOUNDS,
    DEFAULT_PAPER_ETA_GRID_SIZE,
    DEFAULT_PAPER_ETA_LOCAL_TOLERANCE,
    ETA_PARAMETERIZATION,
    ETA_SCORE_SCALE_RULE,
    PaperEtaSpecification,
    attention_support_name,
    file_fingerprints,
)
from scripts.run_datamine_garch_vara import _print_eta_refit as print_garch_eta_refit
from scripts.run_datamine_har_vara import _print_eta_refit as print_har_eta_refit


COMPONENT_COUNT = 21_800


def _specification() -> PaperEtaSpecification:
    return PaperEtaSpecification(
        component_count=COMPONENT_COUNT,
        lower=DEFAULT_PAPER_ETA_BOUNDS[0],
        upper=DEFAULT_PAPER_ETA_BOUNDS[1],
        grid_size=DEFAULT_PAPER_ETA_GRID_SIZE,
        local_tolerance=DEFAULT_PAPER_ETA_LOCAL_TOLERANCE,
    )


def _core_fit_summary(specification: PaperEtaSpecification) -> pd.DataFrame:
    scale = specification.score_scale
    return pd.DataFrame(
        {
            "refit_index": [239, 251, 263],
            "eta_hat": [-2.0 * scale, 0.0, 0.8 * scale],
            "eta_on_boundary": [True, False, False],
            "eta_grid_best": [-2.0 * scale, 0.0, 2.0 * scale / 3.0],
            "eta_grid_best_objective": [-5.0, -4.0, -3.0],
            "eta_objective": [-4.9, -3.9, -2.9],
            "eta_bounds_lower": [-2.0 * scale] * 3,
            "eta_bounds_upper": [2.0 * scale] * 3,
            "eta_grid_size": [7] * 3,
            "eta_local_tolerance": [1e-4 * scale] * 3,
        }
    )


def test_paper_eta_uses_log_of_full_component_count() -> None:
    specification = _specification()
    expected_scale = math.log(COMPONENT_COUNT)

    assert specification.score_scale == pytest.approx(expected_scale)
    assert specification.score_scale != pytest.approx(math.log(500))
    assert specification.effective_bounds == pytest.approx(
        (-2.0 * expected_scale, 2.0 * expected_scale)
    )
    assert specification.effective_local_tolerance == pytest.approx(1e-4 * expected_scale)
    assert specification.paper_eta(specification.effective_slope(0.8)) == pytest.approx(0.8)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"component_count": 1}, "at least two"),
        ({"component_count": True}, "integer"),
        ({"lower": 2.0, "upper": 2.0}, "lower < upper"),
        ({"lower": -np.inf}, "finite"),
        ({"grid_size": 1}, "at least two"),
        ({"local_tolerance": 0.0}, "positive and finite"),
    ],
)
def test_paper_eta_rejects_invalid_configuration(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "component_count": COMPONENT_COUNT,
        "lower": -2.0,
        "upper": 2.0,
        "grid_size": 7,
        "local_tolerance": 1e-4,
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        PaperEtaSpecification(**values)  # type: ignore[arg-type]


def test_eta_contract_records_paper_and_effective_scales() -> None:
    specification = _specification()
    contract = specification.contract(
        attention_support=attention_support_name("exact")
    )

    assert contract["parameterization"] == ETA_PARAMETERIZATION
    assert contract["bounds"] == [-2.0, 2.0]
    assert contract["local_tolerance"] == 1e-4
    assert contract["score_scale"] == {
        "rule": ETA_SCORE_SCALE_RULE,
        "component_count": COMPONENT_COUNT,
        "value": specification.score_scale,
    }
    assert contract["effective_slope"]["bounds"] == pytest.approx(
        specification.effective_bounds
    )
    assert contract["effective_slope"]["local_tolerance"] == pytest.approx(
        specification.effective_local_tolerance
    )
    assert contract["attention_support"] == "deterministic_fixed_smallest_scores"


def test_core_fit_summary_is_converted_once_and_preserves_raw_slope() -> None:
    specification = _specification()
    core = _core_fit_summary(specification)
    original = core.copy(deep=True)

    converted = specification.convert_core_fit_summary(core)

    pd.testing.assert_frame_equal(core, original)
    np.testing.assert_allclose(converted["eta_hat"], [-2.0, 0.0, 0.8])
    np.testing.assert_allclose(
        converted["eta_effective_slope_hat"],
        original["eta_hat"],
    )
    np.testing.assert_allclose(converted["eta_bounds_lower"], -2.0)
    np.testing.assert_allclose(converted["eta_bounds_upper"], 2.0)
    np.testing.assert_allclose(converted["eta_local_tolerance"], 1e-4)
    np.testing.assert_allclose(
        converted["eta_effective_slope_local_tolerance"],
        specification.effective_local_tolerance,
    )
    assert converted["eta_parameterization"].eq(ETA_PARAMETERIZATION).all()
    assert converted["eta_component_count"].eq(COMPONENT_COUNT).all()
    np.testing.assert_allclose(converted["eta_objective"], original["eta_objective"])

    with pytest.raises(ValueError, match="already contains"):
        specification.convert_core_fit_summary(converted)


def test_persisted_summary_rejects_legacy_or_inconsistent_scale() -> None:
    specification = _specification()
    core = _core_fit_summary(specification)
    converted = specification.convert_core_fit_summary(core)

    with pytest.raises(ValueError, match="missing paper-eta columns"):
        specification.validate_persisted_fit_summary(core)

    inconsistent = converted.copy()
    inconsistent.loc[0, "eta_effective_slope_hat"] += 1.0
    with pytest.raises(ValueError, match="inconsistent eta scales"):
        specification.validate_persisted_fit_summary(inconsistent)

    wrong_bounds = converted.copy()
    wrong_bounds["eta_bounds_upper"] = 3.0
    wrong_bounds["eta_effective_slope_bounds_upper"] = 3.0 * specification.score_scale
    with pytest.raises(ValueError, match="configured eta_bounds_upper"):
        specification.validate_persisted_fit_summary(wrong_bounds)

    wrong_boundary = converted.copy()
    wrong_boundary["eta_on_boundary"] = [False, False, False]
    with pytest.raises(ValueError, match="eta_on_boundary"):
        specification.validate_persisted_fit_summary(wrong_boundary)


def test_refit_coverage_must_match_fit_summary() -> None:
    specification = _specification()
    converted = specification.convert_core_fit_summary(_core_fit_summary(specification))
    assignments = np.full(300, -1, dtype=np.int64)
    evaluation = np.arange(240, 270, dtype=np.int64)
    assignments[evaluation[:10]] = 239
    assignments[evaluation[10:20]] = 251
    assignments[evaluation[20:]] = 263

    specification.validate_refit_coverage(converted, assignments, evaluation)

    stale = converted.iloc[:-1].copy()
    with pytest.raises(ValueError, match="do not match assigned"):
        specification.validate_refit_coverage(stale, assignments, evaluation)


def test_file_fingerprints_detect_content_changes(tmp_path) -> None:
    first = tmp_path / "first.bin"
    second = tmp_path / "second.bin"
    first.write_bytes(b"abc")
    second.write_bytes(b"xyz")
    before = file_fingerprints({"second": second, "first": first})

    assert list(before) == ["first", "second"]
    assert before["first"]["size_bytes"] == 3
    assert len(before["first"]["sha256"]) == 64

    first.write_bytes(b"abd")
    after = file_fingerprints({"first": first, "second": second})
    assert after["first"]["sha256"] != before["first"]["sha256"]
    assert after["second"] == before["second"]


@pytest.mark.parametrize("printer", [print_har_eta_refit, print_garch_eta_refit])
def test_application_progress_reports_both_eta_scales(printer, capsys) -> None:
    specification = _specification()
    printer(
        {
            "refit_index": 239,
            "eta_hat": 0.8 * specification.score_scale,
            "eta_grid_best": 2.0 * specification.score_scale / 3.0,
            "eta_on_boundary": False,
        },
        specification,
    )

    output = capsys.readouterr().out
    assert "paper_eta_hat=0.800000" in output
    assert f"effective_slope={0.8 * specification.score_scale:.6f}" in output
    assert "paper_grid_best=0.666667" in output


def test_datamine_config_matches_application_eta_contract() -> None:
    config_path = Path(__file__).parents[1] / "data" / "DataMine" / "config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    specification = _specification()

    assert config["schema_version"] == 2
    assert config["status"] == "frozen_paper_eta_results"
    assert config["application_contract_schema"] == APPLICATION_CONTRACT_SCHEMA
    common = config["attention_parameterization"]
    assert common["name"] == ETA_PARAMETERIZATION
    assert common["component_count"] == COMPONENT_COUNT
    assert common["score_scale_value"] == pytest.approx(specification.score_scale)
    assert common["effective_slope_bounds"] == pytest.approx(
        specification.effective_bounds
    )
    assert common["effective_slope_local_tolerance"] == pytest.approx(
        specification.effective_local_tolerance
    )
    source_workflow = config["source_workflow"]
    assert source_workflow["rank_score_definition"] == (
        "monthly_strict_absolute_error_win_rate"
    )
    assert source_workflow["rank_score_comparators_per_method"] == 9
    assert source_workflow["rank_score_comparison"] == "strict_unrounded_less_than"
    assert source_workflow["rank_score_tie_method"] == "no_win_for_either_method"
    assert source_workflow["rank_score_comparison_methods"] == list(TABLE_METHODS)

    for section_name in (
        "final_har_vara",
        "exploratory_garch_vara",
        "exploratory_riskmetrics_ewma_vara",
        "exploratory_pooled_lightgbm_vara",
        "exploratory_student_t_garch_vara",
    ):
        section = config[section_name]
        assert section["eta_parameterization"] == ETA_PARAMETERIZATION
        assert section["component_count"] == COMPONENT_COUNT
        assert section["eta_bounds"] == list(DEFAULT_PAPER_ETA_BOUNDS)
        assert section["eta_grid_size"] == DEFAULT_PAPER_ETA_GRID_SIZE
        assert section["eta_local_tolerance"] == DEFAULT_PAPER_ETA_LOCAL_TOLERANCE
        assert section["eta_effective_slope_bounds"] == pytest.approx(
            specification.effective_bounds
        )
        assert section["eta_effective_slope_local_tolerance"] == pytest.approx(
            specification.effective_local_tolerance
        )
        assert "papereta_logp" in section["output_dir"]

    har_execution = config["final_har_vara"]
    assert har_execution["parallel_backend"] == "ray"
    assert har_execution["ray_num_cpus"] == 15
    assert har_execution["max_cpus_per_task"] == 1
    assert har_execution["ray_task_unit"] == "rolling_refit_block"

    student_t_execution = config["exploratory_student_t_garch_vara"]
    assert student_t_execution["parallel_backend"] == "ray"
    assert student_t_execution["ray_num_cpus"] == 15
    assert student_t_execution["max_cpus_per_task"] == 1
    assert student_t_execution["ray_task_unit"] == "rolling_refit_block"
    assert "garch11_t_ray15" in student_t_execution["student_t_garch_output"]

    ewma_execution = config["exploratory_riskmetrics_ewma_vara"]
    assert ewma_execution["ewma_riskmetrics"] is True
    assert ewma_execution["ewma_decay"] == pytest.approx(0.97)
    assert ewma_execution["ewma_decay_tuned"] is False
    assert ewma_execution["parallel_backend"] == "ray"
    assert ewma_execution["ray_num_cpus"] == 15
    assert ewma_execution["model_ids"]["validation_offset"] == (
        RISKMETRICS_EWMA_METHOD
    )
    assert ewma_execution["model_ids"]["lambda_vara"] == (
        RISKMETRICS_EWMA_VARA_METHOD
    )

    lightgbm_execution = config["exploratory_pooled_lightgbm_vara"]
    assert lightgbm_execution["parallel_backend"] == "ray"
    assert lightgbm_execution["ray_num_cpus"] == 15
    assert lightgbm_execution["max_cpus_per_task"] == 1
    assert lightgbm_execution["lightgbm_training_window_months"] == 240
    assert lightgbm_execution["lightgbm_hyperparameter_tuning"] == "none"
    assert lightgbm_execution["formula_features_in_base"] is False
    assert lightgbm_execution["model_ids"]["base"] == POOLED_LIGHTGBM_METHOD
    assert lightgbm_execution["model_ids"]["lambda_vara"] == (
        POOLED_LIGHTGBM_VARA_METHOD
    )

    paper_results = config["paper_results"]
    assert paper_results["require_matching_application_contracts"] is True
    assert "papereta_logp" in paper_results["har_vara_output"]
    assert "papereta_logp" in paper_results["garch_vara_output"]
    assert "riskmetrics_ewma_vara_papereta_logp" in (
        paper_results["riskmetrics_ewma_vara_output"]
    )
    assert "pooled_lightgbm_vara_papereta_logp" in (
        paper_results["pooled_lightgbm_vara_output"]
    )
    assert "garch11_t_ray15" in paper_results["student_t_garch_output"]
    assert "garch11_t_vara_papereta_logp" in (
        paper_results["student_t_garch_vara_output"]
    )
    zscore_evaluation = paper_results["training_zscore_evaluation"]
    assert zscore_evaluation == {
        "enabled": True,
        "cli_option": "--write-training-zscore-table",
        "evaluation_only": True,
        "forecast_refit": False,
        "training_window": "fixed_pre_validation",
        "training_window_months": 240,
        "training_start": "1988-01-31",
        "training_end": "2007-12-31",
        "statistics": "finite_only_componentwise_mean_and_population_variance",
        "scale_ddof": 0,
        "scale_floor": TRAINING_ZSCORE_SCALE_FLOOR,
        "crps_included": False,
        "metrics_artifact": "paper_table_metrics_training_zscore.csv",
        "monthly_losses_artifact": (
            "paper_table_monthly_losses_training_zscore.csv"
        ),
        "rankscore_artifact": (
            "paper_table_rankscore_by_month_training_zscore.csv"
        ),
        "scaler_artifact": "paper_table_training_zscore_scalers.csv",
        "markdown_artifact": "paper_table_training_zscore.md",
        "latex_artifact": "paper_table_training_zscore.tex",
    }

    paper_reference = config["paper_reference"]
    assert paper_reference["status"] == "current_paper_eta_results"
    assert paper_reference["eta_parameterization"] == ETA_PARAMETERIZATION
    assert paper_reference["rank_score_definition"] == (
        "monthly_strict_absolute_error_win_rate"
    )
    assert "papereta_logp" in paper_reference["output_dir"]
    assert paper_reference["rank_score_methods"] == 10
    assert paper_reference["rank_score_comparators_per_method"] == 9
    assert paper_reference["rank_score_audit_rows"] == 1200
    assert "papereta_logp" in paper_reference["comparison_results_dir"]
    assert paper_reference["riskmetrics_ewma_vara_lambda"][
        "rank_score"
    ] == pytest.approx(36.7592592593)
    assert paper_reference["student_t_garch_vara_lambda"]["rank_score"] == pytest.approx(
        69.4444444444
    )
    assert paper_reference["pooled_lightgbm_vara_lambda"][
        "log_risk_rmse"
    ] == pytest.approx(1.977029488)


def test_paper_result_cli_defaults_to_legacy_scaling_and_accepts_new_choices() -> None:
    required = [
        "/source",
        "/har",
        "/garch",
        "/output",
        "--student-t-garch-output",
        "/student-t",
        "--student-t-garch-vara-output",
        "/student-t-vara",
        "--riskmetrics-ewma-vara-output",
        "/ewma-vara",
        "--pooled-lightgbm-vara-output",
        "/lightgbm-vara",
    ]
    legacy = build_argument_parser().parse_args(required)
    assert legacy.response_standardization == LEGACY_RESPONSE_STANDARDIZATION
    assert (
        legacy.lightgbm_response_normalization
        == LEGACY_LIGHTGBM_RESPONSE_NORMALIZATION
    )

    normalized = build_argument_parser().parse_args(
        [
            *required,
            "--response-standardization",
            "training_zscore",
            "--lightgbm-response-normalization",
            "training_zscore",
        ]
    )
    assert normalized.response_standardization == "training_zscore"
    assert normalized.lightgbm_response_normalization == "training_zscore"


def test_paper_table_rejects_legacy_or_mixed_eta_contracts() -> None:
    specification = _specification()
    eta_contract = specification.contract(
        attention_support="deterministic_fixed_smallest_scores"
    )
    split = {
        "universe_selection_end": "2007-12-31",
        "validation_start": "2008-01-31",
        "validation_end": "2012-12-31",
        "test_start": "2013-01-31",
        "test_end": "2022-12-31",
    }
    common: dict[str, object] = {
        "application_contract_schema": APPLICATION_CONTRACT_SCHEMA,
        "source_output": "/source",
        "component_count": COMPONENT_COUNT,
        "forecast_scope": "validation_and_test",
        "split": split,
        "eta_estimation": eta_contract,
        "lambda_window_months": 24,
        "lambda_ridge": 0.001,
    }
    har = dict(common)
    garch = {**common, "har_vara_output": "/har"}

    validate_paper_input_contracts(
        har_contract=har,
        garch_contract=garch,
        source_output="/source",
        har_vara_output="/har",
        component_count=COMPONENT_COUNT,
        split=split,
    )

    normalized_har = {**har, "response_standardization": "training_zscore"}
    normalized_garch = {
        **garch,
        "response_standardization": "training_zscore",
    }
    validate_paper_input_contracts(
        har_contract=normalized_har,
        garch_contract=normalized_garch,
        source_output="/source",
        har_vara_output="/har",
        component_count=COMPONENT_COUNT,
        split=split,
        response_standardization="training_zscore",
    )

    with pytest.raises(ValueError, match="response_standardization"):
        validate_paper_input_contracts(
            har_contract=normalized_har,
            garch_contract=garch,
            source_output="/source",
            har_vara_output="/har",
            component_count=COMPONENT_COUNT,
            split=split,
            response_standardization="training_zscore",
        )

    legacy_har = {**har, "eta_estimation": {"parameterization": "legacy"}}
    with pytest.raises(ValueError, match="not paper-eta scaled"):
        validate_paper_input_contracts(
            har_contract=legacy_har,
            garch_contract=garch,
            source_output="/source",
            har_vara_output="/har",
            component_count=COMPONENT_COUNT,
            split=split,
        )

    wrong_garch = {**garch, "har_vara_output": "/other"}
    with pytest.raises(ValueError, match="requested HAR comparator"):
        validate_paper_input_contracts(
            har_contract=har,
            garch_contract=wrong_garch,
            source_output="/source",
            har_vara_output="/har",
            component_count=COMPONENT_COUNT,
            split=split,
        )


def _student_t_vara_contract() -> tuple[dict[str, object], dict[str, object]]:
    specification = _specification()
    eta_contract = specification.contract(
        attention_support="deterministic_fixed_smallest_scores"
    )
    split = {
        "universe_selection_end": "2007-12-31",
        "validation_start": "2008-01-31",
        "validation_end": "2012-12-31",
        "test_start": "2013-01-31",
        "test_end": "2022-12-31",
    }
    contract: dict[str, object] = {
        "application_contract_schema": APPLICATION_CONTRACT_SCHEMA,
        "source_output": "/source",
        "har_vara_output": "/har",
        "garch_variant": "student_t",
        "innovation_distribution": "variance-standardized Student-t",
        "upstream_garch_output": "/student-t-garch",
        "component_count": COMPONENT_COUNT,
        "forecast_scope": "validation_and_test",
        "split": split,
        "eta_estimation": eta_contract,
        "lambda_window_months": 24,
        "lambda_ridge": 0.001,
        "model_ids": {"lambda_vara": STUDENT_T_GARCH_VARA_METHOD},
        "forecast_artifacts": {
            "lambda_vara": "garch11_t_vara_lambda_forecast.npy"
        },
    }
    arguments: dict[str, object] = {
        "source_output": "/source",
        "har_vara_output": "/har",
        "student_t_garch_output": "/student-t-garch",
        "component_count": COMPONENT_COUNT,
        "split": split,
        "eta_estimation": eta_contract,
        "lambda_window_months": 24,
        "lambda_ridge": 0.001,
    }
    return contract, arguments


def test_student_t_vara_contract_accepts_matching_dedicated_refit() -> None:
    contract, arguments = _student_t_vara_contract()

    validate_student_t_vara_contract(contract=contract, **arguments)

    contract["response_standardization"] = "training_zscore"
    arguments["response_standardization"] = "training_zscore"
    validate_student_t_vara_contract(contract=contract, **arguments)


@pytest.mark.parametrize(
    ("section", "replacement", "message"),
    [
        ("field", ("garch_variant", "gaussian"), "garch_variant"),
        (
            "field",
            ("response_standardization", "training_zscore"),
            "response_standardization",
        ),
        ("model_ids", "legacy_t_vara", "wrong lambda model ID"),
        ("forecast_artifacts", "garch_vara_lambda_forecast.npy", "wrong forecast artifact"),
    ],
)
def test_student_t_vara_contract_rejects_mismatched_provenance(
    section: str,
    replacement: object,
    message: str,
) -> None:
    contract, arguments = _student_t_vara_contract()
    if section == "field":
        field, value = replacement  # type: ignore[misc]
        contract[str(field)] = value
    else:
        nested = dict(contract[section])  # type: ignore[arg-type]
        nested["lambda_vara"] = replacement
        contract[section] = nested

    with pytest.raises(ValueError, match=message):
        validate_student_t_vara_contract(contract=contract, **arguments)


def _riskmetrics_ewma_vara_contract() -> tuple[
    dict[str, object], dict[str, object]
]:
    specification = _specification()
    eta_contract = specification.contract(
        attention_support="deterministic_fixed_smallest_scores"
    )
    split = {
        "universe_selection_end": "2007-12-31",
        "validation_start": "2008-01-31",
        "validation_end": "2012-12-31",
        "test_start": "2013-01-31",
        "test_end": "2022-12-31",
    }
    variance_floor = 1e-8
    contract: dict[str, object] = {
        "application_contract_schema": APPLICATION_CONTRACT_SCHEMA,
        "source_output": "/source",
        "har_vara_output": "/har",
        "base_variant": "riskmetrics_ewma",
        "garch_variant": "riskmetrics_ewma",
        "innovation_distribution": "not parametrically specified",
        "upstream_garch_output": None,
        "component_count": COMPONENT_COUNT,
        "forecast_scope": "validation_and_test",
        "split": split,
        "eta_estimation": eta_contract,
        "lambda_window_months": 24,
        "lambda_ridge": 0.001,
        "riskmetrics_ewma": {
            "decay": 0.97,
            "observation_frequency": "monthly",
            "conditional_mean": 0.0,
            "initialization": "first_finite_squared_return_per_series",
            "missing_return_update": "carry_previous_variance_state",
            "one_step_ahead": True,
            "decay_tuned": False,
            "variance_floor": variance_floor,
        },
        "model_ids": {
            "raw": "riskmetrics_ewma_raw",
            "validation_offset": RISKMETRICS_EWMA_METHOD,
            "unit_vara": "riskmetrics_ewma_vara_unit",
            "lambda_vara": RISKMETRICS_EWMA_VARA_METHOD,
            "raw_unit_vara": "riskmetrics_ewma_raw_vara_unit",
        },
        "forecast_artifacts": {
            "raw": "riskmetrics_ewma_raw_forecast.npy",
            "conditional_variance": "riskmetrics_ewma_variance_forecast.npy",
            "validation_offset": (
                "riskmetrics_ewma_validation_offset_forecast.npy"
            ),
            "unit_vara": "riskmetrics_ewma_vara_unit_forecast.npy",
            "lambda_vara": "riskmetrics_ewma_vara_lambda_forecast.npy",
            "raw_unit_vara_correction": (
                "riskmetrics_ewma_raw_vara_unit_correction.npy"
            ),
            "raw_unit_vara": "riskmetrics_ewma_raw_vara_unit_forecast.npy",
        },
    }
    arguments: dict[str, object] = {
        "source_output": "/source",
        "har_vara_output": "/har",
        "component_count": COMPONENT_COUNT,
        "split": split,
        "eta_estimation": eta_contract,
        "lambda_window_months": 24,
        "lambda_ridge": 0.001,
        "variance_floor": variance_floor,
    }
    return contract, arguments


def test_riskmetrics_ewma_vara_contract_accepts_fixed_monthly_decay() -> None:
    contract, arguments = _riskmetrics_ewma_vara_contract()

    validate_riskmetrics_ewma_vara_contract(contract=contract, **arguments)

    contract["response_standardization"] = "training_zscore"
    arguments["response_standardization"] = "training_zscore"
    validate_riskmetrics_ewma_vara_contract(contract=contract, **arguments)


@pytest.mark.parametrize(
    ("section", "key", "replacement", "message"),
    [
        ("field", "base_variant", "gaussian", "base_variant"),
        (
            "field",
            "response_standardization",
            "training_zscore",
            "response_standardization",
        ),
        ("riskmetrics_ewma", "decay", 0.94, "EWMA specification"),
        ("riskmetrics_ewma", "decay_tuned", True, "EWMA specification"),
        ("model_ids", "lambda_vara", "legacy_ewma_vara", "wrong model IDs"),
        (
            "forecast_artifacts",
            "lambda_vara",
            "garch_vara_lambda_forecast.npy",
            "wrong forecast artifacts",
        ),
    ],
)
def test_riskmetrics_ewma_vara_contract_rejects_mismatched_provenance(
    section: str,
    key: str,
    replacement: object,
    message: str,
) -> None:
    contract, arguments = _riskmetrics_ewma_vara_contract()
    if section == "field":
        contract[key] = replacement
    else:
        nested = dict(contract[section])  # type: ignore[arg-type]
        nested[key] = replacement
        contract[section] = nested

    with pytest.raises(ValueError, match=message):
        validate_riskmetrics_ewma_vara_contract(contract=contract, **arguments)


def _pooled_lightgbm_vara_contract() -> tuple[
    dict[str, object], dict[str, object]
]:
    eta_contract = _specification().contract(
        attention_support="deterministic_fixed_smallest_scores"
    )
    split = {
        "universe_selection_end": "2007-12-31",
        "validation_start": "2008-01-31",
        "validation_end": "2012-12-31",
        "test_start": "2013-01-31",
        "test_end": "2022-12-31",
    }
    contract: dict[str, object] = {
        "application_contract_schema": APPLICATION_CONTRACT_SCHEMA,
        "source_output": "/source",
        "har_vara_output": "/har",
        "base_variant": "pooled_lightgbm",
        "component_count": COMPONENT_COUNT,
        "forecast_scope": "validation_and_test",
        "split": split,
        "eta_estimation": eta_contract,
        "error_window_months": 240,
        "refit_frequency_months": 12,
        "lambda_window_months": 24,
        "lambda_ridge": 0.001,
        "target_month_excluded_from_base_and_vara_refits": True,
        "current_or_future_test_target_used_for_base_fit": False,
        "lightgbm": {
            "version": "4.6.0",
            "target": "log(epsilon + squared return)",
            "formula_features_used": False,
            "component_identifiers_used": False,
            "normalization": "none",
            "features": ["lag_1", "mean_3", "mean_12"],
            "feature_windows": [1, 3, 12],
            "training_window_months": 240,
            "refit_frequency_months": 12,
            "hyperparameter_selection": (
                "prespecified single configuration; no validation or test search"
            ),
            "one_step_ahead": True,
            "parameters": {"objective": "regression_l2"},
        },
        "model_ids": {
            "base": POOLED_LIGHTGBM_METHOD,
            "unit_vara": "pooled_lightgbm_vara_unit",
            "lambda_vara": POOLED_LIGHTGBM_VARA_METHOD,
        },
        "forecast_artifacts": {
            "base": "pooled_lightgbm_forecast.npy",
            "base_error": "pooled_lightgbm_error.npy",
            "unit_vara": "pooled_lightgbm_vara_unit_forecast.npy",
            "lambda_vara": "pooled_lightgbm_vara_lambda_forecast.npy",
            "vara_correction": "vara_correction.npy",
        },
    }
    arguments: dict[str, object] = {
        "source_output": "/source",
        "har_vara_output": "/har",
        "component_count": COMPONENT_COUNT,
        "split": split,
        "eta_estimation": eta_contract,
        "lambda_window_months": 24,
        "lambda_ridge": 0.001,
    }
    return contract, arguments


def test_pooled_lightgbm_vara_contract_accepts_causal_fixed_configuration() -> None:
    contract, arguments = _pooled_lightgbm_vara_contract()

    validate_pooled_lightgbm_vara_contract(contract=contract, **arguments)

    legacy_specification = dict(contract["lightgbm"])  # type: ignore[arg-type]
    legacy_specification.pop("normalization")
    contract["lightgbm"] = legacy_specification
    validate_pooled_lightgbm_vara_contract(contract=contract, **arguments)

    contract["response_standardization"] = "training_zscore"
    normalized_specification = dict(contract["lightgbm"])  # type: ignore[arg-type]
    normalized_specification["normalization"] = "training_zscore"
    contract["lightgbm"] = normalized_specification
    arguments["response_standardization"] = "training_zscore"
    arguments["lightgbm_response_normalization"] = "training_zscore"
    validate_pooled_lightgbm_vara_contract(contract=contract, **arguments)


@pytest.mark.parametrize(
    "section,key,replacement,message",
    [
        ("field", "base_variant", "series_lightgbm", "base_variant"),
        (
            "field",
            "response_standardization",
            "training_zscore",
            "response_standardization",
        ),
        ("lightgbm", "formula_features_used", True, "formula_features_used"),
        (
            "lightgbm",
            "normalization",
            "training_zscore",
            "normalization",
        ),
        ("lightgbm", "training_window_months", 120, "training_window_months"),
        ("model_ids", "lambda_vara", "legacy_lightgbm_vara", "wrong model IDs"),
        (
            "forecast_artifacts",
            "lambda_vara",
            "legacy.npy",
            "wrong forecast artifacts",
        ),
    ],
)
def test_pooled_lightgbm_vara_contract_rejects_mismatched_provenance(
    section: str,
    key: str,
    replacement: object,
    message: str,
) -> None:
    contract, arguments = _pooled_lightgbm_vara_contract()
    if section == "field":
        contract[key] = replacement
    else:
        nested = dict(contract[section])  # type: ignore[arg-type]
        nested[key] = replacement
        contract[section] = nested

    with pytest.raises(ValueError, match=message):
        validate_pooled_lightgbm_vara_contract(contract=contract, **arguments)


def test_training_zscore_risk_table_uses_only_fixed_training_statistics() -> None:
    target = np.asarray([
        [0.0, 0.0],
        [2.0, 4.0],
        [20.0, 40.0],
        [30.0, 60.0],
        [3.0, 6.0],
        [5.0, 10.0],
    ])
    forecasts = {
        "baseline": target + np.asarray([1.0, 2.0]),
        "candidate": target + np.asarray([0.5, 1.0]),
    }
    dates = pd.date_range("2020-01-31", periods=len(target), freq="ME")

    metrics, monthly, audit, scalers = evaluate_training_zscore_risk_table(
        target=target,
        dates=dates,
        component_ids=np.asarray(["first", "second"]),
        forecasts=forecasts,
        training_indices=np.asarray([0, 1]),
        test_indices=np.asarray([4, 5]),
    )

    indexed = metrics.set_index("model")
    assert indexed.loc["baseline", "z_log_risk_mse"] == pytest.approx(1.0)
    assert indexed.loc["baseline", "z_log_risk_rmse"] == pytest.approx(1.0)
    assert indexed.loc["baseline", "z_log_risk_mae"] == pytest.approx(1.0)
    assert indexed.loc["baseline", "z_log_risk_bias"] == pytest.approx(1.0)
    assert indexed.loc["baseline", "z_absolute_r2"] == pytest.approx(0.0)
    assert indexed.loc["baseline", "rank_score"] == pytest.approx(0.0)
    assert indexed.loc["candidate", "z_log_risk_mse"] == pytest.approx(0.25)
    assert indexed.loc["candidate", "z_log_risk_rmse"] == pytest.approx(0.5)
    assert indexed.loc["candidate", "z_absolute_r2"] == pytest.approx(0.75)
    assert indexed.loc["candidate", "rank_score"] == pytest.approx(100.0)
    assert monthly["observed_cells"].eq(2).all()
    assert len(audit) == 4
    np.testing.assert_allclose(scalers["training_mean"], [1.0, 2.0])
    np.testing.assert_allclose(scalers["training_variance"], [1.0, 4.0])
    np.testing.assert_allclose(scalers["training_scale"], [1.0, 2.0])
    assert scalers["scale_ddof"].eq(0).all()

    future_perturbed = target.copy()
    future_perturbed[2:4] += 10_000.0
    perturbed_forecasts = {
        name: future_perturbed + (forecast - target)
        for name, forecast in forecasts.items()
    }
    _, _, _, perturbed_scalers = evaluate_training_zscore_risk_table(
        target=future_perturbed,
        dates=dates,
        component_ids=np.asarray(["first", "second"]),
        forecasts=perturbed_forecasts,
        training_indices=np.asarray([0, 1]),
        test_indices=np.asarray([4, 5]),
    )
    np.testing.assert_allclose(
        perturbed_scalers[["training_mean", "training_scale"]],
        scalers[["training_mean", "training_scale"]],
    )


def test_training_zscore_risk_table_rejects_invalid_component_scale() -> None:
    target = np.asarray([
        [1.0, 0.0],
        [1.0, 2.0],
        [2.0, 4.0],
        [3.0, 6.0],
    ])
    with pytest.raises(ValueError, match="near-zero component scales"):
        evaluate_training_zscore_risk_table(
            target=target,
            dates=pd.date_range("2020-01-31", periods=4, freq="ME"),
            component_ids=np.asarray(["constant", "variable"]),
            forecasts={
                "first": target + 0.1,
                "second": target + 0.2,
            },
            training_indices=np.asarray([0, 1]),
            test_indices=np.asarray([2, 3]),
            scale_floor=TRAINING_ZSCORE_SCALE_FLOOR,
        )


def test_table_methods_are_five_base_vara_pairs_in_paper_order() -> None:
    assert TABLE_METHODS == (
        "har_ridge",
        "har_vara_lambda",
        POOLED_LIGHTGBM_METHOD,
        POOLED_LIGHTGBM_VARA_METHOD,
        RISKMETRICS_EWMA_METHOD,
        RISKMETRICS_EWMA_VARA_METHOD,
        "garch11_offset",
        "garch_vara_lambda",
        "garch11_student_t_offset",
        STUDENT_T_GARCH_VARA_METHOD,
    )
    assert len(TABLE_METHODS) == 10
    assert len(TABLE_METHODS) - 1 == 9
    assert 120 * len(TABLE_METHODS) == 1200
    assert len(DM_COMPARISONS) == 10


def test_paper_dm_results_have_two_prespecified_five_pair_panels() -> None:
    dates = pd.date_range("2020-01-31", periods=6, freq="ME")
    loss_shape = np.array([1.0, 1.8, 1.3, 2.4, 1.6, 2.1])
    scales = {
        "ar1_ridge": 1.60,
        "har_ridge": 1.20,
        "har_vara_lambda": 0.80,
        POOLED_LIGHTGBM_METHOD: 1.15,
        POOLED_LIGHTGBM_VARA_METHOD: 0.88,
        RISKMETRICS_EWMA_METHOD: 1.30,
        RISKMETRICS_EWMA_VARA_METHOD: 0.90,
        "garch11_offset": 1.40,
        "garch_vara_lambda": 1.00,
        "garch11_student_t_offset": 1.25,
        STUDENT_T_GARCH_VARA_METHOD: 0.85,
    }
    monthly = pd.concat(
        [
            pd.DataFrame(
                {
                    "date": dates,
                    "model": model,
                    "squared_error": scale * loss_shape,
                    "gaussian_return_crps": 0.1 * scale * loss_shape[::-1],
                }
            )
            for model, scale in scales.items()
        ],
        ignore_index=True,
    )
    metrics = pd.DataFrame(
        {
            "model": list(scales),
            "log_risk_rmse": [math.sqrt(scale) for scale in scales.values()],
        }
    )

    results = build_paper_dm_results(monthly=monthly, metrics=metrics)

    assert len(results) == 10
    assert results.groupby("panel", sort=False).size().tolist() == [5, 5]
    assert list(
        results[["panel", "focal_model", "comparator_model"]].itertuples(
            index=False,
            name=None,
        )
    ) == list(DM_COMPARISONS)
    assert results["n_origins"].eq(6).all()
    assert (results["rmse_gain_percent"] > 0.0).all()
    assert (results["squared_error_dm_statistic"] < 0.0).all()
    assert (results["crps_dm_statistic"] < 0.0).all()
    assert results["squared_error_p_value_two_sided"].between(0.0, 1.0).all()
    assert results["crps_p_value_two_sided"].between(0.0, 1.0).all()
