import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd

from vara.datamine import (
    DataMinePanel,
    audit_return_file,
    build_dense_panel,
    canonical_signal_metadata,
    estimate_global_lambda,
    estimate_group_lambdas,
    estimate_series_lambdas,
    fit_robust_panel_scaler,
    fixed_garch11_grid_forecast,
    formula_feature_blocks,
    load_signal_metadata,
    make_target,
    prevalidation_selection_scores,
    preprocess_feature_blocks,
    relative_panel_split,
    rolling_group_lambdas,
    rolling_group_vara_corrections,
    rolling_har_vara_corrections,
    rolling_ridge_forecast,
    rolling_vara_corrections,
    select_frozen_universe,
)


def _metadata(signal_ids, weighting="ew"):
    return pd.DataFrame({
        "signalid": [str(value) for value in signal_ids],
        "formula_type": ["v1/v2" if value % 2 else "diff(v1)/lag(v2)" for value in signal_ids],
        "numerator": [f"num_{value % 3}" for value in signal_ids],
        "denominator": [f"den_{value % 4}" for value in signal_ids],
        "weighting": weighting,
        "component_id": [f"{weighting.upper()}_{value}" for value in signal_ids],
        "formula_label": [f"formula_{value}" for value in signal_ids],
    })


class DataMineDataTests(unittest.TestCase):
    def test_metadata_aliases_are_mapped_explicitly(self):
        raw = pd.DataFrame({
            "signalid": [2, 1],
            "signal_form": ["v1/v2", "diff(v1)/lag(v2)"],
            "v1": ["sale", "at"],
            "v2": ["at", "sale"],
        })
        result = canonical_signal_metadata(raw, weighting="ew")

        self.assertEqual(list(result["signalid"]), ["1", "2"])
        self.assertEqual(list(result["component_id"]), ["EW_1", "EW_2"])
        self.assertEqual(result.loc[0, "numerator"], "at")
        self.assertIn("delta12", result.loc[0, "formula_label"])

    def test_signal_metadata_is_loaded_from_parquet(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "signals.parquet"
            pd.DataFrame({
                "signalid": [2, 1],
                "signal_form": ["v1/v2", "diff(v1)/lag(v2)"],
                "v1": ["sale", "at"],
                "v2": ["at", "sale"],
            }).to_parquet(path, index=False)

            result = load_signal_metadata(path, weighting="ew")

        self.assertEqual(list(result["component_id"]), ["EW_1", "EW_2"])
        self.assertEqual(list(result["weighting"]), ["ew", "ew"])

    def test_parquet_audit_and_panel_use_single_nstock_and_convert_percent_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "returns.parquet"
            rows = []
            for signalid in (1, 2):
                for month in range(1, 4):
                    rows.append({
                        "signalid": signalid,
                        "year": 2020,
                        "month": month,
                        "ret": float(signalid * month),
                        "nstock": 20 + signalid,
                    })
            pd.DataFrame(rows).to_parquet(path, index=False)
            metadata = _metadata([1, 2, 3])

            audit = audit_return_file(path, metadata, return_unit="percent", chunksize=2)
            panel = build_dense_panel(
                path,
                metadata,
                audit,
                weighting="ew",
                return_unit="percent",
                chunksize=2,
            )

        self.assertEqual(audit["coverage_source"], "nstock")
        self.assertEqual(audit["metadata_signalids_without_returns"], ["3"])
        self.assertEqual(panel.returns.shape, (3, 2))
        self.assertAlmostEqual(float(panel.returns[0, 0]), 0.01)
        self.assertAlmostEqual(float(panel.nstock[0, 0]), 21.0)

    def test_frozen_universe_does_not_use_validation_or_test_values(self):
        dates = pd.date_range("2000-01-31", periods=30, freq="ME")
        returns = np.ones((30, 3), dtype=float)
        nstock = np.full_like(returns, 30.0)
        metadata = _metadata([1, 2, 3])
        base = DataMinePanel(
            dates=dates,
            component_ids=list(metadata["component_id"]),
            signal_ids=list(metadata["signalid"]),
            returns=returns,
            nstock=nstock,
            metadata=metadata,
            weighting="ew",
            coverage_source="nstock",
        )
        split = relative_panel_split(dates, validation_months=5, test_months=5)
        selected_a, _ = select_frozen_universe(
            base,
            split,
            universe_window_months=15,
            min_observed_fraction=0.99,
            min_median_nstock=10,
            min_nstock=5,
            min_nstock_fraction=0.99,
            max_missing_run=1,
            max_series=0,
            seed=1,
        )

        altered_returns = returns.copy()
        altered_nstock = nstock.copy()
        altered_returns[split.validation_indices[0] :, 1:] = np.nan
        altered_nstock[split.validation_indices[0] :, 1:] = 0.0
        altered = DataMinePanel(
            dates=dates,
            component_ids=list(metadata["component_id"]),
            signal_ids=list(metadata["signalid"]),
            returns=altered_returns,
            nstock=altered_nstock,
            metadata=metadata,
            weighting="ew",
            coverage_source="nstock",
        )
        selected_b, _ = select_frozen_universe(
            altered,
            split,
            universe_window_months=15,
            min_observed_fraction=0.99,
            min_median_nstock=10,
            min_nstock=5,
            min_nstock_fraction=0.99,
            max_missing_run=1,
            max_series=0,
            seed=1,
        )

        self.assertEqual(selected_a.component_ids, selected_b.component_ids)

    def test_quality_persistence_scores_reward_coverage_and_low_risk_persistence(self):
        returns = np.asarray([
            [1.0, 1.0, 1.0],
            [-1.0, 2.0, 1.0],
            [1.0, 3.0, 1.0],
            [-1.0, 4.0, 1.0],
            [1.0, 5.0, 1.0],
        ])
        nstock = np.asarray([
            [100.0, 20.0, 5.0],
            [100.0, 20.0, 5.0],
            [100.0, 20.0, 5.0],
            [100.0, 20.0, 5.0],
            [100.0, 20.0, 5.0],
        ])

        scores = prevalidation_selection_scores(
            returns,
            nstock,
            mode="quality_persistence",
        )

        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[0], scores[2])

class DataMineForecastTests(unittest.TestCase):
    def test_formula_svd_can_drop_leading_components(self):
        metadata = _metadata(range(1, 25))

        full, full_info = formula_feature_blocks(
            metadata,
            svd_dim=6,
            seed=17,
        )
        dropped, dropped_info = formula_feature_blocks(
            metadata,
            svd_dim=4,
            svd_drop_components=2,
            seed=17,
        )

        np.testing.assert_allclose(
            dropped["formula_svd"][0],
            full["formula_svd"][0][:, 2:6],
        )
        self.assertEqual(
            dropped["formula_svd"][1],
            ["formula_svd_03", "formula_svd_04", "formula_svd_05", "formula_svd_06"],
        )
        self.assertEqual(dropped_info["svd_dimension"], 4)
        self.assertEqual(dropped_info["svd_fitted_dimension"], 6)
        self.assertEqual(dropped_info["svd_dropped_components"], 2)
        self.assertEqual(full_info["svd_dropped_components"], 0)

    def test_formula_role_embeddings_preserve_role_identity(self):
        metadata = _metadata(range(1, 25))

        blocks, info = formula_feature_blocks(
            metadata,
            svd_dim=4,
            encoding="role_embedding",
            role_embedding_dim=12,
            seed=23,
        )

        numerator = blocks["formula_numerator_embedding"][0]
        denominator = blocks["formula_denominator_embedding"][0]
        np.testing.assert_allclose(numerator[0], numerator[3])
        np.testing.assert_allclose(denominator[0], denominator[4])
        self.assertFalse(np.allclose(numerator[0], numerator[1]))
        self.assertEqual(numerator.shape, (24, 12))
        self.assertEqual(info["encoding"], "role_embedding")
        self.assertNotIn("svd_dimension", info)

    def test_fixed_garch_forecast_is_positive_and_future_perturbation_safe(self):
        rng = np.random.default_rng(19)
        returns = rng.normal(scale=0.03, size=(120, 5))
        original, parameters = fixed_garch11_grid_forecast(
            returns,
            estimation_end_index=59,
            alpha_candidates=(0.05, 0.10),
            beta_candidates=(0.80, 0.90),
        )
        changed = returns.copy()
        changed[90:] *= 20.0
        perturbed, changed_parameters = fixed_garch11_grid_forecast(
            changed,
            estimation_end_index=59,
            alpha_candidates=(0.05, 0.10),
            beta_candidates=(0.80, 0.90),
        )

        self.assertTrue(np.isfinite(original[1:]).all())
        self.assertTrue((original[1:] > 0.0).all())
        pd.testing.assert_frame_equal(parameters, changed_parameters)
        np.testing.assert_allclose(original[:91], perturbed[:91], equal_nan=True)
        self.assertFalse(np.allclose(original[92:], perturbed[92:], equal_nan=True))

    def test_har_forecasts_are_one_step_and_future_perturbation_safe(self):
        rng = np.random.default_rng(7)
        target = rng.normal(size=(90, 4))
        original = rolling_ridge_forecast(
            target,
            baseline_type="har_ridge",
            ridge=0.01,
            minimum_observations=15,
        )
        changed = target.copy()
        changed[60:] += 1000.0
        perturbed = rolling_ridge_forecast(
            changed,
            baseline_type="har_ridge",
            ridge=0.01,
            minimum_observations=15,
        )

        np.testing.assert_allclose(original[:61], perturbed[:61], equal_nan=True)
        self.assertFalse(np.allclose(original[62:], perturbed[62:], equal_nan=True))

    def test_log_squared_return_target_preserves_missing_cells(self):
        returns = np.asarray([[0.1, np.nan], [-0.2, 0.0]])
        target = make_target(returns, target_type="log_squared_return", epsilon=1e-6)

        self.assertTrue(np.isnan(target[0, 1]))
        self.assertAlmostEqual(target[0, 0], np.log(0.010001))
        self.assertAlmostEqual(target[1, 1], np.log(1e-6))

    def test_robust_scaler_inverts_and_uses_positive_floor(self):
        values = np.asarray([
            [0.0, 2.0, 1.0],
            [1.0, 2.0, 2.0],
            [2.0, 2.0, 3.0],
            [1000.0, 2.0, 4.0],
        ])
        scaler = fit_robust_panel_scaler(values, clip=8.0)
        transformed = scaler.transform(values)
        recovered_center = scaler.inverse(np.zeros(3))

        self.assertGreater(scaler.scale_floor, 0.0)
        self.assertTrue(np.isfinite(transformed).all())
        self.assertLessEqual(np.max(np.abs(transformed)), 8.0)
        np.testing.assert_allclose(recovered_center, scaler.center)

    def test_feature_preprocessing_balances_and_row_normalizes(self):
        blocks = {
            "formula": (
                np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                ["a", "b"],
            ),
            "dynamic": (
                np.asarray([[1.0, np.nan], [2.0, 5.0], [3.0, 7.0]]),
                ["mean", "risk"],
            ),
        }
        Z, names, block_names, provenance = preprocess_feature_blocks(
            blocks,
            block_weights={"formula": 1.0, "dynamic": 0.5},
            row_normalize=True,
        )

        self.assertEqual(Z.shape[0], 3)
        self.assertEqual(Z.shape[1], len(names))
        self.assertEqual(len(names), len(block_names))
        self.assertTrue(np.isfinite(Z).all())
        np.testing.assert_allclose(np.linalg.norm(Z, axis=1), 1.0)
        self.assertTrue(provenance["row_normalized"])

    def test_global_lambda_is_bounded(self):
        result = estimate_global_lambda(
            np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        )
        self.assertGreater(result["lambda_unbounded"], 1.0)
        self.assertEqual(result["lambda"], 1.0)

    def test_group_lambdas_use_group_estimates_and_small_group_fallback(self):
        error = np.asarray([[1.0, 1.0, -1.0], [2.0, 2.0, -2.0]])
        correction = np.asarray([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        metadata = pd.DataFrame({"formula_type": ["a", "a", "b"]})

        values, table, global_info = estimate_group_lambdas(
            error,
            correction,
            metadata,
            group_cols=("formula_type",),
            min_group_series=2,
        )

        np.testing.assert_allclose(values[:2], 1.0)
        self.assertAlmostEqual(values[2], global_info["lambda"])
        self.assertTrue(table.loc[table["formula_type"].eq("b"), "fallback_global"].item())

    def test_rolling_group_lambdas_exclude_current_and_future_errors(self):
        error = np.asarray([
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ])
        correction = np.ones_like(error)
        metadata = pd.DataFrame({"formula_type": ["a", "a"]})
        original, _ = rolling_group_lambdas(
            error,
            correction,
            metadata,
            [3, 4, 5],
            window_months=2,
            min_history_months=1,
        )
        changed = error.copy()
        changed[4:] = -1000.0
        perturbed, _ = rolling_group_lambdas(
            changed,
            correction,
            metadata,
            [3, 4, 5],
            window_months=2,
            min_history_months=1,
        )

        np.testing.assert_allclose(original[3:5], perturbed[3:5], equal_nan=True)
        self.assertFalse(np.allclose(original[5], perturbed[5]))

    def test_series_lambdas_shrink_toward_global_with_stronger_prior(self):
        error = np.asarray([[1.0, -1.0], [2.0, -2.0], [3.0, -3.0]])
        correction = np.asarray([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        weak, weak_info = estimate_series_lambdas(
            error,
            correction,
            prior_strength_months=0.0,
        )
        strong, strong_info = estimate_series_lambdas(
            error,
            correction,
            prior_strength_months=1000.0,
        )

        global_lambda = strong_info["lambda"]
        self.assertGreater(np.mean(np.abs(weak - global_lambda)), np.mean(np.abs(strong - global_lambda)))
        self.assertEqual(weak_info["prior_penalty"], 0.0)

    def test_small_rolling_vara_forecast_has_scheduled_refits(self):
        rng = np.random.default_rng(11)
        errors = rng.normal(size=(70, 5))
        metadata = _metadata([1, 2, 3, 4, 5])
        base_features = rng.normal(size=(5, 3))
        base_features /= np.linalg.norm(base_features, axis=1, keepdims=True)

        def feature_builder(origin_index):
            return base_features, {
                "origin_index": origin_index,
                "feature_names": ["a", "b", "c"],
                "block_names": ["formula"] * 3,
            }

        result = rolling_vara_corrections(
            baseline_errors=errors,
            evaluation_indices=np.arange(60, 65),
            feature_builder=feature_builder,
            component_ids=list(metadata["component_id"]),
            metadata=metadata,
            error_window_months=40,
            response_clip=8.0,
            refit_frequency_months=2,
            grid_size=3,
            ridge=0.01,
            eta_bounds=(-1.0, 1.0),
            attention_mode="topk",
            attention_top_k=2,
            attention_cache_mode="exact",
            batch_size=8,
            dense_threshold=100,
            top_neighbors=2,
        )

        self.assertTrue(np.isfinite(result.correction[60:65]).all())
        self.assertEqual(len(result.fit_summary), 3)
        self.assertEqual(
            result.fit_summary["attention_cache_reused"].tolist(),
            [False, True, True],
        )
        self.assertEqual(result.parameter_summary["refit_index"].nunique(), 3)
        self.assertEqual(result.neighbors.groupby(["refit_index", "component_id"]).size().min(), 2)

    def test_group_vara_is_one_step_and_future_perturbation_safe(self):
        rng = np.random.default_rng(31)
        errors = rng.normal(size=(80, 6))
        metadata = _metadata(range(1, 7))
        original = rolling_group_vara_corrections(
            baseline_errors=errors,
            evaluation_indices=np.arange(60, 66),
            metadata=metadata,
            group_cols=("numerator", "denominator"),
            include_global_head=True,
            error_window_months=40,
            response_clip=8.0,
            refit_frequency_months=2,
            ridge=0.01,
        )
        changed = errors.copy()
        changed[63:] += 1000.0
        perturbed = rolling_group_vara_corrections(
            baseline_errors=changed,
            evaluation_indices=np.arange(60, 66),
            metadata=metadata,
            group_cols=("numerator", "denominator"),
            include_global_head=True,
            error_window_months=40,
            response_clip=8.0,
            refit_frequency_months=2,
            ridge=0.01,
        )

        np.testing.assert_allclose(
            original.correction[60:64],
            perturbed.correction[60:64],
            equal_nan=True,
        )
        self.assertFalse(np.allclose(original.correction[65], perturbed.correction[65]))

    def test_har_vara_is_one_step_and_future_perturbation_safe(self):
        rng = np.random.default_rng(37)
        errors = rng.normal(size=(80, 6))
        features = rng.normal(size=(6, 4))
        features /= np.linalg.norm(features, axis=1, keepdims=True)
        original = rolling_har_vara_corrections(
            baseline_errors=errors,
            evaluation_indices=np.arange(60, 66),
            feature_matrix=features,
            error_window_months=40,
            response_clip=8.0,
            refit_frequency_months=2,
            ridge=0.01,
            attention_top_k=2,
            attention_cache_mode="exact",
            attention_batch_size=8,
            eta=-1e-4,
        )
        changed = errors.copy()
        changed[63:] += 1000.0
        perturbed = rolling_har_vara_corrections(
            baseline_errors=changed,
            evaluation_indices=np.arange(60, 66),
            feature_matrix=features,
            error_window_months=40,
            response_clip=8.0,
            refit_frequency_months=2,
            ridge=0.01,
            attention_top_k=2,
            attention_cache_mode="exact",
            attention_batch_size=8,
            eta=-1e-4,
        )

        np.testing.assert_allclose(
            original.correction[60:64],
            perturbed.correction[60:64],
            equal_nan=True,
        )
        self.assertFalse(np.allclose(original.correction[65], perturbed.correction[65]))


if __name__ == "__main__":
    unittest.main()
