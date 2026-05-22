import os
import json
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from scripts.run_full_simulation import (
    _full_simulation_output_paths,
    _load_config_json_defaults,
    _parse_int_list,
    _parse_str_list,
    _simulation_reporting_output_paths,
    draw_simulation_dgp_params,
    make_simulation_reporting_tables,
    make_simulation_setup_table,
)
from scripts.run_vara_forecast import (
    _fit_y_transform,
    _inverse_y_transform,
    _seasonal_naive_forecast,
    _seasonal_naive_residual_panel,
    _tourism_feature_matrix,
    _tourism_specific_feature_matrix,
    _tourism_tsfeature_matrix,
    run_vara_forecast,
)


class ScriptHelperTests(unittest.TestCase):
    def test_seasonal_naive_forecast_and_residual_panel(self):
        history = np.arange(1, 16, dtype=float)[:, None]

        forecast = _seasonal_naive_forecast(history, horizon=3, seasonal_lag=12)
        residuals = _seasonal_naive_residual_panel(history, seasonal_lag=12)

        np.testing.assert_allclose(forecast[:, 0], np.asarray([4.0, 5.0, 6.0]))
        np.testing.assert_allclose(residuals[:, 0], np.asarray([12.0, 12.0, 12.0]))

    def test_tourism_seasonal_diff_inverse_reconstructs_from_lagged_logs(self):
        train = pd.DataFrame({
            "a": range(1, 16),
            "b": range(11, 26),
        }).to_numpy(dtype=float)
        _, transform = _fit_y_transform(train, "log1p-seasonal-diff", seasonal_lag=12)

        target_diff = np.zeros((2, 2))
        model_scale = (target_diff - transform["mean"]) / transform["scale"]
        forecast = _inverse_y_transform(model_scale, transform)

        expected = train[-12:-10]
        pd.testing.assert_frame_equal(
            pd.DataFrame(forecast),
            pd.DataFrame(expected),
            check_dtype=False,
            atol=1e-10,
            rtol=1e-10,
        )

    def test_first_difference_inverse_reconstructs_from_last_level(self):
        train = np.asarray([
            [1.0, 10.0],
            [3.0, 13.0],
            [6.0, 17.0],
            [10.0, 22.0],
        ])
        _, transform = _fit_y_transform(train, "diff", seasonal_lag=12)

        target_diff = np.asarray([
            [5.0, 6.0],
            [7.0, 8.0],
        ])
        model_scale = (target_diff - transform["mean"]) / transform["scale"]
        forecast = _inverse_y_transform(model_scale, transform)

        expected = np.asarray([
            [15.0, 28.0],
            [22.0, 36.0],
        ])
        np.testing.assert_allclose(forecast, expected, atol=1e-10, rtol=1e-10)

    def test_log_first_difference_inverse_reconstructs_from_last_log_level(self):
        train = np.asarray([
            [1.0, 10.0],
            [3.0, 13.0],
            [6.0, 17.0],
            [10.0, 22.0],
        ])
        _, transform = _fit_y_transform(train, "log1p-diff", seasonal_lag=12)

        target_log_diff = np.zeros((2, 2))
        model_scale = (target_log_diff - transform["mean"]) / transform["scale"]
        forecast = _inverse_y_transform(model_scale, transform)

        expected = np.vstack([train[-1], train[-1]])
        np.testing.assert_allclose(forecast, expected, atol=1e-10, rtol=1e-10)

    def test_tourism_tsfeature_matrix_uses_training_window(self):
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        wide = pd.DataFrame({
            "date": dates,
            "AAAAll": np.linspace(10.0, 20.0, 24),
            "AABAll": np.linspace(30.0, 50.0, 24),
        })

        Z, feature_df, feature_cols = _tourism_tsfeature_matrix(
            wide,
            ["AAAAll", "AABAll"],
            horizon=2,
            y_transform="level",
            seasonal_lag=12,
            feature_functions=("series_length",),
        )

        self.assertEqual(feature_cols, ["series_length"])
        self.assertEqual(Z.shape, (2, 1))
        self.assertTrue(np.isfinite(Z).all())
        self.assertEqual(list(feature_df["series_id"]), ["AAAAll", "AABAll"])
        self.assertEqual(list(feature_df["n_obs"]), [22, 22])
        self.assertEqual(list(feature_df["train_start"]), ["2020-01-01", "2020-01-01"])
        self.assertEqual(list(feature_df["train_end"]), ["2021-10-01", "2021-10-01"])

    def test_tourism_specific_feature_matrix_uses_training_window(self):
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        wide = pd.DataFrame({
            "date": dates,
            "AAAAll": np.linspace(10.0, 20.0, 24),
            "AABAll": np.linspace(30.0, 50.0, 24),
        })

        Z, feature_df, feature_cols = _tourism_specific_feature_matrix(
            wide,
            ["AAAAll", "AABAll"],
            horizon=2,
        )

        self.assertEqual(Z.shape[0], 2)
        self.assertTrue(np.isfinite(Z).all())
        self.assertIn("season_m01", feature_cols)
        self.assertIn("growth_1", feature_cols)
        self.assertTrue(any(col.startswith("code_1_") for col in feature_cols))
        self.assertEqual(list(feature_df["series_id"]), ["AAAAll", "AABAll"])
        self.assertEqual(list(feature_df["n_obs"]), [22, 22])
        self.assertEqual(list(feature_df["train_end"]), ["2021-10-01", "2021-10-01"])

    def test_tourism_hybrid_feature_matrix_combines_sources(self):
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        wide = pd.DataFrame({
            "date": dates,
            "AAAAll": np.linspace(10.0, 20.0, 24),
            "AABAll": np.linspace(30.0, 50.0, 24),
        })

        Z, feature_df, feature_cols = _tourism_feature_matrix(
            wide,
            ["AAAAll", "AABAll"],
            horizon=2,
            feature_source="hybrid",
            y_transform="level",
            seasonal_lag=12,
            feature_freq=12,
            feature_functions=("series_length",),
        )

        self.assertEqual(Z.shape[0], 2)
        self.assertTrue(np.isfinite(Z).all())
        self.assertEqual(list(feature_df["transform"]), ["hybrid", "hybrid"])
        self.assertIn("tourism__season_m01", feature_cols)
        self.assertIn("tourism__growth_1", feature_cols)
        self.assertIn("tsfeatures__series_length", feature_cols)

    def test_tourism_seasonal_corr_feature_matrix_is_compact(self):
        dates = pd.date_range("2020-01-01", periods=36, freq="MS")
        seasonal = np.sin(2.0 * np.pi * np.arange(36) / 12.0)
        wide = pd.DataFrame({
            "date": dates,
            "AAAAll": 20.0 + np.linspace(0.0, 3.0, 36) + seasonal,
            "AABAll": 30.0 + np.linspace(0.0, 5.0, 36) + 1.5 * seasonal,
            "ABAAll": 18.0 + np.linspace(0.0, 2.0, 36) - seasonal,
        })

        Z, feature_df, feature_cols = _tourism_feature_matrix(
            wide,
            ["AAAAll", "AABAll", "ABAAll"],
            horizon=2,
            feature_source="seasonal-corr",
            y_transform="level",
            seasonal_lag=12,
            feature_freq=12,
            feature_functions=("series_length",),
        )

        self.assertEqual(Z.shape[0], 3)
        self.assertLessEqual(Z.shape[1], 25)
        self.assertTrue(np.isfinite(Z).all())
        self.assertEqual(list(feature_df["transform"]), ["seasonal_corr", "seasonal_corr", "seasonal_corr"])
        self.assertEqual(list(feature_df["train_end"]), ["2022-10-01", "2022-10-01", "2022-10-01"])
        self.assertTrue(any(col.startswith("corr_embed_") for col in feature_cols))
        self.assertTrue(any(col.startswith("seasonal_pc_") for col in feature_cols))
        self.assertIn("growth_12", feature_cols)
        self.assertIn("code_2_ord", feature_cols)

    def test_tourism_seasonal_predictive_feature_matrix_is_score_matrix(self):
        dates = pd.date_range("2020-01-01", periods=48, freq="MS")
        seasonal = np.sin(2.0 * np.pi * np.arange(48) / 12.0)
        shock = np.sin(2.0 * np.pi * np.arange(48) / 6.0)
        wide = pd.DataFrame({
            "date": dates,
            "AAAAll": 20.0 + np.linspace(0.0, 3.0, 48) + seasonal + 0.2 * shock,
            "AABAll": 30.0 + np.linspace(0.0, 5.0, 48) + 1.5 * seasonal + 0.5 * np.roll(shock, 1),
            "ABAAll": 18.0 + np.linspace(0.0, 2.0, 48) - seasonal + 0.3 * np.roll(shock, 12),
        })

        Z, feature_df, feature_cols = _tourism_feature_matrix(
            wide,
            ["AAAAll", "AABAll", "ABAAll"],
            horizon=2,
            feature_source="seasonal-predictive",
            y_transform="level",
            seasonal_lag=12,
            feature_freq=12,
            feature_functions=("series_length",),
        )

        self.assertEqual(Z.shape, (3, 3))
        self.assertTrue(np.isfinite(Z).all())
        self.assertTrue(np.all(np.abs(Z) <= 1.0 + 1e-8))
        np.testing.assert_allclose(np.diag(Z), np.zeros(3), atol=1e-12)
        self.assertEqual(list(feature_df["transform"]), ["seasonal_predictive"] * 3)
        self.assertEqual(list(feature_df["train_end"]), ["2023-10-01"] * 3)
        self.assertEqual(feature_cols, ["score_from_AAAAll", "score_from_AABAll", "score_from_ABAAll"])

    def test_tourism_rolling_origin_refits_each_test_month(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            output_dir = root / "out"
            data_dir.mkdir()
            dates = pd.date_range("2020-01-01", periods=24, freq="MS")
            pd.DataFrame({
                "date": dates,
                "AAAAll": np.linspace(10.0, 20.0, 24),
                "AABAll": np.linspace(30.0, 50.0, 24),
            }).to_csv(data_dir / "tourism.csv", index=False)

            metrics = run_vara_forecast(
                data_dir=data_dir,
                output_dir=output_dir,
                max_series=0,
                horizon=2,
                grid_size=3,
                y_transform="center",
                seasonal_lag=12,
                ridge=1e-2,
                series_depth=3,
                feature_functions=("series_length",),
                forecast_mode="rolling-origin",
                feature_update_policy="periodic",
                feature_update_interval=2,
                attention_mode="topk",
                attention_top_k=1,
            )

            self.assertEqual(metrics["forecast_type"], "rolling-origin")
            self.assertEqual(metrics["attention_mode"], "topk")
            self.assertEqual(metrics["attention_top_k"], 1)
            self.assertEqual(metrics["attention_feature_update_policy"], "periodic")
            self.assertEqual(metrics["attention_feature_update_interval"], 2)
            self.assertEqual(metrics["attention_feature_exact_origins"], "1")
            self.assertEqual(metrics["attention_feature_exact_origin_count"], 1)
            self.assertEqual(metrics["n_train"], 22)
            self.assertEqual(metrics["n_train_final"], 23)
            self.assertEqual(metrics["n_train_model_final"], 23)
            forecast = pd.read_csv(output_dir / "tourism_forecast.csv")
            diagnostics = pd.read_csv(output_dir / "tourism_rolling_origin_diagnostics.csv")
            features = pd.read_csv(output_dir / "tourism_attention_features.csv")
            metrics_summary = pd.read_csv(output_dir / "metrics_summary.csv")
            metrics_by_series = pd.read_csv(output_dir / "metrics_by_series.csv")
            metrics_by_origin = pd.read_csv(output_dir / "metrics_by_origin.csv")
            self.assertEqual(len(forecast), 2)
            self.assertEqual(list(diagnostics["origin"]), [1, 2])
            self.assertEqual(list(diagnostics["feature_exact"]), [True, False])
            self.assertEqual(list(diagnostics["feature_source_origin"]), [1, 1])
            self.assertEqual(len(features), 4)
            self.assertEqual(sorted(features["feature_exact"].unique().tolist()), [False, True])
            self.assertEqual(sorted(features["feature_source_origin"].unique().tolist()), [1])
            self.assertEqual(list(metrics_summary["model"]), ["vara"])
            self.assertEqual(metrics_summary["forecast_mode"].iloc[0], "rolling-origin")
            self.assertIn("mase", metrics_summary.columns)
            self.assertIn("rmsse", metrics_summary.columns)
            self.assertEqual(set(metrics_by_series["series_id"]), {"AAAAll", "AABAll"})
            self.assertEqual(list(metrics_by_origin["date"]), ["2021-11-01", "2021-12-01"])

    def test_vara_forecast_accepts_generic_wide_csv_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "m4_like.csv"
            output_dir = root / "out"
            dates = pd.date_range("2000-01-01", periods=12, freq="YS")
            pd.DataFrame({
                "date": dates,
                "Y1": np.linspace(10.0, 20.0, 12),
                "Y2": np.linspace(30.0, 40.0, 12),
                "Y3": np.linspace(50.0, 65.0, 12),
            }).to_csv(data_path, index=False)

            metrics = run_vara_forecast(
                data_dir=root,
                data_path=data_path,
                output_dir=output_dir,
                max_series=0,
                horizon=2,
                grid_size=3,
                y_transform="center",
                seasonal_lag=1,
                ridge=1e-2,
                series_depth=None,
                series_suffix=None,
                feature_freq=1,
                feature_functions=("series_length",),
                attention_mode="topk",
                attention_top_k=2,
            )

            self.assertEqual(metrics["source"], str(data_path))
            self.assertEqual(metrics["series_suffix"], "")
            self.assertEqual(metrics["series_depth"], -1)
            self.assertEqual(metrics["n_series"], 3)
            forecast = pd.read_csv(output_dir / "tourism_forecast.csv")
            self.assertEqual(set(forecast.columns), {"date", "Y1", "Y2", "Y3"})

    def test_vara_forecast_accepts_generic_long_csv_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "m4_like_long.csv"
            output_dir = root / "out"
            dates = pd.date_range("2000-01-01", periods=12, freq="YS")
            wide = pd.DataFrame({
                "date": dates,
                "Y1": np.linspace(10.0, 20.0, 12),
                "Y2": np.linspace(30.0, 40.0, 12),
                "Y3": np.linspace(50.0, 65.0, 12),
            })
            wide.melt(id_vars="date", var_name="series_id", value_name="value").to_csv(data_path, index=False)

            metrics = run_vara_forecast(
                data_dir=root,
                data_path=data_path,
                data_format="long",
                output_dir=output_dir,
                max_series=0,
                horizon=2,
                grid_size=3,
                y_transform="center",
                seasonal_lag=1,
                ridge=1e-2,
                series_depth=None,
                series_suffix=None,
                feature_freq=1,
                feature_functions=("series_length",),
                attention_mode="topk",
                attention_top_k=2,
            )

            self.assertEqual(metrics["source"], str(data_path))
            self.assertEqual(metrics["data_format"], "long")
            self.assertEqual(metrics["n_series"], 3)
            forecast = pd.read_csv(output_dir / "tourism_forecast.csv")
            self.assertEqual(set(forecast.columns), {"date", "Y1", "Y2", "Y3"})

    def test_tourism_seasonal_naive_residual_scenario_outputs_components(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            output_dir = root / "out"
            data_dir.mkdir()
            dates = pd.date_range("2020-01-01", periods=36, freq="MS")
            seasonal = np.sin(2.0 * np.pi * np.arange(36) / 12.0)
            pd.DataFrame({
                "date": dates,
                "AAAAll": 20.0 + np.linspace(0.0, 4.0, 36) + seasonal,
                "AABAll": 30.0 + np.linspace(0.0, 6.0, 36) + 1.2 * seasonal,
            }).to_csv(data_dir / "tourism.csv", index=False)

            metrics = run_vara_forecast(
                data_dir=data_dir,
                output_dir=output_dir,
                max_series=0,
                horizon=2,
                grid_size=3,
                y_transform="log1p-seasonal-diff",
                seasonal_lag=12,
                ridge=1e-2,
                series_depth=3,
                feature_source="tourism",
                forecast_mode="rolling-origin",
                forecast_scenario="seasonal-naive-residual",
                residual_transform="center",
            )

            forecast = pd.read_csv(output_dir / "tourism_forecast.csv")
            baseline = pd.read_csv(output_dir / "tourism_baseline_forecast.csv")
            residual = pd.read_csv(output_dir / "tourism_residual_forecast.csv")
            summary = pd.read_csv(output_dir / "metrics_summary.csv")

            self.assertEqual(metrics["forecast_scenario"], "seasonal-naive-residual")
            self.assertEqual(metrics["residual_baseline"], "seasonal_naive")
            self.assertEqual(metrics["residual_transform"], "center")
            self.assertEqual(metrics["n_train"], 34)
            self.assertEqual(metrics["n_train_model_final"], 23)
            pd.testing.assert_frame_equal(
                forecast.drop(columns=["date"]),
                baseline.drop(columns=["date"]) + residual.drop(columns=["date"]),
                check_dtype=False,
                atol=1e-10,
                rtol=1e-10,
            )
            self.assertEqual(list(summary["model"]), ["vara"])

    def test_full_simulation_output_paths_include_major_config(self):
        detail, summary, paper = _full_simulation_output_paths(
            Path("data"),
            R=500,
            T_list=[8],
            settings=["low"],
            q=3,
            seed=2026,
            high_max_p=None,
            run_tag="20260430_12.34.56",
        )

        self.assertEqual(
            summary.name,
            "summary_full_simulation_R=500_p=2_q=3_T=8_seed=2026_20260430_12.34.56.csv",
        )
        self.assertTrue(detail.name.startswith("results_full_simulation_R=500_p=2_q=3_T=8_seed=2026_"))
        self.assertTrue(detail.name.endswith("_details.csv"))
        self.assertTrue(paper.name.startswith("paper_table_full_simulation_R=500_p=2_q=3_T=8_seed=2026_"))

    def test_saved_config_json_can_supply_argparse_defaults(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config_full_simulation_test.json"
            path.write_text(
                json.dumps({
                    "R": 12,
                    "T_list": [100, 200],
                    "settings": ["low"],
                    "dist_list": ["normal", "laplace"],
                    "eta_bounds": [-1.5, 1.5],
                    "ray_num_cpus": 4,
                    "results_dir": "outputs/from_config",
                    "summary_path": "ignored.csv",
                }),
                encoding="utf-8",
            )

            defaults = _load_config_json_defaults(str(path))

        self.assertEqual(defaults["R"], 12)
        self.assertEqual(_parse_int_list(defaults["T_list"]), [100, 200])
        self.assertEqual(_parse_str_list(defaults["settings"]), ["low"])
        self.assertEqual(_parse_str_list(defaults["dist_list"]), ["normal", "laplace"])
        self.assertEqual(defaults["eta_lower"], -1.5)
        self.assertEqual(defaults["eta_upper"], 1.5)
        self.assertEqual(defaults["ray_num_cpus"], 4)
        self.assertEqual(defaults["results_dir"], "outputs/from_config")
        self.assertNotIn("summary_path", defaults)

    def test_simulation_reporting_output_paths_use_expected_prefixes(self):
        paths = _simulation_reporting_output_paths(Path("data"), "R=1_p=2_q=3_T=8_seed=2026_tag")

        self.assertEqual(
            paths["publication_wide"].name,
            "table_simulation_publication_wide_R=1_p=2_q=3_T=8_seed=2026_tag.csv",
        )
        self.assertIn("eta_diagnostics", paths)

    def test_draw_simulation_dgp_params_uses_identification_stable_ranges(self):
        params, stat_info = draw_simulation_dgp_params(jax.random.PRNGKey(2026), p=10)

        self.assertAlmostEqual(float(params["eta_a"]), 1.0)
        self.assertTrue(bool(jnp.all(params["gamma_a"] >= 0.55)))
        self.assertTrue(bool(jnp.all(params["gamma_a"] <= 0.75)))
        self.assertTrue(bool(jnp.all(params["beta"] >= -0.20)))
        self.assertTrue(bool(jnp.all(params["beta"] <= 0.20)))
        self.assertTrue(bool(jnp.all(params["sigma2"] >= 0.50)))
        self.assertTrue(bool(jnp.all(params["sigma2"] <= 1.00)))
        self.assertLessEqual(float(stat_info["lhs_after"]), 0.98 + 1e-6)

    def test_simulation_reporting_tables_split_low_and_high_outputs(self):
        setup = make_simulation_setup_table(
            R=500,
            T_list=[100],
            settings=["low", "high"],
            dist_list=["normal"],
            q=50,
            high_max_p=2000,
            screen_z=True,
            precompute_dgp_attention=True,
            require_good_z=True,
            dgp_eta_a=1.0,
            z_max_tries=200,
            z_score_quantile=0.999,
            z_max_abs_score=5.0,
            z_min_eta_signal=0.05,
            z_min_eta_info=0.05,
            z_min_attention_x2_std=0.02,
            z_min_attention_x2_std_adaptive_factor=1.1,
            z_max_attention_effective_neighbors=1000.0,
            z_max_attention_effective_neighbor_fraction=0.25,
            z_approx_threshold=5000,
            z_approx_rows=512,
            z_approx_pairs=200_000,
            attention_mode="dense",
            attention_top_k=None,
            attention_cache_mode="exact",
            attention_cache_clusters=512,
            attention_cache_probes=4,
            attention_cache_candidate_size=None,
            attention_cache_candidate_multiplier=5.0,
            attention_cache_diagnostic_rows=64,
        )
        summary = pd.DataFrame([
            {
                "setting": "low",
                "dist": "normal",
                "T": 100,
                "p": 10,
                "q": 50,
                "method": "qmle",
                "RMSE_gamma": 0.1,
                "RMSE_alpha": 0.2,
                "RMSE_beta": 0.3,
                "RMSE_sigma2": 0.4,
                "ECP_gamma": 0.95,
                "ECP_alpha": 0.94,
                "ECP_beta": 0.93,
                "ECP_sigma2": 0.92,
                "valid_rate": 1.0,
                "eta_boundary_rate": 0.0,
                "Z_accepted": True,
                "Z_approximate": False,
                "Z_approx_rows": float("nan"),
                "Z_approx_pairs": float("nan"),
                "Z_approx_threshold": 5000,
                "Z_eta_signal": 0.2,
                "Z_attention_x2_std_mean": 0.04,
                "Z_attention_x2_std_effective_min": 0.02,
                "Z_attention_effective_neighbors_mean": 500.0,
                "Z_attention_effective_neighbors_limit": 1000.0,
                "Z_gamma_a_x2_resid_std_mean": 0.03,
                "Z_gamma_a_x1_x2_corr_abs_mean": 0.1,
                "DGP_attention_precomputed": True,
                "TopK_attention_cached": False,
                "TopK_cache_recall_pos_mean": float("nan"),
                "TopK_cache_score_loss_pos_mean": float("nan"),
                "attention_mode": "dense",
                "attention_top_k": float("nan"),
                "attention_cache_mode": "exact",
            },
            {
                "setting": "high",
                "dist": "normal",
                "T": 100,
                "p": 1000,
                "q": 50,
                "method": "two_stage",
                "RMSE_gamma": 0.5,
                "RMSE_alpha": 0.6,
                "RMSE_beta": 0.7,
                "RMSE_sigma2": 0.8,
                "ECP_gamma": float("nan"),
                "ECP_alpha": 0.91,
                "ECP_beta": 0.90,
                "ECP_sigma2": 0.89,
                "valid_rate": 1.0,
                "eta_boundary_rate": 0.1,
                "Z_accepted": True,
                "Z_approximate": True,
                "Z_approx_rows": 512,
                "Z_approx_pairs": 200_000,
                "Z_approx_threshold": 5000,
                "Z_eta_signal": 0.3,
                "Z_attention_x2_std_mean": 0.05,
                "Z_attention_x2_std_effective_min": 0.02,
                "Z_attention_effective_neighbors_mean": 600.0,
                "Z_attention_effective_neighbors_limit": 1000.0,
                "Z_gamma_a_x2_resid_std_mean": 0.04,
                "Z_gamma_a_x1_x2_corr_abs_mean": 0.2,
                "DGP_attention_precomputed": True,
                "TopK_attention_cached": False,
                "TopK_cache_recall_pos_mean": float("nan"),
                "TopK_cache_score_loss_pos_mean": float("nan"),
                "attention_mode": "dense",
                "attention_top_k": float("nan"),
                "attention_cache_mode": "exact",
            },
        ])

        tables = make_simulation_reporting_tables(summary, setup)

        self.assertEqual(list(tables["low_rmse"]["method"]), ["qmle"])
        self.assertEqual(list(tables["high_rmse"]["method"]), ["two_stage"])
        self.assertNotIn("ECP_gamma", tables["high_ecp"].columns)
        self.assertIn("eta_boundary_rate", tables["eta_diagnostics"].columns)
        self.assertIn("Z_attention_x2_std_mean", tables["eta_diagnostics"].columns)
        self.assertIn("Z_attention_x2_std_effective_min", tables["eta_diagnostics"].columns)
        self.assertIn("Z_attention_effective_neighbors_limit", tables["eta_diagnostics"].columns)
        self.assertIn("Z_gamma_a_x2_resid_std_mean", tables["eta_diagnostics"].columns)
        self.assertIn("Z_gamma_a_x1_x2_corr_abs_mean", tables["eta_diagnostics"].columns)
        self.assertIn("Z_approximate", tables["eta_diagnostics"].columns)
        self.assertIn("Z_approx_rows", tables["eta_diagnostics"].columns)
        self.assertIn("DGP_attention_precomputed", tables["eta_diagnostics"].columns)
        self.assertIn("TopK_attention_cached", tables["eta_diagnostics"].columns)
        self.assertIn("attention_mode", tables["eta_diagnostics"].columns)
        self.assertIn("attention_cache_mode", tables["eta_diagnostics"].columns)
        self.assertIn("TopK_cache_recall_pos_mean", tables["eta_diagnostics"].columns)
        self.assertIn("TopK_cache_score_loss_pos_mean", tables["eta_diagnostics"].columns)


if __name__ == "__main__":
    unittest.main()
