"""Forecasting interface for fitted VARA models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import jax.numpy as jnp

from .attention import TopKAttentionCache, attention_matvec
from .simulation import estimate_eta_concentrated
from .two_stage import fit_all_components_two_stage


@dataclass
class VARAForecaster:
    """Small array-based VARA forecaster using the two-stage estimator.

    The public API intentionally starts with arrays only. Data loading,
    transformations, persistence, and command-line orchestration should be
    layered on top after this core is stable.
    """

    params_scale: Dict[str, float] = field(default_factory=lambda: {"eta_a": 1.0})
    eta_bounds: Tuple[float, float] = (-2.0, 2.0)
    grid_size: int = 41
    ridge: float = 1e-10
    batch_size: int = 512
    dense_threshold: int = 2500
    attention_mode: str = "dense"
    attention_top_k: Optional[int] = None
    attention_cache: Optional[TopKAttentionCache] = None

    eta_hat: Optional[jnp.ndarray] = field(default=None, init=False)
    alpha_hat: Optional[jnp.ndarray] = field(default=None, init=False)
    beta_hat: Optional[jnp.ndarray] = field(default=None, init=False)
    sigma2_hat: Optional[jnp.ndarray] = field(default=None, init=False)
    Z_: Optional[jnp.ndarray] = field(default=None, init=False)
    y_last_: Optional[jnp.ndarray] = field(default=None, init=False)
    fitted_values_: Optional[jnp.ndarray] = field(default=None, init=False)
    residuals_: Optional[jnp.ndarray] = field(default=None, init=False)
    fit_diagnostics_: Dict[str, object] = field(default_factory=dict, init=False)

    def fit(self, Y: jnp.ndarray, Z: jnp.ndarray) -> "VARAForecaster":
        """Fit a VARA model from an observed panel ``Y`` and features ``Z``."""
        Y = jnp.asarray(Y)
        Z = jnp.asarray(Z)
        self._validate_training_arrays(Y, Z)
        if int(self.grid_size) < 2:
            raise ValueError("grid_size must be at least 2.")
        lower, upper = self.eta_bounds
        if not lower < upper:
            raise ValueError("eta_bounds must satisfy lower < upper.")

        eta_hat, eta_info = estimate_eta_concentrated(
            Y=Y,
            Z=Z,
            params_scale=self.params_scale,
            eta_bounds=self.eta_bounds,
            grid_size=self.grid_size,
            ridge=self.ridge,
            batch_size=self.batch_size,
            dense_threshold=self.dense_threshold,
            attention_mode=self.attention_mode,
            attention_top_k=self.attention_top_k,
            attention_cache=self.attention_cache,
        )
        fit = fit_all_components_two_stage(
            Y=Y,
            Z=Z,
            eta=float(eta_hat),
            params_scale=self.params_scale,
            ridge=self.ridge,
            batch_size=self.batch_size,
            dense_threshold=self.dense_threshold,
            attention_mode=self.attention_mode,
            attention_top_k=self.attention_top_k,
            attention_cache=self.attention_cache,
        )

        self.eta_hat = eta_hat
        self.alpha_hat = fit["alpha_hat"]
        self.beta_hat = fit["beta_hat"]
        self.sigma2_hat = fit["sigma2_hat"]
        self.Z_ = Z
        self.y_last_ = Y[-1]
        self.residuals_ = fit["resid"]
        self.fitted_values_ = Y[1:] - fit["resid"]
        self.fit_diagnostics_ = self._build_fit_diagnostics(Y, Z, eta_info)
        return self

    def predict(self, h: int = 1, y_last: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """Return recursive forecasts with shape ``(h, p)``."""
        self._require_fitted()
        h_int = int(h)
        if h_int != h or h_int < 1:
            raise ValueError("h must be a positive integer.")

        y = self.y_last_ if y_last is None else jnp.asarray(y_last)
        if y.ndim != 1 or y.shape[0] != self.alpha_hat.shape[0]:
            raise ValueError("y_last must be a vector with length p.")

        forecasts = []
        for _ in range(h_int):
            x2 = attention_matvec(
                self.Z_,
                float(self.eta_hat),
                y,
                self.params_scale,
                batch_size=self.batch_size,
                dense_threshold=self.dense_threshold,
                attention_mode=self.attention_mode,
                attention_top_k=self.attention_top_k,
                attention_cache=self.attention_cache,
            )
            y = self.alpha_hat * y + self.beta_hat * x2
            forecasts.append(y)
        return jnp.stack(forecasts, axis=0)

    def fitted_values(self) -> jnp.ndarray:
        """Return in-sample one-step fitted values with shape ``(T-1, p)``."""
        self._require_fitted()
        return self.fitted_values_

    def residuals(self) -> jnp.ndarray:
        """Return in-sample one-step residuals with shape ``(T-1, p)``."""
        self._require_fitted()
        return self.residuals_

    def diagnostics(self) -> Dict[str, object]:
        """Return fit diagnostics as plain Python scalars where possible."""
        self._require_fitted()
        return dict(self.fit_diagnostics_)

    @staticmethod
    def _validate_training_arrays(Y: jnp.ndarray, Z: jnp.ndarray) -> None:
        if Y.ndim != 2:
            raise ValueError("Y must be a two-dimensional array with shape (T, p).")
        if Z.ndim != 2:
            raise ValueError("Z must be a two-dimensional array with shape (p, q).")
        if Y.shape[0] < 2:
            raise ValueError("Y must contain at least two time points.")
        if Y.shape[1] != Z.shape[0]:
            raise ValueError("Y.shape[1] must match Z.shape[0].")

    def _require_fitted(self) -> None:
        if (
            self.eta_hat is None
            or self.alpha_hat is None
            or self.beta_hat is None
            or self.sigma2_hat is None
            or self.Z_ is None
            or self.y_last_ is None
        ):
            raise RuntimeError("VARAForecaster must be fitted before prediction.")

    def _build_fit_diagnostics(
        self,
        Y: jnp.ndarray,
        Z: jnp.ndarray,
        eta_info: Dict[str, object],
    ) -> Dict[str, object]:
        max_abs_alpha = float(jnp.max(jnp.abs(self.alpha_hat)))
        max_abs_beta = float(jnp.max(jnp.abs(self.beta_hat)))
        sigma2 = jnp.asarray(self.sigma2_hat)
        return {
            "T": int(Y.shape[0]),
            "p": int(Y.shape[1]),
            "q": int(Z.shape[1]),
            "eta_hat": float(self.eta_hat),
            "eta_on_boundary": bool(eta_info.get("on_boundary", False)),
            "grid_best_eta": float(eta_info.get("grid_best_eta", jnp.nan)),
            "objective": float(eta_info.get("objective", jnp.nan)),
            "max_abs_alpha": max_abs_alpha,
            "max_abs_beta": max_abs_beta,
            "stationarity_proxy": max_abs_alpha + max_abs_beta,
            "sigma2_min": float(jnp.min(sigma2)),
            "sigma2_mean": float(jnp.mean(sigma2)),
            "sigma2_max": float(jnp.max(sigma2)),
            "attention_mode": self.attention_mode,
            "attention_top_k": self.attention_top_k,
        }
