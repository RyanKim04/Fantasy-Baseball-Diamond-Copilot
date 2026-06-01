"""M3: CQR + ACI on LightGBM (production candidate).

The ceiling candidate in the three-model bake-off. Reuses M2's tuned
LightGBM quantile boosters and layers conformal calibration (CQR) with
adaptive conformal inference (ACI) for non-stationarity.

Owner: modeler subagent.
Spec: docs/model_bakeoff.md section 3, M3.
References:
  - CQR: Romano, Patterson, Candes (2019) "Conformal Quantile Regression"
  - ACI: Gibbs & Candes (2021) "Adaptive Conformal Inference Under Distribution Shift"
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from packages.ml.models.lgbm_quantile import LGBMQuantileModel

logger = logging.getLogger(__name__)

# ACI gamma candidates (model_bakeoff.md section 3, M3)
ACI_GAMMA_GRID: list[float] = [0.005, 0.01, 0.02, 0.05]

# ACI window size (validation_protocol.md section 4)
ACI_WINDOW_SIZE: int = 200

# Target miscoverage for 80% prediction interval
TARGET_MISCOVERAGE: float = 0.20


class CQRACIModel:
    """CQR + ACI on LightGBM quantile regressors.

    Base learners: identical to M2's three LightGBM quantile boosters.
    CQR step: conformity scores computed on calibration set (last 20% of Train).
    ACI step: online alpha update with gamma tuned on Train+Val.
    """

    name: str = "m3_cqr_aci"
    produces_intervals: bool = True

    def __init__(
        self,
        base_model: LGBMQuantileModel | None = None,
        gamma: float = 0.01,
    ) -> None:
        """Initialize CQR+ACI model.

        Parameters
        ----------
        base_model : LGBMQuantileModel | None
            Pre-trained M2 model to reuse. If None, a new one will be
            created and trained during fit().
        gamma : float
            ACI learning rate for online alpha update.
        """
        self._base_model = base_model
        self._gamma = gamma
        self._conformity_scores: np.ndarray | None = None
        self._cqr_adjustment: float = 0.0
        self._alpha_t: float = TARGET_MISCOVERAGE  # ACI state variable
        self._is_calibrated: bool = False
        self._is_fitted: bool = False

    @property
    def base_model(self) -> LGBMQuantileModel | None:
        """Access the underlying M2 model."""
        return self._base_model

    @property
    def gamma(self) -> float:
        """ACI learning rate."""
        return self._gamma

    @property
    def cqr_adjustment(self) -> float:
        """The CQR conformity score adjustment (quantile of scores)."""
        return self._cqr_adjustment

    @property
    def alpha_t(self) -> float:
        """Current ACI miscoverage rate."""
        return self._alpha_t

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """Fit CQR calibration on top of pre-trained M2 boosters.

        If M2 boosters are not yet trained, this method trains them first
        using the same hyperparameters as M2.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix for fitting the base model (if needed).
        y_train : pd.Series
            Target for fitting the base model (if needed).
        X_val : pd.DataFrame | None
            Validation features for early stopping of base model.
        y_val : pd.Series | None
            Validation target for early stopping.
        """
        if self._base_model is None:
            self._base_model = LGBMQuantileModel()

        if not self._base_model._is_fitted:
            logger.info("Base M2 model not yet fitted. Training now.")
            self._base_model.fit(X_train, y_train, X_val, y_val)

        self._is_fitted = True
        logger.info("CQRACIModel fit complete (base model ready).")

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> None:
        """Compute CQR conformity scores on the calibration set.

        The conformity score for CQR is:
            E_i = max(q_lo_i - y_i, y_i - q_hi_i)

        The CQR adjustment is the ceil((1 - alpha)(n+1)/n)-th quantile
        of these scores.

        Parameters
        ----------
        X_cal : pd.DataFrame
            Features from the calibration portion of Train (last 20% by time).
        y_cal : pd.Series
            Targets from the calibration portion.

        Raises
        ------
        RuntimeError
            If the base model has not been fitted.
        """
        if self._base_model is None or not self._base_model._is_fitted:
            msg = "Base model must be fitted before calibration"
            raise RuntimeError(msg)

        # Get base quantile predictions on calibration set
        all_quantiles = self._base_model.predict_all_quantiles(X_cal)
        q_lo = all_quantiles[0.1]
        q_hi = all_quantiles[0.9]

        y_arr = np.asarray(y_cal, dtype=np.float64)

        # Compute conformity scores (Romano et al. 2019, Eq. 1)
        self._conformity_scores = np.maximum(q_lo - y_arr, y_arr - q_hi)

        # Compute the CQR quantile adjustment
        n = len(self._conformity_scores)
        target_coverage = 1.0 - TARGET_MISCOVERAGE
        # Use the corrected quantile level: ceil((1-alpha)(n+1))/n
        quantile_level = min(np.ceil((target_coverage) * (n + 1)) / n, 1.0)
        self._cqr_adjustment = float(np.quantile(self._conformity_scores, quantile_level))

        self._is_calibrated = True
        self._alpha_t = TARGET_MISCOVERAGE

        logger.info(
            "CQR calibration complete: n_cal=%d, adjustment=%.4f, "
            "score_median=%.4f, score_95pct=%.4f",
            n,
            self._cqr_adjustment,
            float(np.median(self._conformity_scores)),
            float(np.percentile(self._conformity_scores, 95)),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point estimate (same as M2's median booster).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Shape (n_samples,). Point estimates.
        """
        if self._base_model is None or not self._is_fitted:
            msg = "Model must be fit before predict()"
            raise RuntimeError(msg)

        return self._base_model.predict(X)

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Predict CQR-calibrated (p10, p90) intervals.

        The intervals are adjusted by the CQR conformity score quantile:
            [q_lo - Q, q_hi + Q]

        where Q is the CQR adjustment computed during calibrate().

        If ACI has updated alpha_t, the adjustment is recomputed using
        the current alpha_t.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 2) with columns [p10, p90].
            Intervals are adjusted by CQR conformity scores + ACI alpha.
        """
        if self._base_model is None or not self._is_fitted:
            msg = "Model must be fit before predict_quantiles()"
            raise RuntimeError(msg)

        if not self._is_calibrated or self._conformity_scores is None:
            msg = "Model must be calibrated before predict_quantiles()"
            raise RuntimeError(msg)

        # Get base quantile predictions
        all_quantiles = self._base_model.predict_all_quantiles(X)
        q_lo = all_quantiles[0.1]
        q_hi = all_quantiles[0.9]

        # Recompute CQR adjustment using current ACI alpha_t
        adjustment = self._get_current_adjustment()

        # Apply CQR adjustment
        calibrated_lo = q_lo - adjustment
        calibrated_hi = q_hi + adjustment

        return np.column_stack([calibrated_lo, calibrated_hi]).astype(np.float64)

    def _get_current_adjustment(self) -> float:
        """Compute the CQR adjustment using the current ACI alpha_t.

        Returns
        -------
        float
            The conformity score quantile at level (1 - alpha_t).
        """
        if self._conformity_scores is None:
            return self._cqr_adjustment

        n = len(self._conformity_scores)
        target_coverage = 1.0 - self._alpha_t
        quantile_level = min(np.ceil(target_coverage * (n + 1)) / n, 1.0)

        # Clamp to valid range
        quantile_level = np.clip(quantile_level, 0.0, 1.0)

        return float(np.quantile(self._conformity_scores, quantile_level))

    def update_aci(self, y_true: float, q_lo: float, q_hi: float) -> None:
        """Update the ACI miscoverage rate after observing one outcome.

        Update rule (Gibbs & Candes 2021):
            alpha_{t+1} = alpha_t + gamma * (1{y not in [q_lo, q_hi]} - target_miscoverage)

        Parameters
        ----------
        y_true : float
            Observed fantasy points.
        q_lo : float
            Lower bound of the prediction interval.
        q_hi : float
            Upper bound of the prediction interval.
        """
        miscovered = 1.0 if (y_true < q_lo or y_true > q_hi) else 0.0
        self._alpha_t = self._alpha_t + self._gamma * (miscovered - TARGET_MISCOVERAGE)

        # Clamp alpha_t to a reasonable range to prevent degenerate intervals
        self._alpha_t = np.clip(self._alpha_t, 0.01, 0.50)

    def update_aci_batch(
        self,
        y_true: np.ndarray,
        q_lo: np.ndarray,
        q_hi: np.ndarray,
    ) -> list[float]:
        """Update ACI sequentially for a batch of observations.

        Parameters
        ----------
        y_true : np.ndarray
            Observed values (in temporal order).
        q_lo : np.ndarray
            Lower prediction interval bounds.
        q_hi : np.ndarray
            Upper prediction interval bounds.

        Returns
        -------
        list[float]
            History of alpha_t values after each update.
        """
        alpha_history: list[float] = []
        for i in range(len(y_true)):
            self.update_aci(float(y_true[i]), float(q_lo[i]), float(q_hi[i]))
            alpha_history.append(self._alpha_t)
        return alpha_history

    def reset_aci(self) -> None:
        """Reset ACI state to the initial miscoverage rate."""
        self._alpha_t = TARGET_MISCOVERAGE

    def get_params(self) -> dict[str, Any]:
        """Return hyperparameters for MLflow logging."""
        base_params = self._base_model.get_params() if self._base_model else {}
        return {
            "model_type": "cqr_aci",
            "gamma": self._gamma,
            "target_miscoverage": TARGET_MISCOVERAGE,
            "aci_window_size": ACI_WINDOW_SIZE,
            "cqr_adjustment": self._cqr_adjustment,
            "base_model_params": base_params,
        }


def tune_aci_gamma(
    model: CQRACIModel,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    gamma_grid: list[float] | None = None,
) -> float:
    """Tune the ACI gamma parameter on validation data.

    For each gamma candidate, simulate the ACI online update on the
    validation set and measure the empirical coverage. Select the gamma
    whose coverage is closest to the target (80%).

    Parameters
    ----------
    model : CQRACIModel
        A calibrated CQR model.
    X_val : pd.DataFrame
        Validation features (temporal order).
    y_val : pd.Series
        Validation targets.
    gamma_grid : list[float] | None
        Gamma values to try. Defaults to ACI_GAMMA_GRID.

    Returns
    -------
    float
        Best gamma value.
    """
    if gamma_grid is None:
        gamma_grid = ACI_GAMMA_GRID

    y_arr = np.asarray(y_val, dtype=np.float64)
    target_coverage = 1.0 - TARGET_MISCOVERAGE

    best_gamma = gamma_grid[0]
    best_coverage_error = float("inf")

    for gamma in gamma_grid:
        # Reset and set gamma
        model._gamma = gamma
        model.reset_aci()

        # Get base quantile predictions
        all_quantiles = model._base_model.predict_all_quantiles(X_val)  # type: ignore[union-attr]
        base_q_lo = all_quantiles[0.1]
        base_q_hi = all_quantiles[0.9]

        covered_count = 0
        total_count = len(y_arr)

        for i in range(total_count):
            # Get current adjustment
            adj = model._get_current_adjustment()
            cal_lo = base_q_lo[i] - adj
            cal_hi = base_q_hi[i] + adj

            # Check coverage
            if cal_lo <= y_arr[i] <= cal_hi:
                covered_count += 1

            # Update ACI
            model.update_aci(float(y_arr[i]), cal_lo, cal_hi)

        empirical_coverage = covered_count / total_count if total_count > 0 else 0.0
        coverage_error = abs(empirical_coverage - target_coverage)

        logger.info(
            "ACI gamma=%.3f: empirical_coverage=%.4f, error=%.4f",
            gamma,
            empirical_coverage,
            coverage_error,
        )

        if coverage_error < best_coverage_error:
            best_coverage_error = coverage_error
            best_gamma = gamma

    # Reset and apply best gamma
    model._gamma = best_gamma
    model.reset_aci()

    logger.info("Best ACI gamma=%.3f (coverage_error=%.4f)", best_gamma, best_coverage_error)

    return best_gamma
