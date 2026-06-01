"""M3: CQR + ACI on LightGBM (production candidate).

The ceiling candidate in the three-model bake-off. Reuses M2's tuned
LightGBM quantile boosters and layers conformal calibration (CQR) with
adaptive conformal inference (ACI) for non-stationarity.

Owner: modeler subagent.
Spec: docs/model_bakeoff.md section 3, M3.
References:
  - CQR: Romano, Patterson, Candes (2019) "Conformal Quantile Regression"
  - ACI: Gibbs & Candes (2021) "Adaptive Conformal Inference Under Distribution Shift"
Library: MAPIE (mapie.regression.MapieQuantileRegressor)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class CQRACIModel:
    """CQR + ACI on LightGBM quantile regressors.

    Base learners: identical to M2's three LightGBM quantile boosters.
    CQR step: conformity scores computed on calibration set (last 20% of Train).
    ACI step: online alpha update with gamma tuned on Train+Val.
    """

    name: str = "m3_cqr_aci"
    produces_intervals: bool = True

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
        """
        raise NotImplementedError("To be implemented by modeler subagent")

    def calibrate(
        self,
        X_cal: pd.DataFrame,
        y_cal: pd.Series,
    ) -> None:
        """Compute CQR conformity scores on the calibration set.

        Parameters
        ----------
        X_cal : pd.DataFrame
            Features from the calibration portion of Train (last 20% by time).
        y_cal : pd.Series
            Targets from the calibration portion.
        """
        raise NotImplementedError("To be implemented by modeler subagent")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point estimate (same as M2's median booster)."""
        raise NotImplementedError("To be implemented by modeler subagent")

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Predict CQR-calibrated (p10, p90) intervals.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 2) with columns [p10, p90].
            Intervals are adjusted by CQR conformity scores + ACI alpha.
        """
        raise NotImplementedError("To be implemented by modeler subagent")
