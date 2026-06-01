"""M2: LightGBM quantile regression with uncalibrated intervals.

The middle candidate in the three-model bake-off. Produces heteroscedastic
intervals but without conformal calibration, so coverage is expected to
be outside the [0.75, 0.85] acceptable range.

Owner: modeler subagent.
Spec: docs/model_bakeoff.md section 3, M2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class LGBMQuantileModel:
    """LightGBM quantile regression: three boosters at alpha in {0.1, 0.5, 0.9}.

    Hyperparameters shared across the three quantile heads:
    - num_leaves, min_data_in_leaf, learning_rate, feature_fraction, bagging_fraction
    - Tuned via Optuna with 50 trials per population
    - Early stopping on summed Pinball loss, patience 50
    """

    name: str = "m2_lgbm_quantile"
    produces_intervals: bool = True

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """Fit three LightGBM quantile boosters."""
        raise NotImplementedError("To be implemented by modeler subagent")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Median (alpha=0.5) prediction as point estimate."""
        raise NotImplementedError("To be implemented by modeler subagent")

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Predict (p10, p90) quantiles.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 2) with columns [p10, p90].
        """
        raise NotImplementedError("To be implemented by modeler subagent")
