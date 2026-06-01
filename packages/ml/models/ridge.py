"""M1: Ridge regression point estimate model.

The floor of the three-model bake-off. If this doesn't beat baselines,
something is broken upstream in the features or target encoding.

Owner: modeler subagent.
Spec: docs/model_bakeoff.md section 3, M1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


class RidgeProjectionModel:
    """Ridge regression for per-game fantasy points (point estimate only).

    Preprocessing: median-imputation + StandardScaler fit on Train only.
    Hyperparameters: alpha searched over np.logspace(-3, 3, 25) via walk-forward CV.
    """

    name: str = "m1_ridge"
    produces_intervals: bool = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """Fit Ridge regression with StandardScaler."""
        raise NotImplementedError("To be implemented by modeler subagent")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point estimate prediction."""
        raise NotImplementedError("To be implemented by modeler subagent")

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Not supported -- returns empty array."""
        import numpy as np
        return np.array([])
