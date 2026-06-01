"""M1: Ridge regression point estimate model.

The floor of the three-model bake-off. If this doesn't beat baselines,
something is broken upstream in the features or target encoding.

Owner: modeler subagent.
Spec: docs/model_bakeoff.md section 3, M1.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Hyperparameter grid for alpha (model_bakeoff.md section 3, M1)
ALPHA_GRID: np.ndarray = np.logspace(-3, 3, 25)


class RidgeProjectionModel:
    """Ridge regression for per-game fantasy points (point estimate only).

    Preprocessing: median-imputation + StandardScaler fit on Train only.
    Hyperparameters: alpha searched over np.logspace(-3, 3, 25) via walk-forward CV.
    """

    name: str = "m1_ridge"
    produces_intervals: bool = False

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self._pipeline: Pipeline | None = None
        self._is_fitted: bool = False
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """Fit Ridge regression with StandardScaler.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix (no target, no index columns).
        y_train : pd.Series
            Target: per-game fantasy points.
        X_val, y_val : ignored
            Not used for Ridge (no early stopping), accepted for protocol
            compatibility.
        """
        self._feature_names = list(X_train.columns)

        self._pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ])

        self._pipeline.fit(X_train, y_train)
        self._is_fitted = True

        logger.info(
            "RidgeProjectionModel fit: alpha=%.4f, n_features=%d, n_samples=%d",
            self.alpha,
            X_train.shape[1],
            X_train.shape[0],
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point estimate prediction.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Shape (n_samples,). Point estimates of fantasy points.
        """
        if not self._is_fitted or self._pipeline is None:
            msg = "Model must be fit before predict()"
            raise RuntimeError(msg)

        return self._pipeline.predict(X).astype(np.float64)

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Not supported -- returns empty array.

        M1 produces point estimates only (model_bakeoff.md section 3).
        """
        return np.array([]).reshape(0, 2)

    def get_params(self) -> dict[str, Any]:
        """Return hyperparameters for MLflow logging."""
        return {"alpha": self.alpha, "model_type": "ridge"}

    def get_coefficients(self) -> np.ndarray | None:
        """Return fitted Ridge coefficients (for interpretability)."""
        if not self._is_fitted or self._pipeline is None:
            return None
        ridge_step = self._pipeline.named_steps["ridge"]
        return ridge_step.coef_
