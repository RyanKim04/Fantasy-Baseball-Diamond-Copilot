"""Model module for player projection.

This module owns model training and inference. Models consume feature tables
produced by the feature-engineer and output predictions conforming to the
PredictionRow schema.

Three model candidates are defined per docs/model_bakeoff.md:
- M1: Ridge regression (point estimate floor)
- M2: LightGBM quantile (uncalibrated intervals)
- M3: CQR + ACI on LightGBM (production candidate)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


@runtime_checkable
class ProjectionModel(Protocol):
    """Protocol that all projection model implementations must satisfy."""

    @property
    def name(self) -> str:
        """Model identifier matching ModelCandidate enum value."""
        ...

    @property
    def produces_intervals(self) -> bool:
        """Whether this model produces prediction intervals (p10, p90)."""
        ...

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """Fit the model on training data.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix (no target, no index columns).
        y_train : pd.Series
            Target: per-game fantasy points.
        X_val : pd.DataFrame | None
            Validation features for early stopping (M2, M3).
        y_val : pd.Series | None
            Validation target for early stopping.
        """
        ...

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict point estimates (mean/median).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Shape (n_samples,). Point estimates of fantasy points.
        """
        ...

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Predict quantile estimates.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 2) for (p10, p90).
            Returns empty array if produces_intervals is False.
        """
        ...
