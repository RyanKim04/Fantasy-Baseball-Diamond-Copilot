"""M2: LightGBM quantile regression with uncalibrated intervals.

The middle candidate in the three-model bake-off. Produces heteroscedastic
intervals but without conformal calibration, so coverage is expected to
be outside the [0.75, 0.85] acceptable range.

Owner: modeler subagent.
Spec: docs/model_bakeoff.md section 3, M2.
"""

from __future__ import annotations

import logging
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from packages.ml.evaluation.metrics import pinball_loss  # single source of truth

logger = logging.getLogger(__name__)

# Quantile levels for the three boosters
QUANTILE_LEVELS: tuple[float, float, float] = (0.1, 0.5, 0.9)

# Default hyperparameters (overridden by Optuna tuning)
DEFAULT_PARAMS: dict[str, Any] = {
    "num_leaves": 63,
    "min_data_in_leaf": 50,
    "learning_rate": 0.05,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "n_estimators": 2000,
    "verbose": -1,
}


class LGBMQuantileModel:
    """LightGBM quantile regression: three boosters at alpha in {0.1, 0.5, 0.9}.

    Hyperparameters shared across the three quantile heads:
    - num_leaves, min_data_in_leaf, learning_rate, feature_fraction, bagging_fraction
    - Tuned via Optuna with 50 trials per population
    - Early stopping on summed Pinball loss, patience 50
    """

    name: str = "m2_lgbm_quantile"
    produces_intervals: bool = True

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        """Initialize with shared hyperparameters.

        Parameters
        ----------
        params : dict[str, Any] | None
            LightGBM hyperparameters shared across quantile heads.
            If None, uses DEFAULT_PARAMS.
        """
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        self._boosters: dict[float, lgb.LGBMRegressor] = {}
        self._imputer: SimpleImputer | None = None
        self._is_fitted: bool = False
        self._feature_names: list[str] = []
        self._best_iterations: dict[float, int] = {}

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    def boosters(self) -> dict[float, lgb.LGBMRegressor]:
        """Access the fitted LightGBM boosters (used by M3 to reuse M2)."""
        return self._boosters

    @property
    def imputer(self) -> SimpleImputer | None:
        """Access the fitted imputer (used by M3 to reuse M2's preprocessing)."""
        return self._imputer

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """Fit three LightGBM quantile boosters.

        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix.
        y_train : pd.Series
            Target: per-game fantasy points.
        X_val : pd.DataFrame | None
            Validation features for early stopping.
        y_val : pd.Series | None
            Validation target for early stopping.
        """
        self._feature_names = list(X_train.columns)

        # Fit imputer on training data only
        self._imputer = SimpleImputer(strategy="median")
        X_train_imputed = pd.DataFrame(
            self._imputer.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )

        X_val_imputed = None
        if X_val is not None:
            X_val_imputed = pd.DataFrame(
                self._imputer.transform(X_val),
                columns=X_val.columns,
                index=X_val.index,
            )

        for alpha in QUANTILE_LEVELS:
            booster_params = {
                k: v for k, v in self._params.items()
                if k not in ("n_estimators",)
            }

            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=self._params.get("n_estimators", 2000),
                **booster_params,
            )

            fit_kwargs: dict[str, Any] = {}
            if X_val_imputed is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val_imputed, y_val)]
                fit_kwargs["callbacks"] = [
                    lgb.early_stopping(stopping_rounds=50, verbose=False),
                    lgb.log_evaluation(period=0),
                ]

            model.fit(X_train_imputed, y_train, **fit_kwargs)
            self._boosters[alpha] = model

            best_iter = model.best_iteration_ if model.best_iteration_ > 0 else model.n_estimators
            self._best_iterations[alpha] = best_iter

            logger.info(
                "LGBMQuantileModel alpha=%.1f fit: best_iteration=%d, n_features=%d",
                alpha,
                best_iter,
                X_train.shape[1],
            )

        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Median (alpha=0.5) prediction as point estimate.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Shape (n_samples,). Point estimates (median).
        """
        if not self._is_fitted or self._imputer is None:
            msg = "Model must be fit before predict()"
            raise RuntimeError(msg)

        X_imputed = pd.DataFrame(
            self._imputer.transform(X),
            columns=X.columns,
            index=X.index,
        )
        return self._boosters[0.5].predict(X_imputed).astype(np.float64)

    def predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Predict (p10, p90) quantiles.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        np.ndarray
            Shape (n_samples, 2) with columns [p10, p90].
        """
        if not self._is_fitted or self._imputer is None:
            msg = "Model must be fit before predict_quantiles()"
            raise RuntimeError(msg)

        X_imputed = pd.DataFrame(
            self._imputer.transform(X),
            columns=X.columns,
            index=X.index,
        )

        p10 = self._boosters[0.1].predict(X_imputed)
        p90 = self._boosters[0.9].predict(X_imputed)

        return np.column_stack([p10, p90]).astype(np.float64)

    def predict_all_quantiles(self, X: pd.DataFrame) -> dict[float, np.ndarray]:
        """Predict all three quantile levels (used by M3/CQR).

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.

        Returns
        -------
        dict[float, np.ndarray]
            Mapping from quantile level to predictions.
        """
        if not self._is_fitted or self._imputer is None:
            msg = "Model must be fit before predict_all_quantiles()"
            raise RuntimeError(msg)

        X_imputed = pd.DataFrame(
            self._imputer.transform(X),
            columns=X.columns,
            index=X.index,
        )

        return {
            alpha: self._boosters[alpha].predict(X_imputed).astype(np.float64)
            for alpha in QUANTILE_LEVELS
        }

    def get_params(self) -> dict[str, Any]:
        """Return hyperparameters for MLflow logging."""
        return {
            "model_type": "lgbm_quantile",
            **{k: v for k, v in self._params.items()},
            "best_iterations": self._best_iterations,
        }

    def get_feature_importances(self) -> dict[float, np.ndarray]:
        """Return feature importances for each booster."""
        if not self._is_fitted:
            return {}
        return {
            alpha: booster.feature_importances_
            for alpha, booster in self._boosters.items()
        }
