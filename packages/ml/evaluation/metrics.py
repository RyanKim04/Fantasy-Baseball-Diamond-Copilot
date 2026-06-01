"""Metric computation for player projection evaluation.

All metrics defined in validation_protocol.md section 7.

Owner: evaluator subagent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from packages.shared.schemas.ml import PopulationMetrics


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    raise NotImplementedError("To be implemented by evaluator subagent")


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    raise NotImplementedError("To be implemented by evaluator subagent")


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient."""
    raise NotImplementedError("To be implemented by evaluator subagent")


def interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Empirical coverage of prediction intervals.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    lower, upper : np.ndarray
        Lower and upper bounds of prediction intervals.

    Returns
    -------
    float
        Fraction of y_true values within [lower, upper].
    """
    raise NotImplementedError("To be implemented by evaluator subagent")


def sharpness(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of prediction intervals. Lower is better."""
    raise NotImplementedError("To be implemented by evaluator subagent")


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss at quantile level tau.

    Parameters
    ----------
    tau : float
        Quantile level (e.g., 0.1, 0.5, 0.9).
    """
    raise NotImplementedError("To be implemented by evaluator subagent")


def compute_population_metrics(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_p10: np.ndarray | None,
    y_pred_p90: np.ndarray | None,
    player_type: str,
    model_candidate: str,
) -> PopulationMetrics:
    """Compute all headline metrics for one population.

    Returns
    -------
    PopulationMetrics
        Pydantic model with all metrics filled in.
    """
    raise NotImplementedError("To be implemented by evaluator subagent")
