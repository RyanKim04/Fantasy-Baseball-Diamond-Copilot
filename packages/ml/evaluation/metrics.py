"""Metric computation for player projection evaluation.

All metrics defined in validation_protocol.md section 7.

Owner: evaluator subagent.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sp_stats

from packages.shared.schemas.ml import ModelCandidate, PlayerType, PopulationMetrics


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Arrays of equal length.

    Returns
    -------
    float
        RMSE value (non-negative).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length, got {len(y_true)} and {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("Cannot compute RMSE on empty arrays")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Arrays of equal length.

    Returns
    -------
    float
        MAE value (non-negative).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length, got {len(y_true)} and {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("Cannot compute MAE on empty arrays")
    return float(np.mean(np.abs(y_true - y_pred)))


def spearman_rho(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation coefficient.

    Parameters
    ----------
    y_true, y_pred : np.ndarray
        Arrays of equal length.

    Returns
    -------
    float
        Spearman rho in [-1, 1].
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length, got {len(y_true)} and {len(y_pred)}"
        )
    if len(y_true) < 2:
        raise ValueError("Need at least 2 observations for Spearman correlation")
    rho, _ = sp_stats.spearmanr(y_true, y_pred)
    return float(rho)


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
    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if len(y_true) == 0:
        raise ValueError("Cannot compute coverage on empty arrays")
    if not (len(y_true) == len(lower) == len(upper)):
        raise ValueError("All arrays must have the same length")
    covered = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(covered))


def sharpness(lower: np.ndarray, upper: np.ndarray) -> float:
    """Mean width of prediction intervals. Lower is better.

    Parameters
    ----------
    lower, upper : np.ndarray
        Lower and upper bounds of prediction intervals.

    Returns
    -------
    float
        Mean interval width (non-negative).
    """
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    if len(lower) == 0:
        raise ValueError("Cannot compute sharpness on empty arrays")
    if len(lower) != len(upper):
        raise ValueError("lower and upper must have the same length")
    return float(np.mean(upper - lower))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """Pinball (quantile) loss at quantile level tau.

    The pinball loss is defined as:
        L(y, q, tau) = tau * max(y - q, 0) + (1 - tau) * max(q - y, 0)

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted quantile values.
    tau : float
        Quantile level (e.g., 0.1, 0.5, 0.9). Must be in (0, 1).

    Returns
    -------
    float
        Mean pinball loss (non-negative).
    """
    if not 0 < tau < 1:
        raise ValueError(f"tau must be in (0, 1), got {tau}")
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must have same length, got {len(y_true)} and {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("Cannot compute pinball loss on empty arrays")
    diff = y_true - y_pred
    loss = np.where(diff >= 0, tau * diff, (tau - 1) * diff)
    return float(np.mean(loss))


def compute_population_metrics(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_p10: np.ndarray | None,
    y_pred_p90: np.ndarray | None,
    player_type: str,
    model_candidate: str,
) -> PopulationMetrics:
    """Compute all headline metrics for one population.

    Parameters
    ----------
    y_true : np.ndarray
        Actual fantasy points.
    y_pred_mean : np.ndarray
        Predicted mean (point estimate).
    y_pred_p10, y_pred_p90 : np.ndarray | None
        Lower/upper bounds of 80% prediction interval.
        If None, interval metrics are omitted.
    player_type : str
        "hitter" or "pitcher".
    model_candidate : str
        One of the ModelCandidate enum values.

    Returns
    -------
    PopulationMetrics
        Pydantic model with all metrics filled in.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_mean = np.asarray(y_pred_mean, dtype=np.float64)

    metrics_kwargs: dict = {
        "player_type": PlayerType(player_type),
        "model_candidate": ModelCandidate(model_candidate),
        "n_predictions": len(y_true),
        "rmse": rmse(y_true, y_pred_mean),
        "mae": mae(y_true, y_pred_mean),
        "spearman_rho": spearman_rho(y_true, y_pred_mean),
    }

    if y_pred_p10 is not None and y_pred_p90 is not None:
        p10 = np.asarray(y_pred_p10, dtype=np.float64)
        p90 = np.asarray(y_pred_p90, dtype=np.float64)
        metrics_kwargs["coverage_80"] = interval_coverage(y_true, p10, p90)
        metrics_kwargs["sharpness_mean_width"] = sharpness(p10, p90)
        metrics_kwargs["pinball_10"] = pinball_loss(y_true, p10, 0.1)
        metrics_kwargs["pinball_50"] = pinball_loss(y_true, y_pred_mean, 0.5)
        metrics_kwargs["pinball_90"] = pinball_loss(y_true, p90, 0.9)

    return PopulationMetrics(**metrics_kwargs)
