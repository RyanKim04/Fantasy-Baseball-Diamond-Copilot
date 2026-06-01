"""Supplementary diagnostic plots for model evaluation.

Functions here complement calibration.py with additional visualizations
specified in validation_protocol.md section 9.

Owner: evaluator subagent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


def width_vs_predicted(
    y_pred_mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    output_path: Path | None = None,
) -> dict[str, list[float]]:
    """Plot interval width as a function of predicted mean.

    Confirms heteroscedasticity is captured (wider intervals for
    high-variance roles).

    Parameters
    ----------
    y_pred_mean : np.ndarray
        Predicted mean values.
    lower, upper : np.ndarray
        Lower and upper bounds of prediction intervals.
    output_path : Path | None
        If provided, saves the plot.

    Returns
    -------
    dict[str, list[float]]
        Keys: 'predicted_means', 'widths' (binned averages).
    """
    raise NotImplementedError("To be implemented by evaluator subagent")


def residual_vs_predicted(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    output_path: Path | None = None,
) -> None:
    """Residual plot vs predicted value (heteroscedasticity check).

    Per validation_protocol.md section 7.4.
    """
    raise NotImplementedError("To be implemented by evaluator subagent")
