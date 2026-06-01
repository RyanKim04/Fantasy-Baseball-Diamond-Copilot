"""Calibration diagnostics for prediction intervals.

Implements the checks from validation_protocol.md section 9:
1. Reliability diagram
2. Empirical coverage at multiple nominal levels
3. Coverage by predicted-bin
4. Coverage drift over time
5. Width vs. predicted-value plot

Owner: evaluator subagent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np


def reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    output_path: Path | None = None,
) -> dict[str, list[float]]:
    """Create reliability diagram data: predicted vs actual means per bin.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted means.
    n_bins : int
        Number of equal-mass bins.
    output_path : Path | None
        If provided, saves a matplotlib figure.

    Returns
    -------
    dict
        {"bin_centers": [...], "bin_means_pred": [...], "bin_means_actual": [...]}
    """
    raise NotImplementedError("To be implemented by evaluator subagent")


def coverage_at_levels(
    y_true: np.ndarray,
    lower_10: np.ndarray,
    upper_90: np.ndarray,
    lower_25: np.ndarray | None = None,
    upper_75: np.ndarray | None = None,
    lower_15: np.ndarray | None = None,
    upper_85: np.ndarray | None = None,
    lower_05: np.ndarray | None = None,
    upper_95: np.ndarray | None = None,
) -> dict[int, float]:
    """Compute empirical coverage at multiple nominal levels.

    Returns
    -------
    dict[int, float]
        {nominal_coverage_pct: empirical_coverage}. Always includes 80%.
    """
    raise NotImplementedError("To be implemented by evaluator subagent")


def coverage_by_predicted_bin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Coverage within each predicted-value bin.

    Detects miscalibration for high- or low-projection players.

    Returns
    -------
    dict
        {"bin_centers": [...], "coverage": [...]}
    """
    raise NotImplementedError("To be implemented by evaluator subagent")


def coverage_drift_over_time(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    dates: np.ndarray,
    n_bins: int = 4,
) -> dict[str, list]:
    """Coverage computed in temporal bins across the evaluation period.

    Detects whether ACI is maintaining coverage over time.

    Returns
    -------
    dict
        {"bin_labels": [...], "coverage": [...], "n_samples": [...]}
    """
    raise NotImplementedError("To be implemented by evaluator subagent")
