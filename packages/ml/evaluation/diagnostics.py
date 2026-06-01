"""Supplementary diagnostic plots for model evaluation.

Functions here complement calibration.py with additional visualizations
specified in validation_protocol.md section 9.

Owner: evaluator subagent.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime in plot functions

import numpy as np


def width_vs_predicted(
    y_pred_mean: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_bins: int = 10,
    output_path: Path | None = None,
) -> dict[str, list[float]]:
    """Plot interval width as a function of predicted mean.

    Confirms heteroscedasticity is captured (wider intervals for
    high-variance roles). Per validation_protocol.md section 9.5.

    Bins predictions into n_bins equal-mass bins by predicted mean,
    then computes the mean interval width in each bin.

    Parameters
    ----------
    y_pred_mean : np.ndarray
        Predicted mean values.
    lower, upper : np.ndarray
        Lower and upper bounds of prediction intervals.
    n_bins : int
        Number of bins for aggregation.
    output_path : Path | None
        If provided, saves the plot.

    Returns
    -------
    dict[str, list[float]]
        Keys: 'predicted_means', 'widths' (binned averages).
    """
    y_pred_mean = np.asarray(y_pred_mean, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    if len(y_pred_mean) == 0:
        raise ValueError("Cannot compute width_vs_predicted with empty arrays")

    widths = upper - lower
    sorted_indices = np.argsort(y_pred_mean)
    bin_size = len(y_pred_mean) // n_bins
    remainder = len(y_pred_mean) % n_bins

    predicted_means: list[float] = []
    mean_widths: list[float] = []

    start = 0
    for i in range(n_bins):
        end = start + bin_size + (1 if i < remainder else 0)
        if start >= end:
            continue
        idx = sorted_indices[start:end]
        predicted_means.append(float(np.mean(y_pred_mean[idx])))
        mean_widths.append(float(np.mean(widths[idx])))
        start = end

    if output_path is not None:
        _plot_width_vs_predicted(predicted_means, mean_widths, output_path)

    return {
        "predicted_means": predicted_means,
        "widths": mean_widths,
    }


def _plot_width_vs_predicted(
    predicted_means: list[float],
    mean_widths: list[float],
    output_path: Path,
) -> None:
    """Save width vs predicted figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predicted_means, mean_widths, s=60, zorder=5)
    ax.plot(predicted_means, mean_widths, "b-", alpha=0.5)
    ax.set_xlabel("Predicted Mean Fantasy Points")
    ax.set_ylabel("Mean 80% PI Width")
    ax.set_title("Interval Width vs. Predicted Mean")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


def residual_vs_predicted(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    output_path: Path | None = None,
) -> dict[str, list[float]]:
    """Residual plot vs predicted value (heteroscedasticity check).

    Per validation_protocol.md section 7.4: a residual plot vs. predicted
    value is included in the evaluation report.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred_mean : np.ndarray
        Predicted mean values.
    output_path : Path | None
        If provided, saves the plot.

    Returns
    -------
    dict[str, list[float]]
        Keys: 'predicted', 'residuals' (raw values, not binned).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_mean = np.asarray(y_pred_mean, dtype=np.float64)

    if len(y_true) == 0:
        raise ValueError("Cannot compute residuals with empty arrays")

    residuals = y_true - y_pred_mean

    if output_path is not None:
        _plot_residual_vs_predicted(y_pred_mean, residuals, output_path)

    return {
        "predicted": y_pred_mean.tolist(),
        "residuals": residuals.tolist(),
    }


def _plot_residual_vs_predicted(
    predicted: np.ndarray,
    residuals: np.ndarray,
    output_path: Path,
) -> None:
    """Save residual vs predicted figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(predicted, residuals, alpha=0.3, s=10)
    ax.axhline(y=0, color="r", linestyle="--", alpha=0.7)
    ax.set_xlabel("Predicted Mean Fantasy Points")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residuals vs. Predicted")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
