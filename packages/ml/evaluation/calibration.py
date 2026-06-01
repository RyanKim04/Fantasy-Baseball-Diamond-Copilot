"""Calibration diagnostics for prediction intervals.

Implements the checks from validation_protocol.md section 9:
1. Reliability diagram
2. Empirical coverage at multiple nominal levels
3. Coverage by predicted-bin
4. Coverage drift over time
5. Width vs. predicted-value plot (in diagnostics.py)

Owner: evaluator subagent.
"""

from __future__ import annotations

from pathlib import Path  # noqa: TC003 - used at runtime in plot functions

import numpy as np


def reliability_diagram(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    output_path: Path | None = None,
) -> dict[str, list[float]]:
    """Create reliability diagram data: predicted vs actual means per bin.

    Bins predictions into n_bins equal-mass bins by predicted mean,
    then plots mean predicted vs. mean actual per bin. Deviation from
    the y = x diagonal indicates miscalibration of the point estimate.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted means.
    n_bins : int
        Number of equal-mass bins (default 10).
    output_path : Path | None
        If provided, saves a matplotlib figure.

    Returns
    -------
    dict
        {"bin_centers": [...], "bin_means_pred": [...], "bin_means_actual": [...]}
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        raise ValueError("Cannot create reliability diagram with empty arrays")

    # Create equal-mass bins based on predicted values
    sorted_indices = np.argsort(y_pred)
    bin_size = len(y_pred) // n_bins
    remainder = len(y_pred) % n_bins

    bin_centers: list[float] = []
    bin_means_pred: list[float] = []
    bin_means_actual: list[float] = []

    start = 0
    for i in range(n_bins):
        # Distribute remainder across first bins
        end = start + bin_size + (1 if i < remainder else 0)
        if start >= end:
            continue
        idx = sorted_indices[start:end]
        bin_centers.append(float(np.mean(y_pred[idx])))
        bin_means_pred.append(float(np.mean(y_pred[idx])))
        bin_means_actual.append(float(np.mean(y_true[idx])))
        start = end

    if output_path is not None:
        _plot_reliability_diagram(bin_means_pred, bin_means_actual, output_path)

    return {
        "bin_centers": bin_centers,
        "bin_means_pred": bin_means_pred,
        "bin_means_actual": bin_means_actual,
    }


def _plot_reliability_diagram(
    bin_means_pred: list[float],
    bin_means_actual: list[float],
    output_path: Path,
) -> None:
    """Save reliability diagram figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    all_vals = bin_means_pred + bin_means_actual
    lo, hi = min(all_vals), max(all_vals)
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="Perfect calibration")
    ax.scatter(bin_means_pred, bin_means_actual, s=50, zorder=5)
    ax.set_xlabel("Mean Predicted")
    ax.set_ylabel("Mean Actual")
    ax.set_title("Reliability Diagram")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)


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

    The 80% level (p10, p90) is always computed. Other levels are computed
    if the corresponding quantile arrays are provided.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    lower_10, upper_90 : np.ndarray
        Bounds for the 80% interval (always required).
    lower_25, upper_75 : np.ndarray | None
        Bounds for the 50% interval.
    lower_15, upper_85 : np.ndarray | None
        Bounds for the 70% interval.
    lower_05, upper_95 : np.ndarray | None
        Bounds for the 90% interval.

    Returns
    -------
    dict[int, float]
        {nominal_coverage_pct: empirical_coverage}. Always includes 80%.
    """
    from packages.ml.evaluation.metrics import interval_coverage

    y_true = np.asarray(y_true, dtype=np.float64)
    result: dict[int, float] = {}

    # 80% is always computed
    result[80] = interval_coverage(y_true, lower_10, upper_90)

    if lower_25 is not None and upper_75 is not None:
        result[50] = interval_coverage(y_true, lower_25, upper_75)

    if lower_15 is not None and upper_85 is not None:
        result[70] = interval_coverage(y_true, lower_15, upper_85)

    if lower_05 is not None and upper_95 is not None:
        result[90] = interval_coverage(y_true, lower_05, upper_95)

    return result


def coverage_by_predicted_bin(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Coverage within each predicted-value bin.

    Detects miscalibration for high- or low-projection players.
    Per validation_protocol.md section 9: a per-bin coverage outside
    [0.65, 0.95] is flagged for critic review.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Predicted means (used for binning).
    lower, upper : np.ndarray
        Lower and upper bounds of 80% prediction intervals.
    n_bins : int
        Number of equal-mass bins.

    Returns
    -------
    dict
        {"bin_centers": [...], "coverage": [...], "n_samples": [...]}
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    if len(y_true) == 0:
        raise ValueError("Cannot compute coverage by bin with empty arrays")

    sorted_indices = np.argsort(y_pred)
    bin_size = len(y_pred) // n_bins
    remainder = len(y_pred) % n_bins

    bin_centers: list[float] = []
    coverages: list[float] = []
    n_samples: list[int] = []

    start = 0
    for i in range(n_bins):
        end = start + bin_size + (1 if i < remainder else 0)
        if start >= end:
            continue
        idx = sorted_indices[start:end]
        bin_centers.append(float(np.mean(y_pred[idx])))
        covered = (y_true[idx] >= lower[idx]) & (y_true[idx] <= upper[idx])
        coverages.append(float(np.mean(covered)))
        n_samples.append(len(idx))
        start = end

    return {
        "bin_centers": bin_centers,
        "coverage": coverages,
        "n_samples": [float(n) for n in n_samples],
    }


def coverage_drift_over_time(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    dates: np.ndarray,
    n_bins: int = 4,
) -> dict[str, list]:
    """Coverage computed in temporal bins across the evaluation period.

    Detects whether ACI is maintaining coverage over time.
    Per protocol section 9.4: coverage should remain near 0.80 across
    all time bins, not just on average.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    lower, upper : np.ndarray
        Lower and upper bounds of prediction intervals.
    dates : np.ndarray
        Dates corresponding to each prediction (for binning).
    n_bins : int
        Number of temporal bins (default 4 per protocol).

    Returns
    -------
    dict
        {"bin_labels": [...], "coverage": [...], "n_samples": [...]}
    """
    import pandas as pd

    y_true = np.asarray(y_true, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    dates = pd.to_datetime(dates)

    if len(y_true) == 0:
        raise ValueError("Cannot compute coverage drift with empty arrays")

    # Sort by date
    sort_idx = np.argsort(dates)
    y_true_sorted = y_true[sort_idx]
    lower_sorted = lower[sort_idx]
    upper_sorted = upper[sort_idx]
    dates_sorted = dates[sort_idx]

    # Split into n_bins temporal bins of roughly equal size
    bin_size = len(y_true) // n_bins
    remainder = len(y_true) % n_bins

    bin_labels: list[str] = []
    coverages: list[float] = []
    n_samples_list: list[int] = []

    start = 0
    for i in range(n_bins):
        end = start + bin_size + (1 if i < remainder else 0)
        if start >= end:
            continue

        bin_dates = dates_sorted[start:end]
        label = f"{bin_dates[0].strftime('%Y-%m-%d')} to {bin_dates[-1].strftime('%Y-%m-%d')}"
        bin_labels.append(label)

        covered = (y_true_sorted[start:end] >= lower_sorted[start:end]) & (
            y_true_sorted[start:end] <= upper_sorted[start:end]
        )
        coverages.append(float(np.mean(covered)))
        n_samples_list.append(end - start)
        start = end

    return {
        "bin_labels": bin_labels,
        "coverage": coverages,
        "n_samples": n_samples_list,
    }


def pit_histogram(
    y_true: np.ndarray,
    predicted_mean: np.ndarray,
    predicted_p10: np.ndarray,
    predicted_p90: np.ndarray,
    n_bins: int = 10,
    output_path: Path | None = None,
) -> dict[str, list[float]]:
    """Probability Integral Transform (PIT) histogram.

    Approximates the CDF using a piecewise-linear interpolation through
    (p10, 0.1), (mean, 0.5), (p90, 0.9) and computes the PIT value
    for each observation. A well-calibrated model produces a uniform
    PIT histogram.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    predicted_mean : np.ndarray
        Predicted means (used as the median proxy).
    predicted_p10, predicted_p90 : np.ndarray
        10th and 90th percentile predictions.
    n_bins : int
        Number of bins for the histogram.
    output_path : Path | None
        If provided, saves the histogram figure.

    Returns
    -------
    dict
        {"bin_edges": [...], "counts": [...]}
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    predicted_mean = np.asarray(predicted_mean, dtype=np.float64)
    predicted_p10 = np.asarray(predicted_p10, dtype=np.float64)
    predicted_p90 = np.asarray(predicted_p90, dtype=np.float64)

    # Compute PIT values using piecewise linear CDF approximation
    pit_values = np.zeros(len(y_true))
    for i in range(len(y_true)):
        y = y_true[i]
        p10_val = predicted_p10[i]
        med_val = predicted_mean[i]
        p90_val = predicted_p90[i]

        if y <= p10_val:
            # Extrapolate below p10: linear from (p10, 0.1) toward 0
            if med_val > p10_val:
                slope = (0.5 - 0.1) / (med_val - p10_val)
                pit_values[i] = max(0.0, 0.1 - slope * (p10_val - y))
            else:
                pit_values[i] = 0.05
        elif y <= med_val:
            # Interpolate between p10 and median
            if med_val > p10_val:
                pit_values[i] = 0.1 + (0.5 - 0.1) * (y - p10_val) / (med_val - p10_val)
            else:
                pit_values[i] = 0.3
        elif y <= p90_val:
            # Interpolate between median and p90
            if p90_val > med_val:
                pit_values[i] = 0.5 + (0.9 - 0.5) * (y - med_val) / (p90_val - med_val)
            else:
                pit_values[i] = 0.7
        else:
            # Extrapolate above p90
            if p90_val > med_val:
                slope = (0.9 - 0.5) / (p90_val - med_val)
                pit_values[i] = min(1.0, 0.9 + slope * (y - p90_val))
            else:
                pit_values[i] = 0.95

    # Histogram
    counts, bin_edges = np.histogram(pit_values, bins=n_bins, range=(0.0, 1.0))
    counts_normalized = counts / counts.sum() if counts.sum() > 0 else counts

    if output_path is not None:
        _plot_pit_histogram(bin_edges, counts_normalized, output_path)

    return {
        "bin_edges": bin_edges.tolist(),
        "counts": counts_normalized.tolist(),
    }


def _plot_pit_histogram(
    bin_edges: np.ndarray,
    counts: np.ndarray,
    output_path: Path,
) -> None:
    """Save PIT histogram figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    bin_width = bin_edges[1] - bin_edges[0]
    ax.bar(
        bin_edges[:-1],
        counts,
        width=bin_width,
        align="edge",
        edgecolor="black",
        alpha=0.7,
    )
    ax.axhline(y=1.0 / len(counts), color="r", linestyle="--", label="Uniform")
    ax.set_xlabel("PIT Value")
    ax.set_ylabel("Relative Frequency")
    ax.set_title("PIT Histogram")
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
