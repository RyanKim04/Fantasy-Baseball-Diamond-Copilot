"""Temporal data splitting and walk-forward CV fold generation.

All splits are strictly temporal per validation_protocol.md section 2.
No random shuffling, no random k-fold. This module is the single source
of truth for split boundaries.

Owner: modeler subagent.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from packages.shared.schemas.ml import SplitName, WalkForwardFold

# ---------------------------------------------------------------------------
# Split boundaries (frozen per validation_protocol.md section 2)
# ---------------------------------------------------------------------------
SPLIT_BOUNDARIES: dict[SplitName, tuple[date, date]] = {
    SplitName.TRAIN: (date(2019, 1, 1), date(2023, 12, 31)),
    SplitName.VALIDATION: (date(2024, 1, 1), date(2024, 12, 31)),
    SplitName.TEST: (date(2025, 1, 1), date(2025, 12, 31)),
    SplitName.LIVE: (date(2026, 1, 1), date(2099, 12, 31)),
}

# ---------------------------------------------------------------------------
# Walk-forward CV folds (validation_protocol.md section 3)
# ---------------------------------------------------------------------------
WALK_FORWARD_FOLDS: list[WalkForwardFold] = [
    WalkForwardFold(
        fold_number=1,
        train_start=date(2019, 1, 1),
        train_end=date(2019, 12, 31),
        val_start=date(2020, 1, 1),
        val_end=date(2020, 12, 31),
    ),
    WalkForwardFold(
        fold_number=2,
        train_start=date(2019, 1, 1),
        train_end=date(2020, 12, 31),
        val_start=date(2021, 1, 1),
        val_end=date(2021, 12, 31),
    ),
    WalkForwardFold(
        fold_number=3,
        train_start=date(2019, 1, 1),
        train_end=date(2021, 12, 31),
        val_start=date(2022, 1, 1),
        val_end=date(2022, 12, 31),
    ),
    WalkForwardFold(
        fold_number=4,
        train_start=date(2019, 1, 1),
        train_end=date(2022, 12, 31),
        val_start=date(2023, 1, 1),
        val_end=date(2023, 12, 31),
    ),
    WalkForwardFold(
        fold_number=5,
        train_start=date(2019, 1, 1),
        train_end=date(2023, 12, 31),
        val_start=date(2024, 1, 1),
        val_end=date(2024, 12, 31),
    ),
]

# CQR calibration boundary (validation_protocol.md section 4)
# Last 20% of Train by time, approximately 2023-04-01 onward.
# The exact cutoff is computed dynamically by compute_calibration_cutoff().
CQR_CALIBRATION_APPROXIMATE_START = date(2023, 4, 1)

# Headline CV metric uses folds 2-5 (fold 1 excluded because val=2020 COVID).
HEADLINE_FOLD_NUMBERS: list[int] = [2, 3, 4, 5]


def _normalize_date_column(df: pd.DataFrame, date_column: str) -> pd.Series:
    """Ensure the date column contains Python date objects for comparison.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    date_column : str
        Name of the date column.

    Returns
    -------
    pd.Series
        Series of datetime.date objects.

    Raises
    ------
    KeyError
        If date_column is not in df.
    """
    if date_column not in df.columns:
        msg = f"Column '{date_column}' not found in DataFrame. Available: {list(df.columns)}"
        raise KeyError(msg)

    col = df[date_column]

    # If already datetime.date objects, return as-is
    if len(col) > 0 and isinstance(col.iloc[0], date) and not isinstance(col.iloc[0], pd.Timestamp):
        return col

    # Convert pandas Timestamps / datetime64 to date objects
    return pd.to_datetime(col).dt.date


def split_by_date(
    df: pd.DataFrame,
    date_column: str = "game_date",
) -> dict[SplitName, pd.DataFrame]:
    """Split a DataFrame into Train/Val/Test/Live by game_date.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a date column.
    date_column : str
        Name of the date column to split on.

    Returns
    -------
    dict[SplitName, pd.DataFrame]
        One DataFrame per split. Rows are non-overlapping and exhaustive
        (every row that falls within a defined boundary is assigned to
        exactly one split; rows outside all boundaries are dropped).
    """
    dates = _normalize_date_column(df, date_column)

    result: dict[SplitName, pd.DataFrame] = {}
    for split_name, (start, end) in SPLIT_BOUNDARIES.items():
        mask = (dates >= start) & (dates <= end)
        result[split_name] = df.loc[mask].copy()

    return result


def get_walk_forward_folds(
    df: pd.DataFrame,
    date_column: str = "game_date",
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward CV folds from a DataFrame.

    Each fold uses an expanding training window. The training window for
    fold k includes all data from fold 1's train_start through fold k's
    train_end. The validation window is the next season.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (should cover Train + Validation periods).
    date_column : str
        Name of the date column.

    Returns
    -------
    list[tuple[pd.DataFrame, pd.DataFrame]]
        List of (train_fold, val_fold) tuples, one per fold definition.
        Folds are ordered by fold_number (1 through 5).
    """
    dates = _normalize_date_column(df, date_column)

    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    for fold in WALK_FORWARD_FOLDS:
        train_mask = (dates >= fold.train_start) & (dates <= fold.train_end)
        val_mask = (dates >= fold.val_start) & (dates <= fold.val_end)

        train_fold = df.loc[train_mask].copy()
        val_fold = df.loc[val_mask].copy()
        folds.append((train_fold, val_fold))

    return folds


def compute_calibration_cutoff(
    df: pd.DataFrame,
    date_column: str = "game_date",
    train_fraction: float = 0.80,
) -> date:
    """Compute the exact date at which train_fraction of Train rows have been observed.

    Used to define the CQR calibration set boundary. Rows on or before the
    returned date are used for fitting the quantile regressors; rows after
    are the calibration set for computing conformity scores.

    Parameters
    ----------
    df : pd.DataFrame
        Train-split data only (rows in 2019-01-01 to 2023-12-31).
    date_column : str
        Name of the date column.
    train_fraction : float
        Fraction of rows in the "fit" portion (default 0.80).

    Returns
    -------
    date
        The cutoff date. Rows on or before this date are "fit"; rows
        strictly after are "calibration".

    Raises
    ------
    ValueError
        If the DataFrame is empty or train_fraction is out of (0, 1).
    """
    if not 0.0 < train_fraction < 1.0:
        msg = f"train_fraction must be in (0, 1), got {train_fraction}"
        raise ValueError(msg)

    if len(df) == 0:
        msg = "Cannot compute calibration cutoff on an empty DataFrame"
        raise ValueError(msg)

    dates = _normalize_date_column(df, date_column)

    # Sort dates and find the date at which train_fraction of rows have occurred
    sorted_dates = dates.sort_values().reset_index(drop=True)
    cutoff_index = int(len(sorted_dates) * train_fraction) - 1
    cutoff_index = max(0, min(cutoff_index, len(sorted_dates) - 1))

    return sorted_dates.iloc[cutoff_index]


def get_calibration_split(
    df_train: pd.DataFrame,
    date_column: str = "game_date",
    train_fraction: float = 0.80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the Train set into fit and calibration portions for CQR.

    Parameters
    ----------
    df_train : pd.DataFrame
        Train-split data (2019-01-01 to 2023-12-31).
    date_column : str
        Name of the date column.
    train_fraction : float
        Fraction of rows for the "fit" portion.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (fit_df, calibration_df). fit_df contains the first train_fraction
        of rows by time; calibration_df contains the rest.
    """
    cutoff = compute_calibration_cutoff(df_train, date_column, train_fraction)
    dates = _normalize_date_column(df_train, date_column)

    fit_mask = dates <= cutoff
    cal_mask = dates > cutoff

    return df_train.loc[fit_mask].copy(), df_train.loc[cal_mask].copy()
