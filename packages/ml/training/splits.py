"""Temporal data splitting and walk-forward CV fold generation.

All splits are strictly temporal per validation_protocol.md section 2.
No random shuffling, no random k-fold. This module is the single source
of truth for split boundaries.

Owner: modeler subagent.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from packages.shared.schemas.ml import SplitName, WalkForwardFold

if TYPE_CHECKING:
    import pandas as pd

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
        One DataFrame per split. Rows are non-overlapping and exhaustive.
    """
    raise NotImplementedError("To be implemented by modeler subagent")


def get_walk_forward_folds(
    df: pd.DataFrame,
    date_column: str = "game_date",
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Generate walk-forward CV folds from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (Train + Validation).
    date_column : str
        Name of the date column.

    Returns
    -------
    list[tuple[pd.DataFrame, pd.DataFrame]]
        List of (train_fold, val_fold) tuples, one per fold definition.
    """
    raise NotImplementedError("To be implemented by modeler subagent")


def compute_calibration_cutoff(
    df: pd.DataFrame,
    date_column: str = "game_date",
    train_fraction: float = 0.80,
) -> date:
    """Compute the exact date at which 80% of Train rows have been observed.

    Used to define the CQR calibration set boundary.

    Parameters
    ----------
    df : pd.DataFrame
        Train-split data only.
    date_column : str
        Name of the date column.
    train_fraction : float
        Fraction of rows in the "fit" portion (default 0.80).

    Returns
    -------
    date
        The cutoff date. Rows on or before this date are "fit"; after are "calibration".
    """
    raise NotImplementedError("To be implemented by modeler subagent")
