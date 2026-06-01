"""Fantasy points target computation for the ML pipeline.

Computes the fantasy_points target column on batting_stats_daily and
pitching_stats_daily DataFrames using the shared scoring calculator.

Owner: feature-engineer subagent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def compute_target_batting(
    batting_stats: pd.DataFrame,
    rules: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add fantasy_points column to batting stats DataFrame.

    Parameters
    ----------
    batting_stats : pd.DataFrame
        Raw batting stats with standard counting stat columns.
    rules : dict[str, float] | None
        Batting scoring rules. Uses defaults if None.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with fantasy_points column added/updated.
    """
    raise NotImplementedError("To be implemented by feature-engineer subagent")


def compute_target_pitching(
    pitching_stats: pd.DataFrame,
    rules: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add fantasy_points column to pitching stats DataFrame.

    Parameters
    ----------
    pitching_stats : pd.DataFrame
        Raw pitching stats.
    rules : dict[str, float] | None
        Pitching scoring rules. Uses defaults if None.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with fantasy_points column added/updated.
    """
    raise NotImplementedError("To be implemented by feature-engineer subagent")
