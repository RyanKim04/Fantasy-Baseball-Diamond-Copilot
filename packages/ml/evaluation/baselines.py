"""Baseline models for the Phase 1 projection evaluation.

Three baselines per validation_protocol.md section 8:
1. Naive last-game: predict tonight = last game's actual points
2. Trailing-7-day mean: mean per-game points over last 7 calendar days
3. Season-to-date mean: mean per-game points in current season up to (not including) game

Owner: evaluator subagent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


def predict_naive_last_game(
    stats: pd.DataFrame,
    player_id_col: str = "player_id",
    date_col: str = "game_date",
    target_col: str = "fantasy_points",
) -> pd.Series:
    """Predict tonight = most recent game's actual fantasy points.

    Parameters
    ----------
    stats : pd.DataFrame
        Historical stats with fantasy_points computed. Must be sorted by date.
    player_id_col, date_col, target_col : str
        Column names.

    Returns
    -------
    pd.Series
        Predictions indexed same as stats. First game per player is NaN.
    """
    raise NotImplementedError("To be implemented by evaluator subagent")


def predict_trailing_7d_mean(
    stats: pd.DataFrame,
    player_id_col: str = "player_id",
    date_col: str = "game_date",
    target_col: str = "fantasy_points",
) -> pd.Series:
    """Predict = mean per-game fantasy points over last 7 calendar days.

    Requires >= 1 game in the window. NaN if no games in window.

    Returns
    -------
    pd.Series
        Predictions indexed same as stats.
    """
    raise NotImplementedError("To be implemented by evaluator subagent")


def predict_season_to_date_mean(
    stats: pd.DataFrame,
    player_id_col: str = "player_id",
    date_col: str = "game_date",
    target_col: str = "fantasy_points",
    season_col: str = "season_year",
) -> pd.Series:
    """Predict = mean per-game fantasy points in current season up to (not including) game.

    Returns
    -------
    pd.Series
        Predictions indexed same as stats. First game of season is NaN.
    """
    raise NotImplementedError("To be implemented by evaluator subagent")
