"""Baseline models for the Phase 1 projection evaluation.

Three baselines per validation_protocol.md section 8:
1. Naive last-game: predict tonight = last game's actual points
2. Trailing-7-day mean: mean per-game points over last 7 calendar days
3. Season-to-date mean: mean per-game points in current season up to (not including) game

Owner: evaluator subagent.
"""

from __future__ import annotations

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
        Historical stats with fantasy_points computed. Must contain the columns
        specified by player_id_col, date_col, and target_col.
    player_id_col, date_col, target_col : str
        Column names.

    Returns
    -------
    pd.Series
        Predictions indexed same as stats. First game per player is NaN.
    """
    df = stats.sort_values([player_id_col, date_col]).copy()
    # Shift within each player group: the previous row's target becomes the prediction
    preds = df.groupby(player_id_col)[target_col].shift(1)
    # Re-index to match original input index
    return preds.reindex(stats.index)


def predict_trailing_7d_mean(
    stats: pd.DataFrame,
    player_id_col: str = "player_id",
    date_col: str = "game_date",
    target_col: str = "fantasy_points",
) -> pd.Series:
    """Predict = mean per-game fantasy points over last 7 calendar days.

    For each row, computes the mean of target_col for that player over
    games in the 7 calendar days strictly before game_date. Requires >= 1
    game in the window. NaN if no games in window.

    Returns
    -------
    pd.Series
        Predictions indexed same as stats.
    """
    df = stats[[player_id_col, date_col, target_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([player_id_col, date_col])

    result = pd.Series(index=stats.index, dtype=float)

    for _player_id, group in df.groupby(player_id_col):
        dates = group[date_col].values
        targets = group[target_col].values
        preds = []
        for i in range(len(group)):
            current_date = dates[i]
            # Look back 7 calendar days strictly before current_date
            window_start = current_date - pd.Timedelta(days=7)
            mask = (dates[:i] > window_start) & (dates[:i] < current_date)
            if mask.any():
                preds.append(float(targets[:i][mask].mean()))
            else:
                preds.append(float("nan"))
        result.loc[group.index] = preds

    return result


def predict_season_to_date_mean(
    stats: pd.DataFrame,
    player_id_col: str = "player_id",
    date_col: str = "game_date",
    target_col: str = "fantasy_points",
    season_col: str = "season_year",
) -> pd.Series:
    """Predict = mean per-game fantasy points in current season up to (not including) game.

    For each row, computes the mean of target_col for that player in the
    same season, using only games strictly before the current game_date.
    First game of a player's season returns NaN.

    Returns
    -------
    pd.Series
        Predictions indexed same as stats.
    """
    df = stats[[player_id_col, date_col, target_col, season_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([player_id_col, season_col, date_col])

    result = pd.Series(index=stats.index, dtype=float)

    for (_player_id, _season), group in df.groupby([player_id_col, season_col]):
        targets = group[target_col].values
        # Expanding mean of all previous games in the same season
        preds = []
        for i in range(len(group)):
            if i == 0:
                preds.append(float("nan"))
            else:
                preds.append(float(targets[:i].mean()))
        result.loc[group.index] = preds

    return result
