"""Fantasy points target computation for the ML pipeline.

Computes the fantasy_points target column on batting_stats_daily and
pitching_stats_daily DataFrames using the shared scoring calculator.

Owner: feature-engineer subagent.

This is the TARGET variable, not a feature. It is computed from the current
game's box score stats and therefore must NEVER be used as an input feature.
The only legitimate use is as the y column in training / evaluation.
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

    Uses packages.shared.scoring.compute_batting_fantasy_points as the
    single source of truth for the calculation. This function is a thin
    wrapper that adds the result as a column and returns the full DataFrame.

    Parameters
    ----------
    batting_stats : pd.DataFrame
        Raw batting stats with standard counting stat columns
        (h, doubles, triples, hr, r, rbi, bb, hbp, sb, cs, so, ...).
        Keyed on (player_id, game_pk).
    rules : dict[str, float] | None
        Batting scoring rules. Uses DEFAULT_BATTING_RULES if None.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with fantasy_points column added/updated.
    """
    from packages.shared.scoring import compute_batting_fantasy_points

    result = batting_stats.copy()
    result["fantasy_points"] = compute_batting_fantasy_points(result, rules=rules)
    return result


def compute_target_pitching(
    pitching_stats: pd.DataFrame,
    rules: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Add fantasy_points column to pitching stats DataFrame.

    Uses packages.shared.scoring.compute_pitching_fantasy_points as the
    single source of truth for the calculation.

    Parameters
    ----------
    pitching_stats : pd.DataFrame
        Raw pitching stats (ip, so, wins, losses, saves, holds, er, h, bb, ...).
        Keyed on (player_id, game_pk).
    rules : dict[str, float] | None
        Pitching scoring rules. Uses DEFAULT_PITCHING_RULES if None.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with fantasy_points column added/updated.
    """
    from packages.shared.scoring import compute_pitching_fantasy_points

    result = pitching_stats.copy()
    result["fantasy_points"] = compute_pitching_fantasy_points(result, rules=rules)
    return result
