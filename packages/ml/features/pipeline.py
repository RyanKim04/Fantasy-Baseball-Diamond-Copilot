"""Feature assembly pipeline.

Orchestrates all feature builders (P0 + surviving P1) to produce the final
feature tables: features_hitters.parquet and features_pitchers.parquet.

Owner: feature-engineer subagent.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd

    from packages.shared.schemas.ml import PlayerType


def assemble_features(
    batting_stats: pd.DataFrame,
    pitching_stats: pd.DataFrame,
    games: pd.DataFrame,
    park_factors: pd.DataFrame,
    season_batting: pd.DataFrame,
    season_pitching: pd.DataFrame,
    players: pd.DataFrame,
    scoring_rules_batting: dict[str, float],
    scoring_rules_pitching: dict[str, float],
    start_date: date,
    end_date: date,
) -> dict[PlayerType, pd.DataFrame]:
    """Build complete feature tables for hitters and pitchers.

    Parameters
    ----------
    batting_stats : pd.DataFrame
        Raw daily batting stats from batting_stats_daily table.
    pitching_stats : pd.DataFrame
        Raw daily pitching stats from pitching_stats_daily table.
    games : pd.DataFrame
        Game schedule from games table.
    park_factors : pd.DataFrame
        Ballpark factors from park_factors table.
    season_batting : pd.DataFrame
        Prior-season batting aggregates from season_stats_batting table.
    season_pitching : pd.DataFrame
        Prior-season pitching aggregates from season_stats_pitching table.
    players : pd.DataFrame
        Player metadata from players table.
    scoring_rules_batting : dict[str, float]
        Batting scoring rules for target computation.
    scoring_rules_pitching : dict[str, float]
        Pitching scoring rules for target computation.
    start_date : date
        First game_date to include in output features.
    end_date : date
        Last game_date to include in output features.

    Returns
    -------
    dict[PlayerType, pd.DataFrame]
        Keys are PlayerType.HITTER and PlayerType.PITCHER.
        Values are DataFrames keyed on (player_id, game_pk) with all feature
        columns plus the 'fantasy_points' target column.
    """
    raise NotImplementedError("To be implemented by feature-engineer subagent")
