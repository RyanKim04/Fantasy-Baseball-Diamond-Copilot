"""Feature assembly pipeline.

Orchestrates all P0 feature builders to produce the final feature tables:
features_hitters.parquet and features_pitchers.parquet.

Output schema per row: (player_id, game_pk, game_date, season_year,
player_type, fantasy_points, <feature_columns>...).

Owner: feature-engineer subagent.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from datetime import date

    from packages.shared.schemas.ml import PlayerType

logger = logging.getLogger(__name__)


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

    Runs all P0 feature builders in sequence, then merges their outputs
    into two DataFrames keyed on (player_id, game_pk).

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
    from packages.shared.schemas.ml import PlayerType

    from .context import ContextBuilder
    from .lineup import LineupSlotBuilder
    from .opponent import OpponentQualityBuilder
    from .prior_season import PriorSeasonBuilder
    from .rest import RestRecencyBuilder
    from .rolling import RollingProductionBuilder
    from .scoring import compute_target_batting, compute_target_pitching
    from .workload import PitcherWorkloadBuilder

    # --- Step 1: Compute target variable ---
    logger.info("Computing target variables...")
    batting_with_target = compute_target_batting(batting_stats, rules=scoring_rules_batting)
    pitching_with_target = compute_target_pitching(pitching_stats, rules=scoring_rules_pitching)

    # --- Step 2: Build hitter features ---
    logger.info("Building hitter features...")
    hitter_features = _build_hitter_features(
        batting_with_target,
        games=games,
        park_factors=park_factors,
        season_batting=season_batting,
        pitching_stats=pitching_with_target,
        players=players,
        start_date=start_date,
        end_date=end_date,
    )

    # --- Step 3: Build pitcher features ---
    logger.info("Building pitcher features...")
    pitcher_features = _build_pitcher_features(
        pitching_with_target,
        games=games,
        park_factors=park_factors,
        season_pitching=season_pitching,
        batting_stats=batting_with_target,
        players=players,
        start_date=start_date,
        end_date=end_date,
    )

    # --- Step 4: Add metadata columns ---
    for df, ptype in [
        (hitter_features, "hitter"),
        (pitcher_features, "pitcher"),
    ]:
        df["player_type"] = ptype

    return {
        PlayerType.HITTER: hitter_features,
        PlayerType.PITCHER: pitcher_features,
    }


def _build_hitter_features(
    batting_stats: pd.DataFrame,
    games: pd.DataFrame,
    park_factors: pd.DataFrame,
    season_batting: pd.DataFrame,
    pitching_stats: pd.DataFrame,
    players: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Run all P0 builders for hitters and merge."""
    from .context import ContextBuilder
    from .lineup import LineupSlotBuilder
    from .opponent import OpponentQualityBuilder
    from .prior_season import PriorSeasonBuilder
    from .rest import RestRecencyBuilder
    from .rolling import RollingProductionBuilder

    as_of = end_date

    # Filter to date range for output, but keep full history for builders
    bat = batting_stats.copy()
    bat["game_date"] = pd.to_datetime(bat["game_date"]).dt.date

    # Base frame: rows within [start_date, end_date]
    mask = (bat["game_date"] >= start_date) & (bat["game_date"] <= end_date)
    base = bat.loc[mask, ["player_id", "game_pk", "game_date", "fantasy_points"]].copy()
    if "season_year" in bat.columns:
        base["season_year"] = bat.loc[mask, "season_year"]
    else:
        base["season_year"] = pd.to_datetime(base["game_date"]).apply(
            lambda d: d.year if hasattr(d, "year") else pd.Timestamp(d).year
        )

    if base.empty:
        return base

    key = ["player_id", "game_pk"]

    # F1: Rolling production
    rolling = RollingProductionBuilder()
    roll_df = rolling.build(bat, as_of_date=as_of, player_type="hitter")
    base = base.merge(roll_df, on=key, how="left")

    # F2: Rest/recency
    rest = RestRecencyBuilder()
    rest_df = rest.build(bat, as_of_date=as_of)
    base = base.merge(rest_df, on=key, how="left")

    # F3: Context (home/away + park factors)
    ctx = ContextBuilder()
    ctx_df = ctx.build(bat, as_of_date=as_of, games=games, park_factors=park_factors)
    base = base.merge(ctx_df, on=key, how="left")

    # F4: Opponent quality
    # Drop fantasy_points from opponent data to prevent target leakage (M12)
    opp_pitching_safe = pitching_stats.drop(
        columns=["fantasy_points"], errors="ignore"
    )
    opp = OpponentQualityBuilder()
    opp_df = opp.build(
        bat, as_of_date=as_of,
        opponent_pitching=opp_pitching_safe,
        games=games,
        player_type="hitter",
    )
    base = base.merge(opp_df, on=key, how="left")

    # F5: Lineup slot
    lineup = LineupSlotBuilder()
    lineup_df = lineup.build(bat, as_of_date=as_of)
    base = base.merge(lineup_df, on=key, how="left")

    # F7: Prior season
    prior = PriorSeasonBuilder()
    prior_df = prior.build(
        bat, as_of_date=as_of,
        season_batting=season_batting,
        player_type="hitter",
    )
    base = base.merge(prior_df, on=key, how="left")

    return base


def _build_pitcher_features(
    pitching_stats: pd.DataFrame,
    games: pd.DataFrame,
    park_factors: pd.DataFrame,
    season_pitching: pd.DataFrame,
    batting_stats: pd.DataFrame,
    players: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Run all P0 builders for pitchers and merge."""
    from .context import ContextBuilder
    from .opponent import OpponentQualityBuilder
    from .prior_season import PriorSeasonBuilder
    from .rest import RestRecencyBuilder
    from .rolling import RollingProductionBuilder
    from .workload import PitcherWorkloadBuilder

    as_of = end_date

    pitch = pitching_stats.copy()
    pitch["game_date"] = pd.to_datetime(pitch["game_date"]).dt.date

    mask = (pitch["game_date"] >= start_date) & (pitch["game_date"] <= end_date)
    base = pitch.loc[mask, ["player_id", "game_pk", "game_date", "fantasy_points"]].copy()
    if "season_year" in pitch.columns:
        base["season_year"] = pitch.loc[mask, "season_year"]
    else:
        base["season_year"] = pd.to_datetime(base["game_date"]).apply(
            lambda d: d.year if hasattr(d, "year") else pd.Timestamp(d).year
        )

    if base.empty:
        return base

    key = ["player_id", "game_pk"]

    # F1: Rolling production
    rolling = RollingProductionBuilder()
    roll_df = rolling.build(pitch, as_of_date=as_of, player_type="pitcher")
    base = base.merge(roll_df, on=key, how="left")

    # F2: Rest/recency
    rest = RestRecencyBuilder()
    rest_df = rest.build(pitch, as_of_date=as_of)
    base = base.merge(rest_df, on=key, how="left")

    # F3: Context
    ctx = ContextBuilder()
    ctx_df = ctx.build(pitch, as_of_date=as_of, games=games, park_factors=park_factors)
    base = base.merge(ctx_df, on=key, how="left")

    # F4: Opponent quality
    # Drop fantasy_points from opponent data to prevent target leakage (M12)
    opp_batting_safe = batting_stats.drop(
        columns=["fantasy_points"], errors="ignore"
    )
    opp = OpponentQualityBuilder()
    opp_df = opp.build(
        pitch, as_of_date=as_of,
        opponent_batting=opp_batting_safe,
        games=games,
        player_type="pitcher",
    )
    base = base.merge(opp_df, on=key, how="left")

    # F6: Pitcher workload
    workload = PitcherWorkloadBuilder()
    workload_df = workload.build(pitch, as_of_date=as_of)
    base = base.merge(workload_df, on=key, how="left")

    # F7: Prior season
    prior = PriorSeasonBuilder()
    prior_df = prior.build(
        pitch, as_of_date=as_of,
        season_pitching=season_pitching,
        player_type="pitcher",
    )
    base = base.merge(prior_df, on=key, how="left")

    return base
