"""F7: Player identity baseline (prior-season aggregates).

Hitters: prior-season PA, wOBA, ISO, BB%, K%, sprint speed.
Pitchers: prior-season IP, FIP, K%, BB%, GB%, avg fastball velo.

Anti-leakage: strictly prior-season. For 2024 games, use 2023 EOS aggregates.
Rookies get league-average values + missingness indicator.

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class PriorSeasonBuilder:
    """Feature builder for prior-season player identity baseline."""

    name: str = "prior_season"
    requires_history_days: int = 0  # uses prior-season aggregates, not rolling

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
        season_batting: pd.DataFrame | None = None,
        season_pitching: pd.DataFrame | None = None,
        players: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute prior-season identity features.

        Parameters
        ----------
        stats : pd.DataFrame
            Raw stats with (player_id, game_pk, game_date, season_year).
        as_of_date : date
            Latest date for feature computation.
        season_batting : pd.DataFrame | None
            Prior-season batting aggregates from season_stats_batting table.
        season_pitching : pd.DataFrame | None
            Prior-season pitching aggregates from season_stats_pitching table.
        players : pd.DataFrame | None
            Player metadata (for rookie detection via mlb_debut_date).

        Returns
        -------
        pd.DataFrame
            Feature columns prefixed with 'prior_'.
            Includes missingness indicators for rookies.
        """
        raise NotImplementedError("To be implemented by feature-engineer subagent")
