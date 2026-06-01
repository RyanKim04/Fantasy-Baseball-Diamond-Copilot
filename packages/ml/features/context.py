"""F3: Home/away + ballpark factor features.

Family: P0 | Source: MLB schedule + prior-season park factors
Anti-leakage justification: ``is_home`` is observable pre-game (it's the
schedule). Park factors use the **prior season's** values, never the
current season, so there is no information from game_date or later.

Features produced:
- ctx_is_home: 1 if the player's team is the home team, 0 otherwise
- ctx_park_factor_runs: prior-season park factor for runs
- ctx_park_factor_hr: prior-season park factor for home runs
- ctx_park_factor_missing: 1 if park factor data is unavailable

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import date


class ContextBuilder:
    """Feature builder for home/away and park factors.

    requires_history_days: 0 (uses only game metadata, not rolling history).
    """

    name: str = "context"
    requires_history_days: int = 0

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
        games: pd.DataFrame | None = None,
        park_factors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute context features (home/away, park factors).

        Parameters
        ----------
        stats : pd.DataFrame
            Raw stats with at least (player_id, game_pk, game_date).
            Should also have ``team_id`` to determine home/away.
        as_of_date : date
            Guard date.
        games : pd.DataFrame | None
            Game schedule with columns: game_pk, home_team_id, away_team_id,
            venue_id, season_year.
        park_factors : pd.DataFrame | None
            Park factors with columns: venue_id, season_year,
            park_factor_runs, park_factor_hr. Prior-season values are used.

        Returns
        -------
        pd.DataFrame
            Keyed on (player_id, game_pk) with context feature columns.
        """
        df = stats[["player_id", "game_pk", "game_date"]].copy()
        if "team_id" in stats.columns:
            df["team_id"] = stats["team_id"]
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df = df[df["game_date"] <= as_of_date]

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "player_id", "game_pk",
                    "ctx_is_home", "ctx_park_factor_runs",
                    "ctx_park_factor_hr", "ctx_park_factor_missing",
                ]
            )

        # Default: unknown home/away
        df["ctx_is_home"] = np.nan
        df["ctx_park_factor_runs"] = np.nan
        df["ctx_park_factor_hr"] = np.nan

        if games is not None and "team_id" in df.columns:
            games_info = games[["game_pk", "home_team_id", "away_team_id"]].copy()
            if "venue_id" in games.columns:
                games_info["venue_id"] = games["venue_id"]
            if "season_year" in games.columns:
                games_info["season_year"] = games["season_year"]

            df = df.merge(games_info, on="game_pk", how="left")

            # is_home: player's team_id matches home_team_id
            df["ctx_is_home"] = (
                df["team_id"] == df["home_team_id"]
            ).astype(int)

            # Park factors: join on venue_id + prior season
            if (
                park_factors is not None
                and "venue_id" in df.columns
                and "season_year" in df.columns
            ):
                pf = park_factors.copy()
                # Use prior season: shift season_year +1 so that 2023 factors
                # apply to 2024 games
                pf["_join_season"] = pf["season_year"] + 1
                pf_rename = {}
                if "park_factor_runs" in pf.columns:
                    pf_rename["park_factor_runs"] = "_pf_runs"
                if "park_factor_hr" in pf.columns:
                    pf_rename["park_factor_hr"] = "_pf_hr"

                if pf_rename:
                    pf_sub = pf[
                        ["venue_id", "_join_season"] + list(pf_rename.keys())
                    ].rename(columns=pf_rename).drop_duplicates()

                    df = df.merge(
                        pf_sub,
                        left_on=["venue_id", "season_year"],
                        right_on=["venue_id", "_join_season"],
                        how="left",
                    )
                    if "_pf_runs" in df.columns:
                        df["ctx_park_factor_runs"] = df["_pf_runs"]
                    if "_pf_hr" in df.columns:
                        df["ctx_park_factor_hr"] = df["_pf_hr"]

        # Missingness indicator + imputation
        df["ctx_park_factor_missing"] = (
            df["ctx_park_factor_runs"].isna()
        ).astype(int)
        df["ctx_park_factor_runs"] = df["ctx_park_factor_runs"].fillna(1.0)
        df["ctx_park_factor_hr"] = df["ctx_park_factor_hr"].fillna(1.0)
        df["ctx_is_home"] = df["ctx_is_home"].fillna(0).astype(int)

        feature_cols = [c for c in df.columns if c.startswith("ctx_")]
        return df[["player_id", "game_pk"] + feature_cols].copy()
