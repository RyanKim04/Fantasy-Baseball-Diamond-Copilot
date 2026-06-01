"""F7: Player identity baseline (prior-season aggregates).

Family: P0 | Source: Fangraphs / MLB end-of-season tables (prior year only)
Anti-leakage justification: For 2024 games, we use 2023 EOS aggregates.
For 2023 games, we use 2022 EOS aggregates. This is strictly the prior
completed season. No current-season data is used. Rookies with no prior
season get league-average values plus a missingness indicator.

Features produced for hitters:
- prior_pa, prior_woba, prior_iso, prior_bb_pct, prior_k_pct,
  prior_sprint_speed, prior_missing

Features produced for pitchers:
- prior_ip, prior_fip, prior_k_pct, prior_bb_pct, prior_gb_pct,
  prior_fastball_velo, prior_missing

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import date

# League-average defaults for imputation (approximate MLB averages)
HITTER_LEAGUE_AVG: dict[str, float] = {
    "prior_pa": 400.0,
    "prior_woba": 0.320,
    "prior_iso": 0.150,
    "prior_bb_pct": 0.085,
    "prior_k_pct": 0.220,
    "prior_sprint_speed": 27.0,
}

PITCHER_LEAGUE_AVG: dict[str, float] = {
    "prior_ip": 120.0,
    "prior_fip": 4.20,
    "prior_k_pct": 0.220,
    "prior_bb_pct": 0.080,
    "prior_gb_pct": 0.430,
    "prior_fastball_velo": 93.5,
}


class PriorSeasonBuilder:
    """Feature builder for prior-season player identity baseline.

    requires_history_days: 0 (uses prior-season aggregates, not rolling).
    """

    name: str = "prior_season"
    requires_history_days: int = 0

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
        season_batting: pd.DataFrame | None = None,
        season_pitching: pd.DataFrame | None = None,
        players: pd.DataFrame | None = None,
        player_type: str = "hitter",
    ) -> pd.DataFrame:
        """Compute prior-season identity features.

        Parameters
        ----------
        stats : pd.DataFrame
            Raw stats with (player_id, game_pk, game_date, season_year).
        as_of_date : date
            Latest date for feature computation.
        season_batting : pd.DataFrame | None
            Prior-season batting aggregates. Expected columns:
            player_id, season_year, pa, woba, iso, bb_pct, k_pct, sprint_speed.
        season_pitching : pd.DataFrame | None
            Prior-season pitching aggregates. Expected columns:
            player_id, season_year, ip, fip, k_pct, bb_pct, gb_pct,
            fastball_velo.
        players : pd.DataFrame | None
            Player metadata with mlb_debut_date for rookie detection.
        player_type : str
            "hitter" or "pitcher".

        Returns
        -------
        pd.DataFrame
            Feature columns prefixed with 'prior_'.
        """
        df = stats[["player_id", "game_pk", "game_date"]].copy()
        if "season_year" in stats.columns:
            df["season_year"] = stats["season_year"]
        else:
            df["game_date_dt"] = pd.to_datetime(df["game_date"])
            df["season_year"] = df["game_date_dt"].dt.year
            df = df.drop(columns=["game_date_dt"])

        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df = df[df["game_date"] <= as_of_date]

        if df.empty:
            if player_type == "hitter":
                cols = list(HITTER_LEAGUE_AVG.keys()) + ["prior_missing"]
            else:
                cols = list(PITCHER_LEAGUE_AVG.keys()) + ["prior_missing"]
            return pd.DataFrame(columns=["player_id", "game_pk"] + cols)

        # Compute prior season year for join
        df["_prior_season"] = df["season_year"] - 1

        if player_type == "hitter":
            return self._build_hitter(df, season_batting)
        else:
            return self._build_pitcher(df, season_pitching)

    def _build_hitter(
        self, df: pd.DataFrame, season_batting: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Join prior-season batting stats."""
        feature_map = {
            "pa": "prior_pa",
            "woba": "prior_woba",
            "iso": "prior_iso",
            "bb_pct": "prior_bb_pct",
            "k_pct": "prior_k_pct",
            "sprint_speed": "prior_sprint_speed",
        }

        for feat_col in feature_map.values():
            df[feat_col] = np.nan

        if season_batting is not None and not season_batting.empty:
            sb = season_batting.copy()
            rename_map = {}
            for src, dst in feature_map.items():
                if src in sb.columns:
                    rename_map[src] = dst

            sb_sub = sb[["player_id", "season_year"] + list(rename_map.keys())].copy()
            sb_sub = sb_sub.rename(columns=rename_map)

            df = df.merge(
                sb_sub,
                left_on=["player_id", "_prior_season"],
                right_on=["player_id", "season_year"],
                how="left",
                suffixes=("", "_ps"),
            )

            # Fill from merged columns
            for feat_col in feature_map.values():
                ps_col = f"{feat_col}_ps"
                if ps_col in df.columns:
                    df[feat_col] = df[ps_col].combine_first(df[feat_col])
                    df = df.drop(columns=[ps_col])

            # Drop extra season_year column from merge
            if "season_year_ps" in df.columns:
                df = df.drop(columns=["season_year_ps"])

        # Missingness indicator
        df["prior_missing"] = df["prior_woba"].isna().astype(int)

        # Impute with league averages
        for feat_col, default_val in HITTER_LEAGUE_AVG.items():
            df[feat_col] = df[feat_col].fillna(default_val)

        feature_cols = [c for c in df.columns if c.startswith("prior_")]
        return df[["player_id", "game_pk"] + feature_cols].copy()

    def _build_pitcher(
        self, df: pd.DataFrame, season_pitching: pd.DataFrame | None
    ) -> pd.DataFrame:
        """Join prior-season pitching stats."""
        feature_map = {
            "ip": "prior_ip",
            "fip": "prior_fip",
            "k_pct": "prior_k_pct",
            "bb_pct": "prior_bb_pct",
            "gb_pct": "prior_gb_pct",
            "fastball_velo": "prior_fastball_velo",
        }

        for feat_col in feature_map.values():
            df[feat_col] = np.nan

        if season_pitching is not None and not season_pitching.empty:
            sp = season_pitching.copy()
            rename_map = {}
            for src, dst in feature_map.items():
                if src in sp.columns:
                    rename_map[src] = dst

            sp_sub = sp[["player_id", "season_year"] + list(rename_map.keys())].copy()
            sp_sub = sp_sub.rename(columns=rename_map)

            df = df.merge(
                sp_sub,
                left_on=["player_id", "_prior_season"],
                right_on=["player_id", "season_year"],
                how="left",
                suffixes=("", "_ps"),
            )

            for feat_col in feature_map.values():
                ps_col = f"{feat_col}_ps"
                if ps_col in df.columns:
                    df[feat_col] = df[ps_col].combine_first(df[feat_col])
                    df = df.drop(columns=[ps_col])

            if "season_year_ps" in df.columns:
                df = df.drop(columns=["season_year_ps"])

        df["prior_missing"] = df["prior_fip"].isna().astype(int)

        for feat_col, default_val in PITCHER_LEAGUE_AVG.items():
            df[feat_col] = df[feat_col].fillna(default_val)

        feature_cols = [c for c in df.columns if c.startswith("prior_")]
        return df[["player_id", "game_pk"] + feature_cols].copy()
