"""F6: Pitcher expected workload features (pitchers only).

Family: P0 | Source: rolling pitcher_id history
Anti-leakage justification: Rolling averages over the last 5 starts are
computed using only games strictly before the current game (shift by 1).
Role detection uses the same backward-looking approach.

Features produced:
- workload_mean_ip_5starts: rolling 5-start mean innings pitched
- workload_mean_pitches_5starts: rolling 5-start mean pitches thrown
- workload_mean_bf_5starts: rolling 5-start mean batters faced
- workload_is_starter: 1 if classified as SP based on recent IP pattern
- workload_missing: 1 if fewer than 2 prior starts available

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import date


class PitcherWorkloadBuilder:
    """Feature builder for pitcher workload expectations.

    requires_history_days: 45 (approximately 5 starts for a SP on 5-day rotation).
    """

    name: str = "pitcher_workload"
    requires_history_days: int = 45

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Compute pitcher workload features.

        Parameters
        ----------
        stats : pd.DataFrame
            Pitching stats with at least (player_id, game_pk, game_date, ip).
            Optional columns: pitches_thrown, batters_faced.
        as_of_date : date
            Latest date for feature computation.

        Returns
        -------
        pd.DataFrame
            Feature columns prefixed with 'workload_'.
        """
        cols_needed = ["player_id", "game_pk", "game_date"]
        df = stats[cols_needed].copy()
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df = df[df["game_date"] <= as_of_date]

        # Add stat columns
        for col in ["ip", "pitches_thrown", "batters_faced"]:
            if col in stats.columns:
                df[col] = stats.loc[df.index, col]
            else:
                df[col] = np.nan

        # Approximate missing columns
        if df["pitches_thrown"].isna().all() and not df["ip"].isna().all():
            # Rough approximation: ~15 pitches per inning
            df["pitches_thrown"] = df["ip"].fillna(0) * 15
        if df["batters_faced"].isna().all() and not df["ip"].isna().all():
            # Rough approximation: ~4.3 batters per inning
            df["batters_faced"] = (df["ip"].fillna(0) * 4.3).round()

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "player_id", "game_pk",
                    "workload_mean_ip_5starts",
                    "workload_mean_pitches_5starts",
                    "workload_mean_bf_5starts",
                    "workload_is_starter",
                    "workload_missing",
                ]
            )

        df = df.sort_values(["player_id", "game_date", "game_pk"])
        grouped = df.groupby("player_id")

        # Rolling 5-game mean for IP, pitches, BF -- then shift by 1
        df["workload_mean_ip_5starts"] = (
            grouped["ip"]
            .transform(lambda s: s.rolling(5, min_periods=1).mean())
            .groupby(df["player_id"])
            .shift(1)
        )
        df["workload_mean_pitches_5starts"] = (
            grouped["pitches_thrown"]
            .transform(lambda s: s.rolling(5, min_periods=1).mean())
            .groupby(df["player_id"])
            .shift(1)
        )
        df["workload_mean_bf_5starts"] = (
            grouped["batters_faced"]
            .transform(lambda s: s.rolling(5, min_periods=1).mean())
            .groupby(df["player_id"])
            .shift(1)
        )

        # Role flag: if recent mean IP >= 4.0, classify as starter
        df["workload_is_starter"] = (
            df["workload_mean_ip_5starts"].fillna(0) >= 4.0
        ).astype(int)

        # Missingness
        df["workload_missing"] = (
            df["workload_mean_ip_5starts"].isna()
        ).astype(int)

        # Impute missing with zeros
        for col in [
            "workload_mean_ip_5starts",
            "workload_mean_pitches_5starts",
            "workload_mean_bf_5starts",
        ]:
            df[col] = df[col].fillna(0.0)

        feature_cols = [c for c in df.columns if c.startswith("workload_")]
        return df[["player_id", "game_pk"] + feature_cols].copy()
