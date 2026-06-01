"""F5: Lineup slot features (hitters only).

Family: P0 | Source: batting_stats_daily batting_order column
Anti-leakage justification: Uses the player's most recent batting order
position from games BEFORE the current game (shift by 1). This is a proxy
for the projected lineup. The confirmed-lineup override is Phase 2.

Features produced:
- lineup_batting_order_slot: most recent batting order position (1-9)
- lineup_slot_missing: 1 if no prior lineup data is available

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import date


class LineupSlotBuilder:
    """Feature builder for batting order position (hitters only).

    requires_history_days: 14 (look back for most recent lineup slot).
    """

    name: str = "lineup_slot"
    requires_history_days: int = 14

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Compute lineup slot features for hitters.

        Parameters
        ----------
        stats : pd.DataFrame
            Batting stats with at least (player_id, game_pk, game_date).
            Should have ``batting_order`` column (1-9) if available.
        as_of_date : date
            Latest date for feature computation.

        Returns
        -------
        pd.DataFrame
            Feature columns: lineup_batting_order_slot, lineup_slot_missing.
        """
        df = stats[["player_id", "game_pk", "game_date"]].copy()
        if "batting_order" in stats.columns:
            df["batting_order"] = stats["batting_order"]
        elif "batting_order_slot" in stats.columns:
            df["batting_order"] = stats["batting_order_slot"]
        else:
            df["batting_order"] = np.nan

        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df = df[df["game_date"] <= as_of_date]

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "player_id", "game_pk",
                    "lineup_batting_order_slot", "lineup_slot_missing",
                ]
            )

        df = df.sort_values(["player_id", "game_date", "game_pk"])

        # Use the most recent prior game's batting order (shift 1)
        df["lineup_batting_order_slot"] = (
            df.groupby("player_id")["batting_order"].shift(1)
        )

        df["lineup_slot_missing"] = (
            df["lineup_batting_order_slot"].isna()
        ).astype(int)

        # Impute missing with 5 (middle of lineup, neutral position)
        df["lineup_batting_order_slot"] = (
            df["lineup_batting_order_slot"].fillna(5.0)
        )

        return df[
            ["player_id", "game_pk", "lineup_batting_order_slot", "lineup_slot_missing"]
        ].copy()
