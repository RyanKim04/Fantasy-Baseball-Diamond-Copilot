"""F2: Days of rest / recency features.

Family: P0 | Source: MLB Stats API schedules / daily stats
Anti-leakage justification: All features are computed from game_date values
strictly before the current game. ``days_since_last_game`` uses only prior
game dates. ``games_in_last_N_days`` counts games in (game_date - N, game_date),
an open interval excluding the current game.

Features produced:
- rest_days_since_last_game: calendar days since the player's most recent game
- rest_games_in_last_7d: count of games played in the prior 7 calendar days
- rest_games_in_last_14d: count of games played in the prior 14 calendar days
- rest_back_to_back: binary flag, 1 if days_since_last_game == 1

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import date


class RestRecencyBuilder:
    """Feature builder for days-of-rest and recency signals.

    requires_history_days: 14 (enough for the 14-day game count).
    """

    name: str = "rest_recency"
    requires_history_days: int = 14

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Compute rest/recency features.

        Parameters
        ----------
        stats : pd.DataFrame
            Raw stats with at least (player_id, game_pk, game_date).
            game_date can be date or datetime-like.
        as_of_date : date
            Guard: rows after this date are excluded.

        Returns
        -------
        pd.DataFrame
            Keyed on (player_id, game_pk) with rest feature columns.
        """
        df = stats[["player_id", "game_pk", "game_date"]].copy()
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df = df[df["game_date"] <= as_of_date]

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "player_id", "game_pk",
                    "rest_days_since_last_game",
                    "rest_games_in_last_7d",
                    "rest_games_in_last_14d",
                    "rest_back_to_back",
                    "rest_missing",
                ]
            )

        df = df.sort_values(["player_id", "game_date", "game_pk"])
        df = df.drop_duplicates(subset=["player_id", "game_pk"])

        # Days since last game: shift(1) within player group
        df["_prev_game_date"] = df.groupby("player_id")["game_date"].shift(1)
        df["rest_days_since_last_game"] = df.apply(
            lambda row: (row["game_date"] - row["_prev_game_date"]).days
            if pd.notna(row["_prev_game_date"])
            else np.nan,
            axis=1,
        )

        # Games in last N days: for each row, count how many of that player's
        # prior games fall within (game_date - N, game_date) exclusive.
        # We use a merge-based approach for efficiency.
        results = []
        for _, group in df.groupby("player_id"):
            g = group.copy()
            dates = g["game_date"].values
            counts_7 = []
            counts_14 = []
            for i, d in enumerate(dates):
                # Count games strictly before this one within the window
                from datetime import timedelta

                cutoff_7 = d - timedelta(days=7)
                cutoff_14 = d - timedelta(days=14)
                prior_dates = dates[:i]  # all games before this one in sorted order
                c7 = int(np.sum([cutoff_7 < pd <= d for pd in prior_dates]))  # noqa: E741
                c14 = int(np.sum([cutoff_14 < pd <= d for pd in prior_dates]))  # noqa: E741
                counts_7.append(c7)
                counts_14.append(c14)
            g["rest_games_in_last_7d"] = counts_7
            g["rest_games_in_last_14d"] = counts_14
            results.append(g)

        df = pd.concat(results, ignore_index=True)

        df["rest_back_to_back"] = (df["rest_days_since_last_game"] == 1).astype(int)
        df["rest_missing"] = df["rest_days_since_last_game"].isna().astype(int)

        # Impute missing days_since_last_game with median (per global rule 5)
        median_val = df["rest_days_since_last_game"].median()
        if pd.isna(median_val):
            median_val = 1.0
        df["rest_days_since_last_game"] = df["rest_days_since_last_game"].fillna(
            median_val
        )

        feature_cols = [c for c in df.columns if c.startswith("rest_")]
        return df[["player_id", "game_pk"] + feature_cols].copy()
