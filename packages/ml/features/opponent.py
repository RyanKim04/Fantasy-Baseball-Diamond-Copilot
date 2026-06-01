"""F4: Opponent quality features.

Hitters facing pitcher Q tonight: Q's rolling-30-day FIP, K/9, BB/9, HR/9;
Q's career wOBA-against same-handed batters.
Pitchers facing lineup L tonight: L's rolling-30-day team wOBA, team K%, team ISO.

Anti-leakage: opponent stats are computed with window ending game_date - 1 day.
Opponent identity is observed (we know who tonight's SP is); their stats up
through yesterday are the feature.

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class OpponentQualityBuilder:
    """Feature builder for opponent quality signals."""

    name: str = "opponent_quality"
    requires_history_days: int = 30

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
        opponent_pitching: pd.DataFrame | None = None,
        opponent_batting: pd.DataFrame | None = None,
        games: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute opponent quality features.

        Parameters
        ----------
        stats : pd.DataFrame
            Raw stats with (player_id, game_pk, game_date) index columns.
        as_of_date : date
            Latest date for feature computation (window ends here - 1 day).
        opponent_pitching : pd.DataFrame | None
            Pitching stats for opponent SP lookup (for hitter features).
        opponent_batting : pd.DataFrame | None
            Batting stats for opponent lineup lookup (for pitcher features).
        games : pd.DataFrame | None
            Game schedule with probable pitcher assignments.

        Returns
        -------
        pd.DataFrame
            Feature columns prefixed with 'opp_'.
        """
        raise NotImplementedError("To be implemented by feature-engineer subagent")
