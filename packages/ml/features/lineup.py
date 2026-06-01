"""F5: Lineup slot features (hitters only).

Features: batting_order_slot (1-9, integer), lineup_confirmed_flag.
Phase 1 uses projected lineup at training time (rolling-most-recent slot).
The confirmed lineup override is a Phase 2 inference-time enhancement.

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F5.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class LineupSlotBuilder:
    """Feature builder for batting order position (hitters only)."""

    name: str = "lineup_slot"
    requires_history_days: int = 14  # look back for recent lineup slot

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Compute lineup slot features for hitters.

        Parameters
        ----------
        stats : pd.DataFrame
            Batting stats with batting_order_slot column from batting_stats_daily.
        as_of_date : date
            Latest date for feature computation.

        Returns
        -------
        pd.DataFrame
            Feature columns: lineup_batting_order_slot, lineup_confirmed_flag.
        """
        raise NotImplementedError("To be implemented by feature-engineer subagent")
