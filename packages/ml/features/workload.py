"""F6: Pitcher expected workload features (pitchers only).

Features: rolling-5-start mean innings pitched, mean pitches thrown, mean batters
faced; role flag (SP / RP / opener).

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F6.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class PitcherWorkloadBuilder:
    """Feature builder for pitcher workload expectations."""

    name: str = "pitcher_workload"
    requires_history_days: int = 45  # ~5 starts for a SP

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Compute pitcher workload features.

        Parameters
        ----------
        stats : pd.DataFrame
            Pitching stats with ip, pitches_thrown, batters_faced, role_flag columns.
        as_of_date : date
            Latest date for feature computation.

        Returns
        -------
        pd.DataFrame
            Feature columns prefixed with 'workload_'.
        """
        raise NotImplementedError("To be implemented by feature-engineer subagent")
