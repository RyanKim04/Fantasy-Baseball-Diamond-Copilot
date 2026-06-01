"""F1: Rolling production window features.

Hitters: rolling AVG, OBP, SLG, wOBA, ISO, BABIP at windows {7, 14, 30} days.
Pitchers: rolling ERA-equivalent, K/9, BB/9, HR/9, GB%, opponent wOBA-against
at windows {14, 30, 60} days.

Each window also gets PA/BF count (sample-size signal).
Anti-leakage: window ends at game_date - 1 day.

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class RollingProductionBuilder:
    """Feature builder for rolling production windows."""

    name: str = "rolling_production"
    requires_history_days: int = 60  # max window for pitchers

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Compute rolling production features."""
        raise NotImplementedError("To be implemented by feature-engineer subagent")
