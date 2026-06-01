"""F2: Days of rest / recency features.

Features: days_since_last_game, games_in_last_7_days, back_to_back_flag (hitters),
days_since_last_start (SP), pitches_thrown_last_start (SP).

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F2.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class RestRecencyBuilder:
    """Feature builder for days-of-rest and recency signals."""

    name: str = "rest_recency"
    requires_history_days: int = 14

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Compute rest/recency features."""
        raise NotImplementedError("To be implemented by feature-engineer subagent")
