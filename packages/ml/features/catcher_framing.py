"""F13: Catcher framing matchup features (P1, pitchers only).

Features: projected catcher tonight + their rolling-30-day CSAA (called strikes
above average). Pairs with F8's called-strike%.

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F13.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class CatcherFramingBuilder:
    """Feature builder for catcher framing matchup."""

    name: str = "catcher_framing"
    requires_history_days: int = 30

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
        pitches: pd.DataFrame | None = None,
        games: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute catcher framing features."""
        raise NotImplementedError("To be implemented by feature-engineer subagent")
