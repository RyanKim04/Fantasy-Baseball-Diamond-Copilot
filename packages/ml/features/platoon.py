"""F10: Platoon split features (P1).

Hitters: career and rolling-90-day wOBA vs. LHP, vs. RHP; tonight's opponent SP handedness.
Pitchers: wOBA-against by batter handedness (vs. L, vs. R); tonight's opponent
L/R lineup composition.

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F10.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class PlatoonSplitBuilder:
    """Feature builder for platoon (handedness) split metrics."""

    name: str = "platoon_splits"
    requires_history_days: int = 90

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
        pitches: pd.DataFrame | None = None,
        players: pd.DataFrame | None = None,
        games: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute platoon split features."""
        raise NotImplementedError("To be implemented by feature-engineer subagent")
