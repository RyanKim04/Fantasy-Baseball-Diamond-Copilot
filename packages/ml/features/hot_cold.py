"""F11: Recent-form z-scores (hot/cold) features (P1).

Features: (rolling_7d_wOBA - rolling_60d_wOBA) / rolling_60d_std; same for K% and BB%.
Captures deviation from player baseline.

Derived from F1 rolling features.

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F11.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class HotColdZScoreBuilder:
    """Feature builder for hot/cold z-score signals."""

    name: str = "hot_cold"
    requires_history_days: int = 60

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Compute hot/cold z-score features."""
        raise NotImplementedError("To be implemented by feature-engineer subagent")
