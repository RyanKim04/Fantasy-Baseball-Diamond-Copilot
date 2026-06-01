"""F8: Plate discipline features (P1).

Hitters: rolling-30-day BB%, K%, swing% on out-of-zone, contact% on in-zone, chase%.
Pitchers: rolling-30-day called-strike%, swinging-strike%, zone%, first-pitch-strike%.

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class PlateDisciplineBuilder:
    """Feature builder for plate discipline metrics."""

    name: str = "plate_discipline"
    requires_history_days: int = 30

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
        pitches: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute plate discipline features from pitch-level data."""
        raise NotImplementedError("To be implemented by feature-engineer subagent")
