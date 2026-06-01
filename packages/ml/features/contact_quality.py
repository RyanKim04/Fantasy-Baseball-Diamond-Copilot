"""F9: Quality of contact features (P1).

Hitters: rolling-30-day barrel%, hard-hit% (>=95 mph EV), avg exit velocity,
avg launch angle, xwOBA.
Pitchers: rolling-30-day barrel%-allowed, hard-hit%-allowed, xwOBA-against.

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F9.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class ContactQualityBuilder:
    """Feature builder for Statcast quality-of-contact metrics."""

    name: str = "contact_quality"
    requires_history_days: int = 30

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
        pitches: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute quality of contact features from pitch-level data."""
        raise NotImplementedError("To be implemented by feature-engineer subagent")
