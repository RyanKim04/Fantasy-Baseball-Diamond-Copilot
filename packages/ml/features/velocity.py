"""F12: Pitcher velocity and repertoire trend features (P1).

Features: trailing-5-start mean fastball velo, velocity delta vs. season baseline,
pitch-mix entropy (Shannon entropy over pitch_type proportions).

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F12.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class VelocityRepertoireBuilder:
    """Feature builder for pitcher velocity and repertoire trends."""

    name: str = "velocity_repertoire"
    requires_history_days: int = 45

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
        pitches: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute velocity and repertoire features from pitch-level data."""
        raise NotImplementedError("To be implemented by feature-engineer subagent")
