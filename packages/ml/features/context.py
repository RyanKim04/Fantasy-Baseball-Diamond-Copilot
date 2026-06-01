"""F3: Home/away + ballpark factor features.

Features: is_home (binary), park_factor_runs, park_factor_HR, park_factor_BB.
Park factor uses prior-season value (no current-season leakage).

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


class ContextBuilder:
    """Feature builder for home/away and park factors."""

    name: str = "context"
    requires_history_days: int = 0

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
        games: pd.DataFrame | None = None,
        park_factors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Compute context features (home/away, park factors)."""
        raise NotImplementedError("To be implemented by feature-engineer subagent")
