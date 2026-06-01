"""Feature engineering module for player projection.

This module owns all feature computation. Every feature builder follows the
FeatureBuilder protocol: it takes a DataFrame of raw stats keyed on
(player_id, game_pk, game_date) and returns a DataFrame with the same index
plus new feature columns.

Anti-leakage contract: all features for game G must be computable using ONLY
information available before the first pitch of G. Rolling windows end at
game_date - 1 day strictly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


@runtime_checkable
class FeatureBuilder(Protocol):
    """Protocol that all feature builder classes must satisfy.

    Each builder:
    - Declares which history it needs (requires_history_days).
    - Takes raw stats and returns feature columns.
    - Never accesses game_date or later data in its computation.
    """

    @property
    def name(self) -> str:
        """Short identifier for this feature family (e.g., 'rolling_production')."""
        ...

    @property
    def requires_history_days(self) -> int:
        """Number of calendar days of history needed to compute features.

        The pipeline must load at least this many days before the earliest
        game_date in the batch.
        """
        ...

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Compute features for all player-game rows in stats.

        Parameters
        ----------
        stats : pd.DataFrame
            Raw stats with at least (player_id, game_pk, game_date) columns.
            May include historical rows needed for rolling computations.
        as_of_date : date
            The latest date for which features should be computed. Used as a
            guard: no feature may use data from after this date.

        Returns
        -------
        pd.DataFrame
            Same index as input (player_id, game_pk), with new feature columns.
            Column names are prefixed with the builder's name (e.g., 'rolling_7d_avg').
        """
        ...
