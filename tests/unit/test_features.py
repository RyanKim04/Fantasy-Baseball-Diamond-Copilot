"""Unit tests for packages/ml/features/.

Tests each feature builder for correctness, anti-leakage, and null handling.
Owner: feature-engineer subagent (implements tests alongside feature builders).
"""

from __future__ import annotations


# --- Target computation ---


class TestComputeTargetBatting:
    """Tests for features.scoring.compute_target_batting."""

    pass  # To be implemented by feature-engineer subagent


class TestComputeTargetPitching:
    """Tests for features.scoring.compute_target_pitching."""

    pass  # To be implemented by feature-engineer subagent


# --- P0 feature builder tests ---


class TestRollingProductionBuilder:
    """Tests for features.rolling.RollingProductionBuilder."""

    pass  # To be implemented by feature-engineer subagent


class TestRestRecencyBuilder:
    """Tests for features.rest.RestRecencyBuilder."""

    pass  # To be implemented by feature-engineer subagent


class TestContextBuilder:
    """Tests for features.context.ContextBuilder."""

    pass  # To be implemented by feature-engineer subagent


class TestOpponentQualityBuilder:
    """Tests for features.opponent.OpponentQualityBuilder."""

    pass  # To be implemented by feature-engineer subagent


class TestLineupSlotBuilder:
    """Tests for features.lineup.LineupSlotBuilder."""

    pass  # To be implemented by feature-engineer subagent


class TestPitcherWorkloadBuilder:
    """Tests for features.workload.PitcherWorkloadBuilder."""

    pass  # To be implemented by feature-engineer subagent


class TestPriorSeasonBuilder:
    """Tests for features.prior_season.PriorSeasonBuilder."""

    pass  # To be implemented by feature-engineer subagent


# --- Pipeline assembly ---


class TestAssembleFeatures:
    """Tests for features.pipeline.assemble_features."""

    pass  # To be implemented by feature-engineer subagent


# --- Leakage guard tests ---


class TestLeakageGuards:
    """Verify no feature uses game_date or later data.

    These tests construct a DataFrame with known future data and assert that
    feature builders produce the same output whether or not the future rows
    are present.
    """

    pass  # To be implemented by feature-engineer subagent
