"""Unit tests for SQLAlchemy ORM models."""

from __future__ import annotations

from packages.shared.db.models import (
    Base,
    BattingStatsDaily,
    Game,
    LeagueScoringRule,
    ParkFactor,
    Pitch,
    PitchingStatsDaily,
    Player,
    Prediction,
    SeasonStatsBatting,
    SeasonStatsPitching,
    UserLeague,
    UserRoster,
)

EXPECTED_TABLES = {
    "players",
    "games",
    "pitches",
    "batting_stats_daily",
    "pitching_stats_daily",
    "park_factors",
    "season_stats_batting",
    "season_stats_pitching",
    "user_leagues",
    "user_rosters",
    "league_scoring_rules",
    "predictions",
}


class TestModelsImportable:
    """Verify all 12 model classes can be imported."""

    def test_all_models_importable(self) -> None:
        models = [
            Player,
            Game,
            Pitch,
            BattingStatsDaily,
            PitchingStatsDaily,
            ParkFactor,
            SeasonStatsBatting,
            SeasonStatsPitching,
            UserLeague,
            UserRoster,
            LeagueScoringRule,
            Prediction,
        ]
        assert len(models) == 12


class TestBaseMetadata:
    """Verify Base.metadata contains all expected tables."""

    def test_all_tables_registered(self) -> None:
        table_names = set(Base.metadata.tables.keys())
        assert EXPECTED_TABLES.issubset(table_names), (
            f"Missing tables: {EXPECTED_TABLES - table_names}"
        )

    def test_exactly_twelve_tables(self) -> None:
        assert len(Base.metadata.tables) == 12


class TestUniqueConstraints:
    """Verify unique constraints exist on expected columns."""

    def _get_unique_constraint_columns(self, table_name: str) -> list[tuple[str, ...]]:
        """Extract unique constraint column sets for a table."""
        from sqlalchemy import UniqueConstraint

        table = Base.metadata.tables[table_name]
        return [
            tuple(sorted(c.name for c in uc.columns))
            for uc in table.constraints
            if isinstance(uc, UniqueConstraint)
        ]

    def test_pitches_unique_constraint(self) -> None:
        constraints = self._get_unique_constraint_columns("pitches")
        expected = tuple(sorted(("game_pk", "at_bat_number", "pitch_number")))
        assert expected in constraints

    def test_batting_stats_unique_constraint(self) -> None:
        constraints = self._get_unique_constraint_columns("batting_stats_daily")
        expected = tuple(sorted(("player_id", "game_pk")))
        assert expected in constraints

    def test_pitching_stats_unique_constraint(self) -> None:
        constraints = self._get_unique_constraint_columns("pitching_stats_daily")
        expected = tuple(sorted(("player_id", "game_pk")))
        assert expected in constraints

    def test_user_leagues_unique_constraint(self) -> None:
        constraints = self._get_unique_constraint_columns("user_leagues")
        expected = ("yahoo_league_key",)
        assert expected in constraints

    def test_user_rosters_unique_constraint(self) -> None:
        constraints = self._get_unique_constraint_columns("user_rosters")
        expected = tuple(sorted(("league_id", "player_id", "roster_date")))
        assert expected in constraints

    def test_scoring_rules_unique_constraint(self) -> None:
        constraints = self._get_unique_constraint_columns("league_scoring_rules")
        expected = tuple(sorted(("league_id", "stat_category")))
        assert expected in constraints

    def test_predictions_unique_constraint(self) -> None:
        constraints = self._get_unique_constraint_columns("predictions")
        expected = tuple(sorted(("player_id", "game_pk", "model_version")))
        assert expected in constraints

    def test_park_factors_unique_constraint(self) -> None:
        constraints = self._get_unique_constraint_columns("park_factors")
        expected = tuple(sorted(("venue_id", "season_year")))
        assert expected in constraints

    def test_season_batting_unique_constraint(self) -> None:
        constraints = self._get_unique_constraint_columns("season_stats_batting")
        expected = tuple(sorted(("player_id", "season_year", "source")))
        assert expected in constraints

    def test_season_pitching_unique_constraint(self) -> None:
        constraints = self._get_unique_constraint_columns("season_stats_pitching")
        expected = tuple(sorted(("player_id", "season_year", "source")))
        assert expected in constraints
