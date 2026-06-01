"""Unit tests for SQLAlchemy ORM models."""

from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

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


class TestCreateAllTables:
    """Verify all 12 tables can be created from scratch on a fresh engine."""

    def test_create_all_tables(self) -> None:
        """Base.metadata.create_all() should create all 12 tables without error."""
        engine = create_engine("sqlite:///:memory:")

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]  # noqa: ANN001
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")
            cursor.close()

        Base.metadata.create_all(engine)

        from sqlalchemy import inspect

        inspector = inspect(engine)
        table_names = set(inspector.get_table_names())
        assert EXPECTED_TABLES.issubset(table_names)


class TestInsertRows:
    """Verify that a row can be inserted into each table without raising."""

    def _make_session(self, engine: Engine) -> Session:
        return sessionmaker(bind=engine)()

    def test_insert_player(self, test_session: Session) -> None:
        player = Player(
            player_id=545361,
            name_first="Mike",
            name_last="Trout",
            name_display="Mike Trout",
        )
        test_session.add(player)
        test_session.flush()
        assert player.player_id == 545361

    def test_insert_game(self, test_session: Session) -> None:
        game = Game(
            game_pk=717001,
            game_date=date(2026, 5, 20),
            home_team="LAA",
            away_team="SEA",
        )
        test_session.add(game)
        test_session.flush()
        assert game.game_pk == 717001

    def test_insert_pitch(self, test_session: Session) -> None:
        # Need player and game first (FK constraints)
        player = Player(
            player_id=100001,
            name_first="A",
            name_last="B",
            name_display="A B",
        )
        game = Game(
            game_pk=800001,
            game_date=date(2026, 5, 20),
            home_team="NYY",
            away_team="BOS",
        )
        test_session.add_all([player, game])
        test_session.flush()

        pitch = Pitch(
            game_pk=800001,
            at_bat_number=1,
            pitch_number=1,
            batter_id=100001,
            pitcher_id=100001,
            game_date=date(2026, 5, 20),
        )
        test_session.add(pitch)
        test_session.flush()
        assert pitch.id is not None

    def test_insert_batting_stats_daily(self, test_session: Session) -> None:
        player = Player(player_id=100002, name_first="C", name_last="D", name_display="C D")
        game = Game(
            game_pk=800002,
            game_date=date(2026, 5, 20),
            home_team="ATL",
            away_team="NYM",
        )
        test_session.add_all([player, game])
        test_session.flush()

        row = BattingStatsDaily(
            player_id=100002,
            game_pk=800002,
            game_date=date(2026, 5, 20),
        )
        test_session.add(row)
        test_session.flush()
        assert row.id is not None

    def test_insert_pitching_stats_daily(self, test_session: Session) -> None:
        player = Player(player_id=100003, name_first="E", name_last="F", name_display="E F")
        game = Game(
            game_pk=800003,
            game_date=date(2026, 5, 20),
            home_team="SF",
            away_team="LAD",
        )
        test_session.add_all([player, game])
        test_session.flush()

        row = PitchingStatsDaily(
            player_id=100003,
            game_pk=800003,
            game_date=date(2026, 5, 20),
        )
        test_session.add(row)
        test_session.flush()
        assert row.id is not None

    def test_insert_park_factor(self, test_session: Session) -> None:
        row = ParkFactor(venue_id=1, season_year=2026, venue_name="Test Park")
        test_session.add(row)
        test_session.flush()
        assert row.id is not None

    def test_insert_season_stats_batting(self, test_session: Session) -> None:
        player = Player(player_id=100004, name_first="G", name_last="H", name_display="G H")
        test_session.add(player)
        test_session.flush()

        row = SeasonStatsBatting(player_id=100004, season_year=2025, source="fangraphs")
        test_session.add(row)
        test_session.flush()
        assert row.id is not None

    def test_insert_season_stats_pitching(self, test_session: Session) -> None:
        player = Player(player_id=100005, name_first="I", name_last="J", name_display="I J")
        test_session.add(player)
        test_session.flush()

        row = SeasonStatsPitching(player_id=100005, season_year=2025, source="fangraphs")
        test_session.add(row)
        test_session.flush()
        assert row.id is not None

    def test_insert_user_league(self, test_session: Session) -> None:
        row = UserLeague(
            yahoo_league_key="mlb.l.12345",
            league_name="Test League",
            season_year=2026,
            scoring_type="head_to_head",
        )
        test_session.add(row)
        test_session.flush()
        assert row.id is not None

    def test_insert_user_roster(self, test_session: Session) -> None:
        player = Player(player_id=100006, name_first="K", name_last="L", name_display="K L")
        league = UserLeague(
            yahoo_league_key="mlb.l.99999",
            league_name="Test2",
            season_year=2026,
            scoring_type="roto",
        )
        test_session.add_all([player, league])
        test_session.flush()

        row = UserRoster(
            league_id=league.id,
            player_id=100006,
            roster_date=date(2026, 5, 20),
        )
        test_session.add(row)
        test_session.flush()
        assert row.id is not None

    def test_insert_league_scoring_rule(self, test_session: Session) -> None:
        league = UserLeague(
            yahoo_league_key="mlb.l.77777",
            league_name="Test3",
            season_year=2026,
            scoring_type="points",
        )
        test_session.add(league)
        test_session.flush()

        row = LeagueScoringRule(
            league_id=league.id,
            stat_category="HR",
            points_value=4.0,
        )
        test_session.add(row)
        test_session.flush()
        assert row.id is not None

    def test_insert_prediction(self, test_session: Session) -> None:
        player = Player(player_id=100007, name_first="M", name_last="N", name_display="M N")
        game = Game(
            game_pk=800007,
            game_date=date(2026, 5, 20),
            home_team="CHC",
            away_team="STL",
        )
        test_session.add_all([player, game])
        test_session.flush()

        row = Prediction(
            player_id=100007,
            game_pk=800007,
            model_version="v0.1.0",
            prediction_date=date(2026, 5, 20),
            predicted_mean=12.5,
        )
        test_session.add(row)
        test_session.flush()
        assert row.id is not None


class TestUniqueConstraintEnforcement:
    """Verify that unique constraints actually reject duplicate inserts."""

    def test_players_duplicate_rejected(self, test_session: Session) -> None:
        """Inserting two players with the same player_id should raise IntegrityError."""
        p1 = Player(player_id=999001, name_first="A", name_last="A", name_display="A A")
        p2 = Player(player_id=999001, name_first="B", name_last="B", name_display="B B")
        test_session.add(p1)
        test_session.flush()
        test_session.add(p2)
        with pytest.raises(IntegrityError):
            test_session.flush()

    def test_games_duplicate_rejected(self, test_session: Session) -> None:
        """Inserting two games with the same game_pk should raise IntegrityError."""
        g1 = Game(
            game_pk=999001,
            game_date=date(2026, 5, 20),
            home_team="LAA",
            away_team="SEA",
        )
        g2 = Game(
            game_pk=999001,
            game_date=date(2026, 5, 21),
            home_team="NYY",
            away_team="BOS",
        )
        test_session.add(g1)
        test_session.flush()
        test_session.add(g2)
        with pytest.raises(IntegrityError):
            test_session.flush()
