"""Unit tests for Pydantic validation schemas."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from packages.shared.schemas.mlb_schedule import LineupEntry, ScheduledGame
from packages.shared.schemas.pipeline import PipelineResult
from packages.shared.schemas.statcast import (
    StatcastBattingDaily,
    StatcastPitchingDaily,
    StatcastPitchRow,
)
from packages.shared.schemas.yahoo import (
    YahooLeague,
    YahooMatchup,
    YahooRoster,
    YahooScoringRule,
)


class TestStatcastPitchRow:
    """Tests for StatcastPitchRow schema."""

    def test_valid_pitch(self) -> None:
        row = StatcastPitchRow(
            game_pk=717001,
            at_bat_number=1,
            pitch_number=1,
            batter_id=545361,
            pitcher_id=477132,
            game_date=date(2025, 6, 15),
            pitch_type="FF",
            release_speed=95.2,
        )
        assert row.game_pk == 717001

    def test_missing_required_field_rejects(self) -> None:
        with pytest.raises(ValidationError):
            StatcastPitchRow(  # type: ignore[call-arg]
                game_pk=717001,
                at_bat_number=1,
                # pitch_number missing
                batter_id=545361,
                pitcher_id=477132,
                game_date=date(2025, 6, 15),
            )


class TestStatcastBattingDaily:
    """Tests for StatcastBattingDaily schema."""

    def test_valid_batting_line(self) -> None:
        line = StatcastBattingDaily(
            player_id=545361,
            game_pk=717001,
            game_date=date(2025, 6, 15),
            pa=4,
            ab=3,
            h=2,
            hr=1,
            rbi=3,
        )
        assert line.h == 2

    def test_defaults_to_zero(self) -> None:
        line = StatcastBattingDaily(
            player_id=545361,
            game_pk=717001,
            game_date=date(2025, 6, 15),
        )
        assert line.pa == 0
        assert line.fantasy_points is None


class TestStatcastPitchingDaily:
    """Tests for StatcastPitchingDaily schema."""

    def test_valid_pitching_line(self) -> None:
        line = StatcastPitchingDaily(
            player_id=477132,
            game_pk=717001,
            game_date=date(2025, 6, 15),
            ip=7.0,
            so=10,
            er=2,
        )
        assert line.ip == 7.0


class TestScheduledGame:
    """Tests for ScheduledGame schema."""

    def test_valid_game(self) -> None:
        game = ScheduledGame(
            game_pk=717001,
            game_date=date(2025, 6, 15),
            home_team="LAA",
            away_team="NYY",
        )
        assert game.home_team == "LAA"


class TestLineupEntry:
    """Tests for LineupEntry schema."""

    def test_valid_entry(self) -> None:
        entry = LineupEntry(
            player_id=545361,
            game_pk=717001,
            team="LAA",
            batting_order=2,
        )
        assert entry.confirmed is False


class TestYahooSchemas:
    """Tests for Yahoo-related schemas."""

    def test_yahoo_league(self) -> None:
        league = YahooLeague(
            yahoo_league_key="422.l.12345",
            season_year=2025,
        )
        assert league.yahoo_league_key == "422.l.12345"

    def test_yahoo_roster(self) -> None:
        roster = YahooRoster(
            league_id=1,
            player_id=545361,
            roster_date=date(2025, 6, 15),
        )
        assert roster.player_id == 545361

    def test_yahoo_scoring_rule(self) -> None:
        rule = YahooScoringRule(
            league_id=1,
            stat_category="HR",
            points_value=4.0,
        )
        assert rule.is_negative is False

    def test_yahoo_matchup(self) -> None:
        matchup = YahooMatchup(
            league_key="422.l.12345",
            week=10,
            team_key="422.l.12345.t.1",
            opponent_team_key="422.l.12345.t.5",
        )
        assert matchup.is_complete is False


class TestPipelineResult:
    """Tests for the generic PipelineResult schema."""

    def test_success_result(self) -> None:
        result = PipelineResult[int](status="success", rows_upserted=100, data=100)
        assert result.status == "success"
        assert result.data == 100

    def test_failure_result(self) -> None:
        result = PipelineResult[None](
            status="failure",
            errors=["Connection refused"],
        )
        assert len(result.errors) == 1

    def test_invalid_status_rejects(self) -> None:
        with pytest.raises(ValidationError):
            PipelineResult[None](status="unknown")  # type: ignore[arg-type]
