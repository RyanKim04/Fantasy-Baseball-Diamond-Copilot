"""Unit tests for Pydantic validation schemas."""

from __future__ import annotations

from datetime import date

import pytest
from pydantic import ValidationError

from packages.shared.schemas.fangraphs import (
    SeasonStatsBattingSchema,
    SeasonStatsPitchingSchema,
)
from packages.shared.schemas.mlb_schedule import (
    BoxscoreBattingLine,
    BoxscorePitchingLine,
    LineupEntry,
    ScheduledGame,
)
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

    def test_new_fields_default_to_none(self) -> None:
        row = StatcastPitchRow(
            game_pk=717001,
            at_bat_number=1,
            pitch_number=1,
            batter_id=545361,
            pitcher_id=477132,
            game_date=date(2025, 6, 15),
        )
        assert row.barrel is None
        assert row.pfx_x is None
        assert row.fielder_2 is None
        assert row.inning_topbot is None


class TestStatcastBattingDaily:
    """Tests for StatcastBattingDaily schema."""

    def test_valid_batting_line(self) -> None:
        line = StatcastBattingDaily(
            player_id=545361,
            game_pk=717001,
            game_date=date(2025, 6, 15),
            season_year=2025,
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
            season_year=2025,
        )
        assert line.pa == 0
        assert line.gidp == 0
        assert line.fantasy_points is None


class TestStatcastPitchingDaily:
    """Tests for StatcastPitchingDaily schema."""

    def test_valid_pitching_line(self) -> None:
        line = StatcastPitchingDaily(
            player_id=477132,
            game_pk=717001,
            game_date=date(2025, 6, 15),
            season_year=2025,
            ip=7.0,
            so=10,
            er=2,
        )
        assert line.ip == 7.0

    def test_new_fields_default_to_zero(self) -> None:
        line = StatcastPitchingDaily(
            player_id=477132,
            game_pk=717001,
            game_date=date(2025, 6, 15),
            season_year=2025,
        )
        assert line.wins == 0
        assert line.saves == 0
        assert line.quality_starts == 0
        assert line.hbp == 0
        assert line.wp == 0


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

    def test_venue_id_optional(self) -> None:
        game = ScheduledGame(
            game_pk=717001,
            game_date=date(2025, 6, 15),
            home_team="LAA",
            away_team="NYY",
            venue_id=1,
        )
        assert game.venue_id == 1


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


class TestBoxscoreBattingLine:
    """Tests for BoxscoreBattingLine schema."""

    def test_valid_batting_line(self) -> None:
        line = BoxscoreBattingLine(
            player_id=545361,
            game_pk=717001,
            pa=4,
            ab=3,
            h=2,
            hr=1,
            rbi=3,
        )
        assert line.h == 2
        assert line.gidp == 0

    def test_defaults_to_zero(self) -> None:
        line = BoxscoreBattingLine(player_id=545361, game_pk=717001)
        assert line.pa == 0
        assert line.ab == 0
        assert line.hr == 0

    def test_missing_required_rejects(self) -> None:
        with pytest.raises(ValidationError):
            BoxscoreBattingLine(player_id=545361)  # type: ignore[call-arg]


class TestBoxscorePitchingLine:
    """Tests for BoxscorePitchingLine schema."""

    def test_valid_pitching_line(self) -> None:
        line = BoxscorePitchingLine(
            player_id=477132,
            game_pk=717001,
            ip=7.0,
            so=10,
            er=2,
            wins=1,
        )
        assert line.ip == 7.0
        assert line.wins == 1

    def test_defaults_to_zero(self) -> None:
        line = BoxscorePitchingLine(player_id=477132, game_pk=717001)
        assert line.ip == 0.0
        assert line.saves == 0
        assert line.quality_starts == 0

    def test_missing_required_rejects(self) -> None:
        with pytest.raises(ValidationError):
            BoxscorePitchingLine(player_id=477132)  # type: ignore[call-arg]


class TestSeasonStatsBattingSchema:
    """Tests for SeasonStatsBattingSchema."""

    def test_valid_season_batting(self) -> None:
        stats = SeasonStatsBattingSchema(
            player_id=545361,
            season_year=2024,
            avg=0.285,
            obp=0.370,
            slg=0.520,
            woba=0.380,
            war=5.2,
        )
        assert stats.source == "fangraphs"
        assert stats.war == 5.2

    def test_minimal_required_fields(self) -> None:
        stats = SeasonStatsBattingSchema(
            player_id=545361,
            season_year=2024,
        )
        assert stats.pa is None
        assert stats.xwoba is None

    def test_missing_required_rejects(self) -> None:
        with pytest.raises(ValidationError):
            SeasonStatsBattingSchema(player_id=545361)  # type: ignore[call-arg]


class TestSeasonStatsPitchingSchema:
    """Tests for SeasonStatsPitchingSchema."""

    def test_valid_season_pitching(self) -> None:
        stats = SeasonStatsPitchingSchema(
            player_id=477132,
            season_year=2024,
            era=2.89,
            fip=3.10,
            k_pct=0.30,
            whip=1.05,
            war=4.8,
        )
        assert stats.source == "fangraphs"
        assert stats.era == 2.89

    def test_minimal_required_fields(self) -> None:
        stats = SeasonStatsPitchingSchema(
            player_id=477132,
            season_year=2024,
        )
        assert stats.ip is None
        assert stats.xera is None

    def test_missing_required_rejects(self) -> None:
        with pytest.raises(ValidationError):
            SeasonStatsPitchingSchema(player_id=477132)  # type: ignore[call-arg]


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
