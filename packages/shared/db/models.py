"""SQLAlchemy 2.0 declarative models for the Diamond Copilot database.

All 10 tables for Phase 0. Every table has:
- Appropriate primary key
- created_at / updated_at timestamps with server defaults
- Unique constraints for UPSERT idempotency
- Foreign keys where appropriate
"""

from datetime import date, datetime

from sqlalchemy import (
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# ---------------------------------------------------------------------------
# Players
# ---------------------------------------------------------------------------
class Player(Base):
    """MLB player master record. PK is the MLBAM player ID."""

    __tablename__ = "players"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    name_first: Mapped[str] = mapped_column(String(100))
    name_last: Mapped[str] = mapped_column(String(100))
    name_display: Mapped[str] = mapped_column(String(200))
    position_primary: Mapped[str | None] = mapped_column(String(10))
    bats: Mapped[str | None] = mapped_column(String(1))  # L, R, S
    throws: Mapped[str | None] = mapped_column(String(1))  # L, R
    team_abbr: Mapped[str | None] = mapped_column(String(5))
    active: Mapped[bool] = mapped_column(default=True)
    birth_date: Mapped[date | None] = mapped_column()
    mlb_debut_date: Mapped[date | None] = mapped_column()

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Games
# ---------------------------------------------------------------------------
class Game(Base):
    """MLB game record. PK is the MLB game_pk."""

    __tablename__ = "games"

    game_pk: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    game_date: Mapped[date] = mapped_column(index=True)
    home_team: Mapped[str] = mapped_column(String(5))
    away_team: Mapped[str] = mapped_column(String(5))
    venue_name: Mapped[str | None] = mapped_column(String(200))
    status: Mapped[str | None] = mapped_column(String(30))  # scheduled, final, postponed, etc.
    home_score: Mapped[int | None] = mapped_column(SmallInteger)
    away_score: Mapped[int | None] = mapped_column(SmallInteger)
    home_probable_pitcher_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("players.player_id")
    )
    away_probable_pitcher_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("players.player_id")
    )
    game_type: Mapped[str | None] = mapped_column(String(5))  # R, P, W, etc.
    double_header: Mapped[str | None] = mapped_column(String(1))  # Y, N

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Pitches (Statcast pitch-level)
# ---------------------------------------------------------------------------
class Pitch(Base):
    """Statcast pitch-level data. One row per pitch thrown."""

    __tablename__ = "pitches"
    __table_args__ = (
        UniqueConstraint("game_pk", "at_bat_number", "pitch_number", name="uq_pitch_event"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, ForeignKey("games.game_pk"), index=True)
    at_bat_number: Mapped[int] = mapped_column(SmallInteger)
    pitch_number: Mapped[int] = mapped_column(SmallInteger)
    batter_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), index=True)
    pitcher_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), index=True)
    pitch_type: Mapped[str | None] = mapped_column(String(10))
    release_speed: Mapped[float | None] = mapped_column(Float)
    release_spin_rate: Mapped[float | None] = mapped_column(Float)
    plate_x: Mapped[float | None] = mapped_column(Float)
    plate_z: Mapped[float | None] = mapped_column(Float)
    launch_speed: Mapped[float | None] = mapped_column(Float)  # exit velocity
    launch_angle: Mapped[float | None] = mapped_column(Float)
    hit_distance: Mapped[float | None] = mapped_column(Float)
    events: Mapped[str | None] = mapped_column(String(50))  # single, strikeout, etc.
    description: Mapped[str | None] = mapped_column(String(100))  # ball, called_strike, etc.
    zone: Mapped[int | None] = mapped_column(SmallInteger)
    stand: Mapped[str | None] = mapped_column(String(1))  # L, R
    p_throws: Mapped[str | None] = mapped_column(String(1))  # L, R
    inning: Mapped[int | None] = mapped_column(SmallInteger)
    outs_when_up: Mapped[int | None] = mapped_column(SmallInteger)
    balls: Mapped[int | None] = mapped_column(SmallInteger)
    strikes: Mapped[int | None] = mapped_column(SmallInteger)
    estimated_woba_using_speedangle: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Batting Stats Daily
# ---------------------------------------------------------------------------
class BattingStatsDaily(Base):
    """Per-player, per-game batting statistics. Aggregated from pitch-level data."""

    __tablename__ = "batting_stats_daily"
    __table_args__ = (
        UniqueConstraint("player_id", "game_pk", name="uq_batting_player_game"),
        Index("ix_batting_player_date", "player_id", "game_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), index=True)
    game_pk: Mapped[int] = mapped_column(Integer, ForeignKey("games.game_pk"), index=True)
    game_date: Mapped[date] = mapped_column(index=True)

    # Traditional counting stats
    pa: Mapped[int] = mapped_column(SmallInteger, default=0)
    ab: Mapped[int] = mapped_column(SmallInteger, default=0)
    h: Mapped[int] = mapped_column(SmallInteger, default=0)
    doubles: Mapped[int] = mapped_column(SmallInteger, default=0)
    triples: Mapped[int] = mapped_column(SmallInteger, default=0)
    hr: Mapped[int] = mapped_column(SmallInteger, default=0)
    rbi: Mapped[int] = mapped_column(SmallInteger, default=0)
    r: Mapped[int] = mapped_column(SmallInteger, default=0)
    bb: Mapped[int] = mapped_column(SmallInteger, default=0)
    so: Mapped[int] = mapped_column(SmallInteger, default=0)
    sb: Mapped[int] = mapped_column(SmallInteger, default=0)
    cs: Mapped[int] = mapped_column(SmallInteger, default=0)
    hbp: Mapped[int] = mapped_column(SmallInteger, default=0)
    sf: Mapped[int] = mapped_column(SmallInteger, default=0)

    # Statcast quality-of-contact
    exit_velocity_avg: Mapped[float | None] = mapped_column(Float)
    barrel_pct: Mapped[float | None] = mapped_column(Float)
    hard_hit_pct: Mapped[float | None] = mapped_column(Float)
    xwoba: Mapped[float | None] = mapped_column(Float)

    # Fantasy points (null until scoring rules applied)
    fantasy_points: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Pitching Stats Daily
# ---------------------------------------------------------------------------
class PitchingStatsDaily(Base):
    """Per-player, per-game pitching statistics."""

    __tablename__ = "pitching_stats_daily"
    __table_args__ = (
        UniqueConstraint("player_id", "game_pk", name="uq_pitching_player_game"),
        Index("ix_pitching_player_date", "player_id", "game_date"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), index=True)
    game_pk: Mapped[int] = mapped_column(Integer, ForeignKey("games.game_pk"), index=True)
    game_date: Mapped[date] = mapped_column(index=True)

    # Traditional pitching stats
    ip: Mapped[float] = mapped_column(Float, default=0.0)  # innings pitched (e.g. 6.2)
    h: Mapped[int] = mapped_column(SmallInteger, default=0)
    er: Mapped[int] = mapped_column(SmallInteger, default=0)
    r: Mapped[int] = mapped_column(SmallInteger, default=0)
    bb: Mapped[int] = mapped_column(SmallInteger, default=0)
    so: Mapped[int] = mapped_column(SmallInteger, default=0)
    hr_allowed: Mapped[int] = mapped_column(SmallInteger, default=0)
    pitches_thrown: Mapped[int | None] = mapped_column(SmallInteger)
    batters_faced: Mapped[int | None] = mapped_column(SmallInteger)

    # Advanced / Statcast
    gb_pct: Mapped[float | None] = mapped_column(Float)
    fip: Mapped[float | None] = mapped_column(Float)
    xwoba_against: Mapped[float | None] = mapped_column(Float)

    # Fantasy points (null until scoring rules applied)
    fantasy_points: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# User Leagues (Yahoo)
# ---------------------------------------------------------------------------
class UserLeague(Base):
    """A Yahoo Fantasy Baseball league connected by a user."""

    __tablename__ = "user_leagues"
    __table_args__ = (UniqueConstraint("yahoo_league_key", name="uq_yahoo_league_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    yahoo_league_key: Mapped[str] = mapped_column(String(50))
    league_name: Mapped[str | None] = mapped_column(String(200))
    season_year: Mapped[int] = mapped_column(SmallInteger)
    num_teams: Mapped[int | None] = mapped_column(SmallInteger)
    scoring_type: Mapped[str | None] = mapped_column(String(50))  # head-to-head, roto, etc.
    current_week: Mapped[int | None] = mapped_column(SmallInteger)
    user_team_key: Mapped[str | None] = mapped_column(String(50))

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# User Rosters
# ---------------------------------------------------------------------------
class UserRoster(Base):
    """Snapshot of a user's roster on a given date."""

    __tablename__ = "user_rosters"
    __table_args__ = (
        UniqueConstraint("league_id", "player_id", "roster_date", name="uq_roster_slot"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(Integer, ForeignKey("user_leagues.id"), index=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), index=True)
    roster_date: Mapped[date] = mapped_column(index=True)
    roster_position: Mapped[str | None] = mapped_column(String(10))  # C, 1B, OF, BN, DL, etc.
    acquisition_type: Mapped[str | None] = mapped_column(String(20))  # draft, add, trade

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# League Scoring Rules
# ---------------------------------------------------------------------------
class LeagueScoringRule(Base):
    """Scoring rules for a Yahoo fantasy league. One row per stat category."""

    __tablename__ = "league_scoring_rules"
    __table_args__ = (
        UniqueConstraint("league_id", "stat_category", name="uq_scoring_rule"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    league_id: Mapped[int] = mapped_column(Integer, ForeignKey("user_leagues.id"), index=True)
    stat_category: Mapped[str] = mapped_column(String(50))  # HR, RBI, SB, W, K, ERA, etc.
    points_value: Mapped[float] = mapped_column(Float)
    is_negative: Mapped[bool] = mapped_column(default=False)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
class Prediction(Base):
    """Model predictions stored for backtesting and drift detection."""

    __tablename__ = "predictions"
    __table_args__ = (
        UniqueConstraint("player_id", "game_pk", "model_version", name="uq_prediction"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), index=True)
    game_pk: Mapped[int] = mapped_column(Integer, ForeignKey("games.game_pk"), index=True)
    model_version: Mapped[str] = mapped_column(String(50))
    prediction_date: Mapped[date] = mapped_column(index=True)
    predicted_mean: Mapped[float] = mapped_column(Float)
    predicted_p10: Mapped[float | None] = mapped_column(Float)
    predicted_p90: Mapped[float | None] = mapped_column(Float)
    actual_points: Mapped[float | None] = mapped_column(Float)  # backfilled after game

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# News Articles
# ---------------------------------------------------------------------------
class NewsArticle(Base):
    """Scraped player news for RAG pipeline (Phase 6)."""

    __tablename__ = "news_articles"
    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_news_source_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(50))  # espn, rotowire, etc.
    external_id: Mapped[str] = mapped_column(String(200))
    player_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("players.player_id"), index=True
    )
    headline: Mapped[str | None] = mapped_column(String(500))
    body: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column()
    url: Mapped[str | None] = mapped_column(String(500))

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
