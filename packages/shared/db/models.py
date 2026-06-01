"""SQLAlchemy 2.0 declarative models for the Diamond Copilot database.

All 12 tables for Phase 0. Every table has:
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
    yahoo_player_key: Mapped[str | None] = mapped_column(String(50))

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
    venue_id: Mapped[int | None] = mapped_column(Integer)
    season_year: Mapped[int | None] = mapped_column(SmallInteger)
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
        Index("ix_pitch_pitcher_game", "pitcher_id", "game_pk"),
        Index("ix_pitch_batter_game", "batter_id", "game_pk"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    game_pk: Mapped[int] = mapped_column(Integer, ForeignKey("games.game_pk"), index=True)
    at_bat_number: Mapped[int] = mapped_column(SmallInteger)
    pitch_number: Mapped[int] = mapped_column(SmallInteger)
    batter_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), index=True)
    pitcher_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), index=True)
    game_date: Mapped[date] = mapped_column(index=True)
    pitch_type: Mapped[str | None] = mapped_column(String(10))
    release_speed: Mapped[float | None] = mapped_column(Float)
    release_spin_rate: Mapped[float | None] = mapped_column(Float)
    release_extension: Mapped[float | None] = mapped_column(Float)
    pfx_x: Mapped[float | None] = mapped_column(Float)
    pfx_z: Mapped[float | None] = mapped_column(Float)
    plate_x: Mapped[float | None] = mapped_column(Float)
    plate_z: Mapped[float | None] = mapped_column(Float)
    launch_speed: Mapped[float | None] = mapped_column(Float)  # exit velocity
    launch_angle: Mapped[float | None] = mapped_column(Float)
    hit_distance: Mapped[float | None] = mapped_column(Float)
    barrel: Mapped[int | None] = mapped_column(SmallInteger)
    events: Mapped[str | None] = mapped_column(String(50))  # single, strikeout, etc.
    description: Mapped[str | None] = mapped_column(String(100))  # ball, called_strike, etc.
    type: Mapped[str | None] = mapped_column(String(1))  # B, S, X
    zone: Mapped[int | None] = mapped_column(SmallInteger)
    stand: Mapped[str | None] = mapped_column(String(1))  # L, R
    p_throws: Mapped[str | None] = mapped_column(String(1))  # L, R
    inning: Mapped[int | None] = mapped_column(SmallInteger)
    inning_topbot: Mapped[str | None] = mapped_column(String(3))  # Top, Bot
    outs_when_up: Mapped[int | None] = mapped_column(SmallInteger)
    balls: Mapped[int | None] = mapped_column(SmallInteger)
    strikes: Mapped[int | None] = mapped_column(SmallInteger)
    on_1b: Mapped[int | None] = mapped_column(Integer)
    on_2b: Mapped[int | None] = mapped_column(Integer)
    on_3b: Mapped[int | None] = mapped_column(Integer)
    estimated_woba_using_speedangle: Mapped[float | None] = mapped_column(Float)
    estimated_ba_using_speedangle: Mapped[float | None] = mapped_column(Float)
    fielder_2: Mapped[int | None] = mapped_column(Integer)  # catcher player_id

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
    season_year: Mapped[int | None] = mapped_column(SmallInteger)
    team_abbr: Mapped[str | None] = mapped_column(String(5))
    is_home: Mapped[bool | None] = mapped_column()
    batting_order_slot: Mapped[int | None] = mapped_column(SmallInteger)
    opponent_pitcher_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("players.player_id")
    )

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
    gidp: Mapped[int] = mapped_column(SmallInteger, default=0)

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
    season_year: Mapped[int | None] = mapped_column(SmallInteger)
    team_abbr: Mapped[str | None] = mapped_column(String(5))
    is_home: Mapped[bool | None] = mapped_column()
    role_flag: Mapped[str | None] = mapped_column(String(2))  # SP, RP
    opponent_team: Mapped[str | None] = mapped_column(String(5))

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
    hbp: Mapped[int] = mapped_column(SmallInteger, default=0)
    wp: Mapped[int] = mapped_column(SmallInteger, default=0)

    # Fantasy-relevant outcomes
    wins: Mapped[int] = mapped_column(SmallInteger, default=0)
    losses: Mapped[int] = mapped_column(SmallInteger, default=0)
    saves: Mapped[int] = mapped_column(SmallInteger, default=0)
    holds: Mapped[int] = mapped_column(SmallInteger, default=0)
    blown_saves: Mapped[int] = mapped_column(SmallInteger, default=0)
    quality_starts: Mapped[int] = mapped_column(SmallInteger, default=0)

    # Advanced / Statcast
    gb_pct: Mapped[float | None] = mapped_column(Float)
    fip: Mapped[float | None] = mapped_column(Float)
    xwoba_against: Mapped[float | None] = mapped_column(Float)

    # Fantasy points (null until scoring rules applied)
    fantasy_points: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Park Factors
# ---------------------------------------------------------------------------
class ParkFactor(Base):
    """Ballpark factor adjustments by venue and season."""

    __tablename__ = "park_factors"
    __table_args__ = (UniqueConstraint("venue_id", "season_year", name="uq_park_venue_season"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    venue_id: Mapped[int] = mapped_column(Integer, nullable=False)
    venue_name: Mapped[str] = mapped_column(String(200), nullable=False)
    season_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    park_factor_runs: Mapped[float | None] = mapped_column(Float)
    park_factor_hr: Mapped[float | None] = mapped_column(Float)
    park_factor_h: Mapped[float | None] = mapped_column(Float)
    park_factor_2b: Mapped[float | None] = mapped_column(Float)
    park_factor_3b: Mapped[float | None] = mapped_column(Float)
    park_factor_bb: Mapped[float | None] = mapped_column(Float)
    park_factor_so: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Season Stats Batting (FanGraphs / prior-season)
# ---------------------------------------------------------------------------
class SeasonStatsBatting(Base):
    """Prior-season batting stats from FanGraphs or similar sources."""

    __tablename__ = "season_stats_batting"
    __table_args__ = (
        UniqueConstraint("player_id", "season_year", "source", name="uq_season_batting"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), index=True)
    season_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    source: Mapped[str] = mapped_column(String(20), default="fangraphs")

    # Counting
    pa: Mapped[int | None] = mapped_column(Integer)
    ab: Mapped[int | None] = mapped_column(Integer)
    g: Mapped[int | None] = mapped_column(Integer)

    # Rate stats
    avg: Mapped[float | None] = mapped_column(Float)
    obp: Mapped[float | None] = mapped_column(Float)
    slg: Mapped[float | None] = mapped_column(Float)
    woba: Mapped[float | None] = mapped_column(Float)
    wrc_plus: Mapped[float | None] = mapped_column(Float)
    iso: Mapped[float | None] = mapped_column(Float)
    babip: Mapped[float | None] = mapped_column(Float)
    bb_pct: Mapped[float | None] = mapped_column(Float)
    k_pct: Mapped[float | None] = mapped_column(Float)

    # Statcast / expected
    ev_avg: Mapped[float | None] = mapped_column(Float)
    barrel_pct: Mapped[float | None] = mapped_column(Float)
    hard_hit_pct: Mapped[float | None] = mapped_column(Float)
    xwoba: Mapped[float | None] = mapped_column(Float)
    xba: Mapped[float | None] = mapped_column(Float)
    sprint_speed: Mapped[float | None] = mapped_column(Float)

    # Baserunning
    sb: Mapped[int | None] = mapped_column(Integer)
    cs: Mapped[int | None] = mapped_column(Integer)

    # Value
    war: Mapped[float | None] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(server_default=func.now(), onupdate=func.now())


# ---------------------------------------------------------------------------
# Season Stats Pitching (FanGraphs / prior-season)
# ---------------------------------------------------------------------------
class SeasonStatsPitching(Base):
    """Prior-season pitching stats from FanGraphs or similar sources."""

    __tablename__ = "season_stats_pitching"
    __table_args__ = (
        UniqueConstraint("player_id", "season_year", "source", name="uq_season_pitching"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(Integer, ForeignKey("players.player_id"), index=True)
    season_year: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    source: Mapped[str] = mapped_column(String(20), default="fangraphs")

    # Workload
    ip: Mapped[float | None] = mapped_column(Float)
    g: Mapped[int | None] = mapped_column(Integer)
    gs: Mapped[int | None] = mapped_column(Integer)
    batters_faced: Mapped[int | None] = mapped_column(Integer)

    # Rate / advanced
    era: Mapped[float | None] = mapped_column(Float)
    fip: Mapped[float | None] = mapped_column(Float)
    xfip: Mapped[float | None] = mapped_column(Float)
    siera: Mapped[float | None] = mapped_column(Float)
    k_pct: Mapped[float | None] = mapped_column(Float)
    bb_pct: Mapped[float | None] = mapped_column(Float)
    k_bb_pct: Mapped[float | None] = mapped_column(Float)
    hr_per_9: Mapped[float | None] = mapped_column(Float)
    gb_pct: Mapped[float | None] = mapped_column(Float)
    whip: Mapped[float | None] = mapped_column(Float)

    # Stuff / Statcast
    avg_fastball_velo: Mapped[float | None] = mapped_column(Float)
    xwoba_against: Mapped[float | None] = mapped_column(Float)
    barrel_pct_against: Mapped[float | None] = mapped_column(Float)
    hard_hit_pct_against: Mapped[float | None] = mapped_column(Float)
    xera: Mapped[float | None] = mapped_column(Float)

    # Value
    war: Mapped[float | None] = mapped_column(Float)

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
    __table_args__ = (UniqueConstraint("league_id", "stat_category", name="uq_scoring_rule"),)

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
