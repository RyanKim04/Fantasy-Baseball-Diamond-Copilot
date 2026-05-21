"""Pydantic v2 schemas for Statcast data validation."""

from datetime import date

from pydantic import BaseModel


class StatcastPitchRow(BaseModel):
    """A single pitch from the Statcast feed. Validated before DB insert."""

    game_pk: int
    at_bat_number: int
    pitch_number: int
    batter_id: int
    pitcher_id: int
    game_date: date
    pitch_type: str | None = None
    release_speed: float | None = None
    release_spin_rate: float | None = None
    plate_x: float | None = None
    plate_z: float | None = None
    launch_speed: float | None = None
    launch_angle: float | None = None
    hit_distance: float | None = None
    events: str | None = None
    description: str | None = None
    zone: int | None = None
    stand: str | None = None
    p_throws: str | None = None
    inning: int | None = None
    outs_when_up: int | None = None
    balls: int | None = None
    strikes: int | None = None
    estimated_woba_using_speedangle: float | None = None


class StatcastBattingDaily(BaseModel):
    """Daily batting stats aggregated from pitch-level data."""

    player_id: int
    game_pk: int
    game_date: date
    pa: int = 0
    ab: int = 0
    h: int = 0
    doubles: int = 0
    triples: int = 0
    hr: int = 0
    rbi: int = 0
    r: int = 0
    bb: int = 0
    so: int = 0
    sb: int = 0
    cs: int = 0
    hbp: int = 0
    sf: int = 0
    exit_velocity_avg: float | None = None
    barrel_pct: float | None = None
    hard_hit_pct: float | None = None
    xwoba: float | None = None
    fantasy_points: float | None = None


class StatcastPitchingDaily(BaseModel):
    """Daily pitching stats aggregated from pitch-level data."""

    player_id: int
    game_pk: int
    game_date: date
    ip: float = 0.0
    h: int = 0
    er: int = 0
    r: int = 0
    bb: int = 0
    so: int = 0
    hr_allowed: int = 0
    pitches_thrown: int | None = None
    batters_faced: int | None = None
    gb_pct: float | None = None
    fip: float | None = None
    xwoba_against: float | None = None
    fantasy_points: float | None = None
