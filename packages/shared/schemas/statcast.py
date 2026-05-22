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
    release_extension: float | None = None
    pfx_x: float | None = None
    pfx_z: float | None = None
    plate_x: float | None = None
    plate_z: float | None = None
    launch_speed: float | None = None
    launch_angle: float | None = None
    hit_distance: float | None = None
    barrel: int | None = None
    events: str | None = None
    description: str | None = None
    type: str | None = None  # B, S, X
    zone: int | None = None
    stand: str | None = None
    p_throws: str | None = None
    inning: int | None = None
    inning_topbot: str | None = None  # Top, Bot
    outs_when_up: int | None = None
    balls: int | None = None
    strikes: int | None = None
    on_1b: int | None = None
    on_2b: int | None = None
    on_3b: int | None = None
    estimated_woba_using_speedangle: float | None = None
    estimated_ba_using_speedangle: float | None = None
    fielder_2: int | None = None  # catcher player_id


class StatcastBattingDaily(BaseModel):
    """Daily batting stats aggregated from pitch-level data."""

    player_id: int
    game_pk: int
    game_date: date
    season_year: int
    team_abbr: str | None = None
    is_home: bool | None = None
    batting_order_slot: int | None = None
    opponent_pitcher_id: int | None = None
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
    gidp: int = 0
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
    season_year: int
    team_abbr: str | None = None
    is_home: bool | None = None
    role_flag: str | None = None  # SP, RP
    opponent_team: str | None = None
    ip: float = 0.0
    h: int = 0
    er: int = 0
    r: int = 0
    bb: int = 0
    so: int = 0
    hr_allowed: int = 0
    pitches_thrown: int | None = None
    batters_faced: int | None = None
    hbp: int = 0
    wp: int = 0
    wins: int = 0
    losses: int = 0
    saves: int = 0
    holds: int = 0
    blown_saves: int = 0
    quality_starts: int = 0
    gb_pct: float | None = None
    fip: float | None = None
    xwoba_against: float | None = None
    fantasy_points: float | None = None
