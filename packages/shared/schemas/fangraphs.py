"""Pydantic v2 schemas for FanGraphs season-level stats."""

from pydantic import BaseModel


class SeasonStatsBattingSchema(BaseModel):
    """Prior-season batting stats from FanGraphs or similar sources."""

    player_id: int
    season_year: int
    source: str = "fangraphs"

    # Counting
    pa: int | None = None
    ab: int | None = None
    g: int | None = None

    # Rate stats
    avg: float | None = None
    obp: float | None = None
    slg: float | None = None
    woba: float | None = None
    wrc_plus: float | None = None
    iso: float | None = None
    babip: float | None = None
    bb_pct: float | None = None
    k_pct: float | None = None

    # Statcast / expected
    ev_avg: float | None = None
    barrel_pct: float | None = None
    hard_hit_pct: float | None = None
    xwoba: float | None = None
    xba: float | None = None
    sprint_speed: float | None = None

    # Baserunning
    sb: int | None = None
    cs: int | None = None

    # Value
    war: float | None = None


class SeasonStatsPitchingSchema(BaseModel):
    """Prior-season pitching stats from FanGraphs or similar sources."""

    player_id: int
    season_year: int
    source: str = "fangraphs"

    # Workload
    ip: float | None = None
    g: int | None = None
    gs: int | None = None
    batters_faced: int | None = None

    # Rate / advanced
    era: float | None = None
    fip: float | None = None
    xfip: float | None = None
    siera: float | None = None
    k_pct: float | None = None
    bb_pct: float | None = None
    k_bb_pct: float | None = None
    hr_per_9: float | None = None
    gb_pct: float | None = None
    whip: float | None = None

    # Stuff / Statcast
    avg_fastball_velo: float | None = None
    xwoba_against: float | None = None
    barrel_pct_against: float | None = None
    hard_hit_pct_against: float | None = None
    xera: float | None = None

    # Value
    war: float | None = None
