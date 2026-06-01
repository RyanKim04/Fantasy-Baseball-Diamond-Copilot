"""Pydantic v2 schemas for MLB schedule, lineup, and boxscore data."""

from datetime import date

from pydantic import BaseModel


class ScheduledGame(BaseModel):
    """A scheduled MLB game from the MLB Stats API."""

    game_pk: int
    game_date: date
    home_team: str
    away_team: str
    venue_name: str | None = None
    venue_id: int | None = None
    status: str | None = None
    home_probable_pitcher_id: int | None = None
    away_probable_pitcher_id: int | None = None
    game_type: str | None = None
    double_header: str | None = None


class LineupEntry(BaseModel):
    """A single lineup slot (player + batting order position)."""

    player_id: int
    game_pk: int
    team: str
    batting_order: int | None = None
    position: str | None = None
    confirmed: bool = False


class BoxscoreBattingLine(BaseModel):
    """A single player's batting line from a boxscore."""

    player_id: int
    game_pk: int
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
    batting_order_slot: int | None = None


class BoxscorePitchingLine(BaseModel):
    """A single pitcher's line from a boxscore."""

    player_id: int
    game_pk: int
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
