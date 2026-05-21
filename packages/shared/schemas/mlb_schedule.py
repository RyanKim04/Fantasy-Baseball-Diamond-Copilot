"""Pydantic v2 schemas for MLB schedule and lineup data."""

from datetime import date

from pydantic import BaseModel


class ScheduledGame(BaseModel):
    """A scheduled MLB game from the MLB Stats API."""

    game_pk: int
    game_date: date
    home_team: str
    away_team: str
    venue_name: str | None = None
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
