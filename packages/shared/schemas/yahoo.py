"""Pydantic v2 schemas for Yahoo Fantasy Sports API data."""

from datetime import date

from pydantic import BaseModel


class YahooLeague(BaseModel):
    """Yahoo Fantasy Baseball league metadata."""

    yahoo_league_key: str
    league_name: str | None = None
    season_year: int
    num_teams: int | None = None
    scoring_type: str | None = None
    current_week: int | None = None
    user_team_key: str | None = None


class YahooRoster(BaseModel):
    """A player on a user's Yahoo fantasy roster."""

    league_id: int
    player_id: int
    roster_date: date
    roster_position: str | None = None
    acquisition_type: str | None = None


class YahooScoringRule(BaseModel):
    """A single scoring rule from a Yahoo league."""

    league_id: int
    stat_category: str
    points_value: float
    is_negative: bool = False


class YahooMatchup(BaseModel):
    """A head-to-head matchup in a Yahoo league."""

    league_key: str
    week: int
    team_key: str
    opponent_team_key: str
    team_points: float | None = None
    opponent_points: float | None = None
    is_complete: bool = False
