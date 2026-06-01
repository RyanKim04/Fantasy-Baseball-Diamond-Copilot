"""Airflow DAG: ingest MLB schedule, probable pitchers, and player metadata.

Uses the MLB Stats API (statsapi.mlb.com). Idempotent via UPSERT on game_pk / player_id.
Schedule: daily at 06:00 UTC (first DAG in the daily chain).
"""

from __future__ import annotations

import contextlib
import logging
from datetime import date, datetime, timedelta

from airflow.sdk import dag, task

logger = logging.getLogger(__name__)

# Mapping from MLB team full names to standard abbreviations
TEAM_NAME_TO_ABBR: dict[str, str] = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

default_args = {
    "owner": "diamond-copilot",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="ingest_mlb_schedule",
    schedule="0 6 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "mlb-schedule"],
    doc_md=__doc__,
)
def ingest_mlb_schedule(
    start_date: date | None = None,
    end_date: date | None = None,
) -> None:
    """Pull upcoming MLB schedule, probable pitchers, and player metadata.

    Default: next 7 days. Idempotent via UPSERT on natural keys.
    """

    @task(retries=3, retry_delay=timedelta(seconds=60))
    def fetch_schedule(start_date: date | None, end_date: date | None) -> list[dict]:
        """Fetch game schedule from MLB Stats API for the given date range.

        Returns a list of game dicts.
        """
        import statsapi

        if start_date is None:
            start_date = date.today()
        if end_date is None:
            end_date = start_date + timedelta(days=7)

        logger.info("Fetching MLB schedule from %s to %s", start_date, end_date)
        raw_games = statsapi.schedule(start_date=str(start_date), end_date=str(end_date))
        logger.info("Fetched %d games from MLB Stats API", len(raw_games))
        return raw_games

    @task
    def fetch_probable_lineups(games: list[dict]) -> list[int]:
        """Extract probable pitcher player IDs from the schedule response.

        Returns a list of unique player_ids for probable pitchers.
        """
        import statsapi

        player_ids: set[int] = set()

        for game in games:
            game_pk = game.get("game_id")
            if game_pk is None:
                continue

            # Try to get pitcher IDs from hydrated schedule endpoint
            try:
                resp = statsapi.get(
                    "schedule",
                    {"gamePk": game_pk, "hydrate": "probablePitcher"},
                )
                dates = resp.get("dates", [])
                for d in dates:
                    for g in d.get("games", []):
                        for side in ("away", "home"):
                            team_data = g.get("teams", {}).get(side, {})
                            pitcher = team_data.get("probablePitcher", {})
                            pid = pitcher.get("id")
                            if pid:
                                player_ids.add(pid)
            except Exception:
                logger.warning("Failed to fetch probable pitchers for game_pk=%s", game_pk)

        logger.info("Found %d unique probable pitcher IDs", len(player_ids))
        return list(player_ids)

    @task
    def upsert_games(games: list[dict]) -> int:
        """UPSERT game records into the games table.

        Returns the number of rows upserted.
        """
        from sqlalchemy.dialects.postgresql import insert

        from packages.shared.db.engine import get_engine
        from packages.shared.db.models import Game

        if not games:
            logger.info("No games to upsert")
            return 0

        engine = get_engine()
        rows = []
        for g in games:
            game_date_str = g.get("game_date", "")
            try:
                game_date_val = (
                    datetime.strptime(game_date_str, "%Y-%m-%d").date()
                    if isinstance(game_date_str, str)
                    else game_date_str
                )
            except (ValueError, TypeError):
                logger.warning("Skipping game with invalid date: %s", game_date_str)
                continue

            home_team_abbr = TEAM_NAME_TO_ABBR.get(g.get("home_name", ""), "UNK")
            away_team_abbr = TEAM_NAME_TO_ABBR.get(g.get("away_name", ""), "UNK")

            rows.append(
                {
                    "game_pk": g["game_id"],
                    "game_date": game_date_val,
                    "home_team": home_team_abbr,
                    "away_team": away_team_abbr,
                    "venue_name": g.get("venue_name"),
                    "venue_id": g.get("venue_id"),
                    "season_year": game_date_val.year if game_date_val else None,
                    "status": g.get("status"),
                    "home_score": g.get("home_score"),
                    "away_score": g.get("away_score"),
                    "game_type": g.get("game_type"),
                    "double_header": g.get("doubleheader"),
                }
            )

        if not rows:
            return 0

        stmt = insert(Game.__table__).values(rows)
        update_cols = {
            col.name: stmt.excluded[col.name]
            for col in Game.__table__.columns
            if col.name not in ("game_pk", "created_at")
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["game_pk"],
            set_=update_cols,
        )

        with engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

        logger.info("Upserted %d game records", len(rows))
        return len(rows)

    @task
    def upsert_players(player_ids: list[int]) -> int:
        """Fetch player metadata from MLB Stats API and UPSERT into the players table.

        Returns the number of rows upserted.
        """
        import statsapi
        from sqlalchemy.dialects.postgresql import insert

        from packages.shared.db.engine import get_engine
        from packages.shared.db.models import Player

        if not player_ids:
            logger.info("No player IDs to upsert")
            return 0

        # Fetch player metadata in batches
        ids_str = ",".join(str(pid) for pid in player_ids)
        logger.info("Fetching metadata for %d players", len(player_ids))

        try:
            resp = statsapi.get("people", {"personIds": ids_str})
        except Exception:
            logger.exception("Failed to fetch player metadata")
            return 0

        people = resp.get("people", [])
        if not people:
            logger.warning("No player metadata returned for IDs: %s", ids_str)
            return 0

        rows = []
        for person in people:
            birth_date_str = person.get("birthDate")
            birth_date_val = None
            if birth_date_str:
                with contextlib.suppress(ValueError, TypeError):
                    birth_date_val = datetime.strptime(birth_date_str, "%Y-%m-%d").date()

            debut_str = person.get("mlbDebutDate")
            debut_val = None
            if debut_str:
                with contextlib.suppress(ValueError, TypeError):
                    debut_val = datetime.strptime(debut_str, "%Y-%m-%d").date()

            full_name = person.get("fullName", "")
            first_name = person.get("firstName", "")
            last_name = person.get("lastName", "")
            position = person.get("primaryPosition", {}).get("abbreviation")
            bat_side = person.get("batSide", {}).get("code")
            pitch_hand = person.get("pitchHand", {}).get("code")
            team_abbr = person.get("currentTeam", {}).get("abbreviation")
            active = person.get("active", True)

            rows.append(
                {
                    "player_id": person["id"],
                    "name_first": first_name,
                    "name_last": last_name,
                    "name_display": full_name,
                    "position_primary": position,
                    "bats": bat_side,
                    "throws": pitch_hand,
                    "team_abbr": team_abbr,
                    "active": active,
                    "birth_date": birth_date_val,
                    "mlb_debut_date": debut_val,
                }
            )

        if not rows:
            return 0

        engine = get_engine()
        stmt = insert(Player.__table__).values(rows)
        update_cols = {
            col.name: stmt.excluded[col.name]
            for col in Player.__table__.columns
            if col.name not in ("player_id", "created_at")
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["player_id"],
            set_=update_cols,
        )

        with engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

        logger.info("Upserted %d player records", len(rows))
        return len(rows)

    # Task wiring
    games = fetch_schedule(start_date, end_date)
    upsert_games(games)
    pitcher_ids = fetch_probable_lineups(games)
    upsert_players(pitcher_ids)


ingest_mlb_schedule_dag = ingest_mlb_schedule()
