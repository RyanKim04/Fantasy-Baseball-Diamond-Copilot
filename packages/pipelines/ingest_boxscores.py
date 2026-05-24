"""Airflow DAG: ingest MLB boxscore data for completed games.

Fetches yesterday's completed games and extracts batting/pitching lines.
Populates batting_stats_daily and pitching_stats_daily with counting stats
that cannot be derived from Statcast pitch-level data alone (e.g., W, L, SV).
Schedule: daily at 07:00 UTC (after mlb_schedule at 06:00, before statcast at 08:00).
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime, timedelta

from airflow.sdk import dag, task

logger = logging.getLogger(__name__)

default_args = {
    "owner": "diamond-copilot",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _parse_innings_pitched(ip_str: str) -> float:
    """Convert MLB innings pitched string to float.

    MLB convention: "6.1" = 6⅓ innings, "6.2" = 6⅔ innings.
    """
    if not ip_str:
        return 0.0
    parts = str(ip_str).split(".")
    whole = int(parts[0])
    if len(parts) > 1:
        thirds = int(parts[1])
        return whole + thirds / 3
    return float(whole)


@dag(
    dag_id="ingest_boxscores",
    schedule="0 7 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "boxscores"],
    doc_md=__doc__,
)
def ingest_boxscores() -> None:
    """Fetch boxscore data for completed games and upsert batting/pitching stats."""

    @task(retries=3, retry_delay=timedelta(seconds=60))
    def fetch_completed_games() -> list[int]:
        """Query games table for recent completed games without boxscore data."""
        from sqlalchemy import select

        from packages.shared.db.engine import get_engine
        from packages.shared.db.models import BattingStatsDaily, Game

        engine = get_engine()
        cutoff_date = date.today() - timedelta(days=2)

        stmt = (
            select(Game.game_pk)
            .where(Game.status == "Final")
            .where(Game.game_date >= cutoff_date)
            .where(Game.game_pk.not_in(select(BattingStatsDaily.game_pk).distinct()))
        )

        with engine.connect() as conn:
            result = conn.execute(stmt)
            game_pks = [row[0] for row in result]

        logger.info("Found %d completed games needing boxscore data", len(game_pks))
        return game_pks

    @task
    def fetch_boxscore(game_pk: int) -> dict:
        """Fetch boxscore data from MLB Stats API for a single game."""
        import statsapi

        logger.info("Fetching boxscore for game_pk=%d", game_pk)
        box = statsapi.boxscore_data(game_pk)

        batting_lines: list[dict] = []
        pitching_lines: list[dict] = []

        for side in ("away", "home"):
            side_data = box.get(side, {})
            is_home = side == "home"

            # Get team abbreviation from the box data
            team_info = side_data.get("team", {})
            team_name = team_info.get("abbreviation", "")
            if not team_name:
                team_name = side_data.get("team", {}).get("name", "UNK")

            # Track pitcher order for SP/RP determination
            pitcher_order: list[int] = []

            players = side_data.get("players", {})
            for _player_key, player_data in players.items():
                person = player_data.get("person", {})
                player_id = person.get("id")
                if not player_id:
                    continue

                stats = player_data.get("stats", {})

                # --- Batting ---
                batting = stats.get("batting", {})
                pa = batting.get("plateAppearances", 0)
                if pa > 0:
                    batting_order_raw = player_data.get("battingOrder", "")
                    batting_order_slot = None
                    if batting_order_raw:
                        try:
                            slot = int(str(batting_order_raw)[:1])
                            if 1 <= slot <= 9:
                                batting_order_slot = slot
                        except (ValueError, IndexError):
                            pass

                    game_date_str = box.get("gameId", "")[:10]
                    try:
                        game_dt = datetime.strptime(  # noqa: DTZ007
                            game_date_str, "%Y/%m/%d"
                        ).date()
                    except (ValueError, AttributeError):
                        game_dt = date.today()

                    batting_lines.append(
                        {
                            "player_id": player_id,
                            "game_pk": game_pk,
                            "game_date": game_dt,
                            "season_year": game_dt.year,
                            "team_abbr": team_name,
                            "is_home": is_home,
                            "batting_order_slot": batting_order_slot,
                            "pa": pa,
                            "ab": batting.get("atBats", 0),
                            "h": batting.get("hits", 0),
                            "doubles": batting.get("doubles", 0),
                            "triples": batting.get("triples", 0),
                            "hr": batting.get("homeRuns", 0),
                            "rbi": batting.get("rbi", 0),
                            "r": batting.get("runs", 0),
                            "bb": batting.get("baseOnBalls", 0),
                            "so": batting.get("strikeOuts", 0),
                            "sb": batting.get("stolenBases", 0),
                            "cs": batting.get("caughtStealing", 0),
                            "hbp": batting.get("hitByPitch", 0),
                            "sf": batting.get("sacFlies", 0),
                            "gidp": batting.get("groundIntoDoublePlay", 0),
                        }
                    )

                # --- Pitching ---
                pitching = stats.get("pitching", {})
                ip_str = pitching.get("inningsPitched", "0")
                ip = _parse_innings_pitched(ip_str)
                if ip > 0 or pitching.get("battersFaced", 0) > 0:
                    pitcher_order.append(player_id)

                    game_date_str = box.get("gameId", "")[:10]
                    try:
                        game_dt = datetime.strptime(  # noqa: DTZ007
                            game_date_str, "%Y/%m/%d"
                        ).date()
                    except (ValueError, AttributeError):
                        game_dt = date.today()

                    # Parse decision from note field
                    note = pitching.get("note", "")
                    wins = 1 if "(W" in note else 0
                    losses = 1 if "(L" in note else 0
                    saves = 1 if "(S" in note or re.search(r"\bSV\b", note) else 0
                    holds = 1 if "(H" in note or re.search(r"\bHLD\b", note) else 0
                    blown_saves = 1 if "(BS" in note else 0

                    er = pitching.get("earnedRuns", 0)
                    quality_starts = 1 if ip >= 6.0 and er <= 3 else 0

                    # First pitcher is SP, rest are RP
                    role_flag = "SP" if len(pitcher_order) == 1 else "RP"

                    # Opponent team
                    opp_side = "home" if side == "away" else "away"
                    opp_info = box.get(opp_side, {}).get("team", {})
                    opponent_team = opp_info.get("abbreviation", "")

                    pitching_lines.append(
                        {
                            "player_id": player_id,
                            "game_pk": game_pk,
                            "game_date": game_dt,
                            "season_year": game_dt.year,
                            "team_abbr": team_name,
                            "is_home": is_home,
                            "role_flag": role_flag,
                            "opponent_team": opponent_team,
                            "ip": ip,
                            "h": pitching.get("hits", 0),
                            "er": er,
                            "r": pitching.get("runs", 0),
                            "bb": pitching.get("baseOnBalls", 0),
                            "so": pitching.get("strikeOuts", 0),
                            "hr_allowed": pitching.get("homeRuns", 0),
                            "pitches_thrown": pitching.get("numberOfPitches", 0),
                            "batters_faced": pitching.get("battersFaced", 0),
                            "hbp": pitching.get("hitBatsmen", 0),
                            "wp": pitching.get("wildPitches", 0),
                            "wins": wins,
                            "losses": losses,
                            "saves": saves,
                            "holds": holds,
                            "blown_saves": blown_saves,
                            "quality_starts": quality_starts,
                        }
                    )

        # Validate lines against Pydantic schemas
        from packages.shared.schemas.mlb_schedule import (
            BoxscoreBattingLine,
            BoxscorePitchingLine,
        )

        for line in batting_lines:
            try:
                BoxscoreBattingLine(**line)
            except Exception:
                logger.warning(
                    "Batting line failed Pydantic validation: player_id=%s, game_pk=%d",
                    line.get("player_id"),
                    game_pk,
                )

        for line in pitching_lines:
            try:
                BoxscorePitchingLine(**line)
            except Exception:
                logger.warning(
                    "Pitching line failed Pydantic validation: player_id=%s, game_pk=%d",
                    line.get("player_id"),
                    game_pk,
                )

        logger.info(
            "game_pk=%d: %d batting lines, %d pitching lines",
            game_pk,
            len(batting_lines),
            len(pitching_lines),
        )
        return {"batting": batting_lines, "pitching": pitching_lines}

    @task
    def upsert_batting_stats(boxscores: list[dict]) -> int:
        """UPSERT batting lines from boxscores into batting_stats_daily."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from packages.shared.db.engine import get_engine
        from packages.shared.db.models import BattingStatsDaily

        all_lines: list[dict] = []
        for box in boxscores:
            all_lines.extend(box.get("batting", []))

        if not all_lines:
            logger.info("No batting lines to upsert")
            return 0

        engine = get_engine()
        table = BattingStatsDaily.__table__
        update_cols = {c.name: c for c in table.c if c.name not in ("id", "created_at")}

        total = 0
        chunk_size = 500
        for i in range(0, len(all_lines), chunk_size):
            chunk = all_lines[i : i + chunk_size]
            stmt = pg_insert(table).values(chunk)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_batting_player_game",
                set_={k: stmt.excluded[k] for k in update_cols},
            )
            with engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()
            total += len(chunk)

        logger.info("Upserted %d batting stat lines", total)
        return total

    @task
    def upsert_pitching_stats(boxscores: list[dict]) -> int:
        """UPSERT pitching lines from boxscores into pitching_stats_daily."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from packages.shared.db.engine import get_engine
        from packages.shared.db.models import PitchingStatsDaily

        all_lines: list[dict] = []
        for box in boxscores:
            all_lines.extend(box.get("pitching", []))

        if not all_lines:
            logger.info("No pitching lines to upsert")
            return 0

        engine = get_engine()
        table = PitchingStatsDaily.__table__
        update_cols = {c.name: c for c in table.c if c.name not in ("id", "created_at")}

        total = 0
        chunk_size = 500
        for i in range(0, len(all_lines), chunk_size):
            chunk = all_lines[i : i + chunk_size]
            stmt = pg_insert(table).values(chunk)
            stmt = stmt.on_conflict_do_update(
                constraint="uq_pitching_player_game",
                set_={k: stmt.excluded[k] for k in update_cols},
            )
            with engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()
            total += len(chunk)

        logger.info("Upserted %d pitching stat lines", total)
        return total

    # Task wiring
    game_pks = fetch_completed_games()
    boxscores = fetch_boxscore.expand(game_pk=game_pks)
    upsert_batting_stats(boxscores)
    upsert_pitching_stats(boxscores)


ingest_boxscores_dag = ingest_boxscores()
