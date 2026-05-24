"""Airflow DAG: ingest Yahoo Fantasy Sports data (roster, scoring, matchups).

Requires a valid Yahoo OAuth session. See yahoo_auth.py for token management.
Schedule: daily at 09:00 UTC.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta
from typing import Any

from airflow.sdk import dag, task

logger = logging.getLogger(__name__)


def _extract_yahoo_player_id(yahoo_id: str) -> int | None:
    """Extract numeric player ID from Yahoo format 'mlb.p.545361'.

    Yahoo player IDs look like "mlb.p.545361" -- we extract the trailing
    numeric MLBAM ID. Falls back to parsing the raw value as an integer.
    Returns None if extraction fails.
    """
    parts = str(yahoo_id).split(".")
    if len(parts) >= 3:
        try:
            return int(parts[-1])
        except ValueError:
            return None
    try:
        return int(yahoo_id)
    except (ValueError, TypeError):
        return None


default_args = {
    "owner": "diamond-copilot",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="ingest_yahoo",
    schedule="0 9 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "yahoo"],
    doc_md=__doc__,
)
def ingest_yahoo(league_key: str | None = None) -> None:
    """Pull roster, scoring rules, and matchups from Yahoo Fantasy API.

    Idempotent via UPSERT on natural keys. Requires valid OAuth credentials.
    The league_key should be provided via Airflow params or environment variable.
    """

    @task
    def refresh_oauth_token() -> Any:
        """Ensure the Yahoo OAuth token is fresh. Refresh if expired."""
        from packages.pipelines.yahoo_auth import get_oauth_session

        logger.info("Refreshing Yahoo OAuth token...")
        oauth = get_oauth_session()
        logger.info("Yahoo OAuth session acquired successfully.")
        return oauth

    @task
    def fetch_league_info(session: Any, league_key: str) -> dict:
        """Fetch league metadata from Yahoo API."""
        from yahoo_fantasy_api import game  # type: ignore[import-untyped]

        gm = game.Game(session, "mlb")
        lg = gm.to_league(league_key)
        settings = lg.settings()

        # Get current week
        try:
            current_week = lg.current_week()
        except Exception:
            current_week = None

        # Find the user's team key
        user_team_key = None
        try:
            teams = lg.teams()
            if teams:
                # teams() returns a dict of {team_key: team_data}
                # The first team is typically the user's team
                user_team_key = next(iter(teams))
        except Exception:
            logger.warning("Could not determine user team key")

        info = {
            "yahoo_league_key": league_key,
            "league_name": settings.get("name", ""),
            "season_year": int(settings.get("season", datetime.now().year)),  # noqa: DTZ005
            "num_teams": int(settings.get("num_teams", 0)),
            "scoring_type": settings.get("scoring_type", ""),
            "current_week": current_week,
            "user_team_key": user_team_key,
        }
        logger.info("Fetched league info: %s (%s)", info["league_name"], league_key)
        return info

    @task
    def fetch_roster(session: Any, league_info: dict) -> list[dict]:
        """Fetch the current roster for the user's Yahoo fantasy team."""
        from yahoo_fantasy_api import game  # type: ignore[import-untyped]

        team_key = league_info.get("user_team_key")
        if not team_key:
            logger.warning("No user_team_key found in league_info; returning empty roster")
            return []

        league_key = league_info["yahoo_league_key"]
        gm = game.Game(session, "mlb")
        lg = gm.to_league(league_key)
        tm = lg.to_team(team_key)

        try:
            roster = tm.roster()
        except Exception:
            logger.exception("Failed to fetch roster for team %s", team_key)
            return []

        roster_data = []
        for player in roster:
            roster_data.append(
                {
                    "player_id": player.get("player_id", ""),
                    "player_name": player.get("name", ""),
                    "position_slot": player.get("selected_position", {}).get("position", ""),
                    "status": player.get("status", "active"),
                    "acquisition_type": player.get("acquisition_type", ""),
                }
            )

        logger.info("Fetched %d players in roster", len(roster_data))
        return roster_data

    @task
    def fetch_scoring_rules(session: Any, league_key: str) -> list[dict]:
        """Fetch scoring rules for a Yahoo league."""
        from yahoo_fantasy_api import game  # type: ignore[import-untyped]

        gm = game.Game(session, "mlb")
        lg = gm.to_league(league_key)
        settings = lg.settings()

        rules: list[dict] = []
        stat_categories = settings.get("stat_categories", {}).get("stats", [])

        for stat in stat_categories:
            stat_data = stat.get("stat", {})
            display_name = stat_data.get("display_name", "")
            # Point value may be in stat_modifiers
            point_value = float(stat_data.get("point_value", 0))

            rules.append(
                {
                    "stat_category": display_name,
                    "points_value": abs(point_value),
                    "is_negative": point_value < 0,
                }
            )

        # Also check stat_modifiers if available
        stat_modifiers = settings.get("stat_modifiers", {}).get("stats", [])
        modifier_map = {}
        for mod in stat_modifiers:
            mod_data = mod.get("stat", {})
            stat_id = mod_data.get("stat_id")
            value = float(mod_data.get("value", 0))
            if stat_id:
                modifier_map[stat_id] = value

        # Update rules with modifier values if they exist
        if modifier_map and rules:
            for i, stat in enumerate(stat_categories):
                stat_data = stat.get("stat", {})
                stat_id = stat_data.get("stat_id")
                if stat_id and stat_id in modifier_map:
                    val = modifier_map[stat_id]
                    rules[i]["points_value"] = abs(val)
                    rules[i]["is_negative"] = val < 0

        logger.info("Fetched %d scoring rules", len(rules))
        return rules

    @task
    def fetch_matchups(session: Any, league_key: str, week: int | None = None) -> list[dict]:
        """Fetch matchup data for the current (or specified) week."""
        from yahoo_fantasy_api import game  # type: ignore[import-untyped]

        gm = game.Game(session, "mlb")
        lg = gm.to_league(league_key)

        target_week = week
        if target_week is None:
            try:
                target_week = lg.current_week()
            except Exception:
                target_week = 1

        try:
            raw_matchups = lg.matchups(target_week)
        except Exception:
            logger.exception("Failed to fetch matchups for week %d", target_week)
            return []

        matchups: list[dict] = []
        for matchup in raw_matchups.get("fantasy_content", {}).get("matchups", []):
            if not isinstance(matchup, dict):
                continue
            teams = matchup.get("matchup", {}).get("teams", [])
            team_keys = []
            for team_entry in teams:
                if isinstance(team_entry, dict):
                    tk = team_entry.get("team", {}).get("team_key", "")
                    if tk:
                        team_keys.append(tk)
            if len(team_keys) >= 2:
                matchups.append(
                    {
                        "week": target_week,
                        "team_1_key": team_keys[0],
                        "team_2_key": team_keys[1],
                    }
                )

        logger.info("Fetched %d matchups for week %d", len(matchups), target_week)
        return matchups

    @task
    def upsert_yahoo_data(
        league_info: dict,
        roster: list[dict],
        scoring_rules: list[dict],
        matchups: list[dict],
    ) -> int:
        """UPSERT all Yahoo data into their respective tables."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from packages.shared.db.engine import get_engine
        from packages.shared.db.models import (
            LeagueScoringRule,
            UserLeague,
            UserRoster,
        )

        engine = get_engine()
        total = 0

        # --- Use a SINGLE connection for the entire upsert operation ---
        # This avoids cross-connection issues where league_id read could miss
        # the just-upserted row if using separate connections.
        with engine.connect() as conn:
            # --- UPSERT user_leagues ---
            league_row = {
                "yahoo_league_key": league_info["yahoo_league_key"],
                "league_name": league_info.get("league_name"),
                "season_year": league_info.get("season_year"),
                "num_teams": league_info.get("num_teams"),
                "scoring_type": league_info.get("scoring_type"),
                "current_week": league_info.get("current_week"),
                "user_team_key": league_info.get("user_team_key"),
            }
            league_table = UserLeague.__table__
            stmt = pg_insert(league_table).values([league_row])
            update_cols = {
                c.name: stmt.excluded[c.name]
                for c in league_table.c
                if c.name not in ("id", "created_at")
            }
            stmt = stmt.on_conflict_do_update(constraint="uq_yahoo_league_key", set_=update_cols)
            conn.execute(stmt)
            conn.commit()

            # Read back league_id in SAME connection
            result = conn.execute(
                league_table.select().where(
                    league_table.c.yahoo_league_key == league_info["yahoo_league_key"]
                )
            )
            league_row_db = result.fetchone()
            league_id = league_row_db.id if league_row_db else None
            total += 1
            logger.info("Upserted league: %s", league_info.get("league_name"))

            if not league_id:
                logger.warning("Could not determine league_id after upsert")
                return total

            # --- UPSERT user_rosters ---
            if roster:
                today = date.today()
                roster_table = UserRoster.__table__
                skipped = 0
                for player in roster:
                    player_id = _extract_yahoo_player_id(player["player_id"])
                    if player_id is None:
                        logger.warning(
                            "Skipping player with unparseable Yahoo ID: %s",
                            player["player_id"],
                        )
                        skipped += 1
                        continue
                    row = {
                        "league_id": league_id,
                        "player_id": player_id,
                        "roster_date": today,
                        "roster_position": player.get("position_slot"),
                        "acquisition_type": player.get("acquisition_type"),
                    }
                    stmt = pg_insert(roster_table).values([row])
                    update_cols = {
                        c.name: stmt.excluded[c.name]
                        for c in roster_table.c
                        if c.name not in ("id", "created_at")
                    }
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_roster_slot", set_=update_cols
                    )
                    conn.execute(stmt)
                if skipped > 0:
                    logger.warning("Skipped %d players with unparseable Yahoo IDs", skipped)
                conn.commit()
                total += len(roster) - skipped
                logger.info("Upserted %d roster entries", len(roster) - skipped)

            # --- UPSERT league_scoring_rules ---
            if scoring_rules:
                rules_table = LeagueScoringRule.__table__
                for rule in scoring_rules:
                    rule_row = {
                        "league_id": league_id,
                        "stat_category": rule["stat_category"],
                        "points_value": rule["points_value"],
                        "is_negative": rule.get("is_negative", False),
                    }
                    stmt = pg_insert(rules_table).values([rule_row])
                    update_cols = {
                        c.name: stmt.excluded[c.name]
                        for c in rules_table.c
                        if c.name not in ("id", "created_at")
                    }
                    stmt = stmt.on_conflict_do_update(
                        constraint="uq_scoring_rule", set_=update_cols
                    )
                    conn.execute(stmt)
                conn.commit()
                total += len(scoring_rules)
                logger.info("Upserted %d scoring rules", len(scoring_rules))

        # Matchups are logged but not persisted (no matchups table in schema)
        if matchups:
            logger.info("Received %d matchups (logged, not persisted)", len(matchups))

        logger.info("Total Yahoo data rows upserted: %d", total)
        return total

    # Resolve league_key from param or environment
    resolved_league_key = league_key or os.environ.get("YAHOO_LEAGUE_KEY", "")

    # Task wiring
    session = refresh_oauth_token()
    league_info = fetch_league_info(session, resolved_league_key)
    roster = fetch_roster(session, league_info)
    scoring = fetch_scoring_rules(session, resolved_league_key)
    matchup_data = fetch_matchups(session, resolved_league_key)
    upsert_yahoo_data(league_info, roster, scoring, matchup_data)


ingest_yahoo_dag = ingest_yahoo()
