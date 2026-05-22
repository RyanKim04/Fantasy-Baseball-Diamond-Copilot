"""Airflow DAG: ingest Yahoo Fantasy Sports data (roster, scoring, matchups).

Requires a valid Yahoo OAuth session. See yahoo_auth.py for token management.
Schedule: daily at 09:00 UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from airflow.sdk import dag, task

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
        """Ensure the Yahoo OAuth token is fresh. Refresh if expired.

        Returns the authenticated session object.
        """
        raise NotImplementedError("Task 0.8")

    @task
    def fetch_league_info(session: Any, league_key: str) -> dict:
        """Fetch league metadata (name, scoring type, num teams) from Yahoo API.

        Returns a dict matching YahooLeague schema fields.
        """
        raise NotImplementedError("Task 0.8")

    @task
    def fetch_roster(session: Any, league_info: dict) -> list[dict]:
        """Fetch the current roster for a Yahoo fantasy team.

        Extracts team_key from league_info at runtime (league_info contains
        the user_team_key field from the league metadata response).

        Returns a list of dicts matching YahooRoster schema fields.
        """
        raise NotImplementedError("Task 0.8")

    @task
    def fetch_scoring_rules(session: Any, league_key: str) -> list[dict]:
        """Fetch scoring rules for a Yahoo league.

        Returns a list of dicts matching YahooScoringRule schema fields.
        """
        raise NotImplementedError("Task 0.8")

    @task
    def fetch_matchups(session: Any, league_key: str, week: int | None = None) -> list[dict]:
        """Fetch matchup data for the current (or specified) week.

        Returns a list of dicts matching YahooMatchup schema fields.
        """
        raise NotImplementedError("Task 0.8")

    @task
    def upsert_yahoo_data(
        league_info: dict,
        roster: list[dict],
        scoring_rules: list[dict],
        matchups: list[dict],
    ) -> int:
        """UPSERT all Yahoo data into their respective tables.

        Returns total rows upserted.
        """
        raise NotImplementedError("Task 0.8")

    # Task wiring
    session = refresh_oauth_token()
    league_info = fetch_league_info(session, league_key)
    # team_key is extracted from league_info at runtime inside fetch_roster
    roster = fetch_roster(session, league_info)
    scoring = fetch_scoring_rules(session, league_key)
    matchups = fetch_matchups(session, league_key)
    upsert_yahoo_data(league_info, roster, scoring, matchups)


ingest_yahoo_dag = ingest_yahoo()
