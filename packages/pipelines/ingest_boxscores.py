"""Airflow DAG: ingest MLB boxscore data for completed games.

Fetches yesterday's completed games and extracts batting/pitching lines.
Populates batting_stats_daily and pitching_stats_daily with counting stats
that cannot be derived from Statcast pitch-level data alone (e.g., W, L, SV).
Schedule: daily at 07:00 UTC (after mlb_schedule at 06:00, before statcast at 08:00).
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.sdk import dag, task

default_args = {
    "owner": "diamond-copilot",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


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
    """Fetch boxscore data for completed games and upsert batting/pitching stats.

    Default: yesterday's completed games. Idempotent via UPSERT on natural keys.
    """

    @task(retries=3, retry_delay=timedelta(seconds=60))
    def fetch_completed_games() -> list[int]:
        """Query the games table for yesterday's completed games.

        Returns a list of game_pk values for games with status 'Final'.
        """
        raise NotImplementedError("Task 0.5")

    @task
    def fetch_boxscore(game_pk: int) -> dict:
        """Fetch boxscore data from MLB Stats API for a single game.

        Returns a dict containing batting and pitching lines for both teams.
        """
        raise NotImplementedError("Task 0.5")

    @task
    def upsert_batting_stats(boxscores: list[dict]) -> int:
        """UPSERT batting lines from boxscores into batting_stats_daily.

        Returns the number of rows upserted.
        """
        raise NotImplementedError("Task 0.5")

    @task
    def upsert_pitching_stats(boxscores: list[dict]) -> int:
        """UPSERT pitching lines from boxscores into pitching_stats_daily.

        Returns the number of rows upserted.
        """
        raise NotImplementedError("Task 0.5")

    # Task wiring
    game_pks = fetch_completed_games()
    boxscores = fetch_boxscore.expand(game_pk=game_pks)
    upsert_batting_stats(boxscores)
    upsert_pitching_stats(boxscores)


ingest_boxscores_dag = ingest_boxscores()
