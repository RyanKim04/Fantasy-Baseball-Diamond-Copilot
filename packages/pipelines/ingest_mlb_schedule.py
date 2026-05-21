"""Airflow DAG: ingest MLB schedule, probable pitchers, and player metadata.

Uses the MLB Stats API (statsapi.mlb.com). Idempotent via UPSERT on game_pk / player_id.
Schedule: daily at 08:00 UTC.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from airflow.sdk import dag, task

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd

    from packages.shared.schemas.pipeline import PipelineResult


default_args = {
    "owner": "diamond-copilot",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="ingest_mlb_schedule",
    schedule="0 8 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "mlb-schedule"],
    doc_md=__doc__,
)
def ingest_mlb_schedule(
    start_date: date | None = None,
    end_date: date | None = None,
) -> PipelineResult[int]:
    """Pull upcoming MLB schedule, probable pitchers, and player metadata.

    Default: next 7 days. Idempotent via UPSERT on natural keys.
    """

    @task(retries=3, retry_delay=timedelta(seconds=60))
    def fetch_schedule(start_date: date, end_date: date) -> list[dict]:
        """Fetch game schedule from MLB Stats API for the given date range.

        Returns a list of game dicts.
        """
        raise NotImplementedError("Task 0.6")

    @task
    def fetch_probable_lineups(game_pks: list[int]) -> pd.DataFrame:
        """Fetch probable starting lineups for the given games.

        Returns a DataFrame with one row per lineup entry.
        """
        raise NotImplementedError("Task 0.6")

    @task
    def upsert_games(games: list[dict]) -> int:
        """UPSERT game records into the games table.

        Returns the number of rows upserted.
        """
        raise NotImplementedError("Task 0.6")

    @task
    def upsert_players(player_ids: list[int]) -> int:
        """Fetch player metadata from MLB Stats API and UPSERT into the players table.

        Only fetches metadata for players not already in the DB (or stale entries).
        Returns the number of rows upserted.
        """
        raise NotImplementedError("Task 0.6")

    # Task wiring
    games = fetch_schedule(start_date, end_date)
    upsert_games(games)
    lineups = fetch_probable_lineups(games)
    upsert_players(lineups)


ingest_mlb_schedule_dag = ingest_mlb_schedule()
