"""Airflow DAG: ingest Statcast pitch-level data and aggregate to daily stats.

Default behavior: pull last 2 days. Idempotent via UPSERT on natural keys.
Schedule: daily at 08:00 UTC (after mlb_schedule at 06:00 and boxscores at 07:00).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from airflow.sdk import dag, task

if TYPE_CHECKING:
    from datetime import date

    import pandas as pd


default_args = {
    "owner": "diamond-copilot",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


@dag(
    dag_id="ingest_statcast",
    schedule="0 8 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "statcast"],
    doc_md=__doc__,
)
def ingest_statcast(
    start_date: date | None = None,
    end_date: date | None = None,
) -> None:
    """Pull Statcast pitch-level data and aggregate to daily batting/pitching stats.

    Default: last 2 days. Idempotent via UPSERT on natural keys.
    """

    @task(retries=3, retry_delay=timedelta(seconds=60))
    def fetch_statcast_range(start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch raw Statcast data from pybaseball for the given date range.

        Returns a DataFrame with one row per pitch.
        """
        raise NotImplementedError("Task 0.5")

    @task
    def validate_and_transform(raw_df: pd.DataFrame) -> pd.DataFrame:
        """Validate raw Statcast rows against StatcastPitchRow schema.

        Drops invalid rows, logs warnings. Returns cleaned DataFrame.
        """
        raise NotImplementedError("Task 0.5")

    @task
    def upsert_pitches(df: pd.DataFrame) -> int:
        """UPSERT pitch-level rows into the pitches table.

        Returns the number of rows upserted.
        """
        raise NotImplementedError("Task 0.5")

    @task
    def aggregate_daily_batting(pitches_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate pitch-level data to per-player, per-game batting stats.

        Returns a DataFrame matching StatcastBattingDaily schema.
        """
        raise NotImplementedError("Task 0.5")

    @task
    def aggregate_daily_pitching(pitches_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate pitch-level data to per-player, per-game pitching stats.

        Returns a DataFrame matching StatcastPitchingDaily schema.
        """
        raise NotImplementedError("Task 0.5")

    @task
    def upsert_daily_stats(batting_df: pd.DataFrame, pitching_df: pd.DataFrame) -> int:
        """UPSERT daily batting and pitching stats into their respective tables.

        Returns total rows upserted across both tables.
        """
        raise NotImplementedError("Task 0.5")

    # Task wiring
    raw = fetch_statcast_range(start_date, end_date)
    clean = validate_and_transform(raw)
    upsert_pitches(clean)
    batting = aggregate_daily_batting(clean)
    pitching = aggregate_daily_pitching(clean)
    upsert_daily_stats(batting, pitching)


ingest_statcast_dag = ingest_statcast()
