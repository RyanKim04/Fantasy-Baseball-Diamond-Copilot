"""Airflow DAG: one-time historical backfill of Statcast data (2018-2025).

Designed to be run once during initial setup via manual trigger.
Uses dynamic task mapping to process each year in parallel.
Idempotent -- safe to re-run. Not scheduled (manual trigger only).
"""

from __future__ import annotations

from datetime import datetime, timedelta  # noqa: TCH003

from airflow.sdk import dag, task

default_args = {
    "owner": "diamond-copilot",
    "retries": 0,
}


@dag(
    dag_id="seed_historical",
    schedule=None,  # manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["backfill", "statcast"],
    doc_md=__doc__,
)
def seed_historical() -> None:
    """Backfill historical Statcast data for 2018-2025.

    Uses Airflow dynamic task mapping to process each year independently.
    """

    @task(retries=2, retry_delay=timedelta(minutes=2))
    def seed_statcast_month(year: int, month: int) -> int:
        """Fetch and ingest one month of Statcast data.

        Uses the same validation and upsert logic as the daily ingest_statcast DAG.
        Returns the number of rows upserted.
        """
        raise NotImplementedError("Task 0.9")

    @task
    def seed_year(year: int) -> int:
        """Ingest all months of a single MLB season (March through October).

        Delegates to seed_statcast_month for each month.
        Returns total rows upserted for the year.
        """
        raise NotImplementedError("Task 0.9")

    # Task wiring -- dynamic task mapping over year range
    years = list(range(2018, 2026))  # 2018 through 2025 inclusive
    seed_year.expand(year=years)


seed_historical_dag = seed_historical()
