"""Airflow DAG: one-time historical backfill of Statcast data (2018-2025).

Designed to be run once during initial setup via manual trigger.
Processes month-by-month to manage memory and provide progress tracking.
Idempotent -- safe to re-run. Not scheduled (manual trigger only).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from airflow.sdk import dag, task

if TYPE_CHECKING:
    from packages.shared.schemas.pipeline import PipelineResult


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
def seed_historical(
    start_year: int = 2018,
    end_year: int = 2025,
    start_month: int | None = None,
) -> PipelineResult[dict[int, int]]:
    """Backfill historical Statcast data for the specified year range.

    Args:
        start_year: First season to ingest (inclusive).
        end_year: Last season to ingest (inclusive).
        start_month: If provided, resume from this month in start_year.

    Returns:
        PipelineResult with data as a dict of {year: rows_upserted}.
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

    # Task wiring — delegate to seed_year tasks
    # Note: year range is resolved at runtime, not DAG parse time.
    # Individual years are triggered via Airflow params at run time.
    seed_year(start_year)


seed_historical_dag = seed_historical()
