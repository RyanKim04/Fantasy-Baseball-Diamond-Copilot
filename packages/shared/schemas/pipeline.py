"""Generic pipeline result schema used across all Airflow DAGs."""

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class PipelineResult(BaseModel, Generic[T]):
    """Standardized result wrapper for all pipeline flows.

    Attributes:
        status: Overall outcome of the pipeline run.
        rows_upserted: Total rows inserted or updated in the database.
        errors: List of error messages encountered during the run.
        data: Optional payload (e.g., row count, summary dict).
    """

    status: Literal["success", "partial", "failure"]
    rows_upserted: int = 0
    errors: list[str] = []
    data: T | None = None
