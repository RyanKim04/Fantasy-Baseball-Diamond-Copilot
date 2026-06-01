"""Shared pytest fixtures for Diamond Copilot tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from packages.shared.db.models import Base

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="session")
def test_engine() -> Engine:
    """Create an in-memory SQLite engine for unit tests.

    Enables foreign key enforcement via PRAGMA for referential integrity.
    For integration tests requiring Postgres-specific features,
    use a separate fixture that connects to the Docker Postgres.
    """
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return engine


@pytest.fixture()
def test_session(test_engine: Engine) -> Generator[Session, None, None]:
    """Yield a test session that rolls back after each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = sessionmaker(bind=connection)()
    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture()
def clean_tables(test_engine: Engine) -> None:
    """Drop and recreate all tables. Use sparingly for tests that need a fresh DB."""
    Base.metadata.drop_all(test_engine)
    Base.metadata.create_all(test_engine)
