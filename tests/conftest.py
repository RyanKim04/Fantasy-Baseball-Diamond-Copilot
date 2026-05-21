"""Shared pytest fixtures for Diamond Copilot tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from packages.shared.db.models import Base

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture(scope="session")
def test_engine() -> Engine:
    """Create an in-memory SQLite engine for unit tests.

    For integration tests requiring Postgres-specific features,
    use a separate fixture that connects to the Docker Postgres.
    """
    engine = create_engine("sqlite:///:memory:")
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
    """Drop and recreate all tables. Use sparingly."""
    Base.metadata.drop_all(test_engine)
    Base.metadata.create_all(test_engine)
