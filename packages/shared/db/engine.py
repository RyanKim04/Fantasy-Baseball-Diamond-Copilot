"""Database engine and session factory.

Provides a configured SQLAlchemy engine and a context-managed session generator.
"""

from collections.abc import Generator

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from packages.shared.config import get_settings


def get_engine() -> Engine:
    """Create and return a SQLAlchemy engine from the configured DATABASE_URL."""
    settings = get_settings()
    return create_engine(
        settings.database_url,
        echo=(settings.environment == "development"),
        pool_pre_ping=True,
    )


def get_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session, committing on success and rolling back on error."""
    engine = get_engine()
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
