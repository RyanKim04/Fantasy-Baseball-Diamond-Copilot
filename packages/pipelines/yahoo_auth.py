"""Yahoo OAuth 2.0 helper module.

Handles token acquisition, refresh, and storage for the Yahoo Fantasy Sports API.
Used as a utility by the ingest_yahoo DAG — not a DAG itself.
"""

from __future__ import annotations

from typing import Any


def get_oauth_session() -> Any:
    """Return an authenticated Yahoo OAuth session.

    Loads stored tokens if available; otherwise raises an error directing
    the user to run the initial auth flow.
    """
    raise NotImplementedError("Task 0.7")


def refresh_token(token: dict[str, str]) -> dict[str, str]:
    """Refresh an expired Yahoo OAuth access token.

    Args:
        token: Dict containing at least 'refresh_token'.

    Returns:
        Updated token dict with new access_token and expiry.
    """
    raise NotImplementedError("Task 0.7")


def store_token(token: dict[str, str]) -> None:
    """Persist a Yahoo OAuth token securely.

    Storage location determined by Settings (file-based for dev, DB for prod).
    """
    raise NotImplementedError("Task 0.7")


def load_token() -> dict[str, str] | None:
    """Load a previously stored Yahoo OAuth token.

    Returns None if no token is stored.
    """
    raise NotImplementedError("Task 0.7")
