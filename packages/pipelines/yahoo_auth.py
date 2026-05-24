"""Yahoo OAuth 2.0 helper module.

Handles token acquisition, refresh, and storage for the Yahoo Fantasy Sports API.
Used as a utility by the ingest_yahoo DAG — not a DAG itself.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from packages.shared.config import get_settings

logger = logging.getLogger(__name__)

TOKEN_FILE = Path(".yahoo_token.json")


def get_oauth_session() -> Any:
    """Return an authenticated Yahoo OAuth session.

    Loads stored tokens if available; otherwise raises an error directing
    the user to run the initial auth flow.

    Returns:
        An authenticated yahoo_oauth.OAuth2 instance ready for API calls.

    Raises:
        ValueError: If Yahoo OAuth credentials are not configured.
        ImportError: If yahoo_oauth is not installed.
        RuntimeError: If no stored token exists and interactive auth is needed.
    """
    settings = get_settings()

    if not settings.yahoo_client_id or not settings.yahoo_client_secret:
        raise ValueError(
            "Yahoo OAuth credentials not configured. "
            "Set YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET in .env"
        )

    try:
        from yahoo_oauth import OAuth2  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("yahoo_oauth not installed. Run: pip install yahoo_oauth") from e

    # If no token file exists, the user must run the initial interactive flow.
    # yahoo_oauth.OAuth2 would open a browser, which is not safe in headless
    # environments (Airflow workers, CI). Fail fast with a helpful message.
    if not TOKEN_FILE.exists():
        raise RuntimeError(
            "No Yahoo OAuth token found. Run the initial authentication flow:\n"
            "  python -m packages.pipelines.yahoo_auth\n"
            "This will open a browser for Yahoo login and store the token locally."
        )

    oauth = OAuth2(
        settings.yahoo_client_id,
        settings.yahoo_client_secret,
        token_path=str(TOKEN_FILE),
    )

    if not oauth.token_is_valid():
        oauth.refresh_access_token()
        logger.info("Yahoo OAuth token refreshed successfully.")

    return oauth


def refresh_token(token: dict[str, str]) -> dict[str, str]:
    """Refresh an expired Yahoo OAuth access token.

    yahoo_oauth handles refresh internally when using get_oauth_session(),
    but this function provides a manual fallback for programmatic use.

    Args:
        token: Dict containing at least 'refresh_token'.

    Returns:
        Updated token dict with new access_token and expiry.
    """
    settings = get_settings()

    if not settings.yahoo_client_id or not settings.yahoo_client_secret:
        raise ValueError(
            "Yahoo OAuth credentials not configured. "
            "Set YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET in .env"
        )

    try:
        from yahoo_oauth import OAuth2  # type: ignore[import-untyped]
    except ImportError as e:
        raise ImportError("yahoo_oauth not installed. Run: pip install yahoo_oauth") from e

    # Store the token so yahoo_oauth can read it
    store_token(token)

    oauth = OAuth2(
        settings.yahoo_client_id,
        settings.yahoo_client_secret,
        token_path=str(TOKEN_FILE),
    )
    oauth.refresh_access_token()
    logger.info("Yahoo OAuth token refreshed via manual refresh_token() call.")

    # Read back the updated token
    updated = load_token()
    if updated is None:
        logger.warning("Could not read refreshed token; returning original.")
        return token
    return updated


def store_token(token: dict[str, str]) -> None:
    """Persist a Yahoo OAuth token to disk.

    Token is stored as JSON at TOKEN_FILE (default: .yahoo_token.json in
    project root). This path is gitignored.
    """
    TOKEN_FILE.write_text(json.dumps(token, indent=2))
    logger.info("Yahoo OAuth token stored at %s", TOKEN_FILE)


def load_token() -> dict[str, str] | None:
    """Load a previously stored Yahoo OAuth token.

    Returns None if no token is stored or if the file is corrupted.
    """
    if not TOKEN_FILE.exists():
        return None
    try:
        data: dict[str, str] = json.loads(TOKEN_FILE.read_text())
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load Yahoo token from %s: %s", TOKEN_FILE, e)
        return None


if __name__ == "__main__":
    # Interactive initial authentication flow.
    # Run this once to authorize the app and store the token.
    import sys

    logging.basicConfig(level=logging.INFO)

    _settings = get_settings()
    if not _settings.yahoo_client_id or not _settings.yahoo_client_secret:
        print(
            "ERROR: Set YAHOO_CLIENT_ID and YAHOO_CLIENT_SECRET in .env first.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        from yahoo_oauth import OAuth2  # type: ignore[import-untyped]
    except ImportError:
        print(
            "ERROR: yahoo_oauth not installed. Run: pip install yahoo_oauth",
            file=sys.stderr,
        )
        sys.exit(1)

    print("Starting Yahoo OAuth2 authorization flow...")
    print("A browser window will open. Log in to Yahoo and authorize the app.")

    _oauth = OAuth2(
        _settings.yahoo_client_id,
        _settings.yahoo_client_secret,
        token_path=str(TOKEN_FILE),
    )

    if _oauth.token_is_valid():
        print(f"Authentication successful! Token stored at {TOKEN_FILE}")
    else:
        print("WARNING: Token may not be valid. Check your credentials.")
        sys.exit(1)
