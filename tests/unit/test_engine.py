"""Unit tests for database engine and settings singletons."""

from __future__ import annotations

from unittest.mock import patch


class TestGetSettings:
    """Verify get_settings() returns a cached singleton."""

    def test_get_settings_returns_same_object(self) -> None:
        """Calling get_settings() twice should return the same object (lru_cache)."""
        from packages.shared.config import get_settings

        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2

    def test_settings_has_expected_fields(self) -> None:
        """Settings object should expose key configuration fields."""
        from packages.shared.config import get_settings

        settings = get_settings()
        assert hasattr(settings, "database_url")
        assert hasattr(settings, "environment")
        assert hasattr(settings, "log_level")


class TestGetEngine:
    """Verify get_engine() returns a singleton engine."""

    def test_get_engine_returns_singleton(self) -> None:
        """Calling get_engine() twice should return the same Engine instance."""
        from packages.shared.db import engine as engine_module

        # Reset the module-level _engine to None so we can test fresh
        original = engine_module._engine
        engine_module._engine = None

        try:
            with patch.object(
                engine_module,
                "get_settings",
                return_value=type(
                    "FakeSettings",
                    (),
                    {"database_url": "sqlite:///:memory:", "environment": "development"},
                )(),
            ):
                e1 = engine_module.get_engine()
                e2 = engine_module.get_engine()
                assert e1 is e2
        finally:
            # Restore original state
            engine_module._engine = original
