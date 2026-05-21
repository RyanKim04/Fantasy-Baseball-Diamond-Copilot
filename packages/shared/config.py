"""Application configuration via pydantic-settings.

Reads from .env file or environment variables. See .env.example for all fields.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the Diamond Copilot application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Database ---
    database_url: str = "postgresql://user:password@localhost:5432/diamond_copilot"
    supabase_url: str = ""
    supabase_anon_key: str = ""
    supabase_service_role_key: str = ""

    # --- Yahoo Fantasy Sports OAuth ---
    yahoo_client_id: str = ""
    yahoo_client_secret: str = ""
    yahoo_redirect_uri: str = "http://localhost:3000/api/auth/callback/yahoo"

    # --- Anthropic (Phase 6) ---
    anthropic_api_key: str = ""

    # --- MLflow ---
    mlflow_tracking_uri: str = "http://localhost:5000"

    # --- Airflow ---
    airflow_home: str = "./infra/airflow"
    airflow__core__fernet_key: str = ""
    airflow__database__sql_alchemy_conn: str = ""

    # --- Sentry (Phase 4) ---
    sentry_dsn: str = ""

    # --- Misc ---
    environment: Literal["development", "staging", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
