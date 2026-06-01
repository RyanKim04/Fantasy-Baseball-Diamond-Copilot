"""Integration tests for Airflow DAG definitions.

These tests verify that DAGs and tasks are importable and properly defined.
They do NOT execute the actual DAGs (no DB or API calls).
"""

from __future__ import annotations


class TestPipelineImports:
    """Verify all pipeline modules are importable."""

    def test_import_ingest_statcast(self) -> None:
        from packages.pipelines.ingest_statcast import ingest_statcast_dag

        assert ingest_statcast_dag is not None

    def test_import_ingest_mlb_schedule(self) -> None:
        from packages.pipelines.ingest_mlb_schedule import ingest_mlb_schedule_dag

        assert ingest_mlb_schedule_dag is not None

    def test_import_ingest_yahoo(self) -> None:
        from packages.pipelines.ingest_yahoo import ingest_yahoo_dag

        assert ingest_yahoo_dag is not None

    def test_import_seed_historical(self) -> None:
        from packages.pipelines.seed_historical import seed_historical_dag

        assert seed_historical_dag is not None

    def test_import_yahoo_auth(self) -> None:
        from packages.pipelines.yahoo_auth import get_oauth_session

        assert get_oauth_session is not None

    def test_import_ingest_boxscores(self) -> None:
        from packages.pipelines.ingest_boxscores import ingest_boxscores_dag

        assert ingest_boxscores_dag is not None


class TestDagRegistration:
    """Verify Airflow DAG objects have correct metadata."""

    def test_statcast_dag_id(self) -> None:
        from packages.pipelines.ingest_statcast import ingest_statcast_dag

        assert ingest_statcast_dag.dag_id == "ingest_statcast"

    def test_mlb_schedule_dag_id(self) -> None:
        from packages.pipelines.ingest_mlb_schedule import ingest_mlb_schedule_dag

        assert ingest_mlb_schedule_dag.dag_id == "ingest_mlb_schedule"

    def test_yahoo_dag_id(self) -> None:
        from packages.pipelines.ingest_yahoo import ingest_yahoo_dag

        assert ingest_yahoo_dag.dag_id == "ingest_yahoo"

    def test_seed_dag_id(self) -> None:
        from packages.pipelines.seed_historical import seed_historical_dag

        assert seed_historical_dag.dag_id == "seed_historical"

    def test_boxscores_dag_id(self) -> None:
        from packages.pipelines.ingest_boxscores import ingest_boxscores_dag

        assert ingest_boxscores_dag.dag_id == "ingest_boxscores"


class TestDagSchedules:
    """Verify each DAG has the correct cron schedule."""

    def test_mlb_schedule_cron(self) -> None:
        from packages.pipelines.ingest_mlb_schedule import ingest_mlb_schedule_dag

        assert ingest_mlb_schedule_dag.schedule == "0 6 * * *"

    def test_boxscores_cron(self) -> None:
        from packages.pipelines.ingest_boxscores import ingest_boxscores_dag

        assert ingest_boxscores_dag.schedule == "0 7 * * *"

    def test_statcast_cron(self) -> None:
        from packages.pipelines.ingest_statcast import ingest_statcast_dag

        assert ingest_statcast_dag.schedule == "0 8 * * *"

    def test_yahoo_cron(self) -> None:
        from packages.pipelines.ingest_yahoo import ingest_yahoo_dag

        assert ingest_yahoo_dag.schedule == "0 9 * * *"

    def test_seed_historical_manual_only(self) -> None:
        from packages.pipelines.seed_historical import seed_historical_dag

        assert seed_historical_dag.schedule is None


class TestDagTags:
    """Verify each DAG has the correct tags."""

    def test_statcast_tags(self) -> None:
        from packages.pipelines.ingest_statcast import ingest_statcast_dag

        assert "ingestion" in ingest_statcast_dag.tags
        assert "statcast" in ingest_statcast_dag.tags

    def test_mlb_schedule_tags(self) -> None:
        from packages.pipelines.ingest_mlb_schedule import ingest_mlb_schedule_dag

        assert "ingestion" in ingest_mlb_schedule_dag.tags
        assert "mlb-schedule" in ingest_mlb_schedule_dag.tags

    def test_boxscores_tags(self) -> None:
        from packages.pipelines.ingest_boxscores import ingest_boxscores_dag

        assert "ingestion" in ingest_boxscores_dag.tags
        assert "boxscores" in ingest_boxscores_dag.tags

    def test_yahoo_tags(self) -> None:
        from packages.pipelines.ingest_yahoo import ingest_yahoo_dag

        assert "ingestion" in ingest_yahoo_dag.tags
        assert "yahoo" in ingest_yahoo_dag.tags

    def test_seed_historical_tags(self) -> None:
        from packages.pipelines.seed_historical import seed_historical_dag

        assert "backfill" in seed_historical_dag.tags
        assert "statcast" in seed_historical_dag.tags
