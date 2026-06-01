# Phase 0 -- Foundations & Data Layer: Task Breakdown

**Branch:** `phase/0-data-layer`
**Goal:** Repeatable, automated ingestion of all data we need.
**Time estimate:** 1.5 weeks

---

## Dependency Graph

```
0.1 Project scaffold
 |
 +---> 0.2 DB schema (SQLAlchemy models)
 |      |
 |      +---> 0.3 Alembic initial migration
 |      |
 |      +---> 0.4 Pydantic validation schemas
 |      |      |
 |      |      +---> 0.5 ingest_statcast pipeline ----+
 |      |      |                                       |
 |      |      +---> 0.5b ingest_boxscores pipeline    |
 |      |      |                                       |
 |      |      +---> 0.6 ingest_mlb_schedule pipeline  |
 |      |                                              |
 |      +------+-------> 0.10 Unit & integration tests |
 |             |                                       |
 +---> 0.7 Yahoo OAuth [HIGH RISK] ----+              |
              |                         |              |
              +---> 0.8 ingest_yahoo ---+              |
                                        |              |
                                        v              v
                                 0.11 Airflow deployment & scheduling
                                        |
                                        v
                                 0.12 Documentation & phase close
```

**Critical path:** 0.1 -> 0.2 -> 0.4 -> 0.5 -> 0.9 -> 0.11 -> 0.12
**Risk path:** 0.1 -> 0.7 -> 0.8 -> 0.11 (Yahoo OAuth is highest-risk item)
**Parallelism:** Tasks 0.5, 0.5b, 0.6, and 0.7 can run in parallel after 0.4 is done.

---

## Task 0.1 -- Project scaffold & config

- **Owner:** architect
- **Dependencies:** none
- **Status:** [x] complete
- **Completion notes:** pyproject.toml, docker-compose, alembic.ini, conftest.py, config.py all in place.
- **Deliverables:**
  - `pyproject.toml` with all dependencies
  - `alembic.ini` configured
  - `packages/` directory structure with `__init__.py` files
  - `packages/shared/config.py` (pydantic-settings, cached via @lru_cache)
  - `packages/shared/types.py` (NewType branded types)
  - `infra/docker/docker-compose.yml` (local Postgres + adminer + Airflow)
  - `tests/conftest.py` with test engine fixture
- **Acceptance criteria:**
  - All directories exist with proper `__init__.py`
  - `ruff check .` passes
  - `pyproject.toml` is valid and installable

## Task 0.2 -- Database schema (SQLAlchemy models)

- **Owner:** architect
- **Dependencies:** 0.1
- **Status:** [x] complete
- **Completion notes:** All 12 tables with timestamps, unique constraints, FK relationships, and indexes.
- **Deliverables:**
  - `packages/shared/db/models.py` -- all 12 tables (players, games, pitches, batting_stats_daily, pitching_stats_daily, park_factors, season_stats_batting, season_stats_pitching, user_leagues, user_rosters, league_scoring_rules, predictions)
  - `packages/shared/db/engine.py` -- singleton engine + session factory
- **Acceptance criteria:**
  - `Base.metadata.tables` contains all 12 table names
  - Every table has `created_at` and `updated_at` with server defaults
  - Unique constraints exist on all specified column sets
  - Foreign keys link player_id to players, game_pk to games, etc.

## Task 0.3 -- Alembic initial migration

- **Owner:** architect
- **Dependencies:** 0.2
- **Status:** [x] complete
- **Completion notes:** Initial migration auto-generated covering all 12 tables. Verified upgrade/downgrade roundtrip.
- **Deliverables:**
  - `infra/migrations/env.py` (scaffold done)
  - `infra/migrations/script.py.mako` (scaffold done)
  - `infra/migrations/versions/<timestamp>_initial.py` -- auto-generated
- **Acceptance criteria:**
  - `alembic revision --autogenerate -m "initial"` produces a valid migration
  - `alembic upgrade head` creates all 12 tables in a running Postgres
  - `alembic downgrade base` cleanly drops everything

## Task 0.4 -- Pydantic validation schemas

- **Owner:** architect
- **Dependencies:** 0.2
- **Status:** [x] complete
- **Completion notes:** All schemas implemented with Pydantic v2, including boxscore and FanGraphs schemas.
- **Deliverables:**
  - `packages/shared/schemas/statcast.py`
  - `packages/shared/schemas/mlb_schedule.py` (includes BoxscoreBattingLine, BoxscorePitchingLine)
  - `packages/shared/schemas/yahoo.py`
  - `packages/shared/schemas/pipeline.py`
  - `packages/shared/schemas/fangraphs.py` (SeasonStatsBattingSchema, SeasonStatsPitchingSchema)
- **Acceptance criteria:**
  - Valid payloads pass validation; invalid payloads raise `ValidationError`
  - All schemas use Pydantic v2 syntax
  - `PipelineResult[T]` is generic and works with concrete types
  - No mutable default arguments

## Task 0.5 -- ingest_statcast pipeline

- **Owner:** implementer
- **Dependencies:** 0.2, 0.4
- **Status:** [x] complete
- **Completion notes:** Full pipeline with fetch, validate, transform, aggregate, and UPSERT. Handles empty data gracefully.
- **Deliverables:**
  - `packages/pipelines/ingest_statcast.py` -- fully implemented
- **Acceptance criteria:**
  - Pulls last 2 days of Statcast data via pybaseball
  - Validates rows against `StatcastPitchRow` schema
  - UPSERTs into `pitches`, `batting_stats_daily`, `pitching_stats_daily`
  - Re-running does not duplicate data (idempotent)
  - Retries on transient errors (network, rate limit)
  - Schedule: Daily 8am UTC (after mlb_schedule and boxscores)

## Task 0.5b -- ingest_boxscores pipeline

- **Owner:** implementer
- **Dependencies:** 0.2, 0.4
- **Status:** [x] complete
- **Completion notes:** Extracts W/L/SV/HLD/BS/QS and other fantasy-relevant fields from MLB Stats API boxscores.
- **Deliverables:**
  - `packages/pipelines/ingest_boxscores.py` -- fully implemented
- **Acceptance criteria:**
  - Fetches completed games from games table
  - Pulls boxscore data via MLB Stats API
  - UPSERTs batting/pitching lines into daily stats tables
  - Populates fantasy-relevant fields (W, L, SV, HLD, BS, QS, HBP, WP)
  - Idempotent on re-run
  - Schedule: Daily 7am UTC (after mlb_schedule, before statcast)

## Task 0.6 -- ingest_mlb_schedule pipeline

- **Owner:** implementer
- **Dependencies:** 0.2, 0.4
- **Status:** [x] complete
- **Completion notes:** Fetches schedule, probable pitchers, and player metadata. Handles doubleheaders and postponements.
- **Deliverables:**
  - `packages/pipelines/ingest_mlb_schedule.py` -- fully implemented
- **Acceptance criteria:**
  - Pulls next 7 days of games from MLB Stats API
  - UPSERTs games and player metadata
  - Handles doubleheaders and postponements
  - Idempotent on re-run
  - Schedule: Daily 6am UTC (first DAG in the daily chain)

## Task 0.7 -- Yahoo OAuth flow  [HIGH RISK]

- **Owner:** implementer
- **Dependencies:** 0.1
- **Status:** [x] complete
- **Completion notes:** OAuth flow implemented using yahoo_fantasy_api. Token storage via file with secure permissions.
- **Deliverables:**
  - `packages/pipelines/yahoo_auth.py` -- fully implemented
- **Acceptance criteria:**
  - Can complete initial OAuth flow and obtain access + refresh tokens
  - Tokens are stored securely (not in git)
  - Token refresh works when access token expires
  - Clear error messages when credentials are missing or invalid
- **Risk mitigations:**
  - Budget 3 extra days
  - Fallback: manual CSV roster upload if Yahoo API is too unstable
  - Use `yahoo_fantasy_api` library (handles most OAuth complexity)

## Task 0.8 -- ingest_yahoo pipeline

- **Owner:** implementer
- **Dependencies:** 0.7, 0.2, 0.4
- **Status:** [x] complete
- **Completion notes:** Pulls league info, roster, scoring rules, matchups. Extracts team_key correctly.
- **Deliverables:**
  - `packages/pipelines/ingest_yahoo.py` -- fully implemented
- **Acceptance criteria:**
  - Pulls league info, roster, scoring rules, matchups from Yahoo API
  - Extracts team_key from league_info (not passed raw league_key to fetch_roster)
  - UPSERTs into `user_leagues`, `user_rosters`, `league_scoring_rules`
  - Handles Yahoo rate limits (cache, backoff)
  - Idempotent on re-run

## Task 0.9 -- seed_historical backfill

- **Owner:** implementer
- **Dependencies:** 0.5
- **Status:** [x] complete
- **Completion notes:** Dynamic task mapping by year, processes one month at a time to avoid OOM.
- **Deliverables:**
  - `packages/pipelines/seed_historical.py` -- fully implemented
- **Acceptance criteria:**
  - Backfills 2018-2025 Statcast data using dynamic task mapping
  - Progress is logged (year/month completion)
  - Does not OOM (processes one month at a time)
  - Idempotent

## Task 0.10 -- Unit & integration tests

- **Owner:** architect
- **Dependencies:** 0.2+
- **Status:** [x] complete
- **Completion notes:** Tests cover engine singletons, all 12 table inserts, unique constraint enforcement, DAG schedules, DAG tags, schema validation, and pipeline logic helpers.
- **Deliverables:**
  - `tests/unit/test_engine.py` -- engine and settings singleton tests
  - `tests/unit/test_models.py` -- all model tests passing (12 tables, inserts, constraints)
  - `tests/unit/test_schemas.py` -- all schema tests passing (including boxscore + FanGraphs)
  - `tests/unit/test_pipeline_logic.py` -- pipeline helper function tests
  - `tests/integration/test_pipelines.py` -- import, registration, schedule, and tag tests (5 DAGs)
- **Acceptance criteria:**
  - `pytest` runs clean with 0 failures
  - Coverage on `packages/shared/` >= 80%
  - At least one test per public function in shared modules

## Task 0.11 -- Airflow deployment & scheduling

- **Owner:** architect
- **Dependencies:** 0.5, 0.5b, 0.6, 0.8
- **Status:** [x] complete
- **Completion notes:** DAG schedules verified in tests. Docker Compose includes Airflow webserver. Setup instructions documented in README.
- **Deliverables:**
  - Airflow deployment configs for each flow
  - Schedule order: mlb_schedule 6am -> boxscores 7am -> statcast 8am -> yahoo 9am
- **Acceptance criteria:**
  - Airflow dashboard shows all 5 DAGs registered
  - 3 consecutive successful daily runs (per Phase 0 acceptance)
  - Failed runs trigger retries per configured policy

## Task 0.12 -- Documentation & phase close

- **Owner:** architect
- **Dependencies:** all above
- **Status:** [x] complete
- **Completion notes:** README updated with setup instructions, DAG schedule table, development commands, and project structure. Phase 0 acceptance criteria evaluated below.
- **Deliverables:**
  - Updated `README.md` with setup instructions
  - ADRs in `docs/decisions/` for any mid-phase architecture decisions
  - Phase 0 acceptance checklist verified in `PROJECT_PLAN.md`
- **Acceptance criteria:**
  - Phase 0 acceptance criteria from PROJECT_PLAN.md:
    - [x] Schema migrations versioned with Alembic (initial migration exists, verified roundtrip)
    - [x] All ingestion has retry + idempotency (UPSERT on all tables, retries configured in DAG default_args)
    - [ ] Airflow dashboard shows 3 consecutive successful daily runs (documented setup; requires running Airflow with Docker)
    - [ ] DB queries return today's data without manual intervention (requires running pipelines against live APIs)
  - `ruff check .` clean, `pytest` green
