# Fantasy Baseball Diamond Copilot

An AI copilot for fantasy baseball: data-driven roster, trade, and playoff decisions powered by Statcast pitch-level data, MLB schedule feeds, and Yahoo Fantasy league integration.

## Prerequisites

- **Python 3.11+** (use `pyenv` to manage versions)
- **Docker Desktop** (for local Postgres, Airflow, and supporting services)
- **Node.js 18+** (for Claude Code tooling)

## Quick Start

### 1. Clone the repo

```bash
git clone <repo-url> fantasy-baseball-diamond-copilot
cd fantasy-baseball-diamond-copilot
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in:
#   DATABASE_URL          - Postgres connection string
#   YAHOO_CLIENT_ID       - Yahoo Developer App credentials
#   YAHOO_CLIENT_SECRET   - Yahoo Developer App credentials
```

### 3. Install Python dependencies

```bash
pip install -e ".[dev]"
```

### 4. Start local services

```bash
docker compose -f infra/docker/docker-compose.yml up -d
```

This starts:
- **Postgres** on port 5432
- **Adminer** (DB UI) on port 8080
- **Airflow** webserver on port 8081

### 5. Run database migrations

```bash
alembic upgrade head
```

### 6. Trigger DAGs

Open the Airflow UI at [http://localhost:8081](http://localhost:8081) and enable the ingestion DAGs. They will run on their configured schedules, or you can trigger them manually.

## DAG Schedule

| DAG | Schedule (UTC) | Description |
|-----|---------------|-------------|
| `ingest_mlb_schedule` | `0 6 * * *` (6:00 AM) | MLB schedule, probable pitchers, player metadata |
| `ingest_boxscores` | `0 7 * * *` (7:00 AM) | Box score batting/pitching lines for completed games |
| `ingest_statcast` | `0 8 * * *` (8:00 AM) | Statcast pitch-level data, aggregated to daily stats |
| `ingest_yahoo` | `0 9 * * *` (9:00 AM) | Yahoo Fantasy league rosters, scoring rules, matchups |
| `seed_historical` | Manual only | One-time backfill of 2018-2025 Statcast data |

DAGs run in dependency order: schedule first, then boxscores, then statcast, then yahoo. All ingestion is idempotent via UPSERT on natural keys.

## Development

### Linting and formatting

```bash
ruff check .          # lint
ruff check . --fix    # lint with auto-fix
ruff format .         # format
```

### Running tests

```bash
pytest                # run all tests
pytest -v --tb=short  # verbose with short tracebacks
pytest tests/unit/    # unit tests only
```

### Database migrations

```bash
alembic revision --autogenerate -m "description"  # create migration
alembic upgrade head                               # apply migrations
alembic downgrade -1                               # rollback one step
```

## Project Structure

```
fantasy-baseball-diamond-copilot/
├── packages/
│   ├── shared/           # Shared code: DB models, schemas, config
│   │   ├── db/           # SQLAlchemy models, engine, Alembic env
│   │   ├── schemas/      # Pydantic validation schemas
│   │   ├── config.py     # pydantic-settings configuration
│   │   ├── constants.py  # Shared constants (column maps, etc.)
│   │   └── types.py      # Branded NewType definitions
│   ├── pipelines/        # Airflow DAGs for data ingestion
│   └── ml/               # Machine learning (Phase 1+)
│       ├── features/     # Feature engineering
│       ├── models/       # Model definitions
│       ├── training/     # Training scripts
│       └── evaluation/   # Validation and calibration
├── infra/
│   ├── docker/           # Docker Compose for local services
│   └── migrations/       # Alembic migration versions
├── tests/
│   ├── unit/             # Unit tests (models, schemas, engine)
│   └── integration/      # Integration tests (DAG registration, pipelines)
├── notebooks/            # Exploratory analysis (Phase 1+)
├── docs/                 # Architecture decisions, task breakdowns
├── alembic.ini           # Alembic configuration
├── pyproject.toml        # Python project metadata and dependencies
├── CLAUDE.md             # Claude Code operating manual
└── PROJECT_PLAN.md       # Full project plan (Phase 0-6)
```
