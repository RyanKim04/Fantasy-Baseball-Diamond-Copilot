# CI/CD Pipeline

## Overview

GitHub Actions workflow triggered on every push to `main` or `phase/*` branches, and on pull requests to `main`.

## Pipeline Diagram

```
Push to main or phase/* branch (or PR to main)
          │
          ▼
┌─────────────────────────────────────────────────────────┐
│  1. LINT & FORMAT                                       │
│  ─────────────────                                      │
│  • ruff check .          (catch code errors)            │
│  • ruff format --check . (enforce consistent style)     │
│                                                         │
│  Runs in: ~15 seconds                                   │
└────────────────────────┬────────────────────────────────┘
                         │ pass
                    ┌────┴────┐
                    ▼         ▼
┌───────────────────────┐ ┌───────────────────────────────┐
│  2. UNIT TESTS        │ │  3. INTEGRATION TESTS         │
│  ──────────────       │ │  ───────────────────          │
│  • pytest tests/unit/ │ │  • Spins up Postgres 16       │
│  • Uses SQLite        │ │    service container          │
│    (in-memory, fast)  │ │  • Runs Alembic migrations    │
│  • Tests models,      │ │  • pytest tests/integration/  │
│    schemas, types     │ │  • Tests pipeline imports,    │
│                       │ │    DAG registration, DB ops   │
│  Runs in: ~30 seconds │ │                               │
│                       │ │  Runs in: ~2 minutes          │
└──────────┬────────────┘ └──────────────┬────────────────┘
           │                             │
           └──────────┬──────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────┐
│  4. DOCKER BUILD                                        │
│  ────────────────                                       │
│  • docker build -t diamond-copilot:<sha>                │
│  • Verifies the image builds successfully               │
│  • Does NOT push to any registry (yet)                  │
│                                                         │
│  Runs in: ~2-3 minutes                                  │
│  Note: continue-on-error until Dockerfile is created    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼ (Phase 4 — currently commented out)
┌─────────────────────────────────────────────────────────┐
│  5. DEPLOY TO AWS (dormant)                             │
│  ──────────────────────────                             │
│  • Only triggers on push to main (not PRs)              │
│  • Authenticates with AWS via GitHub Secrets             │
│  • Builds & pushes Docker image to Amazon ECR           │
│  • Updates ECS service to deploy new version            │
│                                                         │
│  Requires secrets:                                      │
│    AWS_ACCESS_KEY_ID                                    │
│    AWS_SECRET_ACCESS_KEY                                │
│                                                         │
│  Activate: uncomment deploy job in ci.yml               │
└─────────────────────────────────────────────────────────┘
```

## Concurrency

Duplicate workflow runs on the same branch are automatically cancelled. If you push twice quickly, only the latest push runs.

## Branch Strategy

```
main (protected)
  │
  ├── phase/0-data-layer      ← current
  ├── phase/1-projection      ← future
  ├── phase/2-decision-tools  ← future
  └── ...

PR workflow:
  phase/* branch → PR to main → CI runs → merge when green
```

## Configuration

- Workflow file: `.github/workflows/ci.yml`
- Python version: 3.11
- Postgres version: 16-alpine (in CI service container)
- Dependency caching: pip cache enabled for faster runs
