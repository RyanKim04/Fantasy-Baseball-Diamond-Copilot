# Fantasy Baseball Diamond Copilot — Project Plan

> **An AI copilot for fantasy baseball: data-driven roster, trade, and playoff decisions, with a conversational LLM interface.**

This is a ship-ready engineering plan. It's optimized for two priorities:

1. **Feasibility** — every data source and tool is confirmed accessible (free where possible) for a solo developer.
2. **Skill display for DS + MLE roles** — every phase explicitly maps to interview-grade skills.

---

## 1. Mission & Non-Goals

### Mission
Build a multi-tenant web platform that ingests fantasy baseball data, generates probabilistic player projections, recommends optimal roster decisions, simulates playoff outcomes, and exposes all of it through a Claude-powered chat interface.

### Non-Goals (explicitly out of scope)
- Daily Fantasy Sports (DFS) — different game, different data.
- Real-money betting / odds integration.
- Mobile native apps.
- Commercialization (Yahoo API ToS forbids without partnership).
- Sports other than MLB.

---

## 2. Target Outcomes

By project end, the portfolio should demonstrate, with code on GitHub and a live URL:

| Skill area | Concrete artifact |
|---|---|
| **Data engineering** | Daily ETL pipeline (Prefect), Postgres schema, idempotent ingestion |
| **DS / ML modeling** | Time-series-aware feature engineering, gradient-boosted models with quantile regression for uncertainty, MLflow experiment tracking |
| **Statistical thinking** | Probabilistic projections, Monte Carlo simulation, prediction intervals, drift detection |
| **MLE / Infra** | Dockerized FastAPI model serving, CI/CD, AWS deployment for at least one service, monitoring + drift dashboards |
| **GenAI / LLM** | RAG pipeline, Anthropic API tool use / function calling, conversational memory |
| **Full-stack** | Next.js frontend with auth, BI dashboards, mobile-responsive |
| **Product thinking** | Explainable recommendations grounded in sabermetrics, league-rule-aware scoring |

---

## 3. Data Sources & Feasibility Audit

All data needed for the project is **free and accessible**. No paid APIs required.

| Source | What we get | Access | Risk |
|---|---|---|---|
| **pybaseball** (Python lib) | Statcast pitch-level (2008+), Fangraphs season stats, Baseball Reference historical | Free, no auth | Low — well-maintained, scrapes for us |
| **MLB Stats API** (official) | Live scores, schedules, rosters, basic stats | Free, no auth | Low — official, stable |
| **Yahoo Fantasy Sports API** | User's league: roster, scoring rules, matchups, transactions, free agents | OAuth 2.0, per-user | **Medium** — docs are poor, rate-limited, requires registered Yahoo app |
| **News / Injuries** | Player news, injury reports | Scrape ESPN/RotoWire RSS, or use free news APIs | Low — RSS is stable |

### Yahoo OAuth — the only real risk
The Yahoo Fantasy API is the trickiest piece. Mitigations:
- Use a well-maintained library: `yahoo_fantasy_api` or `yfpy` (Python).
- Start with single-user (your own league) in Phase 0; multi-tenant in Phase 5.
- Cache aggressively — Yahoo data changes slowly (rosters, scoring don't change mid-day).

---

## 4. Tech Stack (Final Decisions)

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ (backend/ML) + TypeScript (frontend) | Standard |
| Data ingestion | **Prefect 3** | Better DX than Airflow, generous free cloud tier |
| Database | **Postgres on Supabase** | Free tier, includes auth + storage |
| ML libs | **LightGBM, scikit-learn, statsmodels, scipy** | Tabular data, fast, well-supported |
| Experiment tracking | **MLflow** (self-hosted on Render OR DagsHub free) | Industry standard for MLE |
| Model serving | **FastAPI** + **Docker** | Standard MLE stack |
| LLM | **Anthropic Claude API** (Sonnet 4) with tool use | Best function calling, pay-per-use |
| RAG / vector DB | **pgvector** (Postgres extension on Supabase) | No separate service needed |
| Frontend | **Next.js 14 (App Router) + TypeScript + Tailwind + shadcn/ui** | Modern, easy to deploy |
| Charts | **Recharts** for app charts, **Plotly** for analytics dashboards | Two tools for two purposes |
| Auth | **NextAuth.js** with Yahoo provider | Handles OAuth |
| Hosting (free tier) | **Vercel** (web), **Render** (API + MLflow), **Supabase** (DB) | All free |
| Hosting (AWS piece) | **AWS ECS Fargate** for one model-serving endpoint | Demonstrates real cloud |
| Monitoring | **Sentry** (errors, free tier), custom **Grafana Cloud** dashboard for model metrics | Real MLE concern |
| CI/CD | **GitHub Actions** | Free for public repos |

### Cloud budget estimate
- First 6 months: **~$0/month** (all free tiers, AWS Free Tier covers ECS)
- After AWS Free Tier expires: **~$10-20/month**
- LLM API: **~$5-15/month** depending on chatbot usage

---

## 5. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Next.js Frontend (Vercel)               │
│   - Roster dashboard  - Projections  - Trade analyzer       │
│   - Chatbot UI        - BI dashboards                       │
└─────────────────────────────────────────────────────────────┘
                         │ HTTPS
                         ▼
┌─────────────────────────────────────────────────────────────┐
│         FastAPI Backend (Render + 1 endpoint on ECS)        │
│   - /predict   - /optimize-lineup  - /trade-analyze         │
│   - /simulate  - /chat (LLM agent w/ tools)                 │
└─────────────────────────────────────────────────────────────┘
         │                │                  │
         ▼                ▼                  ▼
   ┌──────────┐   ┌─────────────┐   ┌──────────────────┐
   │ Postgres │   │   MLflow    │   │  Anthropic API   │
   │(Supabase)│   │ (model reg) │   │  (Claude Sonnet) │
   └──────────┘   └─────────────┘   └──────────────────┘
         ▲
         │ daily
         │
┌─────────────────────────────────────────────────────────────┐
│              Prefect Pipelines (scheduled)                  │
│   - pybaseball ingestion  - Yahoo league sync               │
│   - feature engineering   - nightly retraining              │
└─────────────────────────────────────────────────────────────┘
```

---

## 6. Repository Structure

Monorepo for simplicity:

```
fantasy-baseball-diamond-copilot/
├── README.md                    # Project overview, demo gif, architecture diagram
├── ARCHITECTURE.md              # Detailed system design
├── docs/
│   ├── decisions/               # ADRs (Architecture Decision Records)
│   └── api.md
├── apps/
│   ├── api/                     # FastAPI service
│   │   ├── routers/
│   │   ├── models/              # Pydantic schemas
│   │   ├── services/
│   │   ├── chatbot/             # LLM agent + tools
│   │   └── Dockerfile
│   └── web/                     # Next.js frontend
│       ├── app/
│       ├── components/
│       └── lib/
├── packages/
│   ├── ml/                      # Training code (importable)
│   │   ├── features/
│   │   ├── models/
│   │   ├── training/
│   │   └── evaluation/
│   ├── pipelines/               # Prefect flows
│   │   ├── ingest_statcast.py
│   │   ├── ingest_yahoo.py
│   │   └── retrain_weekly.py
│   └── shared/                  # Types, constants
├── notebooks/                   # EDA, model experiments (committed, version-pinned)
├── tests/
│   ├── unit/
│   └── integration/
├── infra/
│   ├── docker-compose.yml       # Local dev
│   ├── terraform/               # AWS ECS for the one serving endpoint
│   └── github-actions/
└── pyproject.toml
```

---

## 7. Phased Roadmap

Each phase has: **Goal · Skills shown · Deliverables · Acceptance criteria · Est. time**.

### Phase 0 — Foundations & Data Layer
**Time:** 1.5 weeks
**Goal:** Repeatable, automated ingestion of all data we need.
**DS/MLE skills shown:** Data engineering, schema design, idempotent ETL, OAuth integration.

**Deliverables:**
- Postgres schema: `players`, `games`, `pitches`, `batting_stats_daily`, `pitching_stats_daily`, `user_leagues`, `user_rosters`, `league_scoring_rules`, `predictions`, `news_articles`.
- Prefect flow `ingest_statcast.py`: pulls last 2 days of Statcast nightly.
- Prefect flow `ingest_mlb_schedule.py`: pulls upcoming games + lineups.
- Yahoo OAuth flow working for one user (committed but secret).
- `ingest_yahoo.py`: pulls roster, scoring rules, matchups for connected user.
- `seed_historical.py`: one-time backfill of 2018-2025 seasons.

**Acceptance:**
- ✅ Prefect dashboard shows 3 consecutive successful daily runs.
- ✅ DB queries return today's data without manual intervention.
- ✅ Schema migrations versioned with Alembic.
- ✅ All ingestion has retry + idempotency (re-running doesn't duplicate).

---

### Phase 1 — Player Projection Model (MVP)
**Time:** 3 weeks
**Goal:** Predict next-week fantasy points per player, with calibrated uncertainty.
**DS/MLE skills shown:** Feature engineering, time-series validation, gradient boosting, quantile regression, MLflow tracking, model evaluation.

**Deliverables:**
- `packages/ml/features/`: feature builders for rolling averages (7/14/30 day), opponent strength, ballpark factors, days rest, batting order, platoon splits, recent BABIP / xwOBA trends.
- `packages/ml/models/projection.py`: LightGBM regressor for expected points + two quantile regressors (10th, 90th percentile) for prediction interval.
- `packages/ml/training/train.py`: temporal train/val/test split (no leakage), with walk-forward CV for hyperparameter tuning.
- `packages/ml/evaluation/`: RMSE, MAE, **interval coverage** (does the 80% PI actually contain truth 80% of the time?), calibration plot.
- MLflow tracking with at least 5 logged experiments comparing approaches.
- A `notebooks/01_baseline_vs_lgbm.ipynb` documenting the modeling story.

**Acceptance:**
- ✅ Model beats two baselines: (a) last-week's actual points (b) season average. Beat by ≥10% RMSE on held-out games.
- ✅ 80% prediction intervals achieve 75-85% empirical coverage.
- ✅ MLflow has logged ≥5 runs with hyperparameters, metrics, model artifacts.
- ✅ `predict()` API: input player_id + date, output `{mean, p10, p90}`.

**Critical: how this maps to league scoring**
Don't predict raw stats. Predict the **fantasy points under the user's scoring rules**. That requires pulling scoring rules from Yahoo and computing fantasy points as the target variable.

---

### Phase 2 — Decision-Support Tools
**Time:** 3 weeks
**Goal:** Turn projections into actionable recommendations.
**DS/MLE skills shown:** Optimization, expected value reasoning, productization of models.

**Deliverables:**
- `/api/optimize-lineup`: given roster + position constraints, solve for the optimal starting lineup. Use simple ILP via `pulp`. Return top 3 alternatives.
- `/api/trade-analyze`: given two sets of players, compute **expected fantasy points differential** over rest of season (ROS), with confidence interval. Use Phase 3's simulation under the hood (or simpler weighted ROS projection initially).
- `/api/waiver-recommend`: rank free agents by ROS value above roster's worst player at the same position.
- Each endpoint returns **explanations**: top 3 reasons behind each recommendation (e.g., "Trout's xwOBA is 0.420 over last 14 days, vs season .350").

**Acceptance:**
- ✅ Each endpoint responds in <2s for a typical roster (12 players).
- ✅ Recommendations include reasoning, not just numbers.
- ✅ Unit tests cover edge cases (injured players, position eligibility).

---

### Phase 3 — Monte Carlo Season Simulation
**Time:** 2 weeks
**Goal:** Probabilistic forecasts of playoff odds and final standings.
**DS/MLE skills shown:** Probabilistic modeling, simulation, Bayesian thinking.

**Deliverables:**
- `packages/ml/simulation/season_sim.py`: simulate remaining games of the season N=10,000 times. Each player's weekly performance sampled from their predictive distribution.
- Aggregate to: playoff probability per team, expected final rank distribution, "championship odds".
- Endpoint `/api/simulate-season` returns simulation results, cached for 24h.

**Acceptance:**
- ✅ 10,000 sims complete in <30 seconds (use vectorized numpy or numba if needed).
- ✅ Results stable: re-running gives playoff probabilities within ±2pp.
- ✅ Sanity check: leading team has higher playoff odds than trailing teams (matches intuition).

---

### Phase 4 — Production ML Infrastructure
**Time:** 2.5 weeks
**Goal:** Real MLE-grade serving and monitoring.
**DS/MLE skills shown:** Docker, REST API, model versioning, monitoring, drift detection, CI/CD, cloud deployment.

**Deliverables:**
- All services Dockerized; `docker-compose.yml` for local dev.
- FastAPI app deployed to **Render** for most endpoints.
- **One model-serving endpoint deployed to AWS ECS Fargate** (the projection model). Documented in `infra/terraform/`.
- Model registry via MLflow: registered models with stages (Staging, Production), versioned.
- **Drift detection job**: weekly, compares last week's predictions vs actuals, computes drift metrics (PSI, KS statistic on residual distribution). Alerts via Sentry if drift detected.
- **Prediction logging**: every prediction stored in DB with model version. Enables backtesting.
- GitHub Actions: on PR — lint, type check, unit tests. On merge to main — build & push Docker images, run integration tests, deploy.

**Acceptance:**
- ✅ Live URLs work; 99%+ uptime over a 7-day stretch.
- ✅ Drift dashboard exists and is screenshot-able.
- ✅ CI pipeline green on main; failed PR is blocked from merging.
- ✅ Can roll back to previous model version via MLflow with one command.

**Why ECS for one endpoint?** Cost-effective, but more importantly: it lets you say "I have AWS deployment experience" with code to prove it. SageMaker is cleaner but more expensive.

---

### Phase 5 — Frontend & BI Dashboards
**Time:** 3 weeks
**Goal:** A polished web app where projections, recommendations, and analytics are visible.
**DS/MLE skills shown:** Data viz, UX, full-stack.

**Deliverables:**
- Next.js app with pages: Landing, My Team, Projections, Trade Analyzer, Playoff Odds, Chat.
- NextAuth Yahoo OAuth — users can log in with their Yahoo Fantasy account.
- Multi-tenant: every query filtered by `user_id`. Row-level security via Supabase.
- **BI dashboard page**: Plotly charts for team trends, player heat maps, week-over-week performance.
- Mobile-responsive (Tailwind + shadcn handles most of this).
- Deployed to Vercel.

**Acceptance:**
- ✅ Live URL with public access; anyone can sign in with Yahoo and see their own data.
- ✅ Lighthouse score ≥80 on mobile.
- ✅ At least 3 different chart types used meaningfully (not just for show).
- ✅ Loading states + error states handled.

---

### Phase 6 — LLM Chat Interface
**Time:** 2.5 weeks
**Goal:** Conversational interface that uses the rest of the platform via tool calls.
**DS/MLE skills shown:** RAG, agentic LLM design, function calling, prompt engineering, evaluation of LLM systems.

**Deliverables:**
- `apps/api/chatbot/tools.py`: Anthropic-format tool definitions wrapping existing endpoints:
  - `get_player_projection(player_name, week)`
  - `analyze_trade(give, receive)`
  - `get_my_roster()`
  - `simulate_playoffs()`
  - `search_news(query)` ← RAG over scraped news articles
- RAG pipeline: scrape ESPN/RotoWire RSS daily → chunk → embed with `text-embedding-3-small` or open-source model → store in pgvector → retrieve top-k on query.
- Conversation memory per user (stored in Postgres).
- System prompt that grounds Claude in sabermetric reasoning style.
- **Eval suite**: 30 hand-curated questions with expected behaviors. Run via `promptfoo` or custom eval script. Track pass rate over prompt revisions.

**Acceptance:**
- ✅ Chatbot can answer: "Should I start X or Y this Friday?" with a recommendation backed by a tool call to `get_player_projection`.
- ✅ Chatbot uses RAG when asked about recent player news.
- ✅ Eval suite passes ≥80% of 30 test cases.
- ✅ Tool-call traces are logged and inspectable.

---

## 8. Minimum Viable Portfolio (cut version)

If timeline pressure forces a cut, ship in this order. Each cut still produces a coherent portfolio:

| Cut level | Includes | Time | Still hireable for |
|---|---|---|---|
| **Tier 1 (must)** | Phases 0, 1, 4 | ~7 weeks | DS / MLE — proves end-to-end ML system |
| **Tier 2 (strong)** | + Phase 2, 5 | ~13 weeks | DS / MLE / AI Eng — full product |
| **Tier 3 (full)** | + Phase 3, 6 | ~17 weeks | Senior DS / MLE / GenAI roles |

Recommendation: aim for Tier 2, treat Tier 3 as bonus.

---

## 9. Skills Showcase Matrix (for resume bullets)

After completion, you can credibly write:

- *"Built end-to-end ML platform predicting fantasy baseball performance: ingestion (Prefect), training (LightGBM with quantile regression, MLflow), serving (FastAPI on AWS ECS), monitoring (custom drift detection)."*
- *"Designed temporal cross-validation pipeline and uncertainty-calibrated predictions, achieving 80% empirical PI coverage."*
- *"Built RAG-powered conversational agent using Anthropic Claude with function calling over custom-built prediction APIs."*
- *"Deployed multi-tenant Next.js + FastAPI app with OAuth, Postgres + pgvector, and CI/CD via GitHub Actions."*

---

## 10. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Yahoo OAuth complexity blocks Phase 0 | Use `yahoo_fantasy_api` Python lib; budget extra 3 days; have fallback of manual CSV upload of roster |
| Model isn't better than baselines | Better feature engineering (ballpark, opponent SP quality); ensemble; if still bad, document the negative result honestly — that's also a DS skill |
| Scope creep | Hard cut: each phase has a "done" definition. Do not start Phase N+1 until Phase N's acceptance criteria are checked |
| LLM costs balloon | Cache common questions; use Claude Haiku for cheap calls; set monthly budget alert |
| Yahoo rate limits | Cache aggressively; don't re-fetch roster more than daily |

---

## 11. Definition of "Project Done"

The project is shippable to a recruiter when:

1. ✅ Live URL works on phone and desktop.
2. ✅ A stranger can sign in with Yahoo and get personalized projections.
3. ✅ GitHub repo has clear README with architecture diagram + demo GIF.
4. ✅ At least one model-serving endpoint runs on AWS, not just free PaaS.
5. ✅ Code quality: type hints, tests on critical paths, linter passing.
6. ✅ One blog post or ARCHITECTURE.md explaining the system.

---

## 12. Development Methodology — Multi-Agent Workflow

This project is built using **Claude Code's multi-agent (subagent) pattern**. Five specialized subagents collaborate, each with bounded scope and explicit tool restrictions. The agents live in `.claude/agents/` and load automatically when Claude Code starts in this repo.

### The five agents

| Agent | Scope | When invoked |
|---|---|---|
| `architect` | Directory structure, interface contracts, integration | Start and end of every phase |
| `feature-engineer` | `packages/ml/features/` only | Any feature engineering task |
| `modeler` | `packages/ml/models/`, `training/` | Any model training task |
| `evaluator` | `packages/ml/evaluation/` | Validation protocol design (*before* modeling), final eval |
| `critic` | Read-only across whole repo | After every major work unit |

### The critical workflow rule

**The Evaluator commits the validation protocol before the Modeler writes any training code.** This prevents the most common DS failure mode: tuning against the test set. The Modeler's prompt explicitly checks for the existence of `packages/ml/evaluation/validation_protocol.md` and refuses to start training if it's missing.

### Per-phase workflow

```
1. architect          → directory + interface contracts + task breakdown
2. evaluator          → validation protocol committed BEFORE step 3
3. feature-engineer ∥ modeler  → parallel work in bounded scopes
4. evaluator          → run final evaluation on test set (once)
5. critic             → adversarial review with severity-ranked findings
6. architect          → integrate, verify acceptance criteria, close phase
```

### Why this is also a portfolio strength

The workflow itself becomes a story for interviews:

- *"I designed a multi-agent dev workflow where the evaluator pre-registers the validation protocol before any modeling happens, preventing test-set contamination."*
- *"Specialized agents with bounded tool access enforced separation of concerns at the AI-tooling level — features can't accidentally peek at the model, the modeler can't accidentally rewrite evaluation."*

This is rare for portfolio projects and signals modern AI-engineering maturity.

## 13. Suggested Working Cadence with Claude Code

Work **one phase at a time**:

1. *"Start Phase N. Use the architect subagent to define interfaces and task breakdown."*
2. *"Use the evaluator subagent to draft the validation protocol."* (For any modeling phase.)
3. Work sub-tasks with the relevant specialist agents.
4. *"Use the critic subagent to review."*
5. *"Use the architect subagent to verify acceptance criteria and close the phase."*

Do not let Claude Code run ahead into the next phase without explicit go-ahead.
