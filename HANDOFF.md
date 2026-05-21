# Handoff to Cowork — Conversation Context

> This document captures the decisions made during the planning conversation that produced the rest of this starter kit. Read it first so you understand the "why" behind the existing files.

## Project identity

**Name**: Fantasy Baseball Diamond Copilot
**One-liner**: AI copilot for fantasy baseball — data-driven roster, trade, and playoff decisions, with a Claude-powered conversational interface.
**Purpose**: Portfolio project for Data Science + ML Engineering interview preparation.

## Who built what so far

The planning conversation happened in Claude Chat and produced:
- `PROJECT_PLAN.md` — full Phase 0 through Phase 6 spec (read this first)
- `CLAUDE.md` — operating manual for Claude Code
- `.claude/agents/` — 5 subagent definitions (architect, feature-engineer, modeler, evaluator, critic)
- `README.md` — setup guide

These files are **complete and authoritative**. Do not rewrite them; extend them.

## Key decisions already locked in (do not re-debate without reason)

### Strategic
- **Target roles**: Data Science + ML Engineering (not pure analytics, not AI research)
- **Public vs personal**: Public-ready architecture (multi-tenant, OAuth), personal-scale operation (invite-only beta)
- **Cloud budget**: Free tier majority, **one piece (model serving) on AWS ECS** for real cloud experience. Target ~$0-25/month total.

### Architecture
- **Multi-agent at two layers**:
  - Dev-time: 5 subagents (already defined in `.claude/agents/`)
  - Runtime: Phase 6 chatbot is a multi-agent system (Orchestrator, Stats Analyst, Coach, Scout, Critic agents)
- **Stack**: Python 3.11+, FastAPI, Postgres (Supabase), Airflow, MLflow, LightGBM, Next.js, Anthropic Claude API
- **Repo structure**: monorepo with `apps/`, `packages/`, `notebooks/`, `infra/` (see PROJECT_PLAN.md §6)

### Data
- **Sources**: pybaseball (Statcast/Fangraphs/Baseball Reference), MLB Stats API, Yahoo Fantasy API
- All free; no paid APIs needed
- Yahoo API uses OAuth via `yahoo_fantasy_api` Python library

### Phase 1 modeling — critical decisions

The Phase 1 design conversation locked in these choices BEFORE any code is written. The Evaluator subagent will encode them into `packages/ml/evaluation/validation_protocol.md`.

**Uncertainty quantification — chosen approach: CQR + ACI**
- **CQR** = Conformalized Quantile Regression (Romano et al. 2019)
- **ACI** = Adaptive Conformal Inference (Gibbs & Candès 2021)
- Why this combination:
  - Quantile regression captures heteroscedasticity (player-specific interval widths)
  - Conformal calibration provides coverage guarantees
  - ACI handles non-stationarity (league changes over seasons)
- Library: `MAPIE`
- Bake-off comparison set:
  1. Ridge regression (point estimate baseline — floor)
  2. LightGBM quantile (heteroscedastic but no coverage guarantee — middle)
  3. CQR + ACI with LightGBM base (main model — ceiling)

**Data splits**
| Split | Date range | Use |
|---|---|---|
| Train | 2019-01-01 to 2023-12-31 | Model fitting, hyperparameter tuning via walk-forward CV |
| Validation | 2024-01-01 to 2024-12-31 | Architecture comparison, final hyperparameter lock |
| Test | 2025-01-01 to 2025-12-31 | Final evaluation — touched ONCE |
| Live | 2026-01-01 onward | Phase 4+ production monitoring |

- 2020 COVID season (60 games): include in training but flag, exclude from per-season aggregate metrics
- All splits are strictly temporal — no shuffles

**Target variable**
- Primary unit: **per-game fantasy points** (under user's league scoring rules)
- Also evaluated: per-week aggregates (sum of next 6-7 games) — matches actual fantasy decision-making
- Model fits per-game; per-week is a derived aggregate at inference

**Population**
- **Hitters and pitchers modeled separately** (different features, different cadence: pitchers ~every 5 days, hitters ~daily)
- Eligibility for test-set evaluation: ≥100 plate appearances OR ≥30 innings pitched within the test season (filters noise from cup-of-coffee players)
- Rookies / low-sample players: include in training, report as a separate slice in metrics

**Metrics** (Evaluator should expand in `validation_protocol.md`)
- Primary: RMSE
- Secondary: MAE, Spearman rank correlation, interval coverage (target 80%, acceptable 75-85%), sharpness (interval width)
- Slice metrics: hitters vs pitchers, by position, by playing time, by veteran status

**Baselines that must be beaten**
- Naive last-game (predict tonight = last night's actual)
- Naive trailing-7-day mean
- Season-to-date mean
- **Threshold**: main model must beat best baseline by ≥10% RMSE

### What's NOT yet locked in (Cowork should produce these next)

1. **`packages/ml/evaluation/validation_protocol.md`** — full formal protocol document. Outline exists in the conversation; needs to be written as the pre-registered spec.
2. **`docs/feature_plan.md`** — feature family prioritization (P0/P1/P2). Conversation identified these families:
   - Rolling production (7/14/30 day): AVG, OBP, SLG, wOBA, xwOBA
   - Plate discipline: BB%, K%, swing/contact rates
   - Quality of contact: barrel %, hard-hit %, exit velocity
   - Context: opponent SP/RP quality, ballpark factor, home/away, days rest, weather
   - Lineup: batting order slot, projected lineup confirmation
   - Platoon splits: vs LHP / vs RHP
   - Recent trends: hot/cold z-scores
   - Pitcher-specific: K/9, BB/9, HR/9, GB%, FIP, velocity trend
3. **`docs/model_bakeoff.md`** — formalize the 3-model comparison: Ridge / LightGBM quantile / CQR+ACI

After those three docs are in place, the planning is complete and Phase 0 (data layer) can begin in Claude Code.

## Suggested first task in Cowork

```
Read HANDOFF.md, then PROJECT_PLAN.md, then CLAUDE.md, in that order.
Confirm you understand the locked-in decisions.

Then write packages/ml/evaluation/validation_protocol.md as a pre-registered
specification using the decisions in HANDOFF.md §"Phase 1 modeling".

Required sections in the protocol:
1. Objective
2. Data Splits (with the table from HANDOFF.md)
3. Walk-forward CV strategy (5 expanding-window folds within train+val)
4. Calibration set strategy for CQR (last 20% of training window, temporal)
5. Target variable definition (per-game, with per-week aggregation rule)
6. Population eligibility rules
7. Metrics (primary, secondary, slice breakdowns)
8. Baselines that must be beaten + 10% RMSE threshold
9. Calibration checks (reliability diagram, empirical coverage)
10. Disqualifying conditions (leakage indicators, test-set tuning)
11. Test-set discipline (touched once, multiple-comparison penalty for any re-evaluation)

Style: precise, terse, pre-registration tone. After writing it, do not
modify it without explicit user approval.

Do not start any modeling code yet. The next two documents
(feature_plan.md and model_bakeoff.md) come after this one.
```

## After validation_protocol.md is committed

Next two tasks for Cowork (sequential):

1. `docs/feature_plan.md` — feature family prioritization with P0/P1/P2 tags
2. `docs/model_bakeoff.md` — formal bake-off plan for the 3 models

Once all three documents exist and are reviewed, switch from **Cowork tab → Code tab** to begin Phase 0 (data infrastructure) per `PROJECT_PLAN.md` §7.

## Workflow norms

- Cowork should ask before modifying any of the existing locked-in files (CLAUDE.md, PROJECT_PLAN.md, the agent files).
- New files Cowork creates can be edited freely in Cowork; once Phase 0 starts in Claude Code, file ownership shifts to the subagents per `CLAUDE.md`.
- Commit after each new document is finalized. Commit message format: `docs: add validation protocol` etc.
