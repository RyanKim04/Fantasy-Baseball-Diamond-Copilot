# Phase 1 Task Breakdown -- Player Projection Model (MVP)

> Phase task list for Phase 1. Each task is numbered, owner-tagged, and ordered
> to enforce the critical workflow rule: evaluator commits the validation
> protocol BEFORE any modeling starts.

**Branch:** `phase/1-projection`
**Start date:** 2026-06-01
**Target end date:** 2026-06-22 (3 weeks)
**Companion docs:**
- `packages/ml/evaluation/validation_protocol.md` (pre-registered)
- `docs/feature_plan.md` (feature families)
- `docs/model_bakeoff.md` (three-model comparison)
- `packages/shared/schemas/ml.py` (cross-module Pydantic contracts)

---

## Pre-conditions (verified 2026-06-01)

- [x] Phase 0 data layer merged to main (2026-06-01)
- [x] `validation_protocol.md` committed and unmodified since 2026-05-20
- [x] `feature_plan.md` committed with P0/P1/P2 tiers
- [x] `model_bakeoff.md` committed with selection rule
- [x] `packages/shared/schemas/ml.py` defines FeatureRow, PredictionRow,
      PlayerProjection, WalkForwardFold, ScoringRulesSnapshot, PopulationMetrics,
      BaselineComparison
- [x] `packages/shared/scoring.py` implements fantasy points calculator
- [x] DB schema has all required tables: batting_stats_daily, pitching_stats_daily,
      games, park_factors, season_stats_batting, season_stats_pitching, players,
      league_scoring_rules, predictions

---

## Workflow order (per CLAUDE.md)

```
1. architect          -> scaffold, contracts, task breakdown          [CURRENT]
2. evaluator          -> implement evaluation harness (BEFORE modeling)
3. feature-engineer || modeler -> parallel work in bounded scopes
4. evaluator          -> validation-set eval, then single Test touch
5. critic             -> adversarial review
6. architect          -> integrate, verify acceptance criteria, close phase
```

---

## Step 0 -- Architect: scaffold and contracts

| # | Task | Owner | Status |
|---|------|-------|--------|
| 0.1 | Create `phase/1-projection` branch | architect | pending (needs Bash) |
| 0.2 | Audit existing skeletons, confirm interface contracts are complete | architect | done |
| 0.3 | Create test file skeletons under `tests/unit/` for Phase 1 modules | architect | done |
| 0.4 | Write this task breakdown (`docs/phase_1_tasks.md`) | architect | done |
| 0.5 | Create `packages/ml/evaluation/diagnostics.py` stub (width_vs_predicted, residual_vs_predicted) | architect | done |
| 0.6 | Create stubs for missing P0 feature builders (opponent, lineup, workload, prior_season) | architect | done |
| 0.7 | Create stubs for P1 feature builders (plate_discipline, contact_quality, platoon, hot_cold, velocity, catcher_framing) | architect | done |
| 0.8 | Add Phase 1 ML dependencies to pyproject.toml (lightgbm, scikit-learn, mlflow, mapie, optuna, etc.) | architect | done |
| 0.9 | Create `notebooks/01_baseline_vs_lgbm.ipynb` skeleton | architect | done |

---

## Step 1 -- Evaluator: implement evaluation harness

The evaluator implements the metrics, baselines, and report generation code.
This MUST be complete and tested before any model training begins (Step 3).

| # | Task | Owner | Depends on |
|---|------|-------|------------|
| 1.1 | Implement `packages/ml/evaluation/metrics.py`: rmse, mae, spearman_rho, interval_coverage, sharpness, pinball_loss, compute_population_metrics | evaluator | 0.2 |
| 1.2 | Implement `packages/ml/evaluation/baselines.py`: predict_naive_last_game, predict_trailing_7d_mean, predict_season_to_date_mean | evaluator | 0.2 |
| 1.3 | Implement `packages/ml/evaluation/report.py`: generate_report (the single Test touch entry point) | evaluator | 1.1, 1.2 |
| 1.4 | Write unit tests for all evaluation functions in `tests/unit/test_evaluation.py` | evaluator | 1.1, 1.2, 1.3 |
| 1.5 | Implement calibration diagnostic plots in `packages/ml/evaluation/calibration.py` (already stubbed) + `diagnostics.py` (width_vs_predicted, residual_vs_predicted) | evaluator | 1.1 |
| 1.6 | Verify `packages/ml/evaluation/schemas.py` validate_prediction_dataframe works end-to-end with synthetic data | evaluator | 0.2 |

**GATE: Steps 1.1-1.6 must be complete and tests green before Step 3 begins.**

---

## Step 2 -- Feature Engineer: implement feature builders

The feature engineer can work in parallel with Step 1 (evaluator), since
features are independent of evaluation code.

### P0 features (must-have)

| # | Task | Owner | Depends on | Family |
|---|------|-------|------------|--------|
| 2.1 | Implement `packages/ml/features/scoring.py`: compute_target_batting, compute_target_pitching (target variable computation) | feature-engineer | 0.2 | target |
| 2.2 | Implement `packages/ml/features/rolling.py`: RollingProductionBuilder.build() for hitters (7/14/30d) and pitchers (14/30/60d) | feature-engineer | 0.2 | F1 |
| 2.3 | Implement `packages/ml/features/rest.py`: RestRecencyBuilder.build() | feature-engineer | 0.2 | F2 |
| 2.4 | Implement `packages/ml/features/context.py`: ContextBuilder.build() for home/away + park factors | feature-engineer | 0.2 | F3 |
| 2.5 | Implement `packages/ml/features/opponent.py`: OpponentQualityBuilder.build() | feature-engineer | 0.2 | F4 |
| 2.6 | Implement `packages/ml/features/lineup.py`: LineupSlotBuilder.build() (hitters only) | feature-engineer | 0.2 | F5 |
| 2.7 | Implement `packages/ml/features/workload.py`: PitcherWorkloadBuilder.build() (pitchers only) | feature-engineer | 0.2 | F6 |
| 2.8 | Implement `packages/ml/features/prior_season.py`: PriorSeasonBuilder.build() | feature-engineer | 0.2 | F7 |
| 2.9 | Implement `packages/ml/features/pipeline.py`: assemble_features() orchestrator | feature-engineer | 2.1-2.8 | -- |
| 2.10 | Write unit tests for all P0 feature builders in `tests/unit/test_features.py` | feature-engineer | 2.1-2.8 | -- |
| 2.11 | Add leakage guard tests: verify no feature uses game_date or later data | feature-engineer | 2.9 | -- |

### P1 features (stretch)

| # | Task | Owner | Depends on | Family |
|---|------|-------|------------|--------|
| 2.12 | Implement `packages/ml/features/plate_discipline.py`: PlateDisciplineBuilder | feature-engineer | 2.9 | F8 |
| 2.13 | Implement `packages/ml/features/contact_quality.py`: ContactQualityBuilder | feature-engineer | 2.9 | F9 |
| 2.14 | Implement `packages/ml/features/platoon.py`: PlatoonSplitBuilder | feature-engineer | 2.9 | F10 |
| 2.15 | Implement `packages/ml/features/hot_cold.py`: HotColdZScoreBuilder | feature-engineer | 2.2 | F11 |
| 2.16 | Implement `packages/ml/features/velocity.py`: VelocityRepertoireBuilder | feature-engineer | 2.9 | F12 |
| 2.17 | Implement `packages/ml/features/catcher_framing.py`: CatcherFramingBuilder | feature-engineer | 2.9 | F13 |
| 2.18 | Write unit tests for P1 feature builders | feature-engineer | 2.12-2.17 | -- |

**GATE: Steps 2.1-2.11 (P0 features) must be complete and tests green before Step 3 begins.**

---

## Step 3 -- Modeler: implement models and training

The modeler can only start after Gates from Steps 1 and 2-P0 are met.

| # | Task | Owner | Depends on |
|---|------|-------|------------|
| 3.1 | Implement `packages/ml/training/splits.py`: split_by_date, get_walk_forward_folds, compute_calibration_cutoff | modeler | 0.2 |
| 3.2 | Implement `packages/ml/models/ridge.py`: RidgeProjectionModel.fit() and predict() (M1) | modeler | 3.1 |
| 3.3 | Implement `packages/ml/models/lgbm_quantile.py`: LGBMQuantileModel full implementation with Optuna (M2) | modeler | 3.1 |
| 3.4 | Implement `packages/ml/models/cqr_aci.py`: CQRACIModel with MAPIE (M3) | modeler | 3.3 |
| 3.5 | Implement `packages/ml/training/train.py`: train_model() with MLflow logging | modeler | 3.1-3.4 |
| 3.6 | Write unit tests for splits in `tests/unit/test_splits.py` | modeler | 3.1 |
| 3.7 | Write unit tests for each model class in `tests/unit/test_model_candidates.py` | modeler | 3.2-3.4 |
| 3.8 | Implement `packages/ml/models/projection.py`: predict() API (loads from MLflow, returns PlayerProjection) | modeler | 3.5 |
| 3.9 | Run M1 end-to-end on P0 features, log to MLflow as run 1 | modeler | 3.5, 2.9 |
| 3.10 | Run M2 end-to-end on P0 features, log to MLflow as run 2 | modeler | 3.9 |
| 3.11 | Run M3 end-to-end on P0 features, log to MLflow as run 3 | modeler | 3.10 |
| 3.12 | Add P1 features, re-run M2 on P0+P1, log as run 4 | modeler | 3.11, 2.12-2.17 |
| 3.13 | Re-run M3 on P0+P1, log as run 5 | modeler | 3.12 |

**Minimum 5 MLflow runs: M1-P0, M2-P0, M3-P0, M2-P0P1, M3-P0P1.**

---

## Step 4 -- Evaluator: validation-set evaluation (pre-Test)

Before the single Test touch, the evaluator runs on the Validation set to
confirm everything works and to make architecture/feature decisions.

| # | Task | Owner | Depends on |
|---|------|-------|------------|
| 4.1 | Run `report.py` on Validation set for all three candidates (M1, M2, M3 on P0 features) | evaluator | 3.9-3.11 |
| 4.2 | Run `report.py` on Validation set for P0+P1 feature runs | evaluator | 3.12-3.13 |
| 4.3 | Apply P1 feature cut rule: drop any P1 family that does not improve headline RMSE by >=0.5% on Validation | evaluator | 4.2 |
| 4.4 | Lock final feature set + hyperparameters; retrain final M1/M2/M3 on full Train, validate on Val | modeler | 4.3 |
| 4.5 | Produce validation comparison table for user review | evaluator | 4.4 |

---

## Step 5 -- Critic: pre-Test review

The critic reviews all code BEFORE the single Test touch.

| # | Task | Owner | Depends on |
|---|------|-------|------------|
| 5.1 | Review all feature builders for leakage (validation_protocol.md section 10, items 1-3, 5) | critic | 4.4 |
| 5.2 | Review training code for test-set tuning (item 4) and random shuffling (item 6) | critic | 4.4 |
| 5.3 | Review scoring-rule consistency (item 7): same rules used for train target and test target | critic | 4.4 |
| 5.4 | Verify CQR calibration set is disjoint from training set (item 5) | critic | 4.4 |
| 5.5 | Check statistical methodology: proper walk-forward CV, no p-hacking, correct pinball loss usage | critic | 4.4 |
| 5.6 | Severity-ranked findings report | critic | 5.1-5.5 |

**GATE: Critic findings at severity "high" or above must be resolved before Step 6.**

---

## Step 6 -- Evaluator: single Test touch

This is the one-and-only Test evaluation. Per validation_protocol.md section 11.

| # | Task | Owner | Depends on |
|---|------|-------|------------|
| 6.1 | User gives explicit go-ahead for Test touch | user | 5.6 |
| 6.2 | Freeze artifacts: model files, feature builder git SHA, scoring rules snapshot, protocol git SHA | evaluator | 6.1 |
| 6.3 | Run `generate_report()` on Test set for M1, M2, M3 -- single deterministic pass | evaluator | 6.2 |
| 6.4 | Apply selection rule from model_bakeoff.md section 6 mechanically | evaluator | 6.3 |
| 6.5 | Promote selected model to MLflow "Production" stage | evaluator | 6.4 |
| 6.6 | Write `evaluation_report.md` with all headline numbers, diagnostics, slice tables | evaluator | 6.3 |

---

## Step 7 -- Notebook and documentation

| # | Task | Owner | Depends on |
|---|------|-------|------------|
| 7.1 | Create `notebooks/01_baseline_vs_lgbm.ipynb` documenting the modeling story | modeler | 6.6 |
| 7.2 | Include: EDA of feature distributions, baseline vs M1 vs M2 vs M3, calibration plots, slice analysis | modeler | 6.6 |

---

## Step 8 -- Architect: phase close

| # | Task | Owner | Depends on |
|---|------|-------|------------|
| 8.1 | Verify all acceptance criteria from PROJECT_PLAN.md Phase 1 (see checklist below) | architect | 6.6, 7.2 |
| 8.2 | Run integration checks: no broken imports, all tests pass, ruff lint clean | architect | 8.1 |
| 8.3 | Update README.md if user-visible behavior changed | architect | 8.1 |
| 8.4 | Update `docs/decisions/` if any mid-phase architectural decisions were made | architect | 8.1 |
| 8.5 | Confirm critic review is complete (Step 5) and no unresolved high-severity findings | architect | 5.6 |
| 8.6 | PR `phase/1-projection` into main | architect | 8.1-8.5 |

---

## Acceptance criteria checklist (from PROJECT_PLAN.md Phase 1)

- [ ] Model beats two baselines: (a) last-week's actual points (b) season average, by >=10% RMSE on held-out games
- [ ] 80% prediction intervals achieve 75-85% empirical coverage
- [ ] MLflow has logged >=5 runs with hyperparameters, metrics, model artifacts
- [ ] `predict()` API: input player_id + date, output `{mean, p10, p90}`

---

## Interface contracts summary

All contracts are defined in `packages/shared/schemas/ml.py` and the module
`__init__.py` Protocol classes.

### Feature engineer -> Modeler

**File boundary:** `features_hitters.parquet`, `features_pitchers.parquet`

Schema: each row is `(player_id, game_pk, game_date, season_year, player_type,
fantasy_points, <feature_columns>...)`. Keyed on `(player_id, game_pk)`. The
`fantasy_points` column is the target. Feature columns are dynamic but prefixed
by builder name.

Produced by: `assemble_features()` in `packages/ml/features/pipeline.py`
Consumed by: `train_model()` in `packages/ml/training/train.py`

### Modeler -> Evaluator

**File boundary:** Prediction DataFrames conforming to `packages/ml/evaluation/schemas.py`

Required columns: `player_id, game_pk, game_date, player_type, model_candidate, predicted_mean`
Optional columns: `predicted_p10, predicted_p90, actual_points`

Validated by: `validate_prediction_dataframe()` in `schemas.py`
Produced by: model `.predict()` and `.predict_quantiles()` methods
Consumed by: `generate_report()` in `packages/ml/evaluation/report.py`

### Evaluator -> Architect

**File boundary:** `evaluation_report.md` (written by `report.py`)

Contains: `PopulationMetrics` and `BaselineComparison` Pydantic models
serialized to markdown tables.

### Public API

`packages/ml/models/projection.py::predict(player_id, target_date) -> PlayerProjection`

Returns `{player_id, game_date, game_pk, mean, p10, p90, model_version}`.

### Scoring rules

`packages/shared/scoring.py` is the single source of truth for fantasy points
calculation. Both feature engineer (target computation) and evaluator (baseline
computation) use the same functions: `compute_batting_fantasy_points()`,
`compute_pitching_fantasy_points()`.

### Splits

`packages/ml/training/splits.py` is the single source of truth for temporal
split boundaries and walk-forward fold definitions. Both modeler and evaluator
reference the same `SPLIT_BOUNDARIES` and `WALK_FORWARD_FOLDS` constants.

---

## Parallelism map

```
Step 0 (architect) ------>|
                          |---> Step 1 (evaluator)  ---|
                          |---> Step 2 (feat-eng)   ---|---> Step 3 (modeler)
                                                       |---> Step 4 (eval on val)
                                                              |---> Step 5 (critic)
                                                                     |---> Step 6 (test touch)
                                                                            |---> Step 7 (notebook)
                                                                                   |---> Step 8 (close)
```

Steps 1 and 2 run in parallel. Step 3 waits for both gates. Steps 4-8 are sequential.
