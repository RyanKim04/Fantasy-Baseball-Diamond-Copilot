# Validation Protocol — Phase 1 Player Projection Model

> **Status: pre-registered.** This document is the contract under which the Phase 1 projection model is evaluated. It is committed BEFORE any training code is written. Per `CLAUDE.md`, the Modeler subagent must refuse to begin training unless this file exists. Modifications after the first model run require explicit user approval and a multiple-comparison penalty (see §11).

**Owner:** evaluator subagent
**Authored:** 2026-05-20
**Last modified:** 2026-05-20 (initial commit; no modifications)
**Applies to:** Phase 1 player projection model (per-game fantasy points)

---

## 1. Objective

Predict, for each player–game pair, the **fantasy points scored by that player in that game under the user's league scoring rules**, together with a calibrated prediction interval.

Hitters and pitchers are modeled as **two separate models** with disjoint feature sets and disjoint evaluation pools. The deliverable of Phase 1 is two trained models plus this protocol's reports for each.

The projection is the foundation for every downstream Phase 2+ decision tool (lineup, trade, waiver, simulation). For this reason, **calibration of the prediction interval is a first-class requirement, not a nice-to-have**. A model with lower RMSE but worse coverage may be rejected.

---

## 2. Data Splits

All splits are **strictly temporal**. No row may move backward in time during shuffling, k-fold, or any other procedure. Random `train_test_split` is forbidden anywhere in the pipeline.

| Split          | Date range (inclusive)      | Purpose                                                              |
|----------------|-----------------------------|----------------------------------------------------------------------|
| Train          | 2019-01-01 to 2023-12-31    | Model fitting + walk-forward CV for hyperparameter tuning            |
| Validation     | 2024-01-01 to 2024-12-31    | Architecture comparison; final hyperparameter lock                   |
| Test           | 2025-01-01 to 2025-12-31    | Final evaluation. Touched **once**.                                  |
| Live           | 2026-01-01 onward           | Phase 4+ production monitoring; not used in Phase 1 evaluation       |

Notes:
- **2020 COVID season** (60-game schedule): rows are included in Train; per-season aggregate metrics computed for 2020 are excluded from headline summaries (reported separately as a footnote).
- **Off-season rows**: there are no rows during off-season. Spring training and exhibition games are excluded from all splits.
- **Edge of split boundary**: a game played on 2023-12-31 is in Train; a game on 2024-01-01 is in Validation. There is no game that straddles the boundary.
- **Player identity does not partition splits.** The same player appears across multiple splits. We are predicting per-game performance, not generalizing to unseen players.

---

## 3. Walk-Forward CV Strategy (within Train + Val)

Hyperparameter tuning uses **5 expanding-window folds** over the combined Train + Validation window (2019-01-01 to 2024-12-31). Each fold has a strictly earlier training window than its validation window. No fold's validation window overlaps any other fold's training window.

| Fold | Training window (expanding)         | Validation window (1 season) |
|------|-------------------------------------|------------------------------|
| 1    | 2019-01-01 → 2019-12-31             | 2020 (COVID; see note)       |
| 2    | 2019-01-01 → 2020-12-31             | 2021                         |
| 3    | 2019-01-01 → 2021-12-31             | 2022                         |
| 4    | 2019-01-01 → 2022-12-31             | 2023                         |
| 5    | 2019-01-01 → 2023-12-31             | 2024                         |

Aggregation rule:
- Per-fold metrics are computed independently.
- **Headline CV metric** is the mean across folds 2–5 (fold 1 excluded from headline because its validation window is 2020, which has known structural differences).
- Fold 1 is reported separately as a robustness check.

Stopping criteria for hyperparameter search:
- Bayesian search (Optuna) with a budget of **50 trials per architecture per model (hitter, pitcher)**.
- Each trial logs to MLflow with the fold breakdown.
- The hyperparameter set that **minimizes mean Pinball loss across folds 2–5** is selected for the quantile models; **mean RMSE across folds 2–5** for point models.

---

## 4. Calibration Set Strategy for CQR

CQR requires a calibration set that is (a) disjoint from training, (b) drawn from the same distribution as the deployment data, and (c) temporally prior to the evaluation period.

**Calibration set definition:**
- The **last 20% (by time)** of the Train window: **2023-04-01 to 2023-12-31** approximately (exact cutoff: the date at which 80% of Train rows have been observed, computed once and committed as a constant).
- Quantile regressors are fit on the first 80% of Train (2019-01-01 through the cutoff).
- The calibration set is then used to compute the conformity scores and the resulting adjusted quantile width.

**ACI (Adaptive Conformal Inference) addendum:**
- Once we move into Validation (2024) and Test (2025), ACI updates the conformity threshold online using the most recent W=200 prediction–outcome pairs (per-game horizon).
- The ACI learning rate γ is tuned **only within Train + Val folds** (treated as a hyperparameter, see §3).
- **The Test set never feeds back into the ACI threshold update for the headline number.** A separate "ACI-on-test" run is computed as a sensitivity check and reported alongside, but the headline coverage is the prospectively-computed ACI threshold from end-of-2024.

**Refits:**
- The quantile base learners and the CQR calibration are refit at the start of each new season (annual cadence) for the Phase 1 deliverable. Intra-season refits are out of scope for Phase 1 and reserved for Phase 4.

---

## 5. Target Variable Definition

**Primary target:** per-game fantasy points scored by a player in a single MLB game, computed under the user's league scoring rules as pulled from the Yahoo Fantasy API at the time of training. The scoring rules are **frozen at training time** and stored alongside the model artifact in MLflow. Any change to league scoring requires a model retrain and a new MLflow run.

- For hitters: one row per (player_id, game_id) where the player had ≥1 plate appearance.
- For pitchers: one row per (player_id, game_id) where the player recorded ≥1 batter faced.
- A player who did not appear in a game has **no row** for that game (we are not modeling "did the player play"). Inactive-player handling is a separate Phase 2 concern.

**Derived target (reporting only):** per-week fantasy points, defined as the sum of per-game projections for games scheduled within a fantasy-week window (Monday → Sunday, by default; configurable per league).

- Per-week aggregation is performed **at inference time** as a sum of independent per-game predictive samples.
- The per-week prediction interval is computed by **sampling 10,000 per-game outcomes from each game's predictive distribution** (via the quantile estimates, linearly interpolated CDF) and summing them. The 10th and 90th percentiles of the summed distribution form the per-week 80% PI.
- **Phase 1 evaluates the per-game model.** Per-week aggregates are reported as a secondary diagnostic only and are not gating criteria.

**Sign and units:** fantasy points, signed (negative values allowed when scoring rules permit, e.g., pitcher losses). Units are the user's league points; no normalization across leagues in Phase 1.

---

## 6. Population Eligibility Rules

**Training population:** all player-game rows in the Train split. No filtering. Rookies, cup-of-coffee players, and high-variance roles are all included.

**Test-set evaluation population (headline metrics):**
- Hitters with **≥100 plate appearances within the 2025 season** at the time of evaluation.
- Pitchers with **≥30 innings pitched within the 2025 season** at the time of evaluation.
- This filter is computed using **full-season 2025 totals**, applied only at evaluation time, never as a feature.

**Reporting slices (always reported in addition to the headline):**
- All-players slice (no PA/IP filter): documents performance on the cup-of-coffee tail.
- **Rookies slice**: players in their MLB debut season as of the game date.
- **Veteran slice**: ≥3 prior MLB seasons.
- Per-position slices (C, 1B, 2B, 3B, SS, OF, DH for hitters; SP, RP for pitchers).
- Playing-time tertiles within the eligible population.

The headline number determines pass/fail; slice numbers contextualize.

---

## 7. Metrics

### 7.1 Primary metric
- **RMSE** on per-game fantasy points within the eligible test-set population. Computed separately for hitters and pitchers; **two headline RMSE numbers**, not one combined.

### 7.2 Secondary metrics
- **MAE** (more interpretable, less sensitive to outliers).
- **Spearman rank correlation** between predicted and actual per-game fantasy points. This is what fantasy decision-making actually rewards (rank order of players on a given night).
- **Interval coverage** at the nominal 80% level. Acceptable range: **75–85% empirical coverage**. Headline pass condition is in [0.75, 0.85].
- **Sharpness**: mean width of the 80% prediction interval. Lower is better, conditional on coverage being in the acceptable range.
- **Pinball loss** at τ ∈ {0.1, 0.5, 0.9} (proper scoring rule for quantile models; the loss against which quantile heads are trained).

### 7.3 Slice metrics
RMSE, MAE, Spearman, and coverage are reported on each slice from §6. No slice has a hard pass/fail threshold; slices are diagnostic.

### 7.4 Reporting format
A single `evaluation_report.md` per model run, generated by `packages/ml/evaluation/report.py`, containing:
- Headline numbers (hitter RMSE, pitcher RMSE, hitter coverage, pitcher coverage).
- A reliability diagram (§9).
- Slice tables.
- A residual plot vs. predicted value (heteroscedasticity check).
- The MLflow run ID and model version.

---

## 8. Baselines That Must Be Beaten

The main model (CQR + ACI with LightGBM base) must beat **the best of these three baselines** by **≥10% in RMSE** on the headline test-set population (hitters and pitchers, each evaluated separately).

| Baseline                | Definition                                                                                            |
|-------------------------|-------------------------------------------------------------------------------------------------------|
| Naive last-game         | Prediction for tonight = points actually scored in the player's most recent game.                     |
| Trailing-7-day mean     | Prediction = mean per-game fantasy points over the player's last 7 calendar days (≥1 game required). |
| Season-to-date mean     | Prediction = mean per-game fantasy points in the current season up to (but not including) the game.  |

**The 10% threshold is computed per population:**

```
RMSE_model_hitters   ≤ 0.90 × min(RMSE_naive_lastgame_hitters, RMSE_7d_mean_hitters, RMSE_stdate_mean_hitters)
RMSE_model_pitchers  ≤ 0.90 × min(RMSE_naive_lastgame_pitchers, RMSE_7d_mean_pitchers, RMSE_stdate_mean_pitchers)
```

Failing either population fails the headline pass condition.

Baselines are computed on the same eligible test-set population (§6) using the same per-game scoring rules (§5). Baselines do not produce intervals; coverage is not evaluated for them. Baselines are run once on the Test set, as part of the same single Test touch (§11).

---

## 9. Calibration Checks

For each of hitters and pitchers, the report includes:

1. **Reliability diagram**: bin predictions into 10 equal-mass bins by predicted mean, plot mean predicted vs. mean actual. Deviation from the y = x diagonal indicates miscalibration of the point estimate.
2. **Empirical coverage at multiple nominal levels**: 50%, 70%, 80%, 90%. The 80% level is the headline; the others diagnose whether miscalibration is symmetric, in the tails, or in the center.
3. **Coverage by predicted-bin**: empirical coverage within each of the 10 reliability bins. Detects whether the 80% PI is well-calibrated overall but miscalibrated for high-projection or low-projection players.
4. **Coverage drift over time**: empirical coverage computed in 4 weekly bins across the Test season. Detects whether ACI is doing its job (coverage should remain near 0.80 across all weeks, not just on average).
5. **Width vs. predicted-value plot**: interval width as a function of predicted mean. Confirms that the model captures heteroscedasticity (wider intervals for high-variance roles).

A model whose headline coverage is in [0.75, 0.85] but whose per-bin coverage in any bin falls outside [0.65, 0.95] is flagged for critic review; this does not automatically fail Phase 1 but is reported prominently.

---

## 10. Disqualifying Conditions

Any of the following, discovered at any point, **disqualifies the model** and triggers a re-run with the defect fixed. The disqualified run does not consume the single Test touch (§11) — the protocol distinguishes "a real attempt" from "a defective attempt."

1. **Target leakage**: any feature that is a function of the target for that same game, including but not limited to: current-game stat lines, current-game lineup-confirmed result, real-time game state.
2. **Future-information leakage in rolling features**: a rolling-window feature whose window includes the target game's date or any later date. Rolling windows must end at `game_date − 1 day` strictly.
3. **Full-season aggregates as features for that season**: using a player's 2024 season-long wOBA as a feature to predict 2024 games. Permitted: prior-season aggregates (2023 wOBA when predicting 2024).
4. **Test-set tuning**: any hyperparameter change, feature toggle, or architecture change made **after** observing any test-set metric.
5. **Calibration set contamination**: any row in the CQR calibration set that is also in the training set, or that postdates the conformity-score computation date.
6. **Random shuffling at any point**: random shuffles or random k-fold splits applied to the row index before fitting.
7. **Score-rule mismatch**: the scoring rules used to compute the training target differ from the rules used to compute the test target. Both must come from the same frozen `league_scoring_rules` snapshot.
8. **Player-identity leakage via embeddings**: if a player-embedding feature is used, it must be fit on Train-only data and frozen for Validation and Test.

The Critic subagent's review (per `CLAUDE.md` §"Multi-agent development workflow") explicitly checks for items 1–8 before phase close.

---

## 11. Test-Set Discipline

The Test split (2025-01-01 to 2025-12-31) is touched **once**. "Touched" means: any code path that reads test rows and emits a metric, plot, or aggregate.

**One touch = one evaluation run, which produces:**
- The headline RMSE/MAE/Spearman/coverage numbers (§7).
- The reliability and coverage diagnostics (§9).
- The slice tables (§6).
- The baseline comparisons (§8).
- The disqualification check (§10).

All of the above are computed within a single, scripted evaluation pass, deterministic given the model artifact and the protocol.

### 11.1 What counts as "the touch"
- Computing any number, plot, or summary derived from the Test split. Even peeking at a small slice constitutes a touch.
- Loading test rows into a notebook for exploration is a touch.

### 11.2 What does not count
- Disqualified runs (§10), provided the disqualifying defect is identified **independently of the test result** (e.g., found in code review, found by static checks, found by the Critic agent before the test metrics are read).
- Inspecting feature distributions of the Test split for **schema validation only** (e.g., confirming no nulls, no impossible values), provided no target or prediction is read.

### 11.3 Multiple-comparison penalty
If the protocol is modified after the single Test touch and a second touch is performed, the second result is reported **with a Bonferroni-adjusted significance level**:

- For each additional touch *k*, the threshold to claim "model beat baseline" tightens from 10% to:
  ```
  threshold_k = 1 − (0.90)^k
  ```
  i.e., the required RMSE improvement compounds: 10%, 19%, 27.1%, ...
- This is conservative on purpose. The intended behavior is that we touch Test exactly once.

### 11.4 Frozen artifacts
At the time of the single Test touch, the following are frozen and committed:
- The trained model files (LightGBM boosters for point + quantile + CQR calibration array).
- The feature builder version (git SHA of `packages/ml/features/`).
- The league scoring rules snapshot.
- The exact eligibility filter values (≥100 PA, ≥30 IP).
- This protocol file (git SHA).

The evaluation script reads these artifacts, runs the single touch, and writes `evaluation_report.md`. No edits to model, features, scoring rules, or eligibility are made after this point without triggering §11.3.

---

## 12. Out-of-Scope for Phase 1

The following are intentionally deferred and are **not** evaluated by this protocol:
- Intra-season retraining cadence (deferred to Phase 4).
- Lineup confirmation as a real-time feature (Phase 2 decision support).
- Multi-week roster optimization (Phase 2).
- Playoff Monte Carlo simulation calibration (Phase 3).
- LLM agent grounding quality (Phase 6).

Anything in this list will be evaluated under its own protocol, written by the Evaluator subagent at the start of the relevant phase.

---

## 13. Sign-off

By committing this file, the project commits to the following pre-registered claims for Phase 1 success:

1. The CQR + ACI model beats the best of the three baselines (§8) by ≥10% RMSE, separately for hitters and pitchers, on the eligible 2025 test population.
2. The 80% prediction interval achieves empirical coverage in [0.75, 0.85] on each of hitters and pitchers.
3. No disqualifying condition (§10) is present.
4. The single Test touch (§11) produces the deliverables in §11.1 with no subsequent modification.

If any of (1)–(4) fails, Phase 1 is not "done" regardless of how good other numbers look.
