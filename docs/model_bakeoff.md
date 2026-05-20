# Model Bake-off Plan — Phase 1

> Formal comparison plan for the three candidate architectures for the per-game fantasy-points projection model. The bake-off is the contract under which the `modeler` subagent operates. This document declares the candidates, the comparison protocol, and the selection rule **before any training happens**.

**Owner:** modeler subagent (executes); evaluator subagent (judges)
**Authored:** 2026-05-20
**Companion docs:** `packages/ml/evaluation/validation_protocol.md` (governs evaluation), `docs/feature_plan.md` (governs inputs)

---

## 1. The three candidates

| ID  | Name                       | Role     | Outputs            | Calibration source   |
|-----|----------------------------|----------|--------------------|----------------------|
| M1  | Ridge regression           | Floor    | Point only         | None                 |
| M2  | LightGBM quantile          | Middle   | p10, p50, p90      | None (raw quantiles) |
| M3  | CQR + ACI on LightGBM      | Ceiling  | mean, p10, p90     | CQR + ACI            |

The three are not just "different models" — they form a **deliberate ladder of statistical sophistication**:

- **M1** establishes that the features + target encoding actually contain signal beyond the baselines (§ `validation_protocol.md` §8). If M1 doesn't beat baselines, something is broken upstream.
- **M2** demonstrates that a more flexible function class + heteroscedastic intervals improve over the linear-Gaussian assumption.
- **M3** demonstrates that conformal calibration converts uncalibrated quantiles into intervals with finite-sample coverage guarantees, and that ACI adapts to non-stationarity over the season.

The deliverable of Phase 1 is **M3 in production**, with M1 and M2 in MLflow as documented ablations.

---

## 2. What's held constant across all three

To make the comparison interpretable, the following are **fixed** for all three candidates:

- **Features**: exactly the v1 feature set from `feature_plan.md` (P0 + surviving P1 after the cut rule). Same feature tables, same column ordering.
- **Data splits**: per `validation_protocol.md` §2. Train 2019–2023, Val 2024, Test 2025.
- **Walk-forward CV**: per protocol §3 (5 expanding folds; headline = mean of folds 2–5).
- **Target**: per-game fantasy points under the frozen scoring rules (protocol §5).
- **Population for evaluation**: hitters ≥100 PA / pitchers ≥30 IP in 2025 (protocol §6).
- **Two-population modeling**: each candidate is trained as **two separate models** (hitters / pitchers). The "model" referred to throughout this doc is the pair.
- **Evaluation pipeline**: same `report.py` produces the same headline numbers for all three.

Anything else is varied per-candidate. The differences are listed in §3.

---

## 3. Per-candidate specifications

### M1 — Ridge regression (point estimate floor)

- **Library**: `scikit-learn` (`Ridge` from `sklearn.linear_model`).
- **Preprocessing**: median-imputation (per `feature_plan.md` global rule 5), then `StandardScaler` fit on Train only.
- **Hyperparameters**: `alpha` searched over `np.logspace(-3, 3, 25)` via walk-forward CV (protocol §3).
- **Output**: point estimate only. **No prediction interval is produced.**
- **Why include a point-only model**: a coverage-less point estimate is what 90% of published projection systems actually ship; calling it out as a baseline puts the work of M2 and M3 in context.
- **Limitations to disclose**: assumes linear-Gaussian, homoscedastic noise — both clearly violated for fantasy points (a star hitter on a bad day still has bounded downside; a mediocre pitcher's HR rate is right-skewed).

**Expected outcome**: beats the protocol §8 baselines by some margin (likely the smallest of the three); loses to M2 and M3 on RMSE.

### M2 — LightGBM quantile regression (uncalibrated intervals)

- **Library**: `lightgbm` with `objective="quantile"`.
- **Three boosters per population**: one each at α ∈ {0.1, 0.5, 0.9}. The 0.5 booster is the point estimate; (0.1, 0.9) is the 80% PI.
- **Hyperparameters** are **shared across the three quantile heads** within a population (only the α target differs). The shared set is searched once per population:
  - `num_leaves`: 31, 63, 127
  - `min_data_in_leaf`: 20, 50, 100
  - `learning_rate`: 0.05, 0.03, 0.01
  - `feature_fraction`: 0.7, 0.85, 1.0
  - `bagging_fraction`: 0.7, 0.85, 1.0
  - early stopping on Pinball loss summed across α ∈ {0.1, 0.5, 0.9}, patience 50.
- **Tuning**: Optuna, **50 trials per population** (per protocol §3 — "50 trials per architecture per model"). M2 is one architecture; the three boosters share the trial budget by sharing parameters.
- **Output**: p10, p50, p90.
- **Calibration**: **none beyond what quantile loss gives you in-sample**. The 80% PI from raw LightGBM quantiles is **expected to mis-cover** (typically under-covers because of finite-sample bias in the tails).
- **Why include this**: it isolates the contribution of the conformal calibration step. M3 = M2 + CQR/ACI. If M3's coverage looks good but M2's already does too, the conformal layer isn't adding much; if M2 mis-covers and M3 fixes it, the conformal layer is doing its job and we have a clean story.

**Expected outcome**: best raw RMSE among the three (because no calibration bias correction). Coverage likely outside [0.75, 0.85], which is why this candidate does **not** become the production model.

### M3 — CQR + ACI on LightGBM (production candidate)

- **Library**: `MAPIE` (`mapie.regression.MapieQuantileRegressor` for CQR; ACI implemented as a thin online wrapper per Gibbs & Candès 2021).
- **Base learners**: identical to M2's three LightGBM quantile boosters. M3 reuses M2's tuned boosters — no separate hyperparameter search for the base.
- **Calibration set**: per protocol §4 (last 20% of Train by time, ~2023-04-01 to 2023-12-31).
- **CQR step**: compute conformity scores on the calibration set as defined in Romano et al. 2019; adjust the (p10, p90) interval to achieve marginal 80% coverage on the calibration distribution.
- **ACI step**:
  - State variable: `α_t` (effective miscoverage rate). Initialized at 0.20.
  - Update rule: `α_{t+1} = α_t + γ · (1{Y_t ∉ [q_lo, q_hi]} − target_miscoverage)`, with `target_miscoverage = 0.20`.
  - Window: most recent W=200 (player, game) pairs.
  - **γ tuned** on Train+Val (protocol §4); grid: `{0.005, 0.01, 0.02, 0.05}`.
  - Online update during 2024 (Validation) and 2025 (Test); the Test-set ACI state is initialized from the end-of-2024 state, **not re-initialized**, and **not updated using Test residuals in the headline run** (separate sensitivity run permitted per protocol §4).
- **Output**: mean (from M2's p50 booster, kept as point estimate); calibrated p10, p90.
- **Why this is the production pick**: gives the heteroscedastic intervals of M2 plus a coverage guarantee that survives non-stationarity. Loses ~1–2% RMSE vs. M2 in exchange for proper calibration — an acceptable trade because Phase 2 (lineup, trade) depends on the interval, not just the point estimate.

**Expected outcome**: slightly worse RMSE than M2; coverage in [0.75, 0.85] on the headline Test population per protocol §7.2. This is the candidate we expect to ship.

---

## 4. What varies, what doesn't — summary table

|                              | M1 Ridge       | M2 LGBM-Q       | M3 CQR+ACI            |
|------------------------------|----------------|------------------|------------------------|
| Feature set                  | same (v1)      | same (v1)        | same (v1)              |
| Train/Val/Test splits        | same           | same             | same                   |
| Walk-forward CV folds        | same (1–5)     | same (1–5)       | same (1–5)             |
| Target encoding              | same           | same             | same                   |
| Eligibility population       | same           | same             | same                   |
| Preprocessing                | impute + scale | impute only      | impute only            |
| Function class               | linear         | tree ensemble    | tree ensemble          |
| Loss                         | squared (L2)   | pinball at α     | pinball at α + CQR     |
| Hyperparameter search        | 1 param × 25   | Optuna × 50      | inherits M2 + γ × 4    |
| Calibration                  | none           | none             | CQR + ACI              |
| Produces 80% PI              | no             | yes (uncalibrated) | yes (calibrated)     |
| Production deployment        | no             | no               | yes                    |

---

## 5. Comparison criteria

All three models are scored on the **same** Test pass (protocol §11.1 — one touch). The single touch produces one `evaluation_report.md` containing the side-by-side numbers for M1, M2, M3.

### 5.1 Per-population headline metrics
For each of {hitters, pitchers}:
- RMSE, MAE, Spearman ρ — for all three.
- 80% PI empirical coverage, sharpness — for M2 and M3 (M1 has no interval).
- Pinball loss at α ∈ {0.1, 0.5, 0.9} — for M2 and M3.
- Marginal-vs-baseline lift (best of the three baselines, protocol §8) — for all three.

### 5.2 Operational metrics
- **Training wall-clock** per model per population (M1, M2, M3 — though M3 inherits M2's training).
- **Inference latency**: p50 and p99 over 1,000 synthetic prediction requests on the production-target machine size.
- **Artifact size**: pickled model + calibration arrays.

### 5.3 Robustness diagnostics
- **Slice metrics** (protocol §6.5): hitters by position, pitchers by SP/RP, rookies, playing-time tertiles. Reported for all three.
- **Coverage drift over time** (protocol §9.4): weekly bins; ACI is supposed to keep M3 stable here. Critical diagnostic for M3.
- **Calibration plot side-by-side**: M2's reliability vs. M3's reliability. Visualizes the contribution of the conformal layer.

---

## 6. Selection rule

The **production model** for Phase 1 is selected by the following deterministic rule, evaluated **after** the single Test touch:

```
IF M3 satisfies protocol §13 acceptance (beats baselines by ≥10% AND coverage ∈ [0.75, 0.85]):
    SELECT M3
ELIF M2 beats baselines by ≥10% RMSE AND M2's empirical coverage ∈ [0.70, 0.90]:
    SELECT M2 with a documented caveat ("intervals not formally calibrated")
ELIF M1 beats baselines by ≥10% RMSE:
    SELECT M1 with a documented caveat ("no prediction intervals; Phase 2 cannot use uncertainty-aware decision rules")
ELSE:
    PHASE 1 FAILS. Return to feature engineering / data investigation per the critic's report.
```

This rule is committed **before** the Test touch. It is not adjusted after seeing numbers. If the result is borderline (e.g., M3 coverage 0.74 — just outside the band), the rule says M2 wins; we do not bend the rule to favor the more sophisticated candidate.

---

## 7. What this bake-off is *not* doing

To be explicit so we don't drift:

- **Not an exhaustive model search.** No XGBoost, no CatBoost, no neural nets, no quantile forests. Three was chosen for clarity of story, not for chasing the last 1% of RMSE.
- **Not benchmarking against published projection systems** (Steamer, ZiPS, ATC). They use season totals, not per-game; the comparison would be apples-to-oranges. Possible Phase 4 extension if we project per-game → per-week and roll up.
- **Not deciding whether to use LLM-based projections.** Out of scope; LLM enters in Phase 6 as a chat agent over the projection outputs.
- **Not retraining within season.** All three models are trained once per evaluation pass. Intra-season retrain cadence is Phase 4.
- **Not ensembling.** We could stack M1+M2+M3 and probably win 0.5% RMSE; we don't, because the portfolio story is "I understand calibration," not "I can ensemble." If headline goals are not met, ensembling is a documented fallback in the critic's purview, not a default.

---

## 8. Implementation order for the modeler subagent

1. **M1 first**, end-to-end through the evaluation pipeline. Confirms wiring (feature → train → eval → MLflow → report). Cheap and fast.
2. **M2** — train the three quantile boosters, validate Pinball loss converges on Train+Val, log to MLflow with full hyperparameter trace.
3. **M3** — reuse M2's boosters, layer CQR using the calibration window from protocol §4, layer ACI with γ tuned on Train+Val.
4. **All three logged** in MLflow under run group `phase1-bakeoff`; M3 promoted to "Staging" stage; M1 and M2 left as "None" with descriptive tags.
5. **Single Test touch** triggered by `evaluator` subagent on explicit user go-ahead. All three models scored in the same `evaluation_report.md`.
6. **Selection rule §6 applied** mechanically; chosen model promoted to "Production" stage.

---

## 9. Acceptance for the modeler subagent

The bake-off is "done" when:

- M1, M2, M3 each have a registered MLflow run with: hyperparameters, train/val metrics, model artifact, feature-set hash, scoring-rules hash, protocol-doc hash.
- M1 + M2 + M3 each produce a per-population prediction CSV on the Validation set with the column schema documented in `packages/ml/evaluation/schemas.py`.
- The `evaluator` subagent can run `report.py` against all three on Validation and produce a unified comparison table **before** any Test touch.
- The critic subagent's review (per `CLAUDE.md`) finds no leakage, no Test peeking, and confirms the selection rule §6 was not modified.
- The chosen model satisfies the corresponding branch of §6 on the Test pass.
