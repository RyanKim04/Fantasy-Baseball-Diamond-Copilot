# Feature Plan — Phase 1 Player Projection

> Feature family prioritization for the per-game fantasy-points projection model. This is the contract under which the `feature-engineer` subagent works. Anything in P0 ships in v1; P1 is the target stretch for v1; P2 is explicitly deferred.

**Owner:** feature-engineer subagent
**Authored:** 2026-05-20
**Companion docs:** `packages/ml/evaluation/validation_protocol.md`, `docs/model_bakeoff.md`

---

## Global rules for every feature

These hold for **every** feature in every priority tier. The `critic` subagent enforces them.

1. **Causality**: every feature for game G of player P must be computable using **only information available before the first pitch of G**. The rolling window must end at `game_date − 1 day` strictly.
2. **No same-season full aggregates**: 2024 season-long wOBA cannot be a feature when predicting 2024 games. Prior-season aggregates are permitted.
3. **No target-derived features**: if a feature is a function of the row's target, it is leakage. This includes "season-to-date fantasy points" computed using the target column.
4. **Player-context features fit on Train only**: any feature that requires a learned model (e.g., player embeddings, opponent strength regressed onto Train) must be fit on Train and frozen for Validation and Test.
5. **Missing data**: explicit missingness indicator + median/zero imputation. **Never forward-fill across the game-date boundary** (would leak future information).
6. **Two feature sets**: hitter features and pitcher features are built and stored separately. There is no shared "player" feature table.

---

## Priority tiers

- **P0** — minimum viable; the model cannot ship without these. ~15–20 features each side.
- **P1** — strong v1 enhancements; included unless time-cost is prohibitive.
- **P2** — deferred; explicit non-goals for Phase 1. Documented here so we don't forget.

Each family is tagged with: tier, applies-to (H = hitters / P = pitchers / both), source, and effort estimate.

---

## P0 — Foundational

### F1. Rolling production windows
- **Applies to:** both
- **Source:** pybaseball Statcast → daily aggregate
- **Effort:** M
- **Hitters:** rolling AVG, OBP, SLG, wOBA, ISO, BABIP at windows **{7, 14, 30}** days. Each window also gets PA count (sample-size signal).
- **Pitchers:** rolling ERA-equivalent, K/9, BB/9, HR/9, GB%, opponent wOBA-against at **{14, 30, 60}** days (pitcher cadence is slower).
- **Anti-leakage:** window ends at `game_date − 1 day`. If a player has 0 games in the window, the value is null with an explicit missingness indicator.
- **Why P0:** rolling production is the single strongest baseline signal in published projection systems (Marcel, Steamer, ZiPS).

### F2. Days of rest / recency
- **Applies to:** both
- **Source:** MLB Stats API schedules
- **Effort:** S
- **Features:** `days_since_last_game`, `games_in_last_7_days`, `back_to_back_flag` (hitters), `days_since_last_start` (SP), `pitches_thrown_last_start` (SP).
- **Why P0:** pitcher cadence (~every 5 days) makes this binary in importance; hitters have meaningful day-off vs. consecutive-game splits.

### F3. Home / Away + ballpark factor
- **Applies to:** both
- **Source:** MLB schedule; prior-season ballpark factors from Fangraphs / Statcast park factors table
- **Effort:** S
- **Features:** `is_home` (binary), `park_factor_runs`, `park_factor_HR`, `park_factor_BB`. Park factor uses **prior-season** value (no current-season leakage).
- **Why P0:** Coors vs. Petco is a 20%+ swing in offensive environment; cannot be ignored.

### F4. Opponent quality (point estimates)
- **Applies to:** both
- **Source:** pybaseball Statcast season-to-date aggregates with strict date filter
- **Effort:** M
- **Hitters facing pitcher Q tonight:** Q's rolling-30-day FIP, K/9, BB/9, HR/9; Q's career wOBA-against same-handed batters.
- **Pitchers facing lineup L tonight:** L's rolling-30-day team wOBA, team K%, team ISO.
- **Anti-leakage:** opponent stats are computed with window ending `game_date − 1 day`. **Opponent identity is observed** (we know who tonight's SP is); their stats up through yesterday are the feature.
- **Why P0:** opponent quality is the largest source of legitimate variance in single-game projections.

### F5. Lineup slot (hitters only)
- **Applies to:** H
- **Source:** MLB Stats API projected/confirmed lineup
- **Effort:** S (but with a real-time wrinkle, see note)
- **Features:** `batting_order_slot` (1–9, integer), `lineup_confirmed_flag`.
- **Note on confirmation timing:** lineups confirm ~3 hours before first pitch. **Phase 1 model uses projected lineup at training time** (rolling-most-recent slot from the prior week). The "confirmed lineup" override is a Phase 2 inference-time enhancement, not a training feature.
- **Why P0:** leadoff vs. 9-hole is a ~15% PA-volume difference, which directly drives fantasy point variance.

### F6. Pitcher expected workload (pitchers only)
- **Applies to:** P
- **Source:** rolling pitcher_id history
- **Effort:** S
- **Features:** rolling-5-start mean innings pitched, mean pitches thrown, mean batters faced; role flag (SP / RP / opener).
- **Why P0:** fantasy points for pitchers are roughly linear in IP × per-inning quality. Without an IP expectation, the model can't separate "ace going 7" from "opener going 1".

### F7. Player identity baseline (prior-season aggregates)
- **Applies to:** both
- **Source:** Fangraphs end-of-season tables (prior year only)
- **Effort:** S
- **Hitters:** prior-season PA, wOBA, ISO, BB%, K%, sprint speed.
- **Pitchers:** prior-season IP, FIP, K%, BB%, GB%, avg fastball velo.
- **Anti-leakage:** **strictly prior-season**. For 2024 games, use 2023 EOS aggregates. Rookies get league-average values + missingness indicator.
- **Why P0:** anchors the model on player skill level, especially in April when rolling windows are noisy.

---

## P1 — Strong v1 enhancements

### F8. Plate discipline
- **Applies to:** both (hitter-discipline + pitcher-discipline induced)
- **Source:** Statcast pitch-level
- **Effort:** M
- **Hitters:** rolling-30-day BB%, K%, swing% on out-of-zone, contact% on in-zone, chase%.
- **Pitchers:** rolling-30-day called-strike%, swinging-strike%, zone%, first-pitch-strike%.
- **Why P1 (not P0):** strong signal but partly captured by F4. Adds incremental lift in the 5–10% RMSE range based on published literature.

### F9. Quality of contact (Statcast)
- **Applies to:** both
- **Source:** Statcast pitch-level (`pybaseball.statcast`)
- **Effort:** M
- **Hitters:** rolling-30-day barrel%, hard-hit% (≥95 mph EV), avg exit velocity, avg launch angle, **xwOBA**.
- **Pitchers:** rolling-30-day barrel%-allowed, hard-hit%-allowed, **xwOBA-against**.
- **Why P1:** xwOBA is the strongest known single predictor of next-period wOBA; including it should produce most of the gap over the trailing-mean baseline.

### F10. Platoon splits
- **Applies to:** both
- **Source:** pybaseball with batter/pitcher handedness join
- **Effort:** M
- **Hitters:** career and rolling-90-day wOBA vs. LHP, vs. RHP; the night's opponent SP handedness is the gating feature.
- **Pitchers:** wOBA-against by batter handedness (vs. L, vs. R); tonight's opponent L/R lineup composition.
- **Why P1:** matters most for platoon-split players (Joey Gallo vs. lefties, etc.) and for SP-vs-lineup matchups. Lower in priority than F4 because the average effect is smaller across the full population.

### F11. Recent-form z-scores ("hot/cold")
- **Applies to:** both
- **Source:** derived from F1
- **Effort:** S
- **Features:** `(rolling_7d_wOBA − rolling_60d_wOBA) / rolling_60d_std`; same for K% and BB%. Captures deviation from player baseline.
- **Why P1:** market-style signal — fantasy users intuitively reach for "hot" players, and our model should know the same thing. Whether it's actually predictive is an open empirical question (BABIP regression suggests not very); reported as such.

### F12. Pitcher velocity & repertoire trend
- **Applies to:** P
- **Source:** Statcast pitch-level
- **Effort:** L (requires per-pitch aggregation)
- **Features:** trailing-5-start mean fastball velo, velocity delta vs. season baseline, pitch-mix entropy (Shannon entropy over pitch_type proportions).
- **Why P1:** velocity drop is a known leading indicator of injury / decline. High effort but well-bounded.

### F13. Catcher framing matchup (pitchers only)
- **Applies to:** P
- **Source:** Statcast called-strike-above-average tables
- **Effort:** M
- **Features:** projected catcher tonight + their rolling-30-day CSAA. Pairs with F8's called-strike%.
- **Why P1:** ~2–4 fantasy-point swing per start for elite vs. poor framers; small but real.

---

## P2 — Deferred (explicit non-goals for Phase 1)

These are documented so we remember they exist; **do not implement in Phase 1**.

| Family | Why deferred |
|---|---|
| Weather (wind, temperature, humidity) | Data source (weather API + game-time matching) is non-trivial; effect size is small for most parks; Coors-style outliers already partly captured by F3. Revisit in Phase 4. |
| Real-time confirmed lineup override | Inference-time only, not a training feature. Lives in Phase 2 decision-support layer. |
| Umpire identity & zone tendencies | High effort, modest effect. Phase 4 candidate. |
| Sequencing / cluster-luck features | Conceptually suspect — sequencing is largely noise; including it risks chasing variance. |
| News / injury sentiment from RAG | Belongs to Phase 6 LLM layer, not the projection model. |
| Park-handedness interaction (e.g., LHB at Yankee Stadium short porch) | Possible v1.1 if F3 + F10 prove insufficient. Tracked but not built. |
| Catcher game-calling for hitters | Effect size unclear; F13 covers the pitcher side already. |
| Multi-game streak features (current hit streak, etc.) | Almost certainly noise; high overfit risk; skipped on principle. |
| Player age curves | Useful in season-long projections, marginal in per-game. Could add as a P1 feature later if it doesn't introduce leakage; current pass: out. |

---

## Build order

Within v1 implementation:

1. **Pass 1 — P0 only.** Build, unit test, sanity check distributions, train baseline LightGBM, log to MLflow as `v1.0-p0-only`. This is the minimum that should beat `validation_protocol.md` §8 baselines.
2. **Pass 2 — P0 + P1.** Add P1 families incrementally, one family per MLflow run, so we can attribute marginal lift. This is the v1 release candidate.
3. **Cut/keep decision:** any P1 family that doesn't move headline RMSE by ≥0.5% on the **validation** fold is removed before the Test touch (per protocol §11). This is a hyperparameter-like decision, made on Train+Val only.

---

## Acceptance for the feature_engineer subagent

The feature module is "done" when:

- All P0 families have a builder function in `packages/ml/features/`, type-hinted, with at least one unit test per builder.
- Each builder declares its time-window dependency (e.g., `requires_history_days: 30`) so the orchestration layer knows how much history to load.
- A schema check confirms no feature column contains data from `game_date` or later.
- Two assembled feature tables exist: `features_hitters.parquet` and `features_pitchers.parquet`, both keyed on `(player_id, game_id)`.
- The Critic subagent's leakage scan (per `CLAUDE.md` §"Things never to do") passes.
