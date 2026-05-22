# Architecture Diagram

## System Overview

```
+-----------------------------------------------------------------+
|                        DATA SOURCES (External)                   |
+------------------+------------------+----------------+-----------+
|   pybaseball     |  MLB Stats API   |  Yahoo Fantasy |  FanGraphs |
|  (Statcast)      |  (statsapi.mlb)  |  API (OAuth)   |  (via pybb)|
|                  |                  |                |            |
| - Pitch-level    | - Game schedule  | - Roster       | - Season   |
|   data (90+ cols)|   + venue info   | - Scoring rules|   aggregates|
| - 2018-present   | - Boxscores      | - Matchups     | - Park     |
|                  | - Lineups        | - League info  |   factors  |
|                  | - Player info    |                |            |
+--------+---------+--------+---------+-------+--------+------+----+
         |                  |                 |               |
         v                  v                 v               v
+-----------------------------------------------------------------+
|                     AIRFLOW  (Orchestration)                     |
|                     localhost:8081 (Web UI)                       |
+-----------------+-----------------+----------------+-------------+
| ingest_mlb_     | ingest_         | ingest_        |seed_        |
| schedule DAG    | boxscores DAG   | statcast DAG   |historical   |
| Daily 6am UTC   | Daily 7am UTC   | Daily 8am UTC  |DAG          |
|                 |                 |                |Manual trigger|
| - Fetch 7-day   | - Fetch game    | - Fetch pitches|             |
|   schedule      |   boxscores via |   via pybaseball| - Backfill |
| - Probable      |   MLB Stats API | - Validate rows|   2018-2025|
|   pitchers      | - UPSERT daily  | - UPSERT into  | - Pitches + |
| - UPSERT games  |   batting +     |   pitches table|   boxscores+|
|   + players     |   pitching stats| - Fill Statcast|   season    |
|                 |   (counting +   |   columns in   |   stats +   |
|                 |   fantasy-      |   daily stats   |   park      |
|                 |   relevant)     |                |   factors   |
+-----------------+-----------------+                |             |
| ingest_yahoo DAG                  |                |             |
| Daily 9am UTC                     |                |             |
|                                   |                |             |
| - Refresh OAuth                   |                |             |
| - Pull roster, scoring, matchups  |                |             |
| - UPSERT Yahoo tables             |                |             |
+--------+--------+---------+-------+-------+--------+------+------+
         |                 |                |                |
         v                 v                v                v
+-----------------------------------------------------------------+
|                   POSTGRES  (Supabase / Local Docker)            |
|                   localhost:5432                                  |
+-----------------------------------------------------------------+
|                                                                  |
|  +----------------- MLB Data ----------------------------------+ |
|  |                                                              | |
|  |  players              games              pitches             | |
|  |  --------             -----              -------             | |
|  |  player_id (PK)       game_pk (PK)       id (PK)            | |
|  |  name, position,      game_date,         game_pk (FK)       | |
|  |  bats, throws,        home/away team,    batter/pitcher     | |
|  |  team, active,        venue_id/name,     pitch_type,        | |
|  |  birth/debut date,    scores, status,    speed, spin,       | |
|  |  yahoo_player_key     probable pitchers, movement, zone,    | |
|  |                       game_type,         launch EV/angle,   | |
|  |                       season_year        barrel, catcher,   | |
|  |                                          base runners,      | |
|  |                                          xwOBA, xBA         | |
|  |                                                              | |
|  |  batting_stats_daily       pitching_stats_daily              | |
|  |  -------------------       --------------------              | |
|  |  (player_id, game_pk)      (player_id, game_pk)             | |
|  |  game_date, season_year    game_date, season_year           | |
|  |  team, is_home,            team, is_home,                   | |
|  |  batting_order_slot,       role_flag (SP/RP),               | |
|  |  opponent_pitcher_id       opponent_team                    | |
|  |  PA,AB,H,2B,3B,HR,        IP,H,ER,BB,SO,HR,               | |
|  |  RBI,R,BB,SO,SB,CS,       pitches,BF,HBP,WP,              | |
|  |  HBP,SF,GIDP              W,L,SV,HLD,BS,QS                | |
|  |  EV,barrel%,HH%,xwOBA     GB%,FIP,xwOBA-against            | |
|  |  fantasy_points            fantasy_points                   | |
|  |                                                              | |
|  +--------------------------------------------------------------+ |
|                                                                  |
|  +-------------- Reference Data --------------------------------+ |
|  |                                                              | |
|  |  park_factors               season_stats_batting             | |
|  |  ------------               --------------------             | |
|  |  (venue_id, season_year)    (player_id, season_year)         | |
|  |  runs/HR/H/2B/3B/BB/SO     FanGraphs: AVG,OBP,SLG,         | |
|  |  factors per ballpark       wOBA,ISO,BABIP,BB%,K%,          | |
|  |                             EV,barrel%,xwOBA,sprint,WAR     | |
|  |                                                              | |
|  |                             season_stats_pitching            | |
|  |                             ---------------------            | |
|  |                             (player_id, season_year)         | |
|  |                             FanGraphs: ERA,FIP,xFIP,         | |
|  |                             SIERA,K%,BB%,GB%,WHIP,          | |
|  |                             FB velo,xwOBA-against,WAR       | |
|  |                                                              | |
|  +--------------------------------------------------------------+ |
|                                                                  |
|  +-------------- Yahoo Fantasy ---------------------------------+ |
|  |                                                              | |
|  |  user_leagues         user_rosters      scoring_rules        | |
|  |  ------------         ------------      -------------        | |
|  |  yahoo_league_key     (league,player,   (league,stat_cat)    | |
|  |  name, season,         roster_date)     points_value         | |
|  |  scoring_type         position_slot     is_negative          | |
|  |                       acquisition_type                       | |
|  +--------------------------------------------------------------+ |
|                                                                  |
|  +-------------- ML Outputs -----------------------------------+ |
|  |                                                              | |
|  |  predictions                                                 | |
|  |  -----------                                                 | |
|  |  (player_id, game_pk, model_version)                         | |
|  |  predicted_mean, p10, p90, actual_points                     | |
|  |                                                              | |
|  +--------------------------------------------------------------+ |
|                                                                  |
|  Managed by: Alembic migrations (infra/migrations/)              |
|  Total: 12 tables                                                |
+-----------------------------------------------------------------+
         |
         | Phase 1+ reads from here
         v
+-----------------------------------------------------------------+
|                     ML PIPELINE  (Phase 1)                       |
|                     packages/ml/                                 |
+-----------------------------------------------------------------+
|                                                                  |
|  features/                models/              evaluation/       |
|  ----------               -------              -----------       |
|  F1: Rolling production   M1: Ridge baseline   Validation proto  |
|  F2: Days of rest         M2: LightGBM         Walk-forward CV   |
|  F3: Park factors              quantile        Baselines          |
|  F4: Opponent quality     M3: CQR + ACI        Calibration checks|
|  F5: Lineup slot               (production)    Single test touch  |
|  F6: Pitcher workload                                            |
|  F7: Prior-season baseline      training/                        |
|  F8-F13: P1 enhancements       ----------                       |
|                                  MLflow tracking                 |
|  Output:                         Walk-forward CV                 |
|  features_hitters.parquet        Optuna hyperparams              |
|  features_pitchers.parquet                                       |
|                                                                  |
+-----------------------------------------------------------------+
         |
         | Phase 2+
         v
+-----------------------------------------------------------------+
|                     FUTURE PHASES                                |
+-----------------------------------------------------------------+
|  Phase 2: FastAPI endpoints (/predict, /optimize-lineup, /trade) |
|  Phase 3: Monte Carlo season simulation                          |
|  Phase 4: Production ML infra (Docker, AWS ECS, monitoring)      |
|  Phase 5: Next.js frontend + auth                                |
|  Phase 6: Claude chatbot with RAG (+ news_articles table added)  |
+-----------------------------------------------------------------+
```

## DAG Schedule Order

1. `ingest_mlb_schedule` -- Daily 6am UTC (schedule + player metadata)
2. `ingest_boxscores` -- Daily 7am UTC (yesterday's completed game boxscores)
3. `ingest_statcast` -- Daily 8am UTC (Statcast pitch-level + quality-of-contact)
4. `ingest_yahoo` -- Daily 9am UTC (Yahoo Fantasy roster, scoring, matchups)
5. `seed_historical` -- Manual trigger only (2018-2025 backfill)

## Data Flow Summary

External APIs -> Airflow DAGs (scheduled) -> Postgres tables (UPSERT) -> ML feature builders read from DB -> model training -> predictions stored back in DB.
