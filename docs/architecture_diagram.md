# Architecture Diagram

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES (External)                      │
├──────────────────┬──────────────────┬────────────────┬──────────────┤
│   pybaseball     │  MLB Stats API   │  Yahoo Fantasy │  FanGraphs   │
│  (Statcast)      │  (statsapi.mlb)  │  API (OAuth)   │  (via pybb)  │
│                  │                  │                │              │
│ • Pitch-level    │ • Game schedule  │ • Roster       │ • Season     │
│   data (90+ cols)│ • Boxscores      │ • Scoring rules│   aggregates │
│ • 2018-present   │ • Lineups        │ • Matchups     │ • Park       │
│                  │ • Player info    │ • League info  │   factors    │
└────────┬─────────┴────────┬─────────┴───────┬────────┴──────┬───────┘
         │                  │                 │               │
         ▼                  ▼                 ▼               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     AIRFLOW  (Orchestration)                        │
│                     localhost:8081 (Web UI)                          │
├─────────────────┬─────────────────┬────────────────┬────────────────┤
│ ingest_statcast │ ingest_boxscores│ ingest_yahoo   │seed_historical │
│ DAG             │ DAG             │ DAG            │ DAG            │
│ Daily 6am UTC   │ Daily 7am UTC   │ Daily 9am UTC  │ Manual trigger │
│                 │                 │                │                │
│ • Fetch pitches │ • Fetch game    │ • Refresh OAuth│ • Backfill     │
│   via pybaseball│   boxscores via │ • Pull roster, │   2018-2025    │
│ • Validate rows │   MLB Stats API │   scoring,     │ • Pitches +    │
│ • UPSERT into   │ • UPSERT daily  │   matchups     │   boxscores +  │
│   pitches table │   batting +     │ • UPSERT Yahoo │   season stats │
│ • Fill Statcast │   pitching stats│   tables       │   + park       │
│   columns in    │   (counting)    │                │   factors      │
│   daily stats   │                 │                │                │
├─────────────────┼─────────────────┤                │                │
│ ingest_schedule │                 │                │                │
│ DAG             │                 │                │                │
│ Daily 8am UTC   │                 │                │                │
│                 │                 │                │                │
│ • Fetch 7-day   │                 │                │                │
│   schedule      │                 │                │                │
│ • Probable      │                 │                │                │
│   pitchers      │                 │                │                │
│ • UPSERT games  │                 │                │                │
│   + players     │                 │                │                │
└────────┬────────┴────────┬────────┴───────┬────────┴───────┬────────┘
         │                 │                │                │
         ▼                 ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   POSTGRES  (Supabase / Local Docker)               │
│                   localhost:5432                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────── MLB Data ───────────────────────────────┐     │
│  │                                                            │     │
│  │  players              games              pitches           │     │
│  │  ────────             ─────              ───────           │     │
│  │  player_id (PK)       game_pk (PK)       id (PK)          │     │
│  │  name, position,      game_date,         game_pk (FK)     │     │
│  │  bats, throws,        home/away team,    batter/pitcher   │     │
│  │  team, active,        venue_id/name,     pitch_type,      │     │
│  │  birth/debut date,    scores, status,    speed, spin,     │     │
│  │  yahoo_player_key     probable pitchers, movement, zone,  │     │
│  │                       game_type,         launch EV/angle, │     │
│  │                       season_year        barrel, catcher, │     │
│  │                                          base runners,    │     │
│  │                                          xwOBA, xBA       │     │
│  │                                                            │     │
│  │  batting_stats_daily       pitching_stats_daily            │     │
│  │  ───────────────────       ────────────────────            │     │
│  │  (player_id, game_pk)      (player_id, game_pk)           │     │
│  │  game_date, season_year    game_date, season_year         │     │
│  │  team, is_home,            team, is_home,                 │     │
│  │  batting_order_slot,       role_flag (SP/RP),             │     │
│  │  opponent_pitcher_id       opponent_team                  │     │
│  │  PA,AB,H,2B,3B,HR,        IP,H,ER,BB,SO,HR,             │     │
│  │  RBI,R,BB,SO,SB,CS,       pitches,BF,HBP,WP,            │     │
│  │  HBP,SF,GIDP              W,L,SV,HLD,BS,QS              │     │
│  │  EV,barrel%,HH%,xwOBA     GB%,FIP,xwOBA-against         │     │
│  │  fantasy_points            fantasy_points                 │     │
│  │                                                            │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                     │
│  ┌──────────── Reference Data ────────────────────────────────┐     │
│  │                                                            │     │
│  │  park_factors               season_stats_batting           │     │
│  │  ────────────               ──────────────────────         │     │
│  │  (venue_id, season_year)    (player_id, season_year)       │     │
│  │  runs/HR/H/2B/3B/BB/SO     FanGraphs: AVG,OBP,SLG,       │     │
│  │  factors per ballpark       wOBA,ISO,BABIP,BB%,K%,        │     │
│  │                             EV,barrel%,xwOBA,sprint,WAR   │     │
│  │                                                            │     │
│  │                             season_stats_pitching          │     │
│  │                             ───────────────────────        │     │
│  │                             (player_id, season_year)       │     │
│  │                             FanGraphs: ERA,FIP,xFIP,      │     │
│  │                             SIERA,K%,BB%,GB%,WHIP,        │     │
│  │                             FB velo,xwOBA-against,WAR     │     │
│  │                                                            │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                     │
│  ┌──────────── Yahoo Fantasy ─────────────────────────────────┐     │
│  │                                                            │     │
│  │  user_leagues         user_rosters      scoring_rules      │     │
│  │  ────────────         ────────────      ─────────────      │     │
│  │  yahoo_league_key     (league,player,   (league,stat_cat)  │     │
│  │  name, season,         roster_date)     points_value       │     │
│  │  scoring_type         position_slot     is_negative        │     │
│  │                       acquisition_type                     │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                     │
│  ┌──────────── ML Outputs ────────────────────────────────────┐     │
│  │                                                            │     │
│  │  predictions                                               │     │
│  │  ───────────                                               │     │
│  │  (player_id, game_pk, model_version)                       │     │
│  │  predicted_mean, p10, p90, actual_points                   │     │
│  │                                                            │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                     │
│  Managed by: Alembic migrations (infra/migrations/)                 │
│  Total: 12 tables                                                   │
└─────────────────────────────────────────────────────────────────────┘
         │
         │ Phase 1+ reads from here
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     ML PIPELINE  (Phase 1)                          │
│                     packages/ml/                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  features/                models/              evaluation/          │
│  ──────────               ───────              ───────────          │
│  F1: Rolling production   M1: Ridge baseline   Validation protocol  │
│  F2: Days of rest         M2: LightGBM         Walk-forward CV      │
│  F3: Park factors              quantile        Baselines             │
│  F4: Opponent quality     M3: CQR + ACI        Calibration checks   │
│  F5: Lineup slot               (production)    Single test touch     │
│  F6: Pitcher workload                                               │
│  F7: Prior-season baseline      training/                           │
│  F8-F13: P1 enhancements        ──────────                         │
│                                  MLflow tracking                    │
│  Output:                         Walk-forward CV                    │
│  features_hitters.parquet        Optuna hyperparams                 │
│  features_pitchers.parquet                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
         │
         │ Phase 2+
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FUTURE PHASES                                   │
├─────────────────────────────────────────────────────────────────────┤
│  Phase 2: FastAPI endpoints (/predict, /optimize-lineup, /trade)    │
│  Phase 3: Monte Carlo season simulation                             │
│  Phase 4: Production ML infra (Docker, AWS ECS, monitoring)         │
│  Phase 5: Next.js frontend + auth                                   │
│  Phase 6: Claude chatbot with RAG (+ news_articles table added)     │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

External APIs → Airflow DAGs (scheduled) → Postgres tables (UPSERT) → ML feature builders read from DB → model training → predictions stored back in DB.
