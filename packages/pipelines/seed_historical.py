"""Airflow DAG: one-time historical backfill of Statcast data (2018-2025).

Designed to be run once during initial setup via manual trigger.
Uses dynamic task mapping to process each year in parallel.
Idempotent -- safe to re-run. Not scheduled (manual trigger only).
"""

from __future__ import annotations

import calendar
import logging
import time
from datetime import datetime, timedelta

import pandas as pd
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)

default_args = {
    "owner": "diamond-copilot",
    "retries": 0,
}

# Column mapping from pybaseball statcast() output to our Pitch model
_PYBASEBALL_COLUMN_MAP = {
    "game_pk": "game_pk",
    "at_bat_number": "at_bat_number",
    "pitch_number": "pitch_number",
    "batter": "batter_id",
    "pitcher": "pitcher_id",
    "game_date": "game_date",
    "pitch_type": "pitch_type",
    "release_speed": "release_speed",
    "release_spin_rate": "release_spin_rate",
    "release_extension": "release_extension",
    "pfx_x": "pfx_x",
    "pfx_z": "pfx_z",
    "plate_x": "plate_x",
    "plate_z": "plate_z",
    "launch_speed": "launch_speed",
    "launch_angle": "launch_angle",
    "hit_distance_sc": "hit_distance",
    "barrel": "barrel",
    "events": "events",
    "description": "description",
    "type": "type",
    "zone": "zone",
    "stand": "stand",
    "p_throws": "p_throws",
    "inning": "inning",
    "inning_topbot": "inning_topbot",
    "outs_when_up": "outs_when_up",
    "balls": "balls",
    "strikes": "strikes",
    "on_1b": "on_1b",
    "on_2b": "on_2b",
    "on_3b": "on_3b",
    "estimated_woba_using_speedangle": "estimated_woba_using_speedangle",
    "estimated_ba_using_speedangle": "estimated_ba_using_speedangle",
    "fielder_2": "fielder_2",
}

_INT_COLUMNS = [
    "game_pk",
    "at_bat_number",
    "pitch_number",
    "batter_id",
    "pitcher_id",
    "barrel",
    "zone",
    "inning",
    "outs_when_up",
    "balls",
    "strikes",
    "on_1b",
    "on_2b",
    "on_3b",
    "fielder_2",
]


def _safe_float(val: object) -> float | None:
    """Convert a value to float, handling None, NaN, and percentage strings."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    if isinstance(val, str):
        val = val.replace("%", "").replace(" ", "")
        if not val:
            return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _pct_to_float(val: object) -> float | None:
    """Convert a percentage value (e.g., '25.3 %' or 0.253) to a float fraction."""
    f = _safe_float(val)
    if f is None:
        return None
    # If value > 1, assume it's a percentage and divide by 100
    if f > 1.0:
        return f / 100.0
    return f


def _validate_and_transform_statcast(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Validate and transform raw Statcast DataFrame for Pitch table insert."""
    if raw_df.empty:
        return raw_df

    required = ["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"]
    before = len(raw_df)
    df = raw_df.dropna(subset=required)
    dropped = before - len(df)
    if dropped > 0:
        logger.info("Dropped %d rows missing required fields", dropped)

    # Select and rename columns
    available = {k: v for k, v in _PYBASEBALL_COLUMN_MAP.items() if k in df.columns}
    df = df[list(available.keys())].rename(columns=available)

    # Convert game_date to Python date
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

    # Cast integer columns
    for col in _INT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("Int64")

    return df


def _upsert_pitches(df: pd.DataFrame) -> int:
    """UPSERT pitch rows into the pitches table."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from packages.shared.db.engine import get_engine
    from packages.shared.db.models import Pitch

    if df.empty:
        return 0

    engine = get_engine()
    table = Pitch.__table__
    update_cols = {c.name: c for c in table.c if c.name not in ("id", "created_at")}

    # Replace NaN with None
    records = df.where(pd.notna(df), None).to_dict("records")

    total = 0
    chunk_size = 1000
    for i in range(0, len(records), chunk_size):
        chunk = records[i : i + chunk_size]
        stmt = pg_insert(table).values(chunk)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_pitch_event",
            set_={k: stmt.excluded[k] for k in update_cols},
        )
        with engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()
        total += len(chunk)

    return total


def _seed_statcast_month(year: int, month: int) -> int:
    """Fetch and ingest one month of Statcast data."""
    from pybaseball import statcast

    # MLB season: March through October
    if month < 3 or month > 10:
        return 0

    _, last_day = calendar.monthrange(year, month)
    start_dt = f"{year}-{month:02d}-01"
    end_dt = f"{year}-{month:02d}-{last_day:02d}"

    logger.info("Fetching Statcast data for %s to %s", start_dt, end_dt)

    try:
        raw_df = statcast(start_dt=start_dt, end_dt=end_dt)
    except Exception:
        logger.exception("Failed to fetch Statcast for %d-%02d", year, month)
        return 0

    if raw_df is None or raw_df.empty:
        logger.info("No Statcast data for %d-%02d", year, month)
        return 0

    df = _validate_and_transform_statcast(raw_df)
    rows = _upsert_pitches(df)
    logger.info("Seeded %d-%02d: %d pitches upserted", year, month, rows)
    return rows


def _seed_season_batting(year: int) -> int:
    """Seed FanGraphs season batting stats for a given year."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from packages.shared.db.engine import get_engine
    from packages.shared.db.models import SeasonStatsBatting

    try:
        from pybaseball import batting_stats

        df = batting_stats(year)
    except Exception:
        logger.exception("Failed to fetch batting_stats for %d", year)
        return 0

    if df is None or df.empty:
        return 0

    engine = get_engine()
    table = SeasonStatsBatting.__table__

    rows = []
    for _, row in df.iterrows():
        player_id = row.get("IDfg")
        if player_id is None:
            continue
        rows.append(
            {
                "player_id": int(player_id),
                "season_year": year,
                "source": "fangraphs",
                "pa": int(row.get("PA", 0)) if pd.notna(row.get("PA")) else None,
                "ab": int(row.get("AB", 0)) if pd.notna(row.get("AB")) else None,
                "g": int(row.get("G", 0)) if pd.notna(row.get("G")) else None,
                "avg": _safe_float(row.get("AVG")),
                "obp": _safe_float(row.get("OBP")),
                "slg": _safe_float(row.get("SLG")),
                "woba": _safe_float(row.get("wOBA")),
                "wrc_plus": _safe_float(row.get("wRC+")),
                "iso": _safe_float(row.get("ISO")),
                "babip": _safe_float(row.get("BABIP")),
                "bb_pct": _pct_to_float(row.get("BB%")),
                "k_pct": _pct_to_float(row.get("K%")),
                "ev_avg": _safe_float(row.get("EV")),
                "barrel_pct": _pct_to_float(row.get("Barrel%")),
                "hard_hit_pct": _pct_to_float(row.get("HardHit%")),
                "xwoba": _safe_float(row.get("xwOBA")),
                "xba": _safe_float(row.get("xBA")),
                "sprint_speed": _safe_float(row.get("Spd")),
                "sb": int(row.get("SB", 0)) if pd.notna(row.get("SB")) else None,
                "cs": int(row.get("CS", 0)) if pd.notna(row.get("CS")) else None,
                "war": _safe_float(row.get("WAR")),
            }
        )

    if not rows:
        return 0

    update_cols = {c.name: c for c in table.c if c.name not in ("id", "created_at")}
    chunk_size = 500
    total = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        stmt = pg_insert(table).values(chunk)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_season_batting",
            set_={k: stmt.excluded[k] for k in update_cols},
        )
        with engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()
        total += len(chunk)

    logger.info("Seeded %d season batting rows for %d", total, year)
    return total


def _seed_season_pitching(year: int) -> int:
    """Seed FanGraphs season pitching stats for a given year."""
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from packages.shared.db.engine import get_engine
    from packages.shared.db.models import SeasonStatsPitching

    try:
        from pybaseball import pitching_stats

        df = pitching_stats(year)
    except Exception:
        logger.exception("Failed to fetch pitching_stats for %d", year)
        return 0

    if df is None or df.empty:
        return 0

    engine = get_engine()
    table = SeasonStatsPitching.__table__

    rows = []
    for _, row in df.iterrows():
        player_id = row.get("IDfg")
        if player_id is None:
            continue
        rows.append(
            {
                "player_id": int(player_id),
                "season_year": year,
                "source": "fangraphs",
                "ip": _safe_float(row.get("IP")),
                "g": int(row.get("G", 0)) if pd.notna(row.get("G")) else None,
                "gs": int(row.get("GS", 0)) if pd.notna(row.get("GS")) else None,
                "batters_faced": (int(row.get("TBF", 0)) if pd.notna(row.get("TBF")) else None),
                "era": _safe_float(row.get("ERA")),
                "fip": _safe_float(row.get("FIP")),
                "xfip": _safe_float(row.get("xFIP")),
                "siera": _safe_float(row.get("SIERA")),
                "k_pct": _pct_to_float(row.get("K%")),
                "bb_pct": _pct_to_float(row.get("BB%")),
                "k_bb_pct": _pct_to_float(row.get("K-BB%")),
                "hr_per_9": _safe_float(row.get("HR/9")),
                "gb_pct": _pct_to_float(row.get("GB%")),
                "whip": _safe_float(row.get("WHIP")),
                "avg_fastball_velo": _safe_float(row.get("vFA (pi)")),
                "xwoba_against": _safe_float(row.get("xwOBA")),
                "barrel_pct_against": _pct_to_float(row.get("Barrel%")),
                "hard_hit_pct_against": _pct_to_float(row.get("HardHit%")),
                "xera": _safe_float(row.get("xERA")),
                "war": _safe_float(row.get("WAR")),
            }
        )

    if not rows:
        return 0

    update_cols = {c.name: c for c in table.c if c.name not in ("id", "created_at")}
    chunk_size = 500
    total = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        stmt = pg_insert(table).values(chunk)
        stmt = stmt.on_conflict_do_update(
            constraint="uq_season_pitching",
            set_={k: stmt.excluded[k] for k in update_cols},
        )
        with engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()
        total += len(chunk)

    logger.info("Seeded %d season pitching rows for %d", total, year)
    return total


@dag(
    dag_id="seed_historical",
    schedule=None,  # manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["backfill", "statcast"],
    doc_md=__doc__,
)
def seed_historical() -> None:
    """Backfill historical Statcast data for 2018-2025.

    Uses Airflow dynamic task mapping to process each year independently.
    """

    @task(retries=2, retry_delay=timedelta(minutes=2))
    def seed_year(year: int) -> int:
        """Ingest all data for a single MLB season.

        Processes Statcast month-by-month (March through October),
        then seeds FanGraphs season aggregates for batting and pitching.
        """
        total = 0

        # Seed Statcast pitch data month by month
        for month in range(3, 11):  # March through October
            rows = _seed_statcast_month(year, month)
            total += rows
            time.sleep(5)  # rate limiting between months

        # Seed FanGraphs season aggregates
        total += _seed_season_batting(year)
        time.sleep(3)
        total += _seed_season_pitching(year)

        logger.info("Year %d complete: %d total rows", year, total)
        return total

    # Task wiring -- dynamic task mapping over year range
    years = list(range(2018, 2026))  # 2018 through 2025 inclusive
    seed_year.expand(year=years)


seed_historical_dag = seed_historical()
