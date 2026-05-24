"""Airflow DAG: ingest Statcast pitch-level data and aggregate to daily stats.

Default behavior: pull last 2 days. Idempotent via UPSERT on natural keys.
Schedule: daily at 08:00 UTC (after mlb_schedule at 06:00 and boxscores at 07:00).
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import pandas as pd
from airflow.sdk import dag, task
from pybaseball import statcast
from sqlalchemy.dialects.postgresql import insert

from packages.shared.db.engine import get_engine
from packages.shared.db.models import BattingStatsDaily, Pitch, PitchingStatsDaily

logger = logging.getLogger(__name__)

default_args = {
    "owner": "diamond-copilot",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

# Columns we keep from pybaseball, mapped to our Pitch model column names
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

# Critical fields that must not be null
_REQUIRED_FIELDS = ["game_pk", "at_bat_number", "pitch_number", "batter", "pitcher"]

# Events that count as outs for IP calculation
_OUT_EVENTS = {
    "field_out",
    "strikeout",
    "strikeout_double_play",
    "grounded_into_double_play",
    "double_play",
    "triple_play",
    "force_out",
    "fielders_choice",
    "fielders_choice_out",
    "sac_fly",
    "sac_bunt",
    "sac_fly_double_play",
    "sac_bunt_double_play",
    "caught_stealing_2b",
    "caught_stealing_3b",
    "caught_stealing_home",
}


@dag(
    dag_id="ingest_statcast",
    schedule="0 8 * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["ingestion", "statcast"],
    doc_md=__doc__,
)
def ingest_statcast(
    start_date: date | None = None,
    end_date: date | None = None,
) -> None:
    """Pull Statcast pitch-level data and aggregate to daily batting/pitching stats.

    Default: last 2 days. Idempotent via UPSERT on natural keys.
    """

    @task(retries=3, retry_delay=timedelta(seconds=60))
    def fetch_statcast_range(start_date: date | None, end_date: date | None) -> pd.DataFrame:
        """Fetch raw Statcast data from pybaseball for the given date range.

        Returns a DataFrame with one row per pitch.
        """
        if start_date is None:
            end_date = date.today() - timedelta(days=1)
            start_date = end_date - timedelta(days=1)
        elif end_date is None:
            end_date = start_date

        start_str = str(start_date)
        end_str = str(end_date)

        logger.info("Fetching Statcast data from %s to %s", start_str, end_str)
        df = statcast(start_dt=start_str, end_dt=end_str)

        if df is None or df.empty:
            logger.warning("No Statcast data returned for %s to %s", start_str, end_str)
            return pd.DataFrame()

        logger.info("Fetched %d raw pitches", len(df))
        return df

    @task
    def validate_and_transform(raw_df: pd.DataFrame) -> pd.DataFrame:
        """Validate raw Statcast rows against StatcastPitchRow schema.

        Drops invalid rows, logs warnings. Returns cleaned DataFrame.
        """
        if raw_df.empty:
            logger.warning("Empty DataFrame received; nothing to validate.")
            return pd.DataFrame()

        initial_count = len(raw_df)

        # Drop rows missing critical fields
        raw_df = raw_df.dropna(subset=_REQUIRED_FIELDS)
        dropped_count = initial_count - len(raw_df)
        if dropped_count > 0:
            logger.warning(
                "Dropped %d rows with null critical fields (of %d total)",
                dropped_count,
                initial_count,
            )

        # Select only columns we need (those present in pybaseball output)
        available_cols = [c for c in _PYBASEBALL_COLUMN_MAP if c in raw_df.columns]
        df = raw_df[available_cols].copy()

        # Rename to match our model
        rename_map = {k: _PYBASEBALL_COLUMN_MAP[k] for k in available_cols}
        df = df.rename(columns=rename_map)

        # Convert inning_topbot: "Top"/"Bot" stays as-is (model stores String(3))
        # Convert game_date to date type if it's a datetime
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"]).dt.date

        # Convert integer columns that may have float representations
        int_cols = [
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
        for col in int_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                # Keep as nullable int (NaN-safe)
                df[col] = df[col].astype("Int64")

        logger.info("Validated and transformed %d pitches", len(df))
        return df

    @task
    def upsert_pitches(df: pd.DataFrame) -> int:
        """UPSERT pitch-level rows into the pitches table.

        Returns the number of rows upserted.
        """
        if df.empty:
            logger.info("No pitches to upsert.")
            return 0

        engine = get_engine()
        table = Pitch.__table__

        # Replace NaN with None for database insertion
        df = df.where(pd.notna(df), None)
        records = df.to_dict("records")

        total_upserted = 0
        chunk_size = 1000

        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            stmt = insert(table).values(chunk)

            # On conflict, update all non-key columns
            update_cols = {
                col.name: stmt.excluded[col.name]
                for col in table.columns
                if col.name not in ("id", "game_pk", "at_bat_number", "pitch_number", "created_at")
            }
            stmt = stmt.on_conflict_do_update(
                constraint="uq_pitch_event",
                set_=update_cols,
            )

            with engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()
            total_upserted += len(chunk)

        logger.info("Upserted %d pitch rows", total_upserted)
        return total_upserted

    @task
    def aggregate_daily_batting(pitches_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate pitch-level data to per-player, per-game batting stats.

        Returns a DataFrame matching StatcastBattingDaily schema.
        """
        if pitches_df.empty:
            return pd.DataFrame()

        # Work with events column for PA-level aggregation
        # A plate appearance ends when 'events' is not null
        pa_df = pitches_df[pitches_df["events"].notna()].copy()

        if pa_df.empty:
            return pd.DataFrame()

        def _agg_batting(group: pd.DataFrame) -> pd.Series:
            events = group["events"]

            # Count event types
            singles = (events == "single").sum()
            doubles = (events == "double").sum()
            triples = (events == "triple").sum()
            hr = (events == "home_run").sum()
            h = singles + doubles + triples + hr

            bb = (events.isin(["walk"])).sum()
            hbp = (events == "hit_by_pitch").sum()
            so = (events.isin(["strikeout", "strikeout_double_play"])).sum()
            sf = (events.isin(["sac_fly", "sac_fly_double_play"])).sum()
            gidp = (events.isin(["grounded_into_double_play", "double_play"])).sum()

            pa = len(group)
            ab = pa - bb - hbp - sf - (events == "sac_bunt").sum()

            # Quality of contact (from all pitches in the group's at-bats)
            batted = group[group["launch_speed"].notna()]
            exit_velocity_avg = batted["launch_speed"].mean() if len(batted) > 0 else None
            barrel_pct = group["barrel"].mean() if group["barrel"].notna().any() else None
            hard_hit_pct = (batted["launch_speed"] >= 95).mean() if len(batted) > 0 else None
            xwoba = (
                group["estimated_woba_using_speedangle"].mean()
                if group["estimated_woba_using_speedangle"].notna().any()
                else None
            )

            return pd.Series(
                {
                    "pa": int(pa),
                    "ab": int(max(ab, 0)),
                    "h": int(h),
                    "doubles": int(doubles),
                    "triples": int(triples),
                    "hr": int(hr),
                    "bb": int(bb),
                    "so": int(so),
                    "hbp": int(hbp),
                    "sf": int(sf),
                    "gidp": int(gidp),
                    "exit_velocity_avg": exit_velocity_avg,
                    "barrel_pct": barrel_pct,
                    "hard_hit_pct": hard_hit_pct,
                    "xwoba": xwoba,
                }
            )

        result = pa_df.groupby(["batter_id", "game_pk", "game_date"]).apply(
            _agg_batting, include_groups=False
        )
        result = result.reset_index()
        result = result.rename(columns={"batter_id": "player_id"})

        # Add season_year from game_date
        result["season_year"] = result["game_date"].apply(
            lambda d: d.year if d is not None else None
        )

        logger.info("Aggregated batting stats for %d player-games", len(result))
        return result

    @task
    def aggregate_daily_pitching(pitches_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate pitch-level data to per-player, per-game pitching stats.

        Returns a DataFrame matching StatcastPitchingDaily schema.
        """
        if pitches_df.empty:
            return pd.DataFrame()

        def _agg_pitching(group: pd.DataFrame) -> pd.Series:
            events = group["events"]
            events_notna = events[events.notna()]

            # Outs recorded
            outs = events_notna.isin(_OUT_EVENTS).sum()
            # Double plays count as 2 outs
            double_play_events = {
                "grounded_into_double_play",
                "double_play",
                "strikeout_double_play",
                "sac_fly_double_play",
                "sac_bunt_double_play",
            }
            outs += events_notna.isin(double_play_events).sum()  # extra out for DPs
            triple_play_outs = (events_notna == "triple_play").sum()
            outs += triple_play_outs * 2  # triple play already counted once above

            ip = round(outs / 3, 1)

            h = events_notna.isin(["single", "double", "triple", "home_run"]).sum()
            bb = events_notna.isin(["walk"]).sum()
            so = events_notna.isin(["strikeout", "strikeout_double_play"]).sum()
            hr_allowed = (events_notna == "home_run").sum()
            hbp = (events_notna == "hit_by_pitch").sum()

            pitches_thrown = len(group)
            batters_faced = group["at_bat_number"].nunique()

            # Ground ball percentage (using launch angle as proxy: < 10 degrees)
            batted = group[group["launch_speed"].notna()]
            if len(batted) > 0:
                gb_pct = (
                    (batted["launch_angle"] < 10).mean()
                    if "launch_angle" in batted.columns
                    else None
                )
            else:
                gb_pct = None

            xwoba_against = (
                group["estimated_woba_using_speedangle"].mean()
                if group["estimated_woba_using_speedangle"].notna().any()
                else None
            )

            return pd.Series(
                {
                    "ip": float(ip),
                    "h": int(h),
                    "bb": int(bb),
                    "so": int(so),
                    "hr_allowed": int(hr_allowed),
                    "hbp": int(hbp),
                    "pitches_thrown": int(pitches_thrown),
                    "batters_faced": int(batters_faced),
                    "gb_pct": gb_pct,
                    "xwoba_against": xwoba_against,
                }
            )

        result = pitches_df.groupby(["pitcher_id", "game_pk", "game_date"]).apply(
            _agg_pitching, include_groups=False
        )
        result = result.reset_index()
        result = result.rename(columns={"pitcher_id": "player_id"})

        # Add season_year from game_date
        result["season_year"] = result["game_date"].apply(
            lambda d: d.year if d is not None else None
        )

        logger.info("Aggregated pitching stats for %d player-games", len(result))
        return result

    @task
    def upsert_daily_stats(batting_df: pd.DataFrame, pitching_df: pd.DataFrame) -> int:
        """UPSERT daily batting and pitching stats into their respective tables.

        Returns total rows upserted across both tables.
        """
        engine = get_engine()
        total = 0
        chunk_size = 1000

        # Upsert batting stats
        if not batting_df.empty:
            batting_table = BattingStatsDaily.__table__
            batting_df = batting_df.where(pd.notna(batting_df), None)

            # Only keep columns that exist in the table
            valid_batting_cols = [c.name for c in batting_table.columns if c.name != "id"]
            batting_cols_to_use = [c for c in batting_df.columns if c in valid_batting_cols]
            batting_records = batting_df[batting_cols_to_use].to_dict("records")

            for i in range(0, len(batting_records), chunk_size):
                chunk = batting_records[i : i + chunk_size]
                stmt = insert(batting_table).values(chunk)
                update_cols = {
                    col.name: stmt.excluded[col.name]
                    for col in batting_table.columns
                    if col.name not in ("id", "player_id", "game_pk", "created_at")
                }
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_batting_player_game",
                    set_=update_cols,
                )
                with engine.connect() as conn:
                    conn.execute(stmt)
                    conn.commit()
                total += len(chunk)

            logger.info("Upserted %d batting stat rows", total)

        # Upsert pitching stats
        pitching_count = 0
        if not pitching_df.empty:
            pitching_table = PitchingStatsDaily.__table__
            pitching_df = pitching_df.where(pd.notna(pitching_df), None)

            valid_pitching_cols = [c.name for c in pitching_table.columns if c.name != "id"]
            pitching_cols_to_use = [c for c in pitching_df.columns if c in valid_pitching_cols]
            pitching_records = pitching_df[pitching_cols_to_use].to_dict("records")

            for i in range(0, len(pitching_records), chunk_size):
                chunk = pitching_records[i : i + chunk_size]
                stmt = insert(pitching_table).values(chunk)
                update_cols = {
                    col.name: stmt.excluded[col.name]
                    for col in pitching_table.columns
                    if col.name not in ("id", "player_id", "game_pk", "created_at")
                }
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_pitching_player_game",
                    set_=update_cols,
                )
                with engine.connect() as conn:
                    conn.execute(stmt)
                    conn.commit()
                pitching_count += len(chunk)

            logger.info("Upserted %d pitching stat rows", pitching_count)

        total += pitching_count
        logger.info("Total daily stats upserted: %d", total)
        return total

    # Task wiring
    raw = fetch_statcast_range(start_date, end_date)
    clean = validate_and_transform(raw)
    upsert_pitches(clean)
    batting = aggregate_daily_batting(clean)
    pitching = aggregate_daily_pitching(clean)
    upsert_daily_stats(batting, pitching)


ingest_statcast_dag = ingest_statcast()
