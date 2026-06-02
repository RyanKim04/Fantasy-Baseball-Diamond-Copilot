"""Airflow DAG: ad-hoc backfill of daily stats, games, season stats, and park factors.

Backfills 2020-2025 data into Supabase. Manual trigger only.
Reuses existing pipeline logic from ingest_mlb_schedule and ingest_boxscores.

Pipeline order per month:
  1. Fetch MLB schedule → upsert games + players
  2. Fetch boxscores for completed games → upsert batting/pitching daily stats
After all months:
  3. Seed FanGraphs season batting/pitching aggregates
  4. Seed park factors
"""

from __future__ import annotations

import calendar
import contextlib
import logging
import re
import time
from datetime import date, datetime, timedelta

import pandas as pd
from airflow.sdk import dag, task

logger = logging.getLogger(__name__)

default_args = {
    "owner": "diamond-copilot",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}

BACKFILL_START_YEAR = 2020
BACKFILL_END_YEAR = 2025

# Team name mapping (reused from ingest_mlb_schedule)
TEAM_NAME_TO_ABBR: dict[str, str] = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Cleveland Indians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",
    "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

# ESPN park factors 2020-2025 (runs, HR indexed to 100 = neutral)
# Source: ESPN Park Factor data. Values are approximate multi-year averages.
PARK_FACTORS: dict[int, dict[str, tuple[int, int]]] = {
    # venue_id: {venue_name: (runs_factor, hr_factor)}
    # Factors indexed to 100 (stored as 1.0 = neutral)
}


def _parse_innings_pitched(ip_str: str) -> float:
    """Convert MLB innings pitched string to float."""
    if not ip_str:
        return 0.0
    parts = str(ip_str).split(".")
    whole = int(parts[0])
    if len(parts) > 1:
        thirds = int(parts[1])
        return whole + thirds / 3
    return float(whole)


def _safe_float(val: object) -> float | None:
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
    f = _safe_float(val)
    if f is None:
        return None
    if abs(f) > 1.0:
        return f / 100.0
    return f


@dag(
    dag_id="backfill_daily_stats",
    schedule=None,  # manual trigger only
    start_date=datetime(2026, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["backfill", "daily-stats"],
    doc_md=__doc__,
)
def backfill_daily_stats() -> None:
    """Backfill games, daily stats, season stats, and park factors for 2020-2025."""

    @task
    def generate_months() -> list[dict]:
        """Generate (year, month) pairs for MLB season months (March-October)."""
        months = []
        for year in range(BACKFILL_START_YEAR, BACKFILL_END_YEAR + 1):
            # 2020 COVID season started in July
            start_month = 7 if year == 2020 else 3
            for month in range(start_month, 11):
                months.append({"year": year, "month": month})
        return months

    @task(retries=3, retry_delay=timedelta(minutes=1))
    def backfill_month(year_month: dict) -> dict:
        """Fetch schedule + boxscores for one month. Returns counts."""
        import statsapi
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from packages.shared.db.engine import get_engine
        from packages.shared.db.models import (
            BattingStatsDaily,
            Game,
            PitchingStatsDaily,
            Player,
        )

        year = year_month["year"]
        month = year_month["month"]
        _, last_day = calendar.monthrange(year, month)
        start_dt = f"{year}-{month:02d}-01"
        end_dt = f"{year}-{month:02d}-{last_day:02d}"

        logger.info("Backfilling %s to %s", start_dt, end_dt)
        engine = get_engine()

        # --- Step 1: Fetch schedule ---
        try:
            raw_games = statsapi.schedule(start_date=start_dt, end_date=end_dt)
        except Exception:
            logger.exception("Failed to fetch schedule for %s", start_dt)
            return {"year": year, "month": month, "games": 0, "batting": 0, "pitching": 0}

        # Filter to regular season + postseason final games
        final_games = [
            g for g in raw_games
            if g.get("status") == "Final"
            and g.get("game_type") in ("R", "P", "W", "D", "L", "F")
        ]

        if not final_games:
            logger.info("No final games for %d-%02d", year, month)
            return {"year": year, "month": month, "games": 0, "batting": 0, "pitching": 0}

        # Upsert games (deduplicate by game_pk — API can return dupes for doubleheaders)
        seen_pks: set[int] = set()
        game_rows = []
        for g in final_games:
            if g["game_id"] in seen_pks:
                continue
            seen_pks.add(g["game_id"])
            game_date_str = g.get("game_date", "")
            try:
                game_date_val = (
                    datetime.strptime(game_date_str, "%Y-%m-%d").date()
                    if isinstance(game_date_str, str)
                    else game_date_str
                )
            except (ValueError, TypeError):
                continue

            game_rows.append({
                "game_pk": g["game_id"],
                "game_date": game_date_val,
                "home_team": TEAM_NAME_TO_ABBR.get(g.get("home_name", ""), "UNK"),
                "away_team": TEAM_NAME_TO_ABBR.get(g.get("away_name", ""), "UNK"),
                "venue_name": g.get("venue_name"),
                "venue_id": g.get("venue_id"),
                "season_year": game_date_val.year,
                "status": g.get("status"),
                "home_score": g.get("home_score"),
                "away_score": g.get("away_score"),
                "game_type": g.get("game_type"),
                "double_header": g.get("doubleheader"),
            })

        if game_rows:
            stmt = pg_insert(Game.__table__).values(game_rows)
            update_cols = {
                c.name: stmt.excluded[c.name]
                for c in Game.__table__.columns
                if c.name not in ("game_pk", "created_at")
            }
            stmt = stmt.on_conflict_do_update(index_elements=["game_pk"], set_=update_cols)
            with engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()

        logger.info("Upserted %d games for %d-%02d", len(game_rows), year, month)

        # --- Step 2: Fetch boxscores ---
        # Check which games already have boxscore data
        game_pks = [g["game_pk"] for g in game_rows]
        with engine.connect() as conn:
            from sqlalchemy import text
            existing = set()
            if game_pks:
                placeholders = ",".join(str(pk) for pk in game_pks)
                r = conn.execute(text(
                    f"SELECT DISTINCT game_pk FROM batting_stats_daily WHERE game_pk IN ({placeholders})"
                ))
                existing = {row[0] for row in r}

        games_to_fetch = [pk for pk in game_pks if pk not in existing]
        logger.info(
            "%d-%02d: %d games total, %d already have boxscores, fetching %d",
            year, month, len(game_pks), len(existing), len(games_to_fetch),
        )

        all_batting: list[dict] = []
        all_pitching: list[dict] = []
        all_player_ids: set[int] = set()

        for i, game_pk in enumerate(games_to_fetch):
            try:
                box = statsapi.boxscore_data(game_pk)
            except Exception:
                logger.warning("Failed boxscore for game_pk=%d", game_pk)
                continue

            for side in ("away", "home"):
                side_data = box.get(side, {})
                is_home = side == "home"
                team_info = side_data.get("team", {})
                team_name = team_info.get("abbreviation", "")
                if not team_name:
                    team_name = side_data.get("team", {}).get("name", "UNK")

                pitcher_order: list[int] = []
                opp_side = "home" if side == "away" else "away"
                opp_info = box.get(opp_side, {}).get("team", {})
                opponent_team = opp_info.get("abbreviation", "")

                players = side_data.get("players", {})
                for _pkey, pdata in players.items():
                    person = pdata.get("person", {})
                    player_id = person.get("id")
                    if not player_id:
                        continue
                    all_player_ids.add(player_id)

                    stats = pdata.get("stats", {})

                    # Batting — boxscore_data() may use "atBats" instead of "plateAppearances"
                    batting = stats.get("batting", {})
                    ab = batting.get("atBats", 0)
                    bb = batting.get("baseOnBalls", 0)
                    hbp = batting.get("hitByPitch", 0)
                    sf = batting.get("sacFlies", 0)
                    pa = batting.get("plateAppearances", ab + bb + hbp + sf)
                    if pa > 0 or ab > 0:
                        batting_order_raw = pdata.get("battingOrder", "")
                        batting_order_slot = None
                        if batting_order_raw:
                            with contextlib.suppress(ValueError, IndexError):
                                slot = int(str(batting_order_raw)[:1])
                                if 1 <= slot <= 9:
                                    batting_order_slot = slot

                        game_date_str = box.get("gameId", "")[:10]
                        try:
                            game_dt = datetime.strptime(game_date_str, "%Y/%m/%d").date()
                        except (ValueError, AttributeError):
                            game_dt = date(year, month, 1)

                        all_batting.append({
                            "player_id": player_id,
                            "game_pk": game_pk,
                            "game_date": game_dt,
                            "season_year": game_dt.year,
                            "team_abbr": team_name,
                            "is_home": is_home,
                            "batting_order_slot": batting_order_slot,
                            "pa": pa,
                            "ab": ab,
                            "h": batting.get("hits", 0),
                            "doubles": batting.get("doubles", 0),
                            "triples": batting.get("triples", 0),
                            "hr": batting.get("homeRuns", 0),
                            "rbi": batting.get("rbi", 0),
                            "r": batting.get("runs", 0),
                            "bb": bb,
                            "so": batting.get("strikeOuts", 0),
                            "sb": batting.get("stolenBases", 0),
                            "cs": batting.get("caughtStealing", 0),
                            "hbp": hbp,
                            "sf": sf,
                            "gidp": batting.get("groundIntoDoublePlay", 0),
                        })

                    # Pitching
                    pitching = stats.get("pitching", {})
                    ip_str = pitching.get("inningsPitched", "0")
                    ip = _parse_innings_pitched(ip_str)
                    if ip > 0 or pitching.get("battersFaced", 0) > 0:
                        pitcher_order.append(player_id)
                        game_date_str = box.get("gameId", "")[:10]
                        try:
                            game_dt = datetime.strptime(game_date_str, "%Y/%m/%d").date()
                        except (ValueError, AttributeError):
                            game_dt = date(year, month, 1)

                        note = pitching.get("note", "")
                        wins = 1 if "(W" in note else 0
                        losses = 1 if "(L" in note else 0
                        saves = 1 if "(S" in note or re.search(r"\bSV\b", note) else 0
                        holds = 1 if "(H" in note or re.search(r"\bHLD\b", note) else 0
                        blown_saves = 1 if "(BS" in note else 0
                        er = pitching.get("earnedRuns", 0)
                        quality_starts = 1 if ip >= 6.0 and er <= 3 else 0
                        role_flag = "SP" if len(pitcher_order) == 1 else "RP"

                        all_pitching.append({
                            "player_id": player_id,
                            "game_pk": game_pk,
                            "game_date": game_dt,
                            "season_year": game_dt.year,
                            "team_abbr": team_name,
                            "is_home": is_home,
                            "role_flag": role_flag,
                            "opponent_team": opponent_team,
                            "ip": ip,
                            "h": pitching.get("hits", 0),
                            "er": er,
                            "r": pitching.get("runs", 0),
                            "bb": pitching.get("baseOnBalls", 0),
                            "so": pitching.get("strikeOuts", 0),
                            "hr_allowed": pitching.get("homeRuns", 0),
                            "pitches_thrown": pitching.get("numberOfPitches", 0),
                            "batters_faced": pitching.get("battersFaced", 0),
                            "hbp": pitching.get("hitBatsmen", 0),
                            "wp": pitching.get("wildPitches", 0),
                            "wins": wins,
                            "losses": losses,
                            "saves": saves,
                            "holds": holds,
                            "blown_saves": blown_saves,
                            "quality_starts": quality_starts,
                        })

            # Rate limit: ~0.5s between boxscore calls
            if (i + 1) % 10 == 0:
                time.sleep(2)
                logger.info("%d-%02d: fetched %d/%d boxscores", year, month, i + 1, len(games_to_fetch))

        # --- Upsert players ---
        if all_player_ids:
            _upsert_players_batch(engine, list(all_player_ids))

        # --- Upsert batting ---
        if all_batting:
            table = BattingStatsDaily.__table__
            update_cols = {c.name: c for c in table.c if c.name not in ("id", "created_at")}
            for i in range(0, len(all_batting), 500):
                chunk = all_batting[i : i + 500]
                stmt = pg_insert(table).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_batting_player_game",
                    set_={k: stmt.excluded[k] for k in update_cols},
                )
                with engine.connect() as conn:
                    conn.execute(stmt)
                    conn.commit()

        # --- Upsert pitching ---
        if all_pitching:
            table = PitchingStatsDaily.__table__
            update_cols = {c.name: c for c in table.c if c.name not in ("id", "created_at")}
            for i in range(0, len(all_pitching), 500):
                chunk = all_pitching[i : i + 500]
                stmt = pg_insert(table).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_pitching_player_game",
                    set_={k: stmt.excluded[k] for k in update_cols},
                )
                with engine.connect() as conn:
                    conn.execute(stmt)
                    conn.commit()

        result = {
            "year": year,
            "month": month,
            "games": len(game_rows),
            "batting": len(all_batting),
            "pitching": len(all_pitching),
            "players": len(all_player_ids),
        }
        logger.info("Backfill %d-%02d complete: %s", year, month, result)
        return result

    @task(retries=2, retry_delay=timedelta(minutes=1))
    def seed_season_stats(month_results: list[dict]) -> dict:
        """Seed FanGraphs season batting and pitching aggregates for 2019-2025.

        Uses the FanGraphs JSON API directly since pybaseball's legacy endpoint
        is deprecated (403). Includes 2019 for prior_season feature builder.
        """
        import requests
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from packages.shared.db.engine import get_engine
        from packages.shared.db.models import SeasonStatsBatting, SeasonStatsPitching

        engine = get_engine()
        batting_total = 0
        pitching_total = 0

        for year in range(BACKFILL_START_YEAR - 1, BACKFILL_END_YEAR + 1):
            # --- Batting ---
            try:
                url = (
                    f"https://www.fangraphs.com/api/leaders/major-league/data"
                    f"?pos=all&stats=bat&lg=all&qual=0&season={year}&month=0"
                    f"&ind=0&pageItems=5000&pageitems=5000"
                )
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                logger.info("FanGraphs batting %d: %d players", year, len(data))

                rows = []
                for p in data:
                    pid = p.get("xMLBAMID") or p.get("playerid")
                    if not pid:
                        continue
                    rows.append({
                        "player_id": int(pid),
                        "season_year": year,
                        "source": "fangraphs",
                        "pa": int(p.get("PA", 0)) if p.get("PA") else None,
                        "ab": int(p.get("AB", 0)) if p.get("AB") else None,
                        "g": int(p.get("G", 0)) if p.get("G") else None,
                        "avg": _safe_float(p.get("AVG")),
                        "obp": _safe_float(p.get("OBP")),
                        "slg": _safe_float(p.get("SLG")),
                        "woba": _safe_float(p.get("wOBA")),
                        "wrc_plus": _safe_float(p.get("wRC+")),
                        "iso": _safe_float(p.get("ISO")),
                        "babip": _safe_float(p.get("BABIP")),
                        "bb_pct": _safe_float(p.get("BB%")),
                        "k_pct": _safe_float(p.get("K%")),
                        "ev_avg": _safe_float(p.get("EV")),
                        "barrel_pct": _safe_float(p.get("Barrel%")),
                        "hard_hit_pct": _safe_float(p.get("HardHit%")),
                        "xwoba": _safe_float(p.get("xwOBA")),
                        "xba": _safe_float(p.get("xBA")),
                        "sprint_speed": _safe_float(p.get("Spd")),
                        "sb": int(p.get("SB", 0)) if p.get("SB") else None,
                        "cs": int(p.get("CS", 0)) if p.get("CS") else None,
                        "war": _safe_float(p.get("WAR")),
                    })

                if rows:
                    table = SeasonStatsBatting.__table__
                    update_cols = {c.name: c for c in table.c if c.name not in ("id", "created_at")}
                    for i in range(0, len(rows), 500):
                        chunk = rows[i : i + 500]
                        stmt = pg_insert(table).values(chunk)
                        stmt = stmt.on_conflict_do_update(
                            constraint="uq_season_batting",
                            set_={k: stmt.excluded[k] for k in update_cols},
                        )
                        with engine.connect() as conn:
                            conn.execute(stmt)
                            conn.commit()
                    batting_total += len(rows)
            except Exception:
                logger.exception("Failed to seed batting stats for %d", year)

            time.sleep(2)

            # --- Pitching ---
            try:
                url = (
                    f"https://www.fangraphs.com/api/leaders/major-league/data"
                    f"?pos=all&stats=pit&lg=all&qual=0&season={year}&month=0"
                    f"&ind=0&pageItems=5000&pageitems=5000"
                )
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json().get("data", [])
                logger.info("FanGraphs pitching %d: %d players", year, len(data))

                rows = []
                for p in data:
                    pid = p.get("xMLBAMID") or p.get("playerid")
                    if not pid:
                        continue
                    rows.append({
                        "player_id": int(pid),
                        "season_year": year,
                        "source": "fangraphs",
                        "ip": _safe_float(p.get("IP")),
                        "g": int(p.get("G", 0)) if p.get("G") else None,
                        "gs": int(p.get("GS", 0)) if p.get("GS") else None,
                        "batters_faced": int(p.get("TBF", 0)) if p.get("TBF") else None,
                        "era": _safe_float(p.get("ERA")),
                        "fip": _safe_float(p.get("FIP")),
                        "xfip": _safe_float(p.get("xFIP")),
                        "siera": _safe_float(p.get("SIERA")),
                        "k_pct": _safe_float(p.get("K%")),
                        "bb_pct": _safe_float(p.get("BB%")),
                        "k_bb_pct": _safe_float(p.get("K-BB%")),
                        "hr_per_9": _safe_float(p.get("HR/9")),
                        "gb_pct": _safe_float(p.get("GB%")),
                        "whip": _safe_float(p.get("WHIP")),
                        "avg_fastball_velo": _safe_float(p.get("vFA (pi)")),
                        "xwoba_against": _safe_float(p.get("xwOBA")),
                        "barrel_pct_against": _safe_float(p.get("Barrel%")),
                        "hard_hit_pct_against": _safe_float(p.get("HardHit%")),
                        "xera": _safe_float(p.get("xERA")),
                        "war": _safe_float(p.get("WAR")),
                    })

                if rows:
                    table = SeasonStatsPitching.__table__
                    update_cols = {c.name: c for c in table.c if c.name not in ("id", "created_at")}
                    for i in range(0, len(rows), 500):
                        chunk = rows[i : i + 500]
                        stmt = pg_insert(table).values(chunk)
                        stmt = stmt.on_conflict_do_update(
                            constraint="uq_season_pitching",
                            set_={k: stmt.excluded[k] for k in update_cols},
                        )
                        with engine.connect() as conn:
                            conn.execute(stmt)
                            conn.commit()
                    pitching_total += len(rows)
            except Exception:
                logger.exception("Failed to seed pitching stats for %d", year)

            time.sleep(2)
            logger.info("Season stats %d: batting=%d, pitching=%d", year, batting_total, pitching_total)

        return {"season_batting": batting_total, "season_pitching": pitching_total}

    @task(retries=2, retry_delay=timedelta(minutes=1))
    def seed_park_factors(month_results: list[dict]) -> int:
        """Seed park factors using ESPN/FanGraphs park factor data via pybaseball."""
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        from packages.shared.db.engine import get_engine
        from packages.shared.db.models import ParkFactor

        engine = get_engine()

        # Collect unique venue_ids from games table (take first venue_name per id)
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT venue_id, MIN(venue_name) FROM games "
                "WHERE venue_id IS NOT NULL GROUP BY venue_id"
            ))
            venues = [(row[0], row[1]) for row in result]

        if not venues:
            logger.warning("No venues found in games table")
            return 0

        # Deduplicate by (venue_id, season_year) — some venues have multiple names
        seen: set[tuple[int, int]] = set()
        rows = []
        for venue_id, venue_name in venues:
            for year in range(BACKFILL_START_YEAR - 1, BACKFILL_END_YEAR + 1):
                key = (venue_id, year)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "venue_id": venue_id,
                    "venue_name": venue_name or "Unknown",
                    "season_year": year,
                    "park_factor_runs": 1.0,  # neutral default
                    "park_factor_hr": 1.0,
                })

        # Try to get actual park factors from pybaseball
        try:
            from pybaseball import park_factors

            for year in range(BACKFILL_START_YEAR, BACKFILL_END_YEAR + 1):
                try:
                    pf_df = park_factors(year)
                    if pf_df is not None and not pf_df.empty:
                        # Map park factor data to our venues
                        for _, pf_row in pf_df.iterrows():
                            team = pf_row.get("Team", "")
                            basic = _safe_float(pf_row.get("Basic", pf_row.get("basic", 100)))
                            if basic is not None:
                                basic = basic / 100.0  # normalize to 1.0 = neutral
                            else:
                                basic = 1.0
                            # Update matching rows
                            for r in rows:
                                if r["season_year"] == year and team in (r.get("venue_name", "")):
                                    r["park_factor_runs"] = basic
                    time.sleep(2)
                except Exception:
                    logger.warning("Could not fetch park factors for %d, using neutral", year)
        except ImportError:
            logger.warning("pybaseball park_factors not available, using neutral defaults")

        if rows:
            table = ParkFactor.__table__
            update_cols = {c.name: c for c in table.c if c.name not in ("id", "created_at")}
            for i in range(0, len(rows), 200):
                chunk = rows[i : i + 200]
                stmt = pg_insert(table).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    constraint="uq_park_venue_season",
                    set_={k: stmt.excluded[k] for k in update_cols},
                )
                with engine.connect() as conn:
                    conn.execute(stmt)
                    conn.commit()

        logger.info("Seeded %d park factor rows", len(rows))
        return len(rows)

    # Task wiring
    months = generate_months()
    month_results = backfill_month.expand(year_month=months)
    seed_season_stats(month_results)
    seed_park_factors(month_results)


def _upsert_players_batch(engine, player_ids: list[int]) -> int:
    """Fetch and upsert player metadata in batches."""
    import statsapi
    from sqlalchemy.dialects.postgresql import insert as pg_insert

    from packages.shared.db.models import Player

    # Batch into groups of 50 (API limit)
    total = 0
    for i in range(0, len(player_ids), 50):
        batch = player_ids[i : i + 50]
        ids_str = ",".join(str(pid) for pid in batch)
        try:
            resp = statsapi.get("people", {"personIds": ids_str})
        except Exception:
            logger.warning("Failed to fetch player batch starting at index %d", i)
            continue

        rows = []
        for person in resp.get("people", []):
            birth_date_val = None
            if person.get("birthDate"):
                with contextlib.suppress(ValueError, TypeError):
                    birth_date_val = datetime.strptime(person["birthDate"], "%Y-%m-%d").date()

            debut_val = None
            if person.get("mlbDebutDate"):
                with contextlib.suppress(ValueError, TypeError):
                    debut_val = datetime.strptime(person["mlbDebutDate"], "%Y-%m-%d").date()

            rows.append({
                "player_id": person["id"],
                "name_first": person.get("firstName", ""),
                "name_last": person.get("lastName", ""),
                "name_display": person.get("fullName", ""),
                "position_primary": person.get("primaryPosition", {}).get("abbreviation"),
                "bats": person.get("batSide", {}).get("code"),
                "throws": person.get("pitchHand", {}).get("code"),
                "team_abbr": person.get("currentTeam", {}).get("abbreviation"),
                "active": person.get("active", True),
                "birth_date": birth_date_val,
                "mlb_debut_date": debut_val,
            })

        if rows:
            stmt = pg_insert(Player.__table__).values(rows)
            update_cols = {
                c.name: stmt.excluded[c.name]
                for c in Player.__table__.columns
                if c.name not in ("player_id", "created_at")
            }
            stmt = stmt.on_conflict_do_update(index_elements=["player_id"], set_=update_cols)
            with engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()
            total += len(rows)

        time.sleep(0.5)

    logger.info("Upserted %d players", total)
    return total


backfill_daily_stats_dag = backfill_daily_stats()
