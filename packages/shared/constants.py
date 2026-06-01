"""Shared constants used across pipeline modules.

Column mappings and type definitions that are used by both the daily
ingest (ingest_statcast.py) and the historical backfill (seed_historical.py).
"""

from __future__ import annotations

# Columns we keep from pybaseball statcast() output, mapped to our Pitch model column names
PYBASEBALL_COLUMN_MAP: dict[str, str] = {
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

# Integer columns that may arrive as floats from pybaseball and need casting
INT_COLUMNS: list[str] = [
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
