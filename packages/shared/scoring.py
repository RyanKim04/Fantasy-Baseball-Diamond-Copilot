"""Fantasy points calculator.

Computes per-game fantasy points from raw batting/pitching stats using a frozen
set of league scoring rules. This is the single source of truth for the target
variable in the ML pipeline.

The calculator is deterministic: same stats + same rules = same points. It never
accesses the database directly; callers provide both stats and rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from packages.shared.schemas.ml import ScoringRulesSnapshot


# ---------------------------------------------------------------------------
# Default Yahoo scoring rules (standard points league)
# Used as fallback when no league-specific rules are available.
# ---------------------------------------------------------------------------
DEFAULT_BATTING_RULES: dict[str, float] = {
    "H": 1.0,      # singles implicitly (H - 2B - 3B - HR) via decomposition
    "2B": 1.0,     # extra on top of H
    "3B": 2.0,     # extra on top of H
    "HR": 3.0,     # extra on top of H
    "R": 1.0,
    "RBI": 1.0,
    "BB": 1.0,
    "HBP": 1.0,
    "SB": 2.0,
    "CS": -1.0,
    "SO": -0.5,
}

DEFAULT_PITCHING_RULES: dict[str, float] = {
    "IP": 3.0,
    "SO": 1.0,
    "W": 5.0,
    "L": -3.0,
    "SV": 5.0,
    "HLD": 3.0,
    "ER": -2.0,
    "H": -0.5,
    "BB": -0.5,
    "HBP": -0.5,
    "QS": 3.0,
}


# ---------------------------------------------------------------------------
# Column mapping: scoring rule stat_category -> DB/DataFrame column name
# ---------------------------------------------------------------------------
BATTING_STAT_TO_COLUMN: dict[str, str] = {
    "H": "h",
    "2B": "doubles",
    "3B": "triples",
    "HR": "hr",
    "R": "r",
    "RBI": "rbi",
    "BB": "bb",
    "HBP": "hbp",
    "SB": "sb",
    "CS": "cs",
    "SO": "so",
    "SF": "sf",
    "GIDP": "gidp",
}

PITCHING_STAT_TO_COLUMN: dict[str, str] = {
    "IP": "ip",
    "SO": "so",
    "W": "wins",
    "L": "losses",
    "SV": "saves",
    "HLD": "holds",
    "ER": "er",
    "H": "h",
    "BB": "bb",
    "HBP": "hbp",
    "QS": "quality_starts",
    "BS": "blown_saves",
    "WP": "wp",
    "R": "r",
    "HR": "hr_allowed",
}


def compute_batting_fantasy_points(
    stats: pd.DataFrame,
    rules: dict[str, float] | None = None,
) -> pd.Series:
    """Compute per-row batting fantasy points from a stats DataFrame.

    Parameters
    ----------
    stats : pd.DataFrame
        Must contain columns matching BATTING_STAT_TO_COLUMN values.
        Rows are (player_id, game_pk) observations.
    rules : dict[str, float] | None
        Mapping of stat_category -> points_value. Uses DEFAULT_BATTING_RULES if None.

    Returns
    -------
    pd.Series
        Fantasy points per row, same index as input.
    """
    import pandas as pd

    if rules is None:
        rules = DEFAULT_BATTING_RULES

    points = pd.Series(0.0, index=stats.index)

    for stat_cat, pts_value in rules.items():
        col = BATTING_STAT_TO_COLUMN.get(stat_cat)
        if col is None or col not in stats.columns:
            continue
        points = points + stats[col].fillna(0).astype(float) * pts_value

    return points


def compute_pitching_fantasy_points(
    stats: pd.DataFrame,
    rules: dict[str, float] | None = None,
) -> pd.Series:
    """Compute per-row pitching fantasy points from a stats DataFrame.

    Parameters
    ----------
    stats : pd.DataFrame
        Must contain columns matching PITCHING_STAT_TO_COLUMN values.
    rules : dict[str, float] | None
        Uses DEFAULT_PITCHING_RULES if None.

    Returns
    -------
    pd.Series
        Fantasy points per row.
    """
    import pandas as pd

    if rules is None:
        rules = DEFAULT_PITCHING_RULES

    points = pd.Series(0.0, index=stats.index)

    for stat_cat, pts_value in rules.items():
        col = PITCHING_STAT_TO_COLUMN.get(stat_cat)
        if col is None or col not in stats.columns:
            continue
        points = points + stats[col].fillna(0).astype(float) * pts_value

    return points


def scoring_rules_to_dicts(
    snapshot: ScoringRulesSnapshot,
) -> tuple[dict[str, float], dict[str, float]]:
    """Convert a ScoringRulesSnapshot into separate batting and pitching rule dicts.

    Rules whose stat_category appears in BATTING_STAT_TO_COLUMN go to batting;
    those in PITCHING_STAT_TO_COLUMN go to pitching. Some categories (e.g., "BB")
    appear in both; they are included in both dicts with their respective values.

    Parameters
    ----------
    snapshot : ScoringRulesSnapshot
        Frozen scoring rules from a league.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        (batting_rules, pitching_rules)
    """
    batting: dict[str, float] = {}
    pitching: dict[str, float] = {}

    for entry in snapshot.rules:
        cat = entry.stat_category
        val = entry.points_value if not entry.is_negative else -abs(entry.points_value)
        if cat in BATTING_STAT_TO_COLUMN:
            batting[cat] = val
        if cat in PITCHING_STAT_TO_COLUMN:
            pitching[cat] = val

    return batting, pitching
