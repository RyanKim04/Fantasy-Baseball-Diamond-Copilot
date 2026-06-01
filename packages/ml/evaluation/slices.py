"""Eligibility filter and slice generators per validation_protocol.md section 6.

Slices are diagnostic groupings applied at evaluation time. The headline
metrics are computed on the eligible population; slice metrics provide
per-group context but have no hard pass/fail threshold.

Owner: evaluator subagent.
"""

from __future__ import annotations

import pandas as pd  # noqa: TC002 - used at runtime throughout

# ---------------------------------------------------------------------------
# Eligibility filter constants (frozen per protocol section 6 / section 11.4)
# ---------------------------------------------------------------------------
HITTER_MIN_PA: int = 100
PITCHER_MIN_IP: int = 30


def filter_eligible_hitters(
    df: pd.DataFrame,
    season_stats: pd.DataFrame | None = None,
    pa_col: str = "pa",
    player_id_col: str = "player_id",
) -> pd.DataFrame:
    """Filter hitters to eligible population (>= HITTER_MIN_PA plate appearances).

    Parameters
    ----------
    df : pd.DataFrame
        Prediction/actuals DataFrame for hitters.
    season_stats : pd.DataFrame | None
        If provided, contains per-player season totals with at least
        player_id_col and pa_col. If None, pa_col must exist in df
        and is summed per player.
    pa_col : str
        Column name for plate appearances.
    player_id_col : str
        Column name for player ID.

    Returns
    -------
    pd.DataFrame
        Filtered to eligible hitters only.
    """
    if season_stats is not None:
        eligible_ids = set(
            season_stats.loc[season_stats[pa_col] >= HITTER_MIN_PA, player_id_col]
        )
    elif pa_col in df.columns:
        player_pa = df.groupby(player_id_col)[pa_col].sum()
        eligible_ids = set(player_pa[player_pa >= HITTER_MIN_PA].index)
    else:
        # If no PA data, return all rows (caller is responsible for providing data)
        return df

    return df[df[player_id_col].isin(eligible_ids)].copy()


def filter_eligible_pitchers(
    df: pd.DataFrame,
    season_stats: pd.DataFrame | None = None,
    ip_col: str = "ip",
    player_id_col: str = "player_id",
) -> pd.DataFrame:
    """Filter pitchers to eligible population (>= PITCHER_MIN_IP innings pitched).

    Parameters
    ----------
    df : pd.DataFrame
        Prediction/actuals DataFrame for pitchers.
    season_stats : pd.DataFrame | None
        If provided, contains per-player season totals with at least
        player_id_col and ip_col.
    ip_col : str
        Column name for innings pitched.
    player_id_col : str
        Column name for player ID.

    Returns
    -------
    pd.DataFrame
        Filtered to eligible pitchers only.
    """
    if season_stats is not None:
        eligible_ids = set(
            season_stats.loc[season_stats[ip_col] >= PITCHER_MIN_IP, player_id_col]
        )
    elif ip_col in df.columns:
        player_ip = df.groupby(player_id_col)[ip_col].sum()
        eligible_ids = set(player_ip[player_ip >= PITCHER_MIN_IP].index)
    else:
        return df

    return df[df[player_id_col].isin(eligible_ids)].copy()


# ---------------------------------------------------------------------------
# Slice generators per protocol section 6
# ---------------------------------------------------------------------------

def slice_by_position(
    df: pd.DataFrame,
    position_col: str = "position",
) -> dict[str, pd.DataFrame]:
    """Split DataFrame by fielding position.

    Hitter positions: C, 1B, 2B, 3B, SS, OF, DH.
    Pitcher positions: SP, RP.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain position_col.
    position_col : str
        Column name for position.

    Returns
    -------
    dict[str, pd.DataFrame]
        {position_label: filtered_df}.
    """
    if position_col not in df.columns:
        return {"all": df}

    slices: dict[str, pd.DataFrame] = {}
    for pos in df[position_col].unique():
        mask = df[position_col] == pos
        if mask.sum() > 0:
            slices[str(pos)] = df[mask].copy()
    return slices


def slice_by_experience(
    df: pd.DataFrame,
    mlb_debut_col: str = "mlb_debut_year",
    season_col: str = "season_year",
) -> dict[str, pd.DataFrame]:
    """Split into rookie vs veteran slices.

    Rookies: players in their MLB debut season as of the game date.
    Veterans: >= 3 prior MLB seasons.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain mlb_debut_col and season_col.

    Returns
    -------
    dict[str, pd.DataFrame]
        {"rookie": df, "veteran": df, "mid_career": df}
    """
    if mlb_debut_col not in df.columns or season_col not in df.columns:
        return {"all": df}

    years_exp = df[season_col] - df[mlb_debut_col]
    slices: dict[str, pd.DataFrame] = {}

    rookie_mask = years_exp == 0
    veteran_mask = years_exp >= 3
    mid_mask = ~rookie_mask & ~veteran_mask

    if rookie_mask.sum() > 0:
        slices["rookie"] = df[rookie_mask].copy()
    if veteran_mask.sum() > 0:
        slices["veteran"] = df[veteran_mask].copy()
    if mid_mask.sum() > 0:
        slices["mid_career"] = df[mid_mask].copy()

    return slices


def slice_by_playing_time_tertile(
    df: pd.DataFrame,
    games_col: str = "n_games",
    player_id_col: str = "player_id",
) -> dict[str, pd.DataFrame]:
    """Split eligible population into playing-time tertiles.

    Tertiles are based on total games played per player within the
    evaluation period.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain player_id_col. games_col is used if present,
        otherwise row count per player is used.
    player_id_col : str
        Column name for player ID.

    Returns
    -------
    dict[str, pd.DataFrame]
        {"low_pt": df, "mid_pt": df, "high_pt": df}
    """
    if games_col in df.columns:
        player_games = df.groupby(player_id_col)[games_col].sum()
    else:
        player_games = df.groupby(player_id_col).size()

    tercile_bounds = player_games.quantile([1 / 3, 2 / 3])
    t1, t2 = tercile_bounds.iloc[0], tercile_bounds.iloc[1]

    low_ids = set(player_games[player_games <= t1].index)
    mid_ids = set(player_games[(player_games > t1) & (player_games <= t2)].index)
    high_ids = set(player_games[player_games > t2].index)

    slices: dict[str, pd.DataFrame] = {}
    for label, ids in [("low_pt", low_ids), ("mid_pt", mid_ids), ("high_pt", high_ids)]:
        mask = df[player_id_col].isin(ids)
        if mask.sum() > 0:
            slices[label] = df[mask].copy()

    return slices
