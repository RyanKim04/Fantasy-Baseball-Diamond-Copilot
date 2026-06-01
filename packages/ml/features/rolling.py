"""F1: Rolling production window features.

Family: P0 | Source: Statcast daily aggregates
Anti-leakage justification: All rolling windows are computed per-player
sorted by game_date, then shifted by 1 position so that the feature for
game G uses only data from games strictly before G. The shift is applied
AFTER the rolling aggregation, guaranteeing the current game's stats are
never included.

Hitters: rolling fantasy points, H, HR, RBI, SB, OBP, SLG, xwOBA, BABIP
at windows {7, 14, 30} days, plus PA count per window.

Pitchers: rolling fantasy points, ERA, WHIP, K/9, BB/9, xFIP
at windows {14, 30, 60} days, plus BF count per window.

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import date

HITTER_WINDOWS: list[int] = [7, 14, 30]
PITCHER_WINDOWS: list[int] = [14, 30, 60]


class RollingProductionBuilder:
    """Feature builder for rolling production windows.

    requires_history_days: 60 (max pitcher window).
    """

    name: str = "rolling_production"
    requires_history_days: int = 60

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
        player_type: str = "hitter",
    ) -> pd.DataFrame:
        """Compute rolling production features.

        Parameters
        ----------
        stats : pd.DataFrame
            Raw stats with at least (player_id, game_pk, game_date) columns,
            plus the relevant counting stat columns. Must include historical
            rows needed for lookback. ``game_date`` must be a date or datetime.
        as_of_date : date
            Guard date: rows with game_date > as_of_date are excluded from
            computation.
        player_type : str
            "hitter" or "pitcher". Determines which stat set and windows to use.

        Returns
        -------
        pd.DataFrame
            Keyed on (player_id, game_pk) with rolling feature columns.
            Column naming: ``roll_{stat}_{window}d``.
        """
        df = stats.copy()
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        # Exclude any rows beyond as_of_date (guard against future data)
        df = df[df["game_date"] <= as_of_date]

        if df.empty:
            return pd.DataFrame(columns=["player_id", "game_pk"])

        df = df.sort_values(["player_id", "game_date", "game_pk"])

        if player_type == "hitter":
            return self._build_hitter(df)
        else:
            return self._build_pitcher(df)

    def _build_hitter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build rolling features for hitters."""
        # Ensure needed columns exist, fill missing with 0
        needed = [
            "fantasy_points", "h", "hr", "rbi", "sb", "ab", "bb", "hbp",
            "sf", "pa", "doubles", "triples", "so",
        ]
        for col in needed:
            if col not in df.columns:
                df[col] = 0

        # Compute PA if not present (AB + BB + HBP + SF)
        if df["pa"].sum() == 0:
            df["pa"] = (
                df["ab"].fillna(0) + df["bb"].fillna(0)
                + df["hbp"].fillna(0) + df["sf"].fillna(0)
            )

        # Compute rate stat components
        # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
        df["_obp_num"] = df["h"].fillna(0) + df["bb"].fillna(0) + df["hbp"].fillna(0)
        df["_obp_den"] = df["pa"].fillna(0)
        # SLG = TB / AB
        df["_tb"] = (
            df["h"].fillna(0)
            + df["doubles"].fillna(0)
            + 2 * df["triples"].fillna(0)
            + 3 * df["hr"].fillna(0)
        )
        df["_ab"] = df["ab"].fillna(0)
        # BABIP = (H - HR) / (AB - SO - HR + SF)
        df["_babip_num"] = df["h"].fillna(0) - df["hr"].fillna(0)
        df["_babip_den"] = (
            df["ab"].fillna(0) - df["so"].fillna(0)
            - df["hr"].fillna(0) + df["sf"].fillna(0)
        )

        result_frames = []
        grouped = df.groupby("player_id")

        for window in HITTER_WINDOWS:
            suffix = f"_{window}d"

            # Count stats: rolling sum then shift
            for stat in ["fantasy_points", "h", "hr", "rbi", "sb"]:
                col_name = f"roll_{stat}{suffix}"
                df[col_name] = (
                    grouped[stat]
                    .transform(lambda s: s.rolling(window, min_periods=1).sum())
                    .groupby(df["player_id"])
                    .shift(1)
                )

            # PA count in window
            df[f"roll_pa{suffix}"] = (
                grouped["pa"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )

            # OBP: rolling sum of numerator / rolling sum of denominator
            obp_num_roll = (
                grouped["_obp_num"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )
            obp_den_roll = (
                grouped["_obp_den"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )
            df[f"roll_obp{suffix}"] = np.where(
                obp_den_roll > 0, obp_num_roll / obp_den_roll, np.nan
            )

            # SLG: rolling TB / rolling AB
            tb_roll = (
                grouped["_tb"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )
            ab_roll = (
                grouped["_ab"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )
            df[f"roll_slg{suffix}"] = np.where(
                ab_roll > 0, tb_roll / ab_roll, np.nan
            )

            # BABIP
            babip_num_roll = (
                grouped["_babip_num"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )
            babip_den_roll = (
                grouped["_babip_den"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )
            df[f"roll_babip{suffix}"] = np.where(
                babip_den_roll > 0, babip_num_roll / babip_den_roll, np.nan
            )

            # xwOBA if available, else NaN
            if "xwoba" in df.columns:
                df[f"roll_xwoba{suffix}"] = (
                    grouped["xwoba"]
                    .transform(lambda s: s.rolling(window, min_periods=1).mean())
                    .groupby(df["player_id"])
                    .shift(1)
                )
            else:
                df[f"roll_xwoba{suffix}"] = np.nan

        # Collect feature columns
        feature_cols = [c for c in df.columns if c.startswith("roll_")]
        result = df[["player_id", "game_pk"] + feature_cols].copy()

        # Add missingness indicators for each window
        for window in HITTER_WINDOWS:
            pa_col = f"roll_pa_{window}d"
            result[f"roll_missing_{window}d"] = result[pa_col].isna().astype(int)

        return result

    def _build_pitcher(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build rolling features for pitchers."""
        needed = [
            "fantasy_points", "ip", "er", "h", "bb", "so", "hr_allowed",
            "batters_faced",
        ]
        for col in needed:
            if col not in df.columns:
                df[col] = 0

        # If batters_faced is all 0, approximate from IP
        if df["batters_faced"].sum() == 0:
            df["batters_faced"] = (df["ip"].fillna(0) * 3 + 1).astype(int)

        grouped = df.groupby("player_id")

        for window in PITCHER_WINDOWS:
            suffix = f"_{window}d"

            # Rolling fantasy points
            df[f"roll_fantasy_points{suffix}"] = (
                grouped["fantasy_points"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )

            # BF count
            df[f"roll_bf{suffix}"] = (
                grouped["batters_faced"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )

            # ERA = (ER / IP) * 9 over rolling window
            er_roll = (
                grouped["er"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )
            ip_roll = (
                grouped["ip"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )
            df[f"roll_era{suffix}"] = np.where(
                ip_roll > 0, (er_roll / ip_roll) * 9.0, np.nan
            )

            # WHIP = (H + BB) / IP
            h_roll = (
                grouped["h"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )
            bb_roll = (
                grouped["bb"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )
            df[f"roll_whip{suffix}"] = np.where(
                ip_roll > 0, (h_roll + bb_roll) / ip_roll, np.nan
            )

            # K/9 = SO / IP * 9
            so_roll = (
                grouped["so"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )
            df[f"roll_k9{suffix}"] = np.where(
                ip_roll > 0, (so_roll / ip_roll) * 9.0, np.nan
            )

            # BB/9 = BB / IP * 9
            df[f"roll_bb9{suffix}"] = np.where(
                ip_roll > 0, (bb_roll / ip_roll) * 9.0, np.nan
            )

            # xFIP approximation: uses HR_allowed rate
            # xFIP = ((13*(HR/IP*league_avg_HR_FB_rate) + 3*BB - 2*SO) / IP) + constant
            # Simplified: use FIP = ((13*HR + 3*BB - 2*SO) / IP) + 3.2
            hr_roll = (
                grouped["hr_allowed"]
                .transform(lambda s: s.rolling(window, min_periods=1).sum())
                .groupby(df["player_id"])
                .shift(1)
            )
            df[f"roll_xfip{suffix}"] = np.where(
                ip_roll > 0,
                ((13.0 * hr_roll + 3.0 * bb_roll - 2.0 * so_roll) / ip_roll) + 3.2,
                np.nan,
            )

        feature_cols = [c for c in df.columns if c.startswith("roll_")]
        result = df[["player_id", "game_pk"] + feature_cols].copy()

        # Missingness indicators
        for window in PITCHER_WINDOWS:
            bf_col = f"roll_bf_{window}d"
            result[f"roll_missing_{window}d"] = result[bf_col].isna().astype(int)

        return result
