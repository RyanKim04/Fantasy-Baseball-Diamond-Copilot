"""F4: Opponent quality features.

Family: P0 | Source: Statcast season-to-date aggregates with strict date filter
Anti-leakage justification: Opponent stats are computed using a rolling
30-day window ending at game_date - 1 day. Opponent identity is observed
(we know who tonight's SP is); only their stats through yesterday are used.

For hitters: opposing pitcher's recent K/9, BB/9, ERA, WHIP.
For pitchers: opposing team's recent OBP, SLG, K%.

Owner: feature-engineer subagent.
Spec: docs/feature_plan.md, F4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from datetime import date


class OpponentQualityBuilder:
    """Feature builder for opponent quality signals.

    requires_history_days: 30 (rolling window for opponent stats).
    """

    name: str = "opponent_quality"
    requires_history_days: int = 30

    def build(
        self,
        stats: pd.DataFrame,
        as_of_date: date,
        opponent_pitching: pd.DataFrame | None = None,
        opponent_batting: pd.DataFrame | None = None,
        games: pd.DataFrame | None = None,
        player_type: str = "hitter",
    ) -> pd.DataFrame:
        """Compute opponent quality features.

        Parameters
        ----------
        stats : pd.DataFrame
            Raw stats with (player_id, game_pk, game_date, team_id).
        as_of_date : date
            Latest date for feature computation (window ends here).
        opponent_pitching : pd.DataFrame | None
            Pitching daily stats for all pitchers (used for hitter features).
            Columns: player_id, game_pk, game_date, team_id, ip, er, so, bb, h.
        opponent_batting : pd.DataFrame | None
            Batting daily stats for all batters (used for pitcher features).
            Columns: player_id, game_pk, game_date, team_id, ab, h, bb, hbp,
            sf, so, doubles, triples, hr.
        games : pd.DataFrame | None
            Game schedule with: game_pk, home_team_id, away_team_id,
            home_probable_pitcher_id, away_probable_pitcher_id.
        player_type : str
            "hitter" or "pitcher".

        Returns
        -------
        pd.DataFrame
            Feature columns prefixed with 'opp_'.
        """
        df = stats[["player_id", "game_pk", "game_date"]].copy()
        if "team_id" in stats.columns:
            df["team_id"] = stats["team_id"]
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
        df = df[df["game_date"] <= as_of_date]

        if df.empty or games is None:
            return self._empty_result(df, player_type)

        if player_type == "hitter":
            return self._build_hitter_opp(df, games, opponent_pitching, as_of_date)
        else:
            return self._build_pitcher_opp(df, games, opponent_batting, as_of_date)

    def _empty_result(
        self, df: pd.DataFrame, player_type: str
    ) -> pd.DataFrame:
        """Return empty DataFrame with correct columns."""
        if player_type == "hitter":
            cols = [
                "opp_era_30d", "opp_whip_30d", "opp_k9_30d",
                "opp_bb9_30d", "opp_missing",
            ]
        else:
            cols = [
                "opp_obp_30d", "opp_slg_30d", "opp_k_pct_30d", "opp_missing",
            ]
        return pd.DataFrame(
            columns=["player_id", "game_pk"] + cols
        )

    def _build_hitter_opp(
        self,
        df: pd.DataFrame,
        games: pd.DataFrame,
        opponent_pitching: pd.DataFrame | None,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Opponent pitcher quality for hitters."""
        from datetime import timedelta

        # Determine opposing pitcher for each game
        g = games[["game_pk", "home_team_id", "away_team_id"]].copy()
        opp_pitcher_cols = []
        if "home_probable_pitcher_id" in games.columns:
            g["home_probable_pitcher_id"] = games["home_probable_pitcher_id"]
            g["away_probable_pitcher_id"] = games["away_probable_pitcher_id"]
            opp_pitcher_cols = [
                "home_probable_pitcher_id", "away_probable_pitcher_id",
            ]

        df = df.merge(g, on="game_pk", how="left")

        if "team_id" in df.columns and opp_pitcher_cols:
            # If hitter is on home team, opponent pitcher is away_probable_pitcher
            df["opp_pitcher_id"] = np.where(
                df["team_id"] == df["home_team_id"],
                df["away_probable_pitcher_id"],
                df["home_probable_pitcher_id"],
            )
        else:
            df["opp_pitcher_id"] = np.nan

        # Compute rolling 30-day stats for each opposing pitcher
        default_cols = {
            "opp_era_30d": np.nan,
            "opp_whip_30d": np.nan,
            "opp_k9_30d": np.nan,
            "opp_bb9_30d": np.nan,
        }
        for col, val in default_cols.items():
            df[col] = val

        if opponent_pitching is not None and not df["opp_pitcher_id"].isna().all():
            pitching = opponent_pitching.copy()
            pitching["game_date"] = pd.to_datetime(pitching["game_date"]).dt.date
            pitching = pitching[pitching["game_date"] <= as_of_date]

            # For each unique (opp_pitcher_id, game_date), compute their
            # trailing 30-day stats ending at game_date - 1
            unique_matchups = df[["opp_pitcher_id", "game_date"]].drop_duplicates()
            unique_matchups = unique_matchups.dropna(subset=["opp_pitcher_id"])

            opp_stats = {}
            for _, row in unique_matchups.iterrows():
                pid = row["opp_pitcher_id"]
                gd = row["game_date"]
                cutoff = gd - timedelta(days=1)
                start = gd - timedelta(days=31)

                mask = (
                    (pitching["player_id"] == pid)
                    & (pitching["game_date"] >= start)
                    & (pitching["game_date"] <= cutoff)
                )
                subset = pitching.loc[mask]
                if subset.empty:
                    continue

                ip_total = subset["ip"].fillna(0).sum()
                if ip_total > 0:
                    er_total = subset["er"].fillna(0).sum()
                    h_total = subset["h"].fillna(0).sum()
                    bb_total = subset["bb"].fillna(0).sum()
                    so_total = subset["so"].fillna(0).sum()
                    opp_stats[(pid, gd)] = {
                        "opp_era_30d": (er_total / ip_total) * 9.0,
                        "opp_whip_30d": (h_total + bb_total) / ip_total,
                        "opp_k9_30d": (so_total / ip_total) * 9.0,
                        "opp_bb9_30d": (bb_total / ip_total) * 9.0,
                    }

            if opp_stats:
                opp_df = pd.DataFrame.from_dict(opp_stats, orient="index")
                opp_df.index = pd.MultiIndex.from_tuples(
                    opp_df.index, names=["opp_pitcher_id", "game_date"]
                )
                opp_df = opp_df.reset_index()

                df = df.merge(
                    opp_df,
                    on=["opp_pitcher_id", "game_date"],
                    how="left",
                    suffixes=("", "_new"),
                )
                for col in default_cols:
                    new_col = f"{col}_new"
                    if new_col in df.columns:
                        df[col] = df[new_col].combine_first(df[col])
                        df = df.drop(columns=[new_col])

        df["opp_missing"] = (df["opp_era_30d"].isna()).astype(int)
        # Impute with league-average-ish defaults
        df["opp_era_30d"] = df["opp_era_30d"].fillna(4.50)
        df["opp_whip_30d"] = df["opp_whip_30d"].fillna(1.30)
        df["opp_k9_30d"] = df["opp_k9_30d"].fillna(8.5)
        df["opp_bb9_30d"] = df["opp_bb9_30d"].fillna(3.2)

        feature_cols = [c for c in df.columns if c.startswith("opp_")]
        return df[["player_id", "game_pk"] + feature_cols].copy()

    def _build_pitcher_opp(
        self,
        df: pd.DataFrame,
        games: pd.DataFrame,
        opponent_batting: pd.DataFrame | None,
        as_of_date: date,
    ) -> pd.DataFrame:
        """Opponent lineup quality for pitchers."""
        from datetime import timedelta

        g = games[["game_pk", "home_team_id", "away_team_id"]].copy()
        df = df.merge(g, on="game_pk", how="left")

        if "team_id" in df.columns:
            df["opp_team_id"] = np.where(
                df["team_id"] == df["home_team_id"],
                df["away_team_id"],
                df["home_team_id"],
            )
        else:
            df["opp_team_id"] = np.nan

        default_cols = {
            "opp_obp_30d": np.nan,
            "opp_slg_30d": np.nan,
            "opp_k_pct_30d": np.nan,
        }
        for col, val in default_cols.items():
            df[col] = val

        if opponent_batting is not None and not df["opp_team_id"].isna().all():
            batting = opponent_batting.copy()
            batting["game_date"] = pd.to_datetime(batting["game_date"]).dt.date
            batting = batting[batting["game_date"] <= as_of_date]

            unique_matchups = df[["opp_team_id", "game_date"]].drop_duplicates()
            unique_matchups = unique_matchups.dropna(subset=["opp_team_id"])

            opp_stats = {}
            for _, row in unique_matchups.iterrows():
                tid = row["opp_team_id"]
                gd = row["game_date"]
                cutoff = gd - timedelta(days=1)
                start = gd - timedelta(days=31)

                mask = (
                    (batting["team_id"] == tid)
                    & (batting["game_date"] >= start)
                    & (batting["game_date"] <= cutoff)
                )
                subset = batting.loc[mask]
                if subset.empty:
                    continue

                ab_total = subset["ab"].fillna(0).sum()
                h_total = subset["h"].fillna(0).sum()
                bb_total = subset["bb"].fillna(0).sum()
                hbp_total = subset["hbp"].fillna(0).sum() if "hbp" in subset.columns else 0
                sf_total = subset["sf"].fillna(0).sum() if "sf" in subset.columns else 0
                so_total = subset["so"].fillna(0).sum()
                doubles = subset["doubles"].fillna(0).sum() if "doubles" in subset.columns else 0
                triples = subset["triples"].fillna(0).sum() if "triples" in subset.columns else 0
                hr_total = subset["hr"].fillna(0).sum()

                pa_total = ab_total + bb_total + hbp_total + sf_total
                if pa_total > 0:
                    obp = (h_total + bb_total + hbp_total) / pa_total
                    k_pct = so_total / pa_total
                else:
                    obp = np.nan
                    k_pct = np.nan

                if ab_total > 0:
                    tb = h_total + doubles + 2 * triples + 3 * hr_total
                    slg = tb / ab_total
                else:
                    slg = np.nan

                opp_stats[(tid, gd)] = {
                    "opp_obp_30d": obp,
                    "opp_slg_30d": slg,
                    "opp_k_pct_30d": k_pct,
                }

            if opp_stats:
                opp_df = pd.DataFrame.from_dict(opp_stats, orient="index")
                opp_df.index = pd.MultiIndex.from_tuples(
                    opp_df.index, names=["opp_team_id", "game_date"]
                )
                opp_df = opp_df.reset_index()

                df = df.merge(
                    opp_df,
                    on=["opp_team_id", "game_date"],
                    how="left",
                    suffixes=("", "_new"),
                )
                for col in default_cols:
                    new_col = f"{col}_new"
                    if new_col in df.columns:
                        df[col] = df[new_col].combine_first(df[col])
                        df = df.drop(columns=[new_col])

        df["opp_missing"] = (df["opp_obp_30d"].isna()).astype(int)
        df["opp_obp_30d"] = df["opp_obp_30d"].fillna(0.320)
        df["opp_slg_30d"] = df["opp_slg_30d"].fillna(0.400)
        df["opp_k_pct_30d"] = df["opp_k_pct_30d"].fillna(0.22)

        feature_cols = [c for c in df.columns if c.startswith("opp_")]
        return df[["player_id", "game_pk"] + feature_cols].copy()
